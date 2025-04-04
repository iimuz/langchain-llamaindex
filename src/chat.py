"""Ollamaでローカルホストされたサーバーを利用してチャットするLangChainアプリケーション."""

import argparse
import hashlib
import json
import os

import chromadb
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM

# LlamaIndex関連のインポート
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする."""
    parser = argparse.ArgumentParser(
        description="LangChainを使用したOllamaチャットアプリケーション"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hf.co/mmnga/sarashina2.2-3b-instruct-v0.1-gguf:Q4_K_M",
        help="使用するOllamaモデルの名前",
    )
    parser.add_argument(
        "--url", type=str, default="http://localhost:11434", help="OllamaサーバーのURL"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="生成時の温度パラメータ"
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
        help="埋め込みモデルの名前",
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="./documents",
        help="インデックスを作成するドキュメントのディレクトリ",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="./index_storage",
        help="インデックスを保存するディレクトリ",
    )
    parser.add_argument("--rag-mode", action="store_true", help="RAGモードを有効にする")
    parser.add_argument(
        "--force-rebuild", action="store_true", help="インデックスを強制的に再構築する"
    )
    return parser.parse_args()


def create_llm(model_name: str, url: str, temperature: float) -> OllamaLLM:
    """OllamaモデルのLLMインスタンスを作成する."""
    return OllamaLLM(
        model=model_name,
        base_url=url,
        temperature=temperature,
        callbacks=[StreamingStdOutCallbackHandler()],
    )


def create_conversation_chain(llm: OllamaLLM) -> RunnableWithMessageHistory:
    """通常の会話チェーンを作成する."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "以下は人間とAIの親切な会話です。AIは会話の文脈を理解し、詳細かつ役立つ回答を提供します。",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="history",
    )


def create_rag_conversation_chain(llm: OllamaLLM) -> RunnableWithMessageHistory:
    """RAG用の会話チェーンを作成する."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """以下は人間とAIの親切な会話です。AIは会話の文脈を理解し、詳細かつ役立つ回答を提供します。
                関連文書の情報が提供されている場合は、それを参考にして回答してください。
                ただし、関連文書に情報がない場合は、あなた自身の知識で回答してください。""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            (
                "system",
                """### 関連文書:
                {context}

                上記の関連文書と自分の知識を組み合わせて、ユーザーの質問に答えてください。"""
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="history",
    )


def calculate_directory_hash(directory: str) -> str:
    """ディレクトリ内のファイルのハッシュ値を計算する."""
    if not os.path.exists(directory):
        return ""

    hash_dict = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    hash_dict[file_path] = file_hash
            except Exception as e:
                print(f"ファイル '{file_path}' のハッシュ計算中にエラー: {e}")

    # 辞書をソートしてから文字列に変換してハッシュ化
    sorted_dict_str = json.dumps(hash_dict, sort_keys=True)
    return hashlib.md5(sorted_dict_str.encode()).hexdigest()


def save_hash_info(hash_value: str, index_dir: str) -> None:
    """ハッシュ情報をファイルに保存する."""
    os.makedirs(index_dir, exist_ok=True)
    hash_file = os.path.join(index_dir, "document_hash.txt")
    with open(hash_file, "w") as f:
        f.write(hash_value)


def load_hash_info(index_dir: str) -> str:
    """保存されたハッシュ情報を読み込む."""
    hash_file = os.path.join(index_dir, "document_hash.txt")
    if not os.path.exists(hash_file):
        return ""

    with open(hash_file) as f:
        return f.read().strip()


def should_rebuild_index(
    documents_dir: str, index_dir: str, force_rebuild: bool
) -> bool:
    """インデックスを再構築する必要があるか判断する."""
    if force_rebuild:
        print("強制再構築フラグが設定されています。インデックスを再構築します。")
        return True

    # Chromaディレクトリの存在確認
    chroma_dir = os.path.join(index_dir, "chroma_db")
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        print("インデックスが存在しません。新規作成します。")
        return True

    # ドキュメントのハッシュを計算
    current_hash = calculate_directory_hash(documents_dir)
    if not current_hash:
        print("ドキュメントディレクトリが空または存在しません。")
        return False

    # 保存されたハッシュと比較
    saved_hash = load_hash_info(index_dir)
    if current_hash != saved_hash:
        print("ドキュメントに変更が検出されました。インデックスを再構築します。")
        return True

    print("ドキュメントに変更はありません。既存のインデックスを使用します。")
    return False


def create_vector_index(
    documents_dir: str,
    embed_model_name: str,
    url: str,
    temperature: float,
    index_dir: str,
    force_rebuild: bool = False,
) -> VectorStoreIndex | None:
    """RAGのためのベクトルインデックスを作成する."""
    # ドキュメントディレクトリが存在するか確認
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"ディレクトリ '{documents_dir}' が存在しなかったため作成しました。")
        print("このディレクトリにドキュメントを配置してください。")
        return None

    # 埋め込みモデルの設定
    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url=url,
        temperature=temperature,
        request_timeout=120.0,
    )

    # グローバル設定
    Settings.llm = None
    Settings.embed_model = embed_model

    # インデックスの保存先を確保
    os.makedirs(index_dir, exist_ok=True)
    chroma_dir = os.path.join(index_dir, "chroma_db")
    collection_name = "document_collection"

    # インデックスの再構築が必要か確認
    if should_rebuild_index(documents_dir, index_dir, force_rebuild):
        # ドキュメントの読み込み
        documents = SimpleDirectoryReader(documents_dir).load_data()
        if not documents:
            print(f"ディレクトリ '{documents_dir}' にドキュメントが見つかりません。")
            return None

        print(
            f"{len(documents)}個のドキュメントを読み込みました。インデックスを構築中..."
        )

        # 既存のChromaDBがあれば削除して新規作成
        if os.path.exists(chroma_dir):
            try:
                import shutil

                shutil.rmtree(chroma_dir)
                print(f"既存のChromaデータベースを削除しました: {chroma_dir}")
            except Exception as e:
                print(f"既存のChromaデータベース削除中にエラー: {e}")

        # Chroma クライアントとコレクションをセットアップ
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        try:
            chroma_client.delete_collection(collection_name)
            print(f"既存のコレクション '{collection_name}' を削除しました。")
        except:
            pass  # コレクションが存在しない場合は無視

        chroma_collection = chroma_client.create_collection(collection_name)

        # ChromaVectorStoreの設定
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # インデックスの作成
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

        # ドキュメントハッシュを保存
        current_hash = calculate_directory_hash(documents_dir)
        save_hash_info(current_hash, index_dir)

        print(f"インデックスを '{chroma_dir}' に保存しました。")
        return index
    # 既存のインデックスをロード
    try:
        print(f"既存のインデックスを '{chroma_dir}' から読み込みます。")
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        chroma_collection = chroma_client.get_collection(collection_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
    except Exception as e:
        print(f"インデックスの読み込み中にエラーが発生しました: {e}")
        print("インデックスを再構築します...")
        # エラーが発生した場合は再構築を試みる
        return create_vector_index(
            documents_dir,
            embed_model_name,
            url,
            temperature,
            index_dir,
            force_rebuild=True,
        )


def query_vector_index(
    index: VectorStoreIndex, query: str, similarity_top_k: int = 3
) -> dict:
    """ベクトルインデックスにクエリを実行し、関連ドキュメントを取得する"""
    if not index:
        return {"text": "", "sources": []}

    # クエリエンジンの作成
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, similarity_cutoff=0.7
    )

    # クエリの実行
    response = query_engine.query(query)

    # ソースノードの抽出
    sources = []
    if hasattr(response, "source_nodes"):
        for i, node in enumerate(response.source_nodes):
            if isinstance(node, NodeWithScore):
                source = {
                    "score": node.score if hasattr(node, "score") else "不明",
                    "file_name": node.node.metadata.get("file_name", "不明") if hasattr(node.node, "metadata") else "不明",
                    "file_path": node.node.metadata.get("file_path", "不明") if hasattr(node.node, "metadata") else "不明",
                    "text": node.node.text if hasattr(node.node, "text") else "",
                }
                sources.append(source)

    return {
        "text": str(response),
        "sources": sources,
    }


def format_context_from_query_result(query_result: dict) -> str:
    """検索結果からコンテキストを整形する"""
    context = ""

    # 関連テキストがあれば追加
    if query_result.get("text"):
        context += f"{query_result['text']}\n\n"

    # ソース情報があれば追加
    if query_result.get("sources") and len(query_result["sources"]) > 0:
        context += "参照情報:\n"
        for i, source in enumerate(query_result["sources"]):
            context += f"{i+1}. ファイル: {source.get('file_name', '不明')}\n"
            if source.get("text"):
                # テキストの長さを制限（必要に応じて調整）
                text = source["text"]
                if len(text) > 300:
                    text = text[:300] + "..."
                context += f"   内容: {text}\n"
            context += f"   関連度: {source.get('score', '不明')}\n\n"

    return context


def chat_loop(
    conversation: RunnableWithMessageHistory,
    rag_conversation: RunnableWithMessageHistory,
    index: VectorStoreIndex | None = None,
) -> None:
    """チャットのメインループ."""
    rag_mode = index is not None
    session_id = "default"  # セッションID

    if rag_mode:
        print("RAGモードが有効です。ドキュメントベースでの回答を提供します。")
    else:
        print("通常モードで起動しています。RAG機能は無効です。")

    print(
        "チャットを開始します。終了するには 'exit' または 'quit' と入力してください。"
    )

    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["exit", "quit"]:
            print("チャットを終了します。")
            break

        try:
            if rag_mode:
                # インデックスからコンテキストを検索
                print("\n検索中...", end="", flush=True)
                query_result = query_vector_index(index, user_input)
                print("\r" + " " * 10 + "\r", end="")  # 「検索中...」を消す

                # コンテキストを整形
                context = format_context_from_query_result(query_result)

                # RAG用の会話チェーンを実行
                rag_conversation.invoke(
                    {"input": user_input, "context": context},
                    config={"configurable": {"session_id": session_id}},
                )

                # 参照情報があれば表示
                if query_result.get("sources") and len(query_result["sources"]) > 0:
                    print("\n参照ドキュメント:")
                    for i, source in enumerate(query_result["sources"]):
                        print(f"  {i+1}. {source.get('file_name', '不明')}")
            else:
                # 通常の会話チェーンを実行
                conversation.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            print("通常の会話モードにフォールバックします...")
            conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )


def main() -> None:
    """メイン関数."""
    args = parse_args()

    # LLMインスタンスの作成
    llm = create_llm(args.model, args.url, args.temperature)

    # 通常の会話チェーンの作成
    conversation = create_conversation_chain(llm)

    # RAG用の会話チェーンとインデックスの作成
    rag_conversation = create_rag_conversation_chain(llm)
    index = None

    if args.rag_mode:
        index = create_vector_index(
            args.documents,
            args.embed_model,
            args.url,
            args.temperature,
            args.index_dir,
            args.force_rebuild,
        )

    # チャットループの開始
    chat_loop(conversation, rag_conversation, index)


if __name__ == "__main__":
    main()
