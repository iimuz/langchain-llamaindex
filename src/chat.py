"""Ollamaでローカルホストされたサーバーを利用してチャットするLangChainアプリケーション."""

import argparse
import os

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM

# LlamaIndex関連のインポート
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama as LlamaOllama


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする."""
    parser = argparse.ArgumentParser(
        description="LangChainを使用したOllamaチャットアプリケーション"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hf.co/mmnga/sarashina2.2-3b-instruct-v0.1-gguf:Q4_K_M",
        help="使用するOllamaモデルの名前"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:11434",
        help="OllamaサーバーのURL"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成時の温度パラメータ"
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
        help="埋め込みモデルの名前"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="./documents",
        help="インデックスを作成するドキュメントのディレクトリ"
    )
    parser.add_argument(
        "--rag-mode",
        action="store_true",
        help="RAGモードを有効にする"
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
    """会話チェーンを作成する."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "以下は人間とAIの親切な会話です。AIは会話の文脈を理解し、詳細かつ役立つ回答を提供します。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="input",
        history_messages_key="history",
    )


def create_rag_index(documents_dir: str, embed_model_name: str, model_name: str, url: str, temperature: float):
    """RAGのためのインデックスを作成する."""
    # ドキュメントディレクトリが存在するか確認
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"ディレクトリ '{documents_dir}' が存在しなかったため作成しました。")
        print("このディレクトリにドキュメントを配置してください。")
        return None

    # ドキュメントの読み込み
    documents = SimpleDirectoryReader(documents_dir).load_data()
    if not documents:
        print(f"ディレクトリ '{documents_dir}' にドキュメントが見つかりません。")
        return None

    print(f"{len(documents)}個のドキュメントを読み込みました。")

    # LlamaIndexのLLM設定
    llm = LlamaOllama(
        model=model_name,
        base_url=url,
        temperature=temperature,
        request_timeout=120.0,
    )

    # 埋め込みモデルの設定
    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url=url,
        temperature=temperature,
        request_timeout=120.0,
    )

    # グローバル設定
    Settings.llm = llm
    Settings.embed_model = embed_model

    # インデックスの作成
    return VectorStoreIndex.from_documents(documents)


def chat_loop(conversation: RunnableWithMessageHistory, index=None) -> None:
    """チャットのメインループ."""
    rag_mode = index is not None

    if rag_mode:
        print("RAGモードが有効です。ドキュメントベースでの回答を提供します。")
        query_engine = index.as_query_engine()

    print(
        "チャットを開始します。終了するには 'exit' または 'quit' と入力してください。"
    )

    session_id = "default"  # セッションIDを設定

    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["exit", "quit"]:
            print("チャットを終了します。")
            break

        if rag_mode:
            # RAGモードの場合、ドキュメントから情報を検索
            try:
                response = query_engine.query(user_input)
                print(f"\nRAG応答: {response}")
            except Exception as e:
                print(f"RAG検索中にエラーが発生しました: {e}")
                # エラーが発生した場合は通常の会話チェーンにフォールバック
                conversation.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        else:
            # 通常モードの場合
            conversation.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})


def main() -> None:
    """メイン関数."""
    args = parse_args()

    # LLMインスタンスの作成
    llm = create_llm(args.model, args.url, args.temperature)

    # 会話チェーンの作成
    conversation = create_conversation_chain(llm)

    # RAGモードの場合はインデックスを作成
    index = None
    if args.rag_mode:
        index = create_rag_index(
            args.documents,
            args.embed_model,
            args.model,
            args.url,
            args.temperature
        )

    # チャットループの開始
    chat_loop(conversation, index)


if __name__ == "__main__":
    main()
