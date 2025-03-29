"""Ollamaでローカルホストされたサーバーを利用してチャットするLangChainアプリケーション."""

import argparse

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする."""
    parser = argparse.ArgumentParser(
        description="LangChainを使用したOllamaチャットアプリケーション"
    )
    parser.add_argument(
        "--model", type=str, default="hf.co/mmnga/sarashina2.2-3b-instruct-v0.1-gguf:Q4_K_M", help="使用するOllamaモデルの名前"
    )
    parser.add_argument(
        "--url", type=str, default="http://localhost:11434", help="OllamaサーバーのURL"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="生成時の温度パラメータ"
    )
    return parser.parse_args()


def create_llm(model_name: str, url: str, temperature: float) -> Ollama:
    """OllamaモデルのLLMインスタンスを作成する."""
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return Ollama(
        model=model_name,
        base_url=url,
        temperature=temperature,
        callback_manager=callback_manager,
    )


def create_conversation_chain(llm: Ollama) -> ConversationChain:
    """会話チェーンを作成する."""
    template = """以下は人間とAIの親切な会話です。
AIは会話の文脈を理解し、詳細かつ役立つ回答を提供します。

現在の会話:
{history}
人間: {input}
AI: """

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    memory = ConversationBufferMemory(return_messages=True)

    return ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True,
    )


def chat_loop(conversation: ConversationChain) -> None:
    """チャットのメインループ."""
    print(
        "チャットを開始します。終了するには 'exit' または 'quit' と入力してください。"
    )

    while True:
        user_input = input("\n> ")
        if user_input.lower() in ["exit", "quit"]:
            print("チャットを終了します。")
            break

        # 応答はコールバックによって自動的にストリーミング表示されます
        conversation.predict(input=user_input)


def main() -> None:
    """メイン関数."""
    args = parse_args()

    # LLMインスタンスの作成
    llm = create_llm(args.model, args.url, args.temperature)

    # 会話チェーンの作成
    conversation = create_conversation_chain(llm)

    # チャットループの開始
    chat_loop(conversation)


if __name__ == "__main__":
    main()
