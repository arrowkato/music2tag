from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI


def public_sample1():
    """キモいけど、音声ファイルでも image_urlで渡せる
    ref
    https://python.langchain.com/docs/integrations/llms/google_vertex_ai_palm/#using-audio-with-gemini-15-pro
    """
    media_message = {
        "type": "image_url",
        "image_url": {
            "url": "gs://cloud-samples-data/generative-ai/audio/pixel.mp3",
        },
    }

    text_message = {
        "type": "text",
        "text": "何を喋っているかを文字起こしてください。",
    }
    message = HumanMessage(content=[media_message, text_message])

    llm = ChatVertexAI(model_name="gemini-1.5-pro-002")
    response = llm.invoke([message])
    print(response)


def public_sample2():
    """もうちょっとまともぽいサンプル"""

    llm = ChatVertexAI(model="gemini-1.5-flash-002")
    h_msg = HumanMessage(
        [
            "何を喋っているかを文字起こしてください。",
            {
                "type": "media",
                "file_uri": "gs://cloud-samples-data/generative-ai/audio/pixel.mp3",
                "mime_type": "audio/mpeg",
            },
        ]
    )

    response = llm.invoke([h_msg])
    # 注: 日本語でプロンプトを書くと、日本語に翻訳されてました。
    print(response.content)


if __name__ == "__main__":
    # あってもいいはずなんですが、直接音声ファイルを渡すサンプルは見つけられなかったです。
    # public_sample1()
    public_sample2()
