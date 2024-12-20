from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field


# 変数は適当です。
class MovieFeature(BaseModel):
    target: str = Field(..., description="Identify who the commercial is targeting.")
    quality_of_the_video: str = Field(..., description="feature of the music. mood")
    storytelling: str = Field(
        ...,
        description="Identify what message or concept the commercial is trying to convey.",
    )
    quality_of_the_visuals: str = Field(
        ...,
        description="Evaluates the quality of the visuals (image quality, lighting, camera work, etc.",
    )

    visual_elements: str = Field(
        ...,
        description="We check if the visual elements (animation, effects, colors, etc.) complement the message.",
    )

    bgm_category: str = Field(
        ..., description="music genres. e.g., pop, Bossa Nova, Jazz and so on"
    )


# def music2tags(audio_uri_on_gcs: str):
#     pass


def movie2tag(movie_uri_on_gcs: str, model_name: str, prompt: str):
    # Use Gemini 1.5 Pro
    llm = ChatVertexAI(
        model=model_name,
        temparature=0.0,
    ).with_structured_output(MovieFeature)

    # Prepare input for model consumption
    media_message = {
        "type": "image_url",
        "image_url": {
            "url": movie_uri_on_gcs,
        },
    }

    text_message = {
        "type": "text",
        "text": prompt,
    }

    message = HumanMessage(content=[media_message, text_message])

    # invoke a model response
    result: MovieFeature = llm.invoke([message])
    print(result)


if __name__ == "__main__":
    # privateなバケットでも大丈夫です。あらかじめファイルをアップロードして下さい
    uri = "gs://cloud-samples-data/generative-ai/video/pixel8.mp4"
    movie2tag(
        movie_uri_on_gcs=uri,
        model_name="gemini-2.0-flash-exp",
        prompt="このCMの動画の特徴をタグ付けしてください。",
    )
