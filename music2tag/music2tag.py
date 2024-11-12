from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field


class MusicFeature(BaseModel):
    music_feature_word1: str = Field(..., description="feature of the music. mood")
    music_feature_word2: str = Field(
        ..., description="feature of the music.rhythm or tempo"
    )
    music_feature_word3: str = Field(
        ..., description="music genres. e.g., pop, Bossa Nova, Jazz and so on"
    )
    used_instrument1: str = Field(
        ...,
        description="used string instruments in the music. e.g., guitar, piano, violin, harp, Double bass, etc",
    )
    used_instrument2: str = Field(
        ...,
        description="used percussion instruments in the music. e.g., drum, snare drum, cymbals, maracas, xylophone, etc",
    )


def music2tags(audio_uri_on_gcs: str):
    """もうちょっとまともぽいサンプル"""

    llm = ChatVertexAI(model="gemini-1.5-pro-002").with_structured_output(MusicFeature)

    h_msg = HumanMessage(
        [
            "この曲特徴を説明して下さい。ムード、リズム/テンポ、ジャンル、使われている弦楽器、使われている打楽器の系5項目です。",
            {
                "type": "media",
                "file_uri": audio_uri_on_gcs,
                "mime_type": "audio/mpeg",
            },
        ]
    )
    response = llm.invoke([h_msg])
    print(response)
    # music_feature_word1='Hopeful' music_feature_word2='Moderate' music_feature_word3='Classical' used_instrument1='Harp' used_instrument2='Snare drum'


if __name__ == "__main__":
    # privateなバケットでも大丈夫です。あらかじめファイルをアップロードして下さい
    uri = "gs://your_bucket/sample_music.mp3"
    music2tags(uri)
