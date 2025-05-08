import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

audio_file_path = "../resources/test.mp3"

with open(audio_file_path, "rb") as audio_file:
    encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

message = HumanMessage(
    content=[
        {"type": "text", "text": "Transcribe the audio."},
        {"type": "media", "data": encoded_audio, "mime_type": "audio/mp3" }
    ]
)
response = llm.invoke([message])
print(f"Response for audio: {response.content}")