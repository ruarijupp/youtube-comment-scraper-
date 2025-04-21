import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from dotenv import load_dotenv

load_dotenv()  # Load keys from .env

app = FastAPI()
openai_client = OpenAI()

# === Pydantic Input Schema ===
class VideoRequest(BaseModel):
    url: str


# === Core Functions ===
def get_video_id(url: str):
    return url.split("v=")[-1].split("&")[0]

def fetch_transcript(video_url: str) -> str:
    video_id = get_video_id(video_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([entry['text'] for entry in transcript])
    return full_text

def summarize_transcript(text: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarize YouTube transcripts as cleanly and concisely as possible."},
            {"role": "user", "content": f"Summarize this transcript:\n\n{text}"}
        ],
        max_tokens=500,
        temperature=0.5
    )
    return response.choices[0].message.content

def generate_voice(text: str, filename="summary.mp3") -> str:
    eleven_key = os.getenv("ELEVEN_API_KEY")
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice

    if not eleven_key:
        raise ValueError("ELEVEN_API_KEY not set in .env")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": eleven_key,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs error: {response.text}")

    with open(filename, "wb") as f:
        f.write(response.content)

    return f"/static/{filename}"


# === Routes ===

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/summarize")
def summarize_video(req: VideoRequest):
    try:
        transcript = fetch_transcript(req.url)
        summary = summarize_transcript(transcript)
        audio_path = generate_voice(summary)
        return {
            "summary": summary,
            "transcript_excerpt": transcript[:2000] + "...",
            "mp3_url": audio_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
