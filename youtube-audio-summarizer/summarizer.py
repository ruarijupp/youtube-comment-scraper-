from dotenv import load_dotenv
load_dotenv()
import os
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

# === API Clients ===
client = OpenAI()  # Uses OPENAI_API_KEY from env


# === Transcript Logic ===
def get_video_id(url):
    try:
        return url.split("v=")[-1].split("&")[0]
    except Exception as e:
        raise ValueError("Invalid YouTube URL") from e

def fetch_transcript(video_url):
    try:
        video_id = get_video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript])
        return full_text
    except Exception as e:
        return f"Error fetching transcript: {e}"


# === Summarization Logic ===
def summarize_transcript(text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts."},
                {"role": "user", "content": f"Summarize this transcript:\n\n{text}"}
            ],
            max_tokens=500,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {e}"


# === ElevenLabs TTS Logic ===
def generate_voice(summary_text, voice_id="21m00Tcm4TlvDq8ikWAM", filename="summary.mp3"):
    api_key = os.getenv("ELEVEN_API_KEY")

    if not api_key:
        raise ValueError("ELEVEN_API_KEY not set.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    data = {
        "text": summary_text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"\n🔊 Voice summary saved as {filename}")
    else:
        print(f"❌ ElevenLabs failed: {response.status_code}\n{response.text}")


# === Main Runner ===
if __name__ == "__main__":
    url = input("Enter YouTube video URL: ").strip()

    print("\n⏳ Fetching transcript...")
    transcript = fetch_transcript(url)
    print("\n––– TRANSCRIPT (First 2K chars) –––\n")
    print(transcript[:2000] + '...\n')

    print("🤖 Summarizing with OpenAI...")
    summary = summarize_transcript(transcript)
    print("\n––– SUMMARY –––\n")
    print(summary)

    print("\n🎙 Generating voice with ElevenLabs...")
    generate_voice(summary)
