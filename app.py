import streamlit as st
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path('.') / '.env')
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found in .env file.")
else:
    genai.configure(api_key=api_key)

prompt = """
You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. The transcript text will be appended here:
"""

def get_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query).get('v', [None])[0]
        elif query.path.startswith('/embed/'):
            return query.path.split('/')[2]
        elif query.path.startswith('/v/'):
            return query.path.split('/')[2]
    return None

def extract_transcript_details(youtube_video_url):
    try:
        video_id = get_video_id(youtube_video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL format.")
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except Exception as e:
        raise e

def generate_gemini_content(transcript_text, prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        MAX_LENGTH = 4000  # Limit to avoid exceeding quota
        transcript_trimmed = transcript_text[:MAX_LENGTH]
        response = model.generate_content(prompt + transcript_trimmed)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Streamlit interface
st.title("YouTube Transcript to Detailed Notes Converter")

youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = get_video_id(youtube_link)
    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    else:
        st.warning("Invalid YouTube link provided.")

if st.button("Get Detailed Notes"):
    try:
        transcript_text = extract_transcript_details(youtube_link)
        if not transcript_text.strip():
            st.warning("Transcript could not be fetched or is empty.")
        else:
            summary = generate_gemini_content(transcript_text, prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)
    except Exception as e:
        st.error(f"An error occurred: {e}")

