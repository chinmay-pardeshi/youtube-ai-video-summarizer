# 🎥 YouTube Transcript Summarizer with Gemini AI

This tool uses **Google Gemini AI** to turn **YouTube video transcripts** into structured, bullet-point summaries — perfect for students, professionals, or content curators.

---

## 🚀 Features

- 🔗 Paste a YouTube video URL
- 📜 Automatically fetch transcript using `youtube_transcript_api`
- 🧠 Generate smart summaries (max 250 words)
- ✅ Powered by `gemini-1.5-flash`
- 🎛️ Clean UI with Streamlit

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Google Generative AI
- youtube-transcript-api
- dotenv

---

## 💻 How to Run

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/youtube-summarizer.git
cd youtube-summarizer
pip install -r requirements.txt


2. Add API Key
Create a .env file:
env 
GOOGLE_API_KEY=your_api_key_here

3. Run the App
bash 
streamlit run streamlit_app.py