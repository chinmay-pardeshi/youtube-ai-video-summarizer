# 📽️ YouTube Transcript Summarizer with Gemini AI

This AI-powered app lets you **paste any YouTube video URL**, fetch its **transcript**, and generate a **concise summary** using Google Gemini. Ideal for students, researchers, content creators, and busy professionals.

Built with:  
- 🧠 **Google Gemini (gemini-1.5-flash)**  
- 🛠️ **Streamlit**  
- 🧾 **youtube-transcript-api**  
- 🌱 **dotenv**

---

## 🎥 Demo

![App Demo](https://github.com/chinmay-pardeshi/youtube-ai-video-summarizer/blob/main/demo/youtube-ai-video-summarizer-gif.gif)

> Instantly summarize YouTube videos into bite-sized takeaways with Gemini AI!

---

## 🚀 Features

- 🔗 Paste a YouTube video URL  
- 📜 Automatically fetch transcript using `youtube-transcript-api`  
- 🧠 Generate smart summaries (max 250 words) using Gemini AI  
- 🖥️ Clean and responsive UI built with Streamlit  
- ⚠️ Error handling for videos without transcripts  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Google Generative AI (Gemini API)  
- youtube-transcript-api  
- dotenv  

---

## 💻 How to Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/yourusername/youtube-summarizer.git
cd youtube-summarizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your environment variables
# Create a `.env` file with:
GOOGLE_API_KEY=your_google_api_key

# 4. Run the Streamlit app
streamlit run streamlit_app.py
