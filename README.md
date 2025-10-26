# 📄 Chat with Your PDF (Groq + LangChain + Streamlit)

A tiny RAG app to upload a PDF and chat with it. Uses **Streamlit**, **LangChain**, **Chroma**, and **Groq** (no OpenAI).

## Features
- Upload a PDF and ask multi-turn questions
- Choose Groq model: `llama-3.1-8b-instant` or `llama-3.3-70b-versatile`
- Control retriever `top_k`
- See supporting chunks used for answers

## Screenshots
<p align="center">
  <img src="assets/llama8b.png" width="70%" />
</p>
<p align="center">
  <img src="assets/llama70b.png" width="70%" />
</p>
<p align="center">
  <img src="assets/settings.png" width="50%" />
</p>

## Requirements
- Python 3.9+
- Groq API key (free): https://console.groq.com/

## Setup
```bash
cd web_app
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
