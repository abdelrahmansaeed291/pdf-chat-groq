# 📄 Chat with Your PDF (Groq + LangChain + Streamlit)

A tiny RAG app to upload a PDF and chat with it.  
Uses **Streamlit**, **LangChain**, **Chroma**, and **Groq** (no OpenAI).

---

## 🚀 Features
- Upload a PDF and ask multi-turn questions
- Choose Groq model: `llama-3.1-8b-instant` or `llama-3.3-70b-versatile`
- Control retriever `top_k`
- See supporting chunks used for answers

---

## 📸 Screenshots

<p align="center">
  <img src="assets/Screenshot%202025-10-26%20123524.png" width="70%" alt="App view 1"/>
</p>
<p align="center">
  <img src="assets/Screenshot%202025-10-26%20123802.png" width="70%" alt="App view 2"/>
</p>
<p align="center">
  <img src="assets/Screenshot%202025-10-26%20123956.png" width="70%" alt="App view 3"/>
</p>
<p align="center">
  <img src="assets/Screenshot%202025-10-26%20124057.png" width="50%" alt="Settings panel"/>
</p>

---

## 🧰 Requirements
- Python 3.9+
- Groq API key (free): [https://console.groq.com/](https://console.groq.com/)

---

## ⚙️ Setup

```bash
cd web_app
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
