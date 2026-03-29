# AI Resume & Job-Description Analyser
**Gemini API + LangChain + Streamlit | Divyanshu Jaiswal | IIT (BHU) Varanasi**

## What It Does
- Upload your resume (PDF/DOCX/TXT) and paste any job description
- Gemini AI gives you an **ATS score (0–100)** instantly
- Shows matched keywords, missing keywords, strengths, and weaknesses
- Suggests a **rewritten professional summary** tailored to the JD
- **Multi-turn chat** — ask follow-up questions like "What skills should I add?" or "How do I answer the experience gap?"

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## API Key
Get a **free** Google Gemini API key at:
https://aistudio.google.com/app/apikey

Enter it in the sidebar when the app opens.

## Features
| Feature | Details |
|---------|---------|
| Resume upload | PDF, DOCX, TXT |
| ATS Score | 0–100 with colour coding |
| Keyword analysis | Matched + missing vs JD |
| Rewrite tips | Actionable bullet suggestions |
| AI summary | Auto-generated tailored summary |
| Multi-turn chat | Ask anything — full conversation memory |

## Tech Stack
- **Google Gemini 1.5 Flash** — LLM backbone
- **LangChain** — conversation chain + memory management
- **Streamlit** — web UI
- **PyPDF2 / python-docx** — resume text extraction
