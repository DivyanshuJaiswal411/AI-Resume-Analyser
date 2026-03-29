# ============================================================
#  AI Resume & Job-Description Analyser
#  Gemini API + LangChain + Streamlit
#  Author : Divyanshu Jaiswal  |  IIT (BHU) Varanasi
# ============================================================
#
#  SETUP:
#    pip install streamlit langchain langchain-google-genai
#                google-generativeai pypdf2 python-docx
#
#  RUN:
#    streamlit run app.py
#
#  Then paste your GOOGLE_API_KEY when prompted in the sidebar
# ============================================================

import streamlit as st
import os, re, json, textwrap
from io import BytesIO

# PDF / DOCX text extraction
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyser",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header{font-size:2.2rem;font-weight:800;color:#1B3A6B;margin-bottom:0}
    .sub-header{color:#555;margin-top:0;margin-bottom:1.5rem}
    .score-box{background:#f0f4ff;border-left:5px solid #1B3A6B;
               padding:1rem 1.5rem;border-radius:6px;margin:1rem 0}
    .green{color:#16a34a;font-weight:700}
    .red{color:#dc2626;font-weight:700}
    .section-title{font-size:1.1rem;font-weight:700;color:#1B3A6B;
                   border-bottom:2px solid #1B3A6B;padding-bottom:4px;margin-top:1.5rem}
    .chat-bubble-user{background:#e8edf8;padding:0.7rem 1rem;
                      border-radius:12px 12px 4px 12px;margin:0.5rem 0}
    .chat-bubble-ai{background:#f9fafb;border:1px solid #e5e7eb;
                    padding:0.7rem 1rem;border-radius:12px 12px 12px 4px;margin:0.5rem 0}
</style>
""", unsafe_allow_html=True)

# ── FILE EXTRACTION HELPERS ──────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PyPDF2 is None:
        return "[PyPDF2 not installed — pip install pypdf2]"
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(file_bytes: bytes) -> str:
    if DocxDocument is None:
        return "[python-docx not installed — pip install python-docx]"
    doc = DocxDocument(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    elif name.endswith(".docx"):
        return extract_text_from_docx(data)
    else:
        return data.decode("utf-8", errors="ignore")

# ── GEMINI LLM FACTORY ───────────────────────────────────────
@st.cache_resource
def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True
    )

# ── ANALYSIS PROMPT ──────────────────────────────────────────
ANALYSIS_SYSTEM = """You are an expert ATS (Applicant Tracking System) analyst and career coach.
When given a resume and a job description, you produce a STRUCTURED JSON analysis with these exact keys:
{
  "ats_score": <integer 0-100>,
  "verdict": "<one line summary>",
  "matched_keywords": ["kw1","kw2",...],
  "missing_keywords": ["kw1","kw2",...],
  "strengths": ["point1","point2",...],
  "weaknesses": ["point1","point2",...],
  "rewrite_tips": ["tip1","tip2",...],
  "suggested_summary": "<rewritten professional summary tailored to the JD>"
}
Return ONLY the JSON object — no markdown fences, no extra text."""

def analyse(llm, resume_text: str, jd_text: str) -> dict:
    prompt = f"""RESUME:\n{resume_text}\n\n---\nJOB DESCRIPTION:\n{jd_text}"""
    msgs = [
        SystemMessage(content=ANALYSIS_SYSTEM),
        HumanMessage(content=prompt)
    ]
    resp = llm.invoke(msgs)
    raw  = resp.content.strip()
    # Strip markdown fences if model adds them anyway
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password",
                            help="Get free key at https://aistudio.google.com/app/apikey")
    st.markdown("---")
    st.markdown("### 📋 How to use")
    st.markdown("""
1. Enter your **Gemini API key**
2. Upload your **resume** (PDF/DOCX/TXT)
3. Paste the **job description**
4. Click **Analyse**
5. Ask follow-up questions in the **chat**
""")
    st.markdown("---")
    st.caption("Built by Divyanshu Jaiswal · IIT (BHU) Varanasi")

# ── HEADER ───────────────────────────────────────────────────
st.markdown('<p class="main-header">🎯 AI Resume & JD Analyser</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ATS scoring · skill gap analysis · rewrite suggestions · chat Q&A</p>',
            unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────
if "analysis"      not in st.session_state: st.session_state.analysis      = None
if "resume_text"   not in st.session_state: st.session_state.resume_text   = ""
if "jd_text"       not in st.session_state: st.session_state.jd_text       = ""
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []

# ── INPUT COLUMNS ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📄 Upload Resume")
    resume_file = st.file_uploader("PDF, DOCX, or TXT", type=["pdf","docx","txt"])
    if resume_file:
        st.session_state.resume_text = extract_text(resume_file)
        with st.expander("Preview extracted text"):
            st.text(st.session_state.resume_text[:1500] + "…")

with col2:
    st.markdown("#### 💼 Paste Job Description")
    st.session_state.jd_text = st.text_area(
        "Paste the full JD here", height=220,
        value=st.session_state.jd_text,
        placeholder="Paste the job description you are targeting…"
    )

# ── ANALYSE BUTTON ────────────────────────────────────────────
st.markdown("---")
run_btn = st.button("🚀 Analyse Resume", type="primary", use_container_width=True)

if run_btn:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not st.session_state.resume_text:
        st.error("Please upload a resume file.")
    elif not st.session_state.jd_text.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analysing with Gemini AI…"):
            try:
                llm = get_llm(api_key)
                result = analyse(llm, st.session_state.resume_text, st.session_state.jd_text)
                st.session_state.analysis = result
                st.session_state.chat_history = []   # reset chat on new analysis
            except json.JSONDecodeError:
                st.error("Gemini returned unexpected output. Try again.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── RESULTS DASHBOARD ─────────────────────────────────────────
if st.session_state.analysis:
    r = st.session_state.analysis
    score = r.get("ats_score", 0)
    colour = "#16a34a" if score >= 70 else "#d97706" if score >= 50 else "#dc2626"

    st.markdown(f"""
    <div class="score-box">
      <span style="font-size:2.8rem;font-weight:900;color:{colour}">{score}/100</span>
      <span style="font-size:1.1rem;color:#333;margin-left:1rem">{r.get('verdict','')}</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<p class="section-title">✅ Matched Keywords</p>', unsafe_allow_html=True)
        for kw in r.get("matched_keywords", []):
            st.markdown(f'<span class="green">✓</span> {kw}', unsafe_allow_html=True)

    with c2:
        st.markdown('<p class="section-title">❌ Missing Keywords</p>', unsafe_allow_html=True)
        for kw in r.get("missing_keywords", []):
            st.markdown(f'<span class="red">✗</span> {kw}', unsafe_allow_html=True)

    with c3:
        st.markdown('<p class="section-title">💪 Strengths</p>', unsafe_allow_html=True)
        for s in r.get("strengths", []):
            st.markdown(f"• {s}")

    st.markdown('<p class="section-title">⚠️ Weaknesses & Gaps</p>', unsafe_allow_html=True)
    for w in r.get("weaknesses", []):
        st.markdown(f"• {w}")

    st.markdown('<p class="section-title">🛠️ Rewrite Tips</p>', unsafe_allow_html=True)
    for t in r.get("rewrite_tips", []):
        st.markdown(f"→ {t}")

    st.markdown('<p class="section-title">✍️ AI-Suggested Professional Summary</p>', unsafe_allow_html=True)
    st.info(r.get("suggested_summary", ""))

# ── MULTI-TURN CHAT ──────────────────────────────────────────
if st.session_state.analysis:
    st.markdown("---")
    st.markdown("### 💬 Ask Follow-Up Questions")
    st.caption("Ask anything about your resume, the JD match, how to improve, interview tips…")

    CHAT_SYSTEM = f"""You are an expert career coach and ATS specialist.
You already analysed a resume against a job description.

RESUME SUMMARY (first 1200 chars):
{st.session_state.resume_text[:1200]}

JOB DESCRIPTION SUMMARY (first 800 chars):
{st.session_state.jd_text[:800]}

ATS ANALYSIS RESULT:
{json.dumps(st.session_state.analysis, indent=2)}

Answer the user's follow-up questions specifically, concisely, and helpfully.
Use bullet points when listing items. Be direct and actionable."""

    # Render chat history
    for msg in st.session_state.chat_history:
        role, text = msg
        if role == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">🤖 {text}</div>', unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask a question about your resume or the role…")
    if user_input and api_key:
        st.session_state.chat_history.append(("user", user_input))

        # Build message list
        msgs = [SystemMessage(content=CHAT_SYSTEM)]
        for role, text in st.session_state.chat_history[:-1]:
            if role == "user":
                msgs.append(HumanMessage(content=text))
            else:
                msgs.append(AIMessage(content=text))
        msgs.append(HumanMessage(content=user_input))

        with st.spinner("Thinking…"):
            llm  = get_llm(api_key)
            resp = llm.invoke(msgs)
            answer = resp.content.strip()

        st.session_state.chat_history.append(("ai", answer))
        st.rerun()
    elif user_input and not api_key:
        st.warning("Please add your API key in the sidebar to use the chat.")
