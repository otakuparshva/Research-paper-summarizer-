"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     INTELLIGENT RESEARCH PAPER SUMMARIZATION, RAG CHAT & SPEECH SYSTEM       ║
║     Production-Grade · Single-File Deployment                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  INSTALLATION  (run once in terminal)                                        ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  pip install streamlit google-generativeai pymupdf faiss-cpu                 ║
║              sentence-transformers rouge-score edge-tts gTTS                 ║
║              fpdf2 numpy requests python-dotenv                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import os
import re
import time
import base64
import logging
import asyncio
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# ─── Load .env FIRST ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv, find_dotenv, dotenv_values
    _dotenv_path = find_dotenv(usecwd=True)
    _dotenv_found = bool(_dotenv_path)
    load_dotenv(_dotenv_path, override=False)
    DOTENV_OK = True
except ImportError:
    DOTENV_OK = False
    _dotenv_found = False
    _dotenv_path = ""

# ─── Third Party ──────────────────────────────────────────────────────────────
import requests
import numpy as np
import streamlit as st

# PyMuPDF
try:
    import fitz
    PYMUPDF_OK = True
except ImportError:
    PYMUPDF_OK = False

# Google Gemini
try:
    import google.generativeai as genai
    from google.generativeai.types import generation_types
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

# FAISS
try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

# SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False

# ROUGE
try:
    from rouge_score import rouge_scorer
    ROUGE_OK = True
except ImportError:
    ROUGE_OK = False

# edge-tts
try:
    import edge_tts
    EDGE_TTS_OK = True
except ImportError:
    EDGE_TTS_OK = False

# gTTS fallback
try:
    from gtts import gTTS
    GTTS_OK = True
except ImportError:
    GTTS_OK = False

# fpdf2
try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ResearchAI")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL = "gemini-2.5-flash" 
GEMINI_FLASH_MODEL = "gemini-2.5-flash" 
MAX_OUTPUT_TOKENS = 4096

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

SARVAM_LANGUAGES = {
    "Hindi (हिन्दी)": "hi-IN", "Tamil (தமிழ்)": "ta-IN",
    "Telugu (తెలుగు)": "te-IN", "Kannada (ಕನ್ನಡ)": "kn-IN",
    "Malayalam (മലയാളം)": "ml-IN", "Bengali (বাংলা)": "bn-IN",
    "Gujarati (ગુજરાતી)": "gu-IN", "Marathi (मराठी)": "mr-IN",
    "Odia (ଓଡ଼ିଆ)": "od-IN", "Punjabi (ਪੰਜਾਬੀ)": "pa-IN",
    "English India": "en-IN",
}

SARVAM_SPEAKERS = {
    "hi-IN": ["meera", "pavithra", "maitreyi", "arvind", "amol"],
    "ta-IN": ["meera", "pavithra", "maitreyi", "arvind"],
    "te-IN": ["meera", "pavithra", "maitreyi", "arvind"],
    "kn-IN": ["meera", "pavithra", "arvind"],
    "ml-IN": ["meera", "pavithra", "arvind"],
    "bn-IN": ["amartya", "maitreyi", "arvind"],
    "gu-IN": ["meera", "pavithra", "arvind"],
    "mr-IN": ["meera", "amol", "arvind"],
    "od-IN": ["meera", "arvind"],
    "pa-IN": ["meera", "arvind"],
    "en-IN": ["neel", "harsh", "manisha", "vidya", "arjun", "siya"],
}

EDGE_TTS_VOICES = {
    "US Female (Aria)": "en-US-AriaNeural", "US Male (Guy)": "en-US-GuyNeural",
    "UK Female (Sonia)": "en-GB-SoniaNeural", "UK Male (Ryan)": "en-GB-RyanNeural",
    "AU Female (Natasha)": "en-AU-NatashaNeural", "IN Male (Prabhat)": "en-IN-PrabhatNeural",
}

CHAT_LANGUAGES = [
    "English", "Hindi (हिन्दी)", "Tamil (தமிழ்)", "Telugu (తెలుగు)", 
    "Kannada (ಕನ್ನಡ)", "Malayalam (മലയാളം)", "Bengali (বাংলা)", 
    "Gujarati (ગુજરાતી)", "Marathi (मराठी)", "Odia (ଓଡ଼ିଆ)", 
    "Punjabi (ਪੰਜਾਬੀ)", "Spanish", "French", "German", "Chinese", "Japanese"
]

MAX_FILE_SIZE_MB = 25
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 60
TOP_K_RETRIEVAL = 7
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SECTION_HEADERS = [
    "abstract", "introduction", "background", "related work",
    "literature review", "methodology", "methods", "approach",
    "proposed method", "system design", "framework",
    "experiment", "experimental setup", "experimental results",
    "results", "evaluation", "performance", "discussion",
    "analysis", "conclusion", "conclusions", "future work",
    "limitations", "acknowledgement", "acknowledgements", "references", "bibliography",
]

# ─── System Prompts ───
SUMMARIZATION_SYSTEM_PROMPT = """
You are a world-class academic research analyst.
Produce a structured professional research summary in EXACTLY this format:

**1. Research Objective**
- [Primary goal of the research]

**2. Problem Statement**
- [Research gap or challenge being addressed]

**3. Proposed Methodology**
- [Core methods, models, datasets, architectures used]

**4. Key Contributions**
- [Numbered list of novel contributions]

**5. Experimental Results**
- [Key findings with quantitative metrics where stated]

**6. Limitations**
- [Weaknesses or constraints in the paper]

**7. Future Work**
- [Directions proposed or logically inferred]

RULES:
- Total word count should be concise (around 400 words).
- You MUST output ALL 7 sections fully. Do not truncate or stop early.
- Use concise bullet points inside each numbered section.
- Professional academic tone throughout.
- NEVER fabricate statistics, names, or methods not present in the source.
- If a section cannot be determined, write: "Not explicitly stated in this paper."
- Begin directly with the format — no preamble.
"""

SECTION_SUMMARY_PROMPT = """
You are a precise academic summarizer.
Given a section of a research paper, write a concise 2-4 sentence summary.
Include only information explicitly present in the text. Never hallucinate.
"""

REFLECTION_PROMPT = """
You are a rigorous academic fact-checker.
Review the research paper summary for:
1. Factual consistency with the source excerpts
2. Hallucinated claims not found in source text
3. Logical coherence between sections

Return the CORRECTED summary in the EXACT SAME structured format.
You MUST return ALL 7 sections fully completed. Do not truncate the response under any circumstances.
If a claim has no source support, replace with: "[Unverifiable - removed]"
If no corrections are needed, return the original unchanged.
"""

CHAT_SYSTEM_PROMPT = """
You are an expert research assistant with deep knowledge of the uploaded research paper.
Answer questions accurately using ONLY the provided context excerpts.
Be specific, cite relevant sections, and maintain academic precision.
If the answer is not in the provided context, say so clearly.
Never fabricate information not present in the paper.
"""

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES & API SETUP
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    keys = ["chat_history", "result", "filename", "prefill"]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = [] if key == "chat_history" else None

def get_env_key(var: str) -> Optional[str]:
    val = os.environ.get(var, "").strip()
    return val if val else None

def configure_gemini(key: str) -> bool:
    if not GEMINI_OK or not key:
        return False
    try:
        genai.configure(api_key=key)
        return True
    except Exception as e:
        logger.error(f"Gemini config failed: {e}")
        return False

def clean_text_for_speech(text: str) -> str:
    """Strips Markdown symbols so TTS engines don't read asterisks or crash."""
    if not text: return ""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)     # Remove italic
    text = re.sub(r'__(.*?)__', r'\1', text)     # Remove bold
    text = re.sub(r'_(.*?)_', r'\1', text)       # Remove italic
    text = re.sub(r'#+\s*', '', text)            # Remove headers
    text = text.replace('•', '')                 # Remove bullet points
    text = text.replace('- ', '')                # Remove list dashes
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) # Remove links
    text = re.sub(r'\n+', '. ', text)            # Replace newlines with periods for breathing
    text = re.sub(r'\s+', ' ', text)             # Clean up double spaces
    return text.strip()

# ══════════════════════════════════════════════════════════════════════════════
# PDF EXTRACTION & CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf(pdf_bytes: bytes) -> Tuple[str, Dict[int, str], int]:
    if not PYMUPDF_OK:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise ValueError("PDF has no pages.")
        
        pages = {}
        parts = []
        for i in range(doc.page_count):
            raw = doc[i].get_text("text")
            cleaned = re.sub(r'[ \t]+', ' ', raw)
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
            pages[i + 1] = cleaned
            parts.append(cleaned)
        doc.close()
        
        full = "\n\n".join(parts)
        if len(full.split()) < 80:
            raise ValueError("Too little text extracted. Ensure it's not a scanned image PDF.")
        return full, pages, len(pages)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")

def detect_sections(text: str) -> Dict[str, str]:
    header_re = re.compile(
        r'^(?:\d+[\.\s\-]+)?(' + '|'.join(re.escape(h) for h in SECTION_HEADERS) + r')[\s\.:]*$',
        re.IGNORECASE | re.MULTILINE,
    )
    lines = text.split('\n')
    sections = {}
    current = "preamble"
    buf = []
    
    for line in lines:
        stripped = line.strip()
        m = header_re.match(stripped)
        if m and len(stripped) < 80:
            txt = "\n".join(buf).strip()
            if txt:
                sections[current] = txt
            current = m.group(1).lower().strip()
            buf = []
        else:
            buf.append(line)
            
    if buf:
        txt = "\n".join(buf).strip()
        if txt:
            sections[current] = txt

    named = [k for k in sections if k != "preamble"]
    if len(named) < 2:
        return _proportional_sections(text)

    if "abstract" not in sections and "preamble" in sections:
        m = re.search(r'abstract\s*[:\n]+(.+?)(?=\n\n|\Z)', sections["preamble"], re.IGNORECASE | re.DOTALL)
        if m:
            sections["abstract"] = m.group(1).strip()
    return sections

def _proportional_sections(text: str) -> Dict[str, str]:
    words = text.split()
    q = max(1, len(words) // 6)
    return {
        "introduction": " ".join(words[:q]),
        "related work": " ".join(words[q:2*q]),
        "methodology": " ".join(words[2*q:3*q]),
        "results": " ".join(words[3*q:4*q]),
        "discussion": " ".join(words[4*q:5*q]),
        "conclusion": " ".join(words[5*q:]),
    }

def semantic_chunk(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_words = 0
            overlap_chunk = []
            for s in reversed(current_chunk):
                overlap_chunk.insert(0, s)
                overlap_words += len(s.split())
                if overlap_words >= overlap:
                    break
            current_chunk = overlap_chunk
            current_length = sum(len(s.split()) for s in current_chunk)
            
        current_chunk.append(sentence)
        current_length += len(words)
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS + FAISS RAG
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_embedder():
    if not ST_OK: return None
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Embedder load failed: {e}")
        return None

def build_faiss_index(chunks: List[str], embedder) -> Tuple[Any, np.ndarray]:
    embs = embedder.encode(chunks, show_progress_bar=False, batch_size=32).astype(np.float32)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs

def retrieve_chunks(query: str, chunks: List[str], index, embedder, top_k: int = TOP_K_RETRIEVAL) -> List[str]:
    q = embedder.encode([query], show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(q)
    _, idxs = index.search(q, min(top_k, len(chunks)))
    return [chunks[i] for i in idxs[0] if 0 <= i < len(chunks)]

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI LLM CALLS
# ══════════════════════════════════════════════════════════════════════════════

def make_model(system_prompt: str, model_name: str = GEMINI_MODEL, temp: float = 0.3):
    cfg = genai.types.GenerationConfig(
        temperature=temp, top_p=0.95, top_k=40, max_output_tokens=MAX_OUTPUT_TOKENS
    )
    return genai.GenerativeModel(model_name=model_name, generation_config=cfg, system_instruction=system_prompt)

def gemini_call(model, prompt: str, retries: int = 4) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = model.generate_content(prompt)
            try:
                text = resp.text.strip()
                return text if text else "[Empty response received.]"
            except ValueError:
                return "[Content blocked by Gemini Safety Settings]"
        except Exception as e:
            error_msg = str(e).lower()
            if attempt == retries:
                raise RuntimeError(f"Gemini API failed after {retries} attempts. Last Error: {e}")
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                time.sleep(10 * attempt)
            else:
                time.sleep(2 ** attempt)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def summarize_section(name: str, text: str, model) -> str:
    prompt = f"Section: **{name.upper()}**\n\n{text[:3000]}\n\nWrite a concise 2-4 sentence academic summary of this section."
    return gemini_call(model, prompt)

def generate_consolidated_summary(sec_summaries: Dict[str, str], retrieved: List[str], model) -> str:
    sec_blob = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in sec_summaries.items())
    ctx_blob = "\n\n".join(f"[CONTEXT {i+1}]\n{c}" for i, c in enumerate(retrieved))
    prompt = (
        "Using the section summaries and supporting context below, generate the final structured research summary.\n\n"
        "=== SECTION SUMMARIES ===\n" + sec_blob + "\n\n"
        "=== RAG CONTEXT ===\n" + ctx_blob + "\n\n"
        "Produce the structured 7-section summary following the required format exactly."
    )
    return gemini_call(model, prompt)

def self_reflect(draft: str, retrieved: List[str], model) -> str:
    source = "\n\n".join(retrieved[:10]) 
    prompt = (
        "=== SOURCE (ground truth) ===\n" + source + "\n\n"
        "=== DRAFT TO VALIDATE ===\n" + draft + "\n\n"
        "Validate. Remove hallucinated claims not in source. Return corrected summary in the EXACT same structured format."
    )
    return gemini_call(model, prompt)

def extractive_summary(text: str, n: int = 8) -> str:
    STOP = {"the","a","an","and","or","but","in","on","at","to","of","for","is","are","was","were","be","been","by","with","this","that","from","it","its","as","have","has","had","we","our","their","which","also","can","may","these","those","such","not","no","paper","show","using","based"}
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.split()) > 8]
    freq = {}
    for w in re.findall(r'\b[a-zA-Z]+\b', text.lower()):
        if w not in STOP: freq[w] = freq.get(w, 0) + 1
    
    scored = []
    for s in sents:
        ws = re.findall(r'\b[a-zA-Z]+\b', s.lower())
        score = sum(freq.get(w, 0) for w in ws) / max(len(ws), 1)
        scored.append((score, s))
        
    scored.sort(reverse=True)
    top = {s for _, s in scored[:n]}
    return " ".join(s for s in sents if s in top)

def compute_rouge(hypothesis: str, reference: str) -> Dict[str, Any]:
    if not ROUGE_OK: return {"error": "rouge-score not installed."}
    sc = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    scores = sc.score(reference, hypothesis)
    return {
        k: {"precision": round(v.precision, 4), "recall": round(v.recall, 4), "f1": round(v.fmeasure, 4)}
        for k, v in scores.items()
    }

# ══════════════════════════════════════════════════════════════════════════════
# TTS ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def sarvam_tts(text: str, lang: str, speaker: str, api_key: str) -> bytes:
    MAX_CHARS = 400
    
    # 1. Strip ALL markdown to prevent 400 Bad Request
    clean_text = clean_text_for_speech(text)

    # 2. Strict chunking (Max 400 chars per API payload)
    sents = re.split(r'(?<=[।.!?\n])\s+', clean_text)
    chunks = []
    cur = ""
    for s in sents:
        s = s.strip()
        if not s: continue
        
        # If a single sentence without punctuation is too long, force split it
        if len(s) > MAX_CHARS:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(s), MAX_CHARS):
                chunks.append(s[i:i+MAX_CHARS])
            continue

        if len(cur) + len(s) + 1 <= MAX_CHARS:
            cur += (" " + s if cur else s)
        else:
            if cur: chunks.append(cur)
            cur = s
            
    if cur: chunks.append(cur)

    all_audio = BytesIO()
    headers = {"Content-Type": "application/json", "API-Subscription-Key": api_key}

    with requests.Session() as session:
        for chunk_text in chunks:
            payload = {
                "inputs": [chunk_text],
                "target_language_code": lang,
                "speaker": speaker,
                "model": "bulbul:v1",
                "enable_preprocessing": True,
            }
            try:
                resp = session.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                b64 = data.get("audios", [None])[0]
                if b64:
                    all_audio.write(base64.b64decode(b64))
            except Exception as e:
                raise RuntimeError(f"Sarvam TTS HTTP Error: {e} | Text Chunk: {chunk_text[:50]}...")

    result = all_audio.getvalue()
    if not result: raise RuntimeError("Sarvam returned empty audio.")
    return result

def sarvam_translate(text: str, target: str, api_key: str, source: str = "en-IN") -> str:
    clean_text = clean_text_for_speech(text) # Translate clean text only
    payload = {
        "input": clean_text[:1000],
        "source_language_code": source,
        "target_language_code": target,
        "model": "mayura:v1",
        "enable_preprocessing": False,
    }
    headers = {"Content-Type": "application/json", "API-Subscription-Key": api_key}
    try:
        resp = requests.post(SARVAM_TRANSLATE_URL, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json().get("translated_text", text)
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def edge_tts_generate(text: str, voice: str, speed: float = 1.0) -> bytes:
    clean_text = clean_text_for_speech(text)
    pct = int((speed - 1.0) * 100)
    rate = f"+{pct}%" if pct >= 0 else f"{pct}%"
    
    async def _generate() -> bytes:
        communicate = edge_tts.Communicate(clean_text, voice, rate=rate)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_generate())
    finally:
        loop.close()

def export_pdf(text: str, title: str = "Research Summary") -> bytes:
    if not FPDF_OK: raise ImportError("fpdf2 not installed.")
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': "-", '\u2014': "--", '\u2026': "...", '\u2022': "-",
        '\u00a0': " ", '\t': "    "
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
        title = title.replace(char, rep)
    
    text = text.encode('latin-1', 'replace').decode('latin-1')
    title = title.encode('latin-1', 'replace').decode('latin-1')

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 52, 96)
    pdf.multi_cell(0, 10, title, align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}", ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)
    
    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    pdf.set_text_color(0, 0, 0)
    
    for line in clean.split('\n'):
        line = line.strip()
        if not line: pdf.ln(3)
        elif re.match(r'^\d+\.', line):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(15, 52, 96)
            pdf.multi_cell(0, 8, line)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith('- ') or line.startswith('•'):
            pdf.set_x(20)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 7, "  • " + line.lstrip('-• '))
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 7, line)
            
    return bytes(pdf.output())

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(pdf_bytes: bytes, gemini_key: str, embedder, run_rouge: bool) -> Dict[str, Any]:
    result = {}
    bar = st.progress(0, text="Initialising pipeline…")

    bar.progress(5, text="Configuring Gemini API…")
    if not configure_gemini(gemini_key):
        raise RuntimeError("Gemini configuration failed. Check your API key in the backend environment variables.")

    summ_model = make_model(SUMMARIZATION_SYSTEM_PROMPT, GEMINI_MODEL, 0.25)
    section_model = make_model(SECTION_SUMMARY_PROMPT, GEMINI_FLASH_MODEL, 0.2)
    reflect_model = make_model(REFLECTION_PROMPT, GEMINI_MODEL, 0.1)

    bar.progress(15, text="Extracting text from PDF…")
    full_text, pages, page_count = extract_pdf(pdf_bytes)
    result.update({
        "full_text": full_text, "pages": pages,
        "page_count": page_count, "word_count": len(full_text.split())
    })

    bar.progress(25, text="Detecting paper sections…")
    sections = detect_sections(full_text)
    result["sections"] = sections

    bar.progress(35, text="Semantic chunking…")
    chunks = semantic_chunk(full_text)
    result["chunks"] = chunks

    bar.progress(45, text="Building FAISS vector index…")
    rag_ok = bool(embedder and FAISS_OK)
    if rag_ok:
        faiss_index, _ = build_faiss_index(chunks, embedder)
        result["faiss_index"] = faiss_index
    else:
        faiss_index = None
        logger.warning("RAG unavailable — FAISS or embedder missing.")

    bar.progress(55, text="Summarising sections…")
    sec_sums = {}
    key_secs = [s for s in sections if s not in ("preamble", "references", "bibliography")]
    for sec in key_secs[:8]:
        txt = sections[sec]
        if len(txt.split()) >= 30:
            sec_sums[sec] = summarize_section(sec, txt, section_model)
            time.sleep(1.5) 

    result["section_summaries"] = sec_sums

    bar.progress(65, text="RAG retrieval…")
    rag_q = "research objective problem statement methodology key contributions results limitations future work"
    retrieved = retrieve_chunks(rag_q, chunks, faiss_index, embedder) if rag_ok else (chunks[:3] + chunks[-3:])
    result["retrieved_chunks"] = retrieved

    bar.progress(75, text="Generating consolidated summary…")
    draft = generate_consolidated_summary(sec_sums, retrieved, summ_model)
    time.sleep(1) 
    
    bar.progress(85, text="Self-reflection & hallucination check…")
    final = self_reflect(draft, retrieved, reflect_model)
    result["final"] = final

    bar.progress(95, text="Computing ROUGE baseline scores…")
    if run_rouge and "abstract" in sections and ROUGE_OK:
        ext = extractive_summary(full_text)
        result["extractive"] = ext
        result["rouge_abstractive"] = compute_rouge(final, sections["abstract"])
        result["rouge_extractive"] = compute_rouge(ext, sections["abstract"])

    bar.progress(100, text="Done!")
    time.sleep(0.5)
    bar.empty()
    return result

def chat_qa(question: str, chunks: List[str], faiss_index, embedder, history: List[Dict], gemini_key: str, language: str) -> str:
    retrieved = retrieve_chunks(question, chunks, faiss_index, embedder, top_k=5)
    ctx = "\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(retrieved))

    hist_str = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        hist_str += f"{role}: {msg['content']}\n\n"

    prompt = (
        "=== PAPER EXCERPTS ===\n" + ctx + "\n\n"
        "=== CONVERSATION HISTORY ===\n" + hist_str +
        f"User: {question}\n\n"
        "Answer based strictly on the paper excerpts. Be specific and cite details.\n"
        f"IMPORTANT: You MUST generate your final response entirely in the {language} language."
    )
    
    model = make_model(CHAT_SYSTEM_PROMPT, GEMINI_FLASH_MODEL, temp=0.4)
    return gemini_call(model, prompt)

# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}

    .app-hdr {
        background: linear-gradient(135deg, #0a0a1a, #0f1e3d 50%, #1a0a3d);
        color: white; padding: 2.5rem 2rem; border-radius: 16px;
        margin-bottom: 1.5rem; text-align: center;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.07);
    }
    .app-hdr h1 { font-size: 2.1rem; font-weight: 700; margin: 0; }
    .app-hdr .sub { font-size: 0.85rem; opacity: 0.8; margin-top: 10px; line-height: 1.8; }
    
    .bdg {
        display: inline-block; background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2); padding: 3px 12px;
        border-radius: 20px; font-size: 0.74rem; margin: 3px;
        backdrop-filter: blur(4px);
    }

    .mgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 1rem 0; }
    .mcard {
        background: white; border: 1px solid #e8ecf3; border-radius: 12px;
        padding: 1.1rem; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        transition: transform 0.2s;
    }
    .mcard:hover { transform: translateY(-3px); }
    .ml { font-size: 0.7rem; color: #8b9ab1; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }
    .mv { font-size: 1.6rem; font-weight: 700; color: #0f1e3d; margin-top: 4px; }

    .sbadge {
        display: inline-block; background: #eff6ff; color: #1d4ed8;
        border: 1px solid #bfdbfe; padding: 4px 14px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 500; margin: 4px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #0f3460, #533483);
        color: white; border: none; border-radius: 10px; font-weight: 600;
        padding: 0.55rem 1.6rem; box-shadow: 0 4px 14px rgba(15,52,96,0.3);
        transition: all 0.2s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(15,52,96,0.4); color: white; }

    .warn { background: #fffbeb; border: 1px solid #f59e0b; border-radius: 10px; padding: 1rem; font-size: 0.88rem; color: #92400e; }
    .info { background: #eff6ff; border: 1px solid #93c5fd; border-radius: 10px; padding: 1rem; font-size: 0.88rem; color: #1e40af; }
    hr.div { border: none; border-top: 1px solid #e5e7eb; margin: 2rem 0; }
    </style>
    """, unsafe_allow_html=True)

def render_rouge_table(scores: dict, label: str):
    if not scores or "error" in scores: return
    rows = ""
    for k, v in scores.items():
        pct = int(v["f1"] * 100)
        bar = (f'<div style="background:#e5e7eb;border-radius:4px;height:8px;margin-top:2px;">'
               f'<div style="width:{pct}%;background:linear-gradient(90deg,#0f3460,#533483);height:100%;border-radius:4px"></div></div>')
        rows += (f"<tr><td style='padding:10px 8px;border-bottom:1px solid #eee;'><b>{k.upper()}</b></td>"
                 f"<td style='padding:10px 8px;border-bottom:1px solid #eee;'>{v['precision']:.3f}</td>"
                 f"<td style='padding:10px 8px;border-bottom:1px solid #eee;'>{v['recall']:.3f}</td>"
                 f"<td style='padding:10px 8px;border-bottom:1px solid #eee;'><b>{v['f1']:.3f}</b></td>"
                 f"<td style='padding:10px 8px;width:120px;border-bottom:1px solid #eee;'>{bar}</td></tr>")
    
    st.markdown(
        f"<p style='font-weight:600;margin:0 0 8px;color:#1e293b;font-size:1.05rem;'>{label}</p>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.9rem;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.05);'>"
        f"<tr style='background:#f8fafc;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.5px;color:#64748b;'>"
        f"<th style='padding:12px 8px;text-align:left'>Metric</th>"
        f"<th style='padding:12px 8px;text-align:left'>Precision</th>"
        f"<th style='padding:12px 8px;text-align:left'>Recall</th>"
        f"<th style='padding:12px 8px;text-align:left'>F1 Score</th>"
        f"<th style='padding:12px 8px;text-align:left'>Strength</th></tr>"
        f"{rows}</table>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Research Paper AI Analyzer", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    inject_css()
    
    st.markdown("""
    <div class="app-hdr">
        <h1>🔬 Intelligent Research Paper Analyzer</h1>
    </div>""", unsafe_allow_html=True)

    gemini_key = get_env_key("GEMINI_API_KEY")
    sarvam_key = get_env_key("SARVAM_API_KEY")

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.markdown("### 📊 Options")
        run_rouge = st.toggle("ROUGE Evaluation", value=True, help="Requires detectable abstract section.")

        st.markdown("---")
        st.markdown("### 🌐 English TTS")
        if EDGE_TTS_OK:
            eng_voice = st.selectbox("Voice", list(EDGE_TTS_VOICES.keys()))
            eng_speed = st.slider("Speed", 0.5, 1.8, 1.0, 0.1)
        else:
            eng_voice, eng_speed = "US Female (Aria)", 1.0
            st.warning("Install edge-tts: pip install edge-tts")

        st.markdown("---")
        st.markdown("### 🇮🇳 Indian TTS")
        lang_label = st.selectbox("Language", list(SARVAM_LANGUAGES.keys()))
        lang_code = SARVAM_LANGUAGES[lang_label]
        ind_speaker = st.selectbox("Speaker", SARVAM_SPEAKERS.get(lang_code, ["meera"]))
        auto_trans = st.toggle("Auto-Translate Text", value=True)

    t1, t2, t3 = st.tabs(["📄 Upload & Analyze", "💬 Chat with Document", "🔊 Text-to-Speech"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — UPLOAD & ANALYZE
    # ══════════════════════════════════════════════════════════════════════════
    with t1:
        st.markdown("## 📂 Upload Research Paper")
        uploaded = st.file_uploader("Drop PDF here (max 25 MB)", type=["pdf"], label_visibility="collapsed")

        if uploaded:
            raw = uploaded.getvalue()
            size_mb = len(raw) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File too large ({size_mb:.1f} MB). Max is {MAX_FILE_SIZE_MB} MB.")
                st.stop()

            st.markdown(f'<div class="info">📄 <b>{uploaded.name}</b> &nbsp;|&nbsp; {size_mb:.2f} MB</div>', unsafe_allow_html=True)

            with st.spinner("Loading embedding model…"):
                embedder = load_embedder()

            if not gemini_key:
                st.error("GEMINI_API_KEY is not found in the backend environment. Please configure your .env file.")
                st.stop()

            if st.button("🚀 Analyze & Summarize Paper", use_container_width=True):
                st.session_state["chat_history"] = []
                try:
                    with st.status("Running analysis pipeline...", expanded=True):
                        st.session_state["result"] = run_pipeline(raw, gemini_key, embedder, run_rouge)
                    st.session_state["filename"] = Path(uploaded.name).stem
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"⚠️ Error during processing: {str(e)}")
                    logger.exception("Pipeline Exception")

        if st.session_state.get("result"):
            res = st.session_state["result"]
            fname = st.session_state.get("filename", "paper")
            
            st.markdown('<hr class="div">', unsafe_allow_html=True)
            st.markdown("## 📈 Results Overview")
            st.markdown(f"""
            <div class="mgrid">
                <div class="mcard"><div class="ml">📄 Pages</div><div class="mv">{res.get('page_count',0)}</div></div>
                <div class="mcard"><div class="ml">📝 Words</div><div class="mv">{res.get('word_count',0):,}</div></div>
                <div class="mcard"><div class="ml">🗂️ Sections</div><div class="mv">{len(res.get('sections',{}))}</div></div>
                <div class="mcard"><div class="ml">🧩 Vector Chunks</div><div class="mv">{len(res.get('chunks',[]))}</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>**Detected Sections:**", unsafe_allow_html=True)
            badges = "".join(f'<span class="sbadge">{s.title()}</span>' for s in res.get("sections", {}) if s != "preamble")
            st.markdown(badges, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("📋 View Per-Section Summaries", expanded=False):
                for sec, txt in res.get("section_summaries", {}).items():
                    st.markdown(f"**{sec.title()}**")
                    st.info(txt)

            # Native Markdown Rendering for clean bullet points and bolding
            st.markdown("## 📝 Final Structured Summary")
            with st.container(border=True):
                st.markdown(res.get("final", ""))
            
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("### ⬇️ Export Downloads")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📄 Download as TXT", data=res.get("final", "").encode("utf-8"), file_name=f"{fname}_summary.txt", mime="text/plain", use_container_width=True)
            with c2:
                if FPDF_OK:
                    try:
                        pdf_b = export_pdf(res.get("final", ""), title=f"Summary – {fname}")
                        st.download_button("📑 Download as PDF", data=pdf_b, file_name=f"{fname}_summary.pdf", mime="application/pdf", use_container_width=True)
                    except Exception as e:
                        st.warning(f"PDF export failed: {e}")
                else:
                    st.info("pip install fpdf2 for PDF export.")

            if res.get("rouge_abstractive"):
                st.markdown('<hr class="div">', unsafe_allow_html=True)
                st.markdown("## 📊 Quality Evaluation (ROUGE)")
                st.caption("Compared against the paper's original abstract to measure information retention.")
                r1, r2 = st.columns(2)
                with r1: render_rouge_table(res["rouge_abstractive"], "🤖 AI Abstractive Summary")
                with r2: render_rouge_table(res.get("rouge_extractive", {}), "📌 Extractive Baseline")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — CHAT WITH DOCUMENT
    # ══════════════════════════════════════════════════════════════════════════
    with t2:
        c_head1, c_head2 = st.columns([3, 1])
        with c_head1:
            st.markdown("## 💬 Chat with Your Research Paper")
            st.caption("Powered by RAG Vector Search & Gemini 2.5 Flash.")
        with c_head2:
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            chat_lang = st.selectbox("Response Language", CHAT_LANGUAGES, index=0)

        if not st.session_state.get("result"):
            st.markdown('<div class="warn">⚠️ Analyze a paper first in the <b>Upload & Analyze</b> tab.</div>', unsafe_allow_html=True)
        else:
            res = st.session_state["result"]
            chunks = res.get("chunks", [])
            f_idx = res.get("faiss_index")
            embedder = load_embedder()

            with st.expander("💡 Suggested Questions", expanded=True):
                suggestions = [
                    "What is the main research objective?", "What methodology was used?",
                    "What are the key contributions?", "What datasets were used?",
                    "What are the limitations?", "What future work is proposed?"
                ]
                cols = st.columns(2)
                for i, q in enumerate(suggestions):
                    if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                        st.session_state["prefill"] = q

            # Input area
            c1, c2 = st.columns([4, 1])
            with c1:
                user_q = st.text_input("Your question:", value=st.session_state.get("prefill", ""), key="chat_in", label_visibility="collapsed", placeholder="Ask a question about the paper...")
            with c2:
                send_btn = st.button("📨 Send", use_container_width=True)
            
            st.session_state["prefill"] = "" # Reset after use

            if send_btn and user_q.strip():
                if not gemini_key or not (embedder and f_idx):
                    st.error("Missing Gemini API Key or RAG functionality (FAISS/Embedder).")
                else:
                    st.session_state["chat_history"].append({"role": "user", "content": user_q.strip()})
                    with st.spinner(f"Searching document context & generating {chat_lang} response…"):
                        try:
                            ans = chat_qa(user_q.strip(), chunks, f_idx, embedder, st.session_state["chat_history"][:-1], gemini_key, chat_lang)
                            st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                        except Exception as e:
                            st.error(f"Chat error: {e}")
                    st.rerun()

            st.button("🗑️ Clear Chat History", on_click=lambda: st.session_state.update(chat_history=[]))

            if st.session_state.get("chat_history"):
                st.markdown('<hr class="div">', unsafe_allow_html=True)
                
                # Use native beautiful chat elements
                for msg in st.session_state["chat_history"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        
    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 3 — TEXT-TO-SPEECH
    # ══════════════════════════════════════════════════════════════════════════════
    with t3:
        st.markdown("## 🔊 Text-to-Speech Generation")
        
        # Pull text and CLEAN IT of all asterisks and markdown symbols
        res_state = st.session_state.get("result") or {}
        raw_final_text = res_state.get("final", "")
        default_tts = clean_text_for_speech(raw_final_text)
        
        st.info("💡 The text below has been automatically stripped of markdown formatting to ensure the text-to-speech engine speaks it fluently without errors.")
        tts_text = st.text_area("Text to convert to speech:", value=default_tts, height=220)
        
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        
        # English TTS
        st.markdown("### 🌐 English Neural TTS")
        c1, c2 = st.columns([3, 1])
        c1.markdown(f'<div class="info">Voice: <b>{eng_voice}</b> &nbsp;|&nbsp; Speed: <b>{eng_speed}x</b></div>', unsafe_allow_html=True)
        if c2.button("🎙️ Generate English", use_container_width=True):
            if not tts_text.strip():
                st.warning("No text provided.")
            elif EDGE_TTS_OK:
                with st.spinner("Generating Neural English TTS..."):
                    try:
                        audio = edge_tts_generate(tts_text, EDGE_TTS_VOICES[eng_voice], eng_speed)
                        st.audio(audio, format="audio/mp3")
                    except Exception as e:
                        st.error(f"Edge TTS Error: {e}")
            else:
                st.error("Install edge-tts to use this feature.")

        st.markdown('<hr class="div">', unsafe_allow_html=True)

        # Indian TTS
        st.markdown("### 🇮🇳 Indian Language TTS (Sarvam AI)")
        st.markdown(f'<div class="info">Language: <b>{lang_label}</b> &nbsp;|&nbsp; Speaker: <b>{ind_speaker}</b> &nbsp;|&nbsp; Auto-Translate: <b>{"ON" if auto_trans else "OFF"}</b></div>', unsafe_allow_html=True)
        
        if st.button(f"🎙️ Generate {lang_label} Audio", use_container_width=True):
            if not sarvam_key:
                st.error("SARVAM_API_KEY is missing from backend environment variables. Cannot generate Indian TTS.")
            elif not tts_text.strip():
                st.warning("No text provided.")
            else:
                with st.spinner(f"Processing via Sarvam AI API..."):
                    try:
                        target_text = tts_text
                        if auto_trans and lang_code != "en-IN":
                            with st.status("Translating to target language..."):
                                target_text = sarvam_translate(tts_text, lang_code, sarvam_key)
                            with st.expander("View Translated Text"):
                                st.write(target_text)
                                
                        with st.status(f"Generating {lang_label} speech..."):
                            audio = sarvam_tts(target_text, lang_code, ind_speaker, sarvam_key)
                        
                        st.audio(audio, format="audio/wav")
                    except Exception as e:
                        st.error(f"Sarvam AI Error: {e}")

if __name__ == "__main__":
    main()