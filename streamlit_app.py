import time
import google.generativeai as genai
import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from tempfile import NamedTemporaryFile
from gtts import gTTS
import tempfile
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

# Set the new API key
GEMINI_API_KEY = "AIzaSyBGK_8Doan5w6XCjorUczMxyM9S4fShY5s"
genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using PyMuPDF."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def upload_to_gemini(path, mime_type=None):
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        st.success(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except GoogleAPIError as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def wait_for_files_active(files, timeout=300):
    st.write("Waiting for file processing...")
    start_time = time.time()
    for file in files:
        while file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                st.error(f"Timeout exceeded for file {file.name}.")
                return False
            st.write(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(file.name)
        if file.state.name != "ACTIVE":
            st.error(f"File {file.name} failed to process.")
            return False
    st.write("...all files ready.")
    return True

def create_model():
    generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
        "response_mime_type": "text/plain",
    }

    return genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        generation_config=generation_config,
        system_instruction=(
            "Summarize research papers in a concise, professional format with key points in under 300 words."
        ),
    )

def main():
    st.title("Research Paper Summarizer")

    uploaded_file = st.file_uploader("Upload your Research Paper (PDF)", type="pdf")

    if st.button("Summarize") and uploaded_file:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        extracted_text = extract_text_from_pdf(temp_file_path)
        if not extracted_text:
            return  # Exit if text extraction failed

        try:
            chat_session = model.start_chat(
                history=[{"role": "user", "content": extracted_text}]
            )
            response = chat_session.send_message("Summarize this paper.")
            cleaned_summary = response.text.replace('*', '').strip()
            st.session_state.summary_text = cleaned_summary
            st.header("Summary")
            st.text_area("Summarized Text", value=cleaned_summary, height=300, disabled=True)
        except ResourceExhausted:
            st.error("Resource limit reached. Try again later or reduce input size.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.header("Text-to-Speech")
    if st.button("Convert Summary to Speech"):
        if 'summary_text' in st.session_state:
            tts = gTTS(text=st.session_state.summary_text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
                tts.save(audio_file.name)
                st.audio(audio_file.name, format='audio/mp3')
        else:
            st.warning("Please summarize the paper first.")

if __name__ == "__main__":
    model = create_model()
    main()
