import time
import google.generativeai as genai
import streamlit as st
from tempfile import NamedTemporaryFile
from gtts import gTTS
import tempfile
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

# Set the API key (replace with your actual API key)
GEMINI_API_KEY = "AIzaSyBGK_8Doan5w6XCjorUczMxyM9S4fShY5s"
genai.configure(api_key=GEMINI_API_KEY)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        st.success(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except GoogleAPIError as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def wait_for_files_active(files, timeout=300):
    """Waits for the given files to be active with a timeout."""
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
    """Creates the generative model for summarization and chat."""
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
            "You are an AI assistant for summarizing and answering questions about research papers. "
            "Provide clear, concise, and professional responses based on the content of the uploaded paper."
        ),
    )

def main():
    st.title("Research Paper Summarizer & Chat Assistant")

    uploaded_file = st.file_uploader("Upload your Research Paper (PDF)", type="pdf")

    if st.button("Summarize") and uploaded_file:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        uploaded_file_obj = upload_to_gemini(temp_file_path, mime_type="application/pdf")
        if not uploaded_file_obj:
            return

        if not wait_for_files_active([uploaded_file_obj]):
            return

        with open(temp_file_path, 'rb') as file:
            pdf_text = file.read().decode('utf-8')

        try:
            st.session_state.chat_session = model.start_chat(
                history=[{"role": "user", "content": pdf_text}]
            )
            response = st.session_state.chat_session.send_message("Summarize this paper.")
            cleaned_summary = response.text.replace('*', '').strip()
            st.session_state.summary_text = cleaned_summary
            st.header("Summary")
            st.text_area("Summarized Text", value=cleaned_summary, height=300, disabled=True)
        except ResourceExhausted:
            st.error("Resource limit reached. Try again later or reduce input size.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.header("Chat with the Paper")
    if 'chat_session' in st.session_state:
        user_query = st.text_input("Ask a question about the research paper:")
        if st.button("Ask") and user_query:
            try:
                chat_response = st.session_state.chat_session.send_message(user_query)
                st.write(f"**AI Response:** {chat_response.text.strip()}")
            except ResourceExhausted:
                st.error("Resource limit reached. Try again later.")
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
