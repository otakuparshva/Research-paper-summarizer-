import time
import google.generativeai as genai
import streamlit as st
from tempfile import NamedTemporaryFile
from gtts import gTTS
import tempfile

# Directly set the API key
GEMINI_API_KEY = "AIzaSyDBzwMBeIQp7iYwtHcfXmtZkKt6xAZgmGM"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        st.success(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    st.write("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            st.write(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            st.error(f"File {file.name} failed to process.")
            return False
    st.write("...all files ready.")
    return True

def create_model():
    """Creates the generative model for summarization."""
    generation_config = {
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    return genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        generation_config=generation_config,
        system_instruction=(
            "only allow research paper or something related to that. "
            "summary should be well written and precise that explaining "
            "whole purpose of research paper in summary format and summary "
            "should be in points and written in proper professional manner "
            "and under 300 words." 
            "Consider key point  of research paper and make sure that "

        ),
    )

def main():
    st.title("Research Paper Summarizer")

    uploaded_file = st.file_uploader("Upload your Research Paper (PDF)", type="pdf")
    
    if st.button("Summarize") and uploaded_file:
        # Create a temporary file to save the uploaded PDF
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        # Upload the file
        files = [upload_to_gemini(temp_file_path, mime_type="application/pdf")]
        if files[0] is None:
            return  # Exit if file upload failed

        # Wait for the files to be processed
        if not wait_for_files_active(files):
            return  # Exit if file processing failed

        # Create the chat session
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                        "Summarize and explain this research paper in detail.",
                    ],
                },
            ]
        )

        # Send the message to the model
        response = chat_session.send_message("Summarize this paper.")

        # Summary Section
        st.header("Summary")
        # Clean summary text by removing any asterisks
        cleaned_summary = response.text.replace('*', '').strip()
        st.session_state.summary_text = cleaned_summary  # Store cleaned summary in session state
        st.text_area("Summarized Text", value=cleaned_summary, height=300, disabled=True)

    # Text-to-Speech Section
    st.header("Text-to-Speech")
    if st.button("Convert Summary to Speech"):
        if 'summary_text' in st.session_state:
            # Generate audio file from summary text
            tts = gTTS(text=st.session_state.summary_text, lang='en')
            # Save the audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
                tts.save(audio_file.name)
                st.audio(audio_file.name, format='audio/mp3')
        else:
            st.warning("Please summarize the paper first.")

if __name__ == "__main__":
    model = create_model()
    main()
