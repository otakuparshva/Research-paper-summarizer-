

# Research Paper Summarizer

## Overview

The **Research Paper Summarizer** is a web application that leverages Google Generative AI to provide concise summaries of research papers. Users can upload their research papers in PDF format, and the application generates a well-structured summary. Additionally, the app offers a feature to convert the generated summary into speech using the Google Text-to-Speech (gTTS) library, enhancing accessibility for users who prefer audio formats.

### Demo Video

[Insert Demo Video Link Here]

## Getting Started with Gemini API

To use the Google Generative AI in this application, you need to obtain an API key from Google AI Studio:

1. Go to the [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create a new project or select an existing one.
3. Navigate to the **API & Services** dashboard.
4. Enable the **Generative AI API** for your project.
5. Copy the API key generated for your project and replace the placeholder in the code.

## Languages, Libraries, Tools, and Technologies Used

- **Languages**: 
  - Python

- **Libraries**:
  - `Streamlit`: For creating the web interface.
  - `google.generativeai`: To interact with the Google Generative AI API.
  - `gtts`: For converting text to speech.
  - `tempfile`: For handling temporary file storage.

- **Tools**:
  - Google AI Studio (for API access)
  - Git (for version control)

## Project Summary

The **Research Paper Summarizer** application follows these key steps:

1. **User Interface**: The application is built using Streamlit, allowing users to upload their research papers in PDF format through a simple and intuitive web interface.

2. **File Upload**: Once a user uploads a PDF, the application temporarily saves the file and uploads it to the Gemini API for processing.

3. **File Processing**: The application monitors the status of the uploaded files to ensure they are processed correctly.

4. **Summary Generation**: Upon successful processing, the application generates a summary of the research paper using the Google Generative AI. The summary is tailored to be concise and structured, focusing on key points.

5. **Text-to-Speech Conversion**: Users have the option to convert the generated summary into audio format, enabling them to listen to the summary instead of reading it.

6. **User Feedback**: The application provides users with immediate feedback on the status of their uploads and the generated summary.

## Conclusion

The **Research Paper Summarizer** is a powerful tool for researchers and students looking to streamline their reading process. By harnessing the capabilities of Google Generative AI, this application simplifies the task of summarizing lengthy research papers and enhances accessibility through its text-to-speech feature. This project showcases the potential of AI in education and research, making it easier for users to grasp essential information quickly.

Feel free to explore the code, contribute to its development, or use it as a foundation for your own projects!
