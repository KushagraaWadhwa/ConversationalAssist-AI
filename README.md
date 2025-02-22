# Real-Time Speech-to-Text and Query Processing with Gemini AI & RAG

## Overview
This project integrates real-time speech-to-text transcription with **Whisper**, **Retrieval-Augmented Generation (RAG)**, and **Gemini AI** to process user queries. It detects questions, retrieves answers from a knowledge base, and falls back to Gemini AI when necessary.

## Features
- **Real-time Audio Transcription** using OpenAI's Whisper model
- **Question Detection** with NLP techniques (Spacy, NLTK, regex)
- **RAG-based Query Engine** for retrieving relevant answers
- **Fallback to Gemini AI** if RAG retrieval fails
- **Multi-modal Retrieval** using **Vector Store Index** and **Keyword-based Index**
- **Logging and Debugging** for query response tracking

## Tech Stack
- **Speech-to-Text:** OpenAI's Whisper
- **NLP & Question Detection:** NLTK, Spacy, Regex
- **Query Processing:** LlamaIndex (RAG), Gemini AI
- **Database:** SimpleDirectoryReader for document storage
- **Audio Handling:** PyAudio
- **Task Management:** Logging for query responses

## Installation
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. **Create a Virtual Environment & Install Dependencies**:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. **Set Up Environment Variables**:
   Add up you api keys in the `.env` file.
   
5. **Download Spacy Model & NLTK Stopwords**:
   ```sh
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('stopwords')"
   ```
6. **Run the Application**:
   ```sh
   python integration2.py
   ```

## How It Works
1. **Audio Capture**:
   - Captures system audio using PyAudio’s Stereo Mix feature.
   - Converts raw audio into a transcribable format.
2. **Speech Transcription**:
   - Whisper model transcribes the audio into text.
3. **Question Detection**:
   - Uses regex, Spacy NLP, and dependency parsing to detect if a sentence is a question.
4. **Query Processing**:
   - If a question is detected, it queries the RAG system using LlamaIndex.
   - If RAG doesn’t find an answer, it falls back to Gemini AI.
5. **Logging & Debugging**:
   - Logs Gemini AI fallback responses for debugging.

## Future Enhancements
- Support for **real-time WebSockets-based query handling**.
- Integration with **external databases** for improved knowledge retrieval.
- Enhanced **multi-turn conversation tracking**.
- Deployment as an API service with **FastAPI**.
