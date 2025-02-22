from llama_index.core import SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core import QueryBundle, get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import whisper
import pyaudio
import numpy as np
import logging
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import nltk
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model=genai.GenerativeModel("gemini-pro")

def get_gemini_response(text,context):
    prompt=context+"\n"+text
    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt) 
    return response.text

whisper_model = whisper.load_model("base")

# Set up PyAudio with system audio
audio = pyaudio.PyAudio()
stereo_mix_index = None

for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i)
    if "Stereo Mix" in device_info["name"] or "What U Hear" in device_info["name"]:
        stereo_mix_index = i
        break

if stereo_mix_index is None:
    raise Exception("Stereo Mix device not found. Make sure it’s enabled in the Recording devices.")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=stereo_mix_index, frames_per_buffer=CHUNK)

print("Listening to system audio...")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)

def initialize_query_engine():
    documents = SimpleDirectoryReader('data').load_data()
    for doc in documents:
        doc.text = preprocess_text(doc.text)

    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GEMINI_API_KEY)
    llm = Gemini(api_key=GEMINI_API_KEY)
    Settings.embed_model = embed_model
    Settings.llm = llm

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index, top_k=5)

    class CustomRetriever(BaseRetriever):
        def __init__(self, vector_retriever, keyword_retriever, mode="AND"):
            self._vector_retriever = vector_retriever
            self._keyword_retriever = keyword_retriever
            self._mode = mode
            super().__init__()

        def _retrieve(self, query_bundle):
            vector_nodes = self._vector_retriever.retrieve(query_bundle)
            keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

            combined_dict = {n.node.node_id: n for n in vector_nodes + keyword_nodes}
            vector_ids = {n.node.node_id for n in vector_nodes}
            keyword_ids = {n.node.node_id for n in keyword_nodes}
            retrieve_ids = vector_ids & keyword_ids if self._mode == "AND" else vector_ids | keyword_ids

            return [combined_dict[r_id] for r_id in retrieve_ids]

    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)
    response_synthesizer = get_response_synthesizer()

    return RetrieverQueryEngine(retriever=custom_retriever, response_synthesizer=response_synthesizer)

query_engine = initialize_query_engine()


def detect_question(text):
    text = text.lower().strip()
    if text.endswith("?"):
        return True
    
    question_starters = [
        "who", "what", "when", "where", "why", "how", "is", "are", "can", "could", 
        "do", "does", "did", "should", "would", "will", "may", "might", "am", "have", "has"
    ]

    if any(text.startswith(word) for word in question_starters):
        return True

    question_patterns = [
        r"\bdo you know\b", r"\bcould you tell me\b", r"\bis it possible\b",
        r"\bcan you explain\b", r"\bwhat is\b", r"\bhow do\b", r"\bwhy does\b",
        r"\bis there\b", r"\bare there\b"
    ]
    if any(re.search(pattern, text) for pattern in question_patterns):
        return True

    doc = nlp(text)
    
    for token in doc:
        if token.dep_ == "ROOT" and token.tag_ in ["VB", "VBP", "VBZ"]:
            if token.lemma_ in question_starters:
                return True
            
        if token.tag_ in ["WP", "WRB"]:
            return True

    question_words = ["anyone", "someone", "anybody", "somebody", "anything", "something", "guess"]
    if any(word in text for word in question_words):
        return True

    return False

context_text = ""
frames = []
try:
    while True:
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, np.int16))

        if len(frames) >= RATE // CHUNK * 7 :
            audio_chunk = np.concatenate(frames)
            frames = []

            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            transcription = whisper_model.transcribe(audio_chunk)
            text = transcription["text"]
            print("Transcription:", text)

            if detect_question(text):
                print("Question detected:", text)
                query_bundle = QueryBundle(text)
                result = query_engine.query(query_bundle)

                if result.response =="Empty Response":
                    print("RAG Response: No answer found in the RAG docs")
                    print("Fallback to Gemini AI:")
                    fallback_response = get_gemini_response(text,context_text)
                    print("Gemini Response:", fallback_response)
                else:
                    print("RAG Response:", result.response)

                context_text = ""
            else:
                context_text += text + " "

except KeyboardInterrupt:
    print("Stopping transcription.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()


import logging
logging.basicConfig(level=logging.DEBUG, filename="gemini_fallback_test.log", filemode="w")

logging.debug("RAG response was empty or insufficient. Calling Gemini Pro fallback...")
logging.debug("Gemini Response Generated: %s", fallback_response)

