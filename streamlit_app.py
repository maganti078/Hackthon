# Python script for StudyMate: an AI-Powered PDF Q&A System
# This application uses Streamlit for the UI, PyMuPDF for PDF processing,
# Sentence-Transformers and FAISS for semantic search, and the IBM Watsonx
# foundation model API for generating answers.

# --- Core Library Imports ---
import streamlit as st
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.credentials import Credentials
import os
from dotenv import load_dotenv
import textwrap
from datetime import datetime
import io

# --- Environment Variable Setup ---
# This loads environment variables from a .env file for secure API key handling.
load_dotenv()

# IBM Watsonx credentials
IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")
IBM_URL = os.getenv("IBM_URL")

# --- Streamlit Session State Initialization ---
# This ensures that key variables persist across user interactions.
if "history" not in st.session_state:
    st.session_state.history = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# --- Backend: Core Functions for PDF Processing and AI ---

@st.cache_resource
def get_embedding_model():
    """
    Loads the Sentence-Transformer model and caches it to prevent
    re-loading on every interaction.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def process_and_index_pdfs(uploaded_files, embedding_model):
    """
    Processes a list of uploaded PDF files, extracts text,
    creates chunks, and builds a FAISS index.
    """
    all_chunks = []
    
    # 1. Text Extraction & Chunking
    for file in uploaded_files:
        with st.spinner(f"Extracting text from {file.name}..."):
            pdf_document = fitz.open(stream=file.read(), filetype="pdf")
            file_text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                file_text += page.get_text() + "\n"

            # Split text into chunks with a sliding window
            # 500 word chunk size with 100 word overlap
            words = file_text.split()
            chunk_size = 500
            overlap = 100
            
            chunks = [
                " ".join(words[i:i + chunk_size])
                for i in range(0, len(words), chunk_size - overlap)
            ]
            
            # Add source metadata to each chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": file.name,
                    "chunk_id": i
                })

    # 2. Embedding Generation
    with st.spinner("Generating embeddings..."):
        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')

    # 3. FAISS Index Construction
    with st.spinner("Building semantic index..."):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
    
    return index, all_chunks

def get_watsonx_model():
    """
    Initializes and returns the IBM Watsonx foundation model client.
    """
    credentials = Credentials(
        url=IBM_URL,
        api_key=IBM_API_KEY
    )
    
    # Define generation parameters
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.TEMPERATURE: 0.5
    }

    # Initialize the model object
    model = Model(
        model_id=ModelTypes.GRANITE_13B_INSTRUCT_V2,
        credentials=credentials,
        params=params,
        project_id=IBM_PROJECT_ID
    )
    return model

def get_answer_from_watsonx(query, retrieved_chunks):
    """
    Constructs a prompt with retrieved context and sends it to the LLM.
    """
    # Create the prompt from the retrieved chunks and user query
    context_text = "\n\n".join([c["text"] for c in retrieved_chunks])
    
    prompt_template = textwrap.dedent(f"""
    <|system|>
    You are a helpful and knowledgeable academic assistant. Answer the user's question
    strictly based on the following context. If the answer is not in the context,
    state that you cannot find the answer in the provided documents.
    Do not add any information not present in the context.

    Context:
    {context_text}
    <|end|>
    <|user|>
    {query}
    <|end|>
    <|assistant|>
    """)

    try:
        # Call the IBM Watsonx model
        llm = get_watsonx_model()
        response = llm.generate_text(prompt=prompt_template)
        return response
    except Exception as e:
        st.error(f"Error calling IBM Watsonx API: {e}")
        return "Sorry, I am unable to generate a response at this time."

# --- Frontend: Streamlit UI Components and Layout ---

def display_history():
    """
    Displays the session's Q&A history.
    """
    for item in reversed(st.session_state.history):
        # Question display
        with st.chat_message("user"):
            st.write(item["question"])
        
        # Answer display
        with st.chat_message("assistant"):
            st.write(item["answer"])
            # Display source references in an expandable section
            with st.expander("Referenced Paragraphs"):
                for chunk in item["chunks"]:
                    st.markdown(f"**Source:** {chunk['source']} (Chunk {chunk['chunk_id'] + 1})")
                    st.text(textwrap.fill(chunk["text"], width=80))

# --- Main Application Layout ---

st.set_page_config(
    page_title="StudyMate",
    layout="wide"
)

# Custom CSS for a modern look
st.markdown("""
<style>
    .st-emotion-cache-1f8u91d {
        background-image: url('https://placehold.co/1920x1080/4f5053/b7c2cc?text=StudyMate');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .st-emotion-cache-1f8u91d .st-emotion-cache-1cpx922{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .st-emotion-cache-1f8u91d h1 {
        color: #000;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .st-emotion-cache-1f8u91d h2 {
        color: #333;
        font-family: 'Inter', sans-serif;
    }
    .st-emotion-cache-1f8u91d .st-emotion-cache-1p6y5l4 {
        border-radius: 10px;
    }
    .st-emotion-cache-1p6y5l4:hover {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1p6y5l4 span {
        font-family: 'Inter', sans-serif;
    }
    .st-emotion-cache-13k9l8d {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Main page header
st.markdown(
    """
    <div style="text-align:center; padding: 20px; border-radius: 15px; background-color: rgba(255, 255, 255, 0.8);">
        <h1>📚 StudyMate</h1>
        <p>An AI-powered Q&A system for your academic PDFs.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# PDF Upload section
uploaded_files = st.file_uploader(
    "Upload one or more academic PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Main logic to run after file upload
if uploaded_files:
    # Process files and build index if not already done
    if not st.session_state.faiss_index or len(uploaded_files) != len(st.session_state.doc_chunks):
        st.session_state.faiss_index, st.session_state.doc_chunks = process_and_index_pdfs(
            uploaded_files, get_embedding_model()
        )
        st.session_state.model_loaded = True
        st.success("Documents processed and ready for questions!")
    
    # Create the user query input box
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # 1. Embed the query
        query_embedding = get_embedding_model().encode([user_query])
        query_embedding = np.array(query_embedding).astype('float32')

        # 2. Retrieve top-k chunks using FAISS
        k = 3
        distances, indices = st.session_state.faiss_index.search(query_embedding, k)
        
        retrieved_chunks = [
            st.session_state.doc_chunks[i] for i in indices[0]
        ]
        
        # 3. Generate answer from LLM with retrieved context
        with st.spinner("Generating answer..."):
            answer = get_answer_from_watsonx(user_query, retrieved_chunks)

        # 4. Store and display the Q&A pair
        st.session_state.history.append({
            "question": user_query,
            "answer": answer,
            "chunks": retrieved_chunks
        })
        
    # Display the Q&A history
    st.subheader("Q&A History")
    display_history()

    # Create the downloadable log button
    if st.session_state.history:
        log_content = ""
        for item in st.session_state.history:
            log_content += f"Q: {item['question']}\n"
            log_content += f"A: {item['answer']}\n"
            log_content += f"Sources:\n"
            for chunk in item['chunks']:
                log_content += f"- {chunk['source']}\n"
            log_content += "----------------------------------------\n\n"
        
        st.download_button(
            label="Download Q&A History",
            data=log_content.encode('utf-8'),
            file_name=f"studymate_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
else:
    st.info("Please upload a PDF to begin.")
