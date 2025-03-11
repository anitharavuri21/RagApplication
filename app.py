import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import chromadb
from chromadb.utils import embedding_functions
from google.generativeai import configure, GenerativeModel
import os

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCwkfBb9cc8TlqpI4FszfP8-lTcB0RtCac"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="pdf_chunks",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def store_text_in_chromadb(text):
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[str(i)])

st.title("RAG with ChromaDB & Gemini")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file type.")
        text = None
    
    if text:
        store_text_in_chromadb(text)
        st.success("File processed and stored in ChromaDB.")

query = st.text_input("Ask a question based on the document:")
if query:
    results = collection.query(query_texts=[query], n_results=5)
    retrieved_texts = "\n".join([doc for doc in results["documents"][0]])
    
    # Generate response using Gemini
    model = GenerativeModel("gemini-2.0-flash-exp")
    prompt=f'''answer the questions {query} from this {retrieved_texts}
    if the query is not relevant to the provided text.. you can answer the query.
    don't provide any explanations, introductions or whether the answer to that question is provided in the text or not.
    only output the answer to the question directly.'''
    response = model.generate_content(prompt)
    
    st.subheader("Answer:")
    st.write(response.text)
