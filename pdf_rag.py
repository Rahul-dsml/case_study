import os
import base64
import streamlit as st
import faiss
import numpy as np
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize model and FAISS index globally
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
faiss_index = None
chunks = []



def parse_pdf(file_path):
    """Extracts text and images from the PDF."""
    elements = partition_pdf(file_path, strategy="hi_res", infer_table_structure=True)
    text_chunks, images = [], []

    for element in elements:
        if hasattr(element, "text") and element.text:
            text_chunks.append(element.text)
        elif isinstance(element, Image):  # Correct way to handle images
            image_base64 = base64.b64encode(element.to_bytes()).decode("utf-8")
            images.append(image_base64)

    return text_chunks, images


def chunk_text(text_chunks, images, chunk_size=500, overlap=100):
    """Chunks text with sliding window and attaches metadata."""
    chunks = []
    for i, text in enumerate(text_chunks):
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            metadata = {"chunk_id": i, "image": images[i] if i < len(images) else None}
            chunks.append(Document(page_content=chunk, metadata=metadata))
            start = end - overlap
    return chunks

def generate_embeddings(chunks):
    """Generates embeddings for text chunks."""
    return model.encode([chunk.page_content for chunk in chunks])

def create_faiss_index(embeddings):
    """Creates a FAISS index with L2 distance."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_chunks(query, index, chunks, top_k=3):
    """Retrieves top-k relevant chunks based on query."""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Streamlit UI
st.title("📄 PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        text_chunks, images = parse_pdf(file_path)
        chunks = chunk_text(text_chunks, images)
        embeddings = generate_embeddings(chunks)
        faiss_index = create_faiss_index(embeddings)
        st.success("PDF processed successfully! Ask a question below.")

query = st.text_input("Ask a question about the document:")
if query and faiss_index:
    with st.spinner("Retrieving relevant information..."):
        relevant_chunks = retrieve_chunks(query, faiss_index, chunks)
        
        for i, chunk in enumerate(relevant_chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(chunk.page_content)
            if chunk.metadata["image"]:
                st.image(base64.b64decode(chunk.metadata["image"]), caption=f"Image from Chunk {i+1}")
                st.write("\n")
