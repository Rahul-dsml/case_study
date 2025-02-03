
import streamlit as st
from pdf2image import convert_from_path
# from unstructured.partition.image import partition_image
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import base64
from groq import Groq

# Constants
API_KEY = st.secrets["API_KEY"]  
MODEL_NAME = st.secrets["MODEL_NAME"]  
EMBEDDING_MODEL = st.secrets["EMBEDDING_MODEL"]   
MAX_TOKENS = st.secrets["MAX_TOKENS"] 


class GroqImageQuery:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def query_model(self, image_path, text_prompt, extracted_text="", model=MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS):
        base64_image = self.encode_image(image_path)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"""You are an expert in identifying observations and insights from a visualization or plots or charts.
                                          Your task is to understand the user's query given below, and based on the additional context and image of visualization or chart,
                                          generate short, accurate and crisp answer along with the reasoning.
                                          Your answer must not exceed the word limit of 150 words and must contain only the generated answer with reasoning.
                                          Do not include the additional context as it is.

                                          ```User's Query: {text_prompt}```
                                          ```Additional Context: {extracted_text}```
                                          """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]
        chat_completion = self.client.chat.completions.create(
            messages=messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        return chat_completion.choices[0].message.content.strip()

# def extract_text_from_image(image_path):
#     elements = partition_image(filename=image_path)
#     return " ".join([element.text for element in elements if element.text])

def describe_image(image_path, groq_query):
    base64_image = groq_query.encode_image(image_path)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text":  """You are an expert data analyst. Your task is to describe the given image of visualization plot or chart.
                    The generated description should have plot title analysis, trend analysis, key insights, scale information, etc.
                    OUTPUT MUST NOT EXCEED THE WORD LIMIT OF 300 WORDS."""},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ]
    chat_completion = groq_query.client.chat.completions.create(
        messages=messages, model=MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS
    )
    return chat_completion.choices[0].message.content.strip()

def embed_text(text_list):
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(text_list)

def process_uploaded_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    images = convert_from_path("temp.pdf")
    return images

def main():
    st.title("Conversational PDF Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_reference" not in st.session_state:
        st.session_state.selected_reference = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = []
    if "extracted_texts" not in st.session_state:
        st.session_state.extracted_texts = []

    groq_query = GroqImageQuery(API_KEY)
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if st.button("Process PDF"):
    # if uploaded_file:
        st.write("Processing PDF...")
        images = process_uploaded_pdf(uploaded_file)
        st.session_state.extracted_texts, st.session_state.image_paths = [], []

        for i, image in enumerate(images):
            image_path = f"page_{i}.jpg"
            image.save(image_path, "JPEG")
            st.session_state.image_paths.append(image_path)
            st.session_state.extracted_texts.append(describe_image(image_path, groq_query))

        embeddings = embed_text(st.session_state.extracted_texts)
        st.session_state.index = faiss.IndexFlatL2(embeddings.shape[1])
        st.session_state.index.add(embeddings)
        print(len(embeddings), len(st.session_state.image_paths), len(st.session_state.extracted_texts))
    if st.session_state.index:
        st.write("Ready for queries!")
        user_query = st.chat_input("Ask a question about your document:")

        if user_query:
            query_embedding = embed_text([user_query])
            distances, indices = st.session_state.index.search(query_embedding, 1)
            idx = indices[0][0]
            selected_image, selected_text = st.session_state.image_paths[idx], st.session_state.extracted_texts[idx]

            st.session_state.chat_history.append({"role": "user", "text": user_query})
            response = groq_query.query_model(selected_image, user_query, selected_text)
            st.session_state.chat_history.append({"role": "assistant", "text": response, "reference": idx})

        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message(chat["role"]):
                st.write(chat["text"])
                if "reference" in chat:
                    unique_key = f"ref_{chat['reference']}_{i}"
                    if st.button(f"🔍 View Reference (Page {chat['reference'] + 1})", key=unique_key):
                        st.session_state.selected_reference = chat["reference"]

        if st.session_state.selected_reference is not None:
            with st.sidebar:
                ref_idx = st.session_state.selected_reference
                st.image(st.session_state.image_paths[ref_idx], caption=f"Reference: Page {ref_idx + 1}", use_container_width=True)
                st.write("**Extracted Text:**")
                st.write(st.session_state.extracted_texts[ref_idx])

if __name__ == "__main__":
    main()
