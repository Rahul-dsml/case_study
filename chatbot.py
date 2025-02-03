from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np
import base64
from unstructured.partition.pdf import partition_pdf

# Step 1: Parse the PDF and Extract Text and Images
def parse_pdf(file_path):
    elements = partition_pdf(file_path, strategy="hi_res", infer_table_structure=True, extract_images_in_pdf=True)
    text_chunks = []
    images = []

    for element in elements:
        if hasattr(element, "text"):
            text_chunks.append(element.text)
        elif hasattr(element, "image"):
            image_base64 = base64.b64encode(element.image).decode("utf-8")
            images.append(image_base64)

    return text_chunks, images

# Step 2: Chunk the Text and Attach Metadata
def chunk_text(text_chunks, images, chunk_size=500, overlap=100):
    chunks = []
    for i, text in enumerate(text_chunks):
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            metadata = {
                "chunk_id": i,
                "image": images[i] if i < len(images) else None,
            }
            chunks.append({"text": chunk, "metadata": metadata})
            start = end - overlap

    return chunks

# Step 3: Generate Embeddings
def generate_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([chunk["text"] for chunk in chunks])
    return embeddings

# Step 4: Create FAISS Vector Store
def create_faiss_store(chunks, embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Step 5: Build Conversational Chatbot
def build_chatbot(file_path):
    # Parse PDF
    text_chunks, images = parse_pdf(file_path)

    # Chunk Text
    chunks = chunk_text(text_chunks, images)

    # Generate Embeddings
    embeddings = generate_embeddings(chunks)

    # Create FAISS Vector Store
    vector_store = FAISS.from_texts(
        texts=[chunk["text"] for chunk in chunks],
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        metadatas=[chunk["metadata"] for chunk in chunks],
    )

    # Initialize Conversation Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Define Prompt Template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question. If you don't know the answer, say you don't know.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
    )

    # Initialize Chat Model
    chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-4")

    # Create Conversational Retrieval Chain
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
    )

    return chatbot

# Step 6: Run Chatbot
def run_chatbot(chatbot):
    print("Welcome to the PDF Chatbot! Type 'exit' to end the conversation.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        # Get response from chatbot
        response = chatbot({"question": query})
        print(f"Bot: {response['answer']}")

# Main Execution
if __name__ == "__main__":
    # Path to the PDF file
    pdf_file_path = "image.png"  # Replace with your PDF file path

    # Build Chatbot
    chatbot = build_chatbot(pdf_file_path)

    # Run Chatbot
    run_chatbot(chatbot)