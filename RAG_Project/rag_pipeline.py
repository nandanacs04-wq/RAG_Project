# rag_pipeline.py
# Retrieval-Augmented Generation (RAG) Pipeline using FAISS, HuggingFace, and Groq

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# -------------------------------
# Step 1: Load environment variables
# -------------------------------
# This loads the GROQ_API_KEY from the .env file
print("DEBUG: Loaded GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))
load_dotenv()


# -------------------------------
# Step 2: Initialize embedding model
# -------------------------------
# Lightweight embedding model for converting text into vectors
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}
)


# -------------------------------
# Step 3: Load FAISS vector database
# -------------------------------
# Loads previously created FAISS index
vector_db = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever from vector database
retriever = vector_db.as_retriever()


# -------------------------------
# Step 4: Initialize Groq LLM
# -------------------------------
# Load Groq model using API key from environment variable
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)


# -------------------------------
# Step 5: Question Answering Function
# -------------------------------
def ask_question(question: str):
    """
    Retrieves relevant documents from FAISS and generates an answer using Groq LLM.

    Args:
        question (str): User question
    """

    print("\nSearching for relevant documents...")

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Handle case when no documents are found
    if not documents:
        print("No relevant documents found.")
        return

    # Combine retrieved document contents
    context = "\n".join([doc.page_content for doc in documents])

    # Create prompt for LLM
    prompt = f"""
You are an AI assistant. Answer the question based only on the context provided below.

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate response using Groq model
    response = llm.invoke(prompt)

    # Display result
    print("\nAnswer:")
    print(response.content)


# -------------------------------
# Optional: Test run
# -------------------------------
if __name__ == "__main__":
    user_question = input("Enter your question: ")
    ask_question(user_question)