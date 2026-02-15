from ingestion import load_documents, split_documents
from vectorstore import create_vector_store

def main():
    print("Loading documents...")
    docs = load_documents("Data")

    print("Splitting documents...")
    split_docs = split_documents(docs)

    print("Creating vector database...")
    create_vector_store(split_docs)

    print("✅ Module 1 Completed Successfully!")

if __name__ == "__main__":
    main()