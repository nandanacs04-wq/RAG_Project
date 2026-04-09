from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

# Load vector database
db = FAISS.load_local("faiss_index", embeddings=None)

def get_rag_response(query, chat_history):
    # Combine last few messages for context
    context = ""
    for msg in chat_history[-3:]:
        context += msg["role"] + ": " + msg["content"] + "\n"

    retriever = db.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=retriever
    )

    response = qa.run(context + query)

    return response
