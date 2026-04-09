import streamlit as st
from rag_pipeline import get_rag_response

st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("📄 AI-Based Document Search Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Ask your question...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # Get response from RAG
    response = get_rag_response(user_input, st.session_state.messages)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)
