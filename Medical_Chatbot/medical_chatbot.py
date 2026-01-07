import streamlit as st
import os
from memory_llm import get_qa_chain

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="⚕️",
    layout="centered"
)

# -------------------------
# Load Resources
# -------------------------
@st.cache_resource
def load_chain():
    """
    Caches the QA chain to avoid reloading on every interaction.
    """
    return get_qa_chain()

try:
    qa_chain = load_chain()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------
# Session State
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("⚕️ Medical Chatbot")
    st.markdown("""
    This chatbot uses RAG (Retrieval-Augmented Generation) to provide medical information based on verified documents.
    
    **Note:** This is an AI tool. Always consult a medical professional for serious health concerns.
    """)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -------------------------
# Chat Interface
# -------------------------
st.title("How can I help you today?")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Enter your medical query here..."):
    # Clearer way to handle role and content
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical records..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    pass
