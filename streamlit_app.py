import streamlit as st
from logic import process_query
import os
from dotenv import load_dotenv

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

st.set_page_config(page_title="AI Code Coach", page_icon="🤖", layout="wide")

st.title("🤖 AI Code Coach")
st.markdown("Your RAG-powered assistant for debugging, translating, and explaining code.")

# Sidebar for configuration/info
with st.sidebar:
    st.header("Settings")
    task_type = st.radio(
        "Choose Task:",
        ["Debug Code", "Translate Code", "Explain Algorithm"],
        index=0
    )
    
    st.divider()
    st.info("Make sure you have run `python ingest.py` to index your codebase.")

# Mapping task names to choice strings
task_map = {
    "Debug Code": "1",
    "Translate Code": "2",
    "Explain Algorithm": "3"
}

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about your code..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            choice = task_map[task_type]
            # Call our shared logic
            result, docs = process_query(choice, prompt)
            
            st.markdown(result)
            
            # Show sources in an expander
            with st.expander("View Retrieved Context"):
                for doc in docs:
                    source = doc.metadata.get('source', 'Unknown')
                    st.caption(f"Source: {source}")
                    st.code(doc.page_content, language='python')

    st.session_state.messages.append({"role": "assistant", "content": result})
