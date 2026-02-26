import streamlit as st
from logic import process_query, apply_fix
import os
from dotenv import load_dotenv
import difflib

# Load environment variables (for GROQ_API_KEY)
load_dotenv()

st.set_page_config(page_title="AI Code Coach", page_icon="🤖", layout="wide")

st.title("🤖 AI Code Coach")
st.markdown("Your RAG-powered assistant with **One-Click Fix**.")

# Sidebar for configuration/info
with st.sidebar:
    st.header("Settings")
    task_type = st.radio(
        "Choose Task:",
        ["Debug Code", "Translate Code", "Explain Algorithm"],
        index=0
    )
    
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Mapping task names to choice strings
task_map = {
    "Debug Code": "1",
    "Translate Code": "2",
    "Explain Algorithm": "3"
}

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If this message was an assistant response with a fix, we don't re-render the apply button 
        # for old messages to avoid confusion, but we could if we wanted to.
        # For simplicity, we only show 'Apply' on the latest response.

# User input
if prompt := st.chat_input("Ask about your code..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            choice = task_map[task_type]
            result, docs, fix_info = process_query(choice, prompt)
            
            st.markdown(result)
            
            # One-Click Fix UI
            if fix_info:
                st.info(f"💡 AI suggested a fix for `{fix_info['file_path']}`")
                
                # Show Diff
                if os.path.exists(fix_info["file_path"]):
                    with open(fix_info["file_path"], 'r') as f:
                        old_content = f.read()
                    
                    diff = difflib.unified_diff(
                        old_content.splitlines(),
                        fix_info["new_content"].splitlines(),
                        fromfile="original",
                        tofile="fixed",
                        lineterm=""
                    )
                    diff_text = "\n".join(list(diff))
                    
                    if diff_text:
                        with st.expander("🔍 View Proposed Changes"):
                            st.code(diff_text, language='diff')
                        
                        if st.button("Apply Fix", key=f"apply_{len(st.session_state.messages)}"):
                            success, msg = apply_fix(fix_info["file_path"], fix_info["new_content"])
                            if success:
                                st.success(msg)
                            else:
                                st.error(f"Error applying fix: {msg}")
                    else:
                        st.write("Current file already matches the suggested fix.")
                else:
                    st.warning(f"File `{fix_info['file_path']}` not found on disk.")

            # Show context sources
            with st.expander("View Retrieved Context"):
                for doc in docs:
                    source = doc.metadata.get('source', 'Unknown')
                    st.caption(f"Source: {source}")
                    st.code(doc.page_content, language='python')

    st.session_state.messages.append({"role": "assistant", "content": result})
