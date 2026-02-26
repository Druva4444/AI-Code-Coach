# AI Code Coach

AI Code Coach is a Retrieval-Augmented Generation (RAG) based tool designed to help developers debug, translate, and understand their codebase. It indexes your local Python files and uses a Large Language Model (LLM) to provide context-aware suggestions and explanations.

## 🚀 Features

- **Debug Code**: Identify root causes and suggest fixes for issues in your codebase.
- **Translate Code**: Translate code snippets into different programming languages or paradigms.
- **Explain Algorithm**: Get clear explanations of the logic and design patterns used in your code.
- **Local Indexing**: Rapidly searches your codebase using FAISS and HuggingFace embeddings.

## 🛠️ Prerequisites

- Python 3.10+
- A Groq API Key (get one at [console.groq.com](https://console.groq.com/))

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Druva4444/AI-Code-Coach.git
   cd AI-Code-Coach
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file or export your Groq API key:
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

## 📖 Usage

### 1. Index your Codebase
Place the code you want to analyze in the `codebase/` directory. Then, run the ingestion script to build the vector index:
```bash
python3 ingest.py
```

### 2. Run the AI Code Coach
Start the interactive CLI tool:
```bash
python3 app.py
```
Follow the on-screen prompts to choose a task (Debug, Translate, or Explain) and describe your problem.

## 📁 Project Structure

- `app.py`: Main entry point for the CLI application.
- `ingest.py`: Script to build the FAISS vector index from the `codebase/` directory.
- `retriever.py`: Handles loading the vector index and retrieving relevant code snippets.
- `llm.py`: Configuration for the Groq LLM (using `llama-3.1-8b-instant`).
- `prompts.py`: Logic for different AI task prompt templates.
- `config.py`: Centralized configuration for directory paths and API keys.
- `codebase/`: Put the source files you want the AI to "read" here.
- `vector_index/`: Stores the generated FAISS index files.

## 🧰 Technologies Used

- [LangChain](https://github.com/langchain-ai/langchain): Framework for building LLM applications.
- [FAISS](https://github.com/facebookresearch/faiss): Efficient similarity search and clustering of dense vectors.
- [Groq](https://groq.com/): High-performance LLM inference.
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): For generating code vector representations.
