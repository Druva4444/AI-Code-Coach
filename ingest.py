import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import CODEBASE_DIR, INDEX_DIR

def build_index():
    loader = DirectoryLoader(
        path=CODEBASE_DIR,
        glob="**/*.py",
        loader_cls=TextLoader,
        exclude=["**/venv/**", "**/__pycache__/**", "**/.git/**"]
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)

    print(f"✅ Indexed {len(chunks)} code chunks.")

if __name__ == "__main__":
    build_index()