print("🔥 INGEST FILE STARTED 🔥")

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH = "data/pdfs"
DB_PATH = "embeddings/vector_store"

def load_documents():
    documents = []
    print("📂 Looking for PDFs in:", DATA_PATH)

    if not os.path.exists(DATA_PATH):
        print("❌ PDF folder not found!")
        return documents

    files = os.listdir(DATA_PATH)
    print("📄 Files found:", files)

    for file in files:
        if file.lower().endswith(".pdf"):
            print("➡️ Loading:", file)
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)

if __name__ == "__main__":

    docs = load_documents()

    if len(docs) == 0:
        print("❌ No documents loaded. Stopping.")
    else:
        print(f"✅ Loaded {len(docs)} pages")

        chunks = split_documents(docs)
        print(f"✂️ Created {len(chunks)} chunks")

        create_vector_store(chunks)
        print("🎉 VECTOR STORE CREATED SUCCESSFULLY!")
