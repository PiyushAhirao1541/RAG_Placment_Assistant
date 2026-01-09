# RAG Placement Assistant 🤖

A Retrieval-Augmented Generation (RAG) based AI assistant that helps students prepare for placement interviews using interview preparation PDFs.

---

## 📌 What this project does
- Takes interview-related PDFs (HR, Technical, Company-wise)
- Converts them into embeddings
- Stores them in a FAISS vector database
- Retrieves relevant content based on user questions
- Generates short, professional answers using a FREE local LLM

---

## 🚀 Features
- 100% FREE (No OpenAI / API key required)
- Uses FAISS for fast document retrieval
- Uses HuggingFace sentence embeddings
- Uses FLAN-T5 Large as a local LLM
- Clean, bullet-point answers
- CLI + Streamlit interface

---

## 🧠 Tech Stack
- Python
- LangChain
- FAISS
- HuggingFace Transformers
- Sentence Transformers
- Streamlit

---

## 📁 Project Structure

RAG_Placment_Assistant/
├── data/
│ └── pdfs/ # Interview PDFs
├── embeddings/ # FAISS vector store
├── ingest.py # PDF ingestion & embedding creation
├── rag_pipeline.py # RAG pipeline logic
├── test_rag.py # Test RAG via terminal
├── app.py # Streamlit web app
├── requirements.txt
└── README.md
