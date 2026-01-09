from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
import os

DB_PATH = "embeddings/vector_store"

# 1️⃣ Embeddings (FREE)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2️⃣ Load vector store
vectorstore = FAISS.load_local(
    DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# 3️⃣ Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4️⃣ Prompt (Grounded RAG)
prompt = PromptTemplate.from_template(
    """
You are an expert placement interview assistant.

Your task:
- Read the context
- Understand the meaning
- Rewrite the answer clearly
- Use bullet points (max 5)
- Remove duplicates and repeated lines
- Do NOT copy text word by word
- Make the response professional and short

If answer is not found in context:
say: "I don't have enough information from the documents."

Context:
{context}

Question:
{question}

Answer:
"""
)


# 5️⃣ FREE LOCAL LLM (FLAN-T5)
llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=512
)

def local_llm(prompt_text: str):
    result = llm_pipeline(prompt_text)
    text = result[0]["generated_text"]

    # ➤ Cleanup formatting:
    lines = text.split(". ")
    bullets = []
    for line in lines:
        cleaned = line.strip()
        if len(cleaned) > 5:
            bullets.append(f"- {cleaned}")

    return "\n".join(bullets[:5])


# 6️⃣ RAG Function
def ask_rag(question: str):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    return local_llm(final_prompt)
