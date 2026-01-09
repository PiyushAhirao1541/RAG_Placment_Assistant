from rag_pipeline import ask_rag

question = "What are common HR interview questions?"
answer = ask_rag(question)

print("\nANSWER:\n")
print(answer)
