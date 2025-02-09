RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following retrieved context to answer the user's question. 
If the context does not contain relevant information, return "I do not know".
    
Context:
{context}

Question:
{question}

Answer:
"""