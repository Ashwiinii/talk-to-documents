import streamlit as st
from typing import List
from openai import OpenAI

# Define the RAG prompt template
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following retrieved context to answer the user's question. 
If the context does not contain relevant information, respond based on general knowledge.
    
Context:
{context}

Question:
{question}

Answer:
"""

# Define LLMInterface
class LLMInterface:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )

    def query(
        self,
        retrieved_responses: List[str],
        user_query: str,
        llm_name: str = "gemma:2b",
        system_role: str = "You are a helpful assistant.",
        temperature: float = 0.0,
    ) -> str:
        # Format context for the prompt
        context_str = "\n".join(retrieved_responses) if retrieved_responses else "No relevant context found."
        prompt = RAG_PROMPT_TEMPLATE.format(question=user_query, context=context_str)

        # Create message structure
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=llm_name, messages=messages, temperature=temperature
            )
            response = completion.choices[0].message.content  # Fix attribute access
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            response = "API call encountered an error."

        return response


# Initialize the LLM
llm = LLMInterface()

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot with Streamlit")

# Sidebar settings
st.sidebar.header("Model Settings")
llm_name = st.sidebar.text_input("Model Name", "gemma:2b")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

# User input
user_query = st.text_area("Enter your question:")
retrieved_responses = st.text_area("Enter retrieved context (comma-separated):").split(",")

if st.button("Generate Response"):
    if user_query.strip():
        with st.spinner("Generating response..."):
            response = llm.query(retrieved_responses, user_query, llm_name, temperature=temperature)
            st.subheader("💡 Response")
            st.write(response)
    else:
        st.warning("Please enter a question before generating a response.")
