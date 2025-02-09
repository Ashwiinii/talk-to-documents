import streamlit as st
from llm_file import LLMInterface
import os


st.session_state["vector_store"] = None

llm = LLMInterface()



st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot with Streamlit")

st.sidebar.header("Model Settings")
llm_name = st.sidebar.text_input("Model Name", "phi3")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

# st.write("### Upload the files")


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
