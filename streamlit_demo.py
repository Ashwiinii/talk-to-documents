import streamlit as st
from llm_file import LLMInterface
import os
import PyPDF2

st.session_state["vector_store"] = None

llm = LLMInterface()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot with Streamlit")

st.sidebar.header("Model Settings")
llm_name = st.sidebar.text_input("Model Name", "phi3")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

st.write("### Upload the files")
uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

# Extract text from uploaded PDFs
def extract_text_from_pdfs(files):
    extracted_texts = []
    for file in files:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        extracted_texts.append(text)
    return extracted_texts

retrieved_context = []
if uploaded_files:
    retrieved_context = extract_text_from_pdfs(uploaded_files)
    st.success(f"Extracted text from {len(uploaded_files)} PDFs.")

user_query = st.text_area("Enter your question:")
# retrieved_responses = st.text_area("Enter retrieved context (comma-separated):").split(",")
# retrieved_responses.extend(retrieved_context)  # Add extracted text to the context

st.write(retrieved_context)

# if st.button("Generate Response"):
    # if user_query.strip():
    #     with st.spinner("Generating response..."):
    #         response = llm.query(retrieved_responses, user_query, llm_name, temperature=temperature)
    #         st.subheader("💡 Response")
    #         st.write(response)
    # else:
    #     st.warning("Please enter a question before generating a response.")
