import streamlit as st
from llm_file import LLMInterface
import os
import PyPDF2
from RAG_pipeline import FAISSVectorStore

llm = LLMInterface()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot with Streamlit")

st.sidebar.header("Model Settings")
llm_name = st.sidebar.text_input("Model Name", "phi3")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

st.write("### Upload the files")
uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"])

def extract_text_from_pdfs(file):
    
    reader = PyPDF2.PdfReader(file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

texts = []
if uploaded_files:
    texts = extract_text_from_pdfs(uploaded_files)
    # st.success(f"Extracted text from {len(uploaded_files)} PDFs.")

user_query = st.text_area("Enter your question:")
# retrieved_responses = st.text_area("Enter retrieved context (comma-separated):").split(",")
# retrieved_responses.extend(retrieved_context)  # Add extracted text to the context

# st.write(retrieved_context)

#     vector_store = FAISSVectorStore()
#     texts = ["Example document text 1", "Example document text 2"]  # Replace with extracted text
#     vector_store.add_texts(texts)
    
#     query = "What is in the document?"
#     response = vector_store.retrieve(query)
#     print("Retrieved Context:", response)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore()

vector_store = FAISSVectorStore()

# st.write(texts)
vector_store.add_texts(texts)
context = vector_store.retrieve(user_query)
st.write(context)

# if st.button("Generate Response"):
#     if user_query.strip():
#         with st.spinner("Generating response..."):
#             response = llm.query(retrieved_context, user_query, llm_name, temperature=temperature)
#             st.subheader("💡 Response")
#             st.write(response)
#     else:
#         st.warning("Please enter a question before generating a response.")
