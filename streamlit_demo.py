import streamlit as st
from llm_file import LLMInterface
from RAG_pipeline import FAISSVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import os

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Initialize LLM interface
llm = LLMInterface()

# Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 Talk to your docs")

# Sidebar settings
st.sidebar.header("Model Settings")

# Display a dropdown to select the model name
llm_name = st.sidebar.selectbox("Model Name", ("phi4"))

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

# File upload section
st.write("### Upload the files")
uploaded_file = st.file_uploader("Upload PDF documents", type=["pdf"])


# Function to extract text from PDFs
def extract_text_from_pdfs(temp_file):
    loader = PyPDFLoader(temp_file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    texts = [str(doc) for doc in documents]
    return texts


# Process uploaded file
texts = []
if uploaded_file:
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
    texts = extract_text_from_pdfs(temp_file)

# User query input
user_query = st.text_area("Enter your question:")

# Initialize vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore()

vector_store = st.session_state.vector_store

# Generate response
if st.button("Generate Response"):
    if user_query.strip():
        with st.spinner("Generating response..."):
            vector_store.add_texts(texts)
            context = vector_store.retrieve(user_query)
            response = llm.query(context, user_query, llm_name, temperature=temperature)
            st.subheader("💡 Response")
            st.write(response)
    else:
        st.warning("Please enter a question before generating a response.")
