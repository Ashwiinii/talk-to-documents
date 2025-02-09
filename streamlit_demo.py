import streamlit as st
from llm_file import LLMInterface
import os
import PyPDF2
from RAG_pipeline import FAISSVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

llm = LLMInterface()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 Talk to your docs")

st.sidebar.header("Model Settings")
llm_name = st.sidebar.text_input("Model Name", "phi3")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)

st.write("### Upload the files")
uploaded_file = st.file_uploader("Upload PDF documents", type=["pdf"])
if uploaded_file :

   temp_file = "./temp.pdf"
   with open(temp_file, "wb") as file:
       file.write(uploaded_file.getvalue())
       file_name = uploaded_file.name

def extract_text_from_pdfs(temp_file):

    loader = PyPDFLoader(temp_file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    texts = [str(doc) for doc in documents]
    return texts


texts = []
if uploaded_file:
    texts = extract_text_from_pdfs(uploaded_file, temp_file)
    # st.success(f"Extracted text from {len(uploaded_files)} PDFs.")

user_query = st.text_area("Enter your question:")
# retrieved_responses = st.text_area("Enter retrieved context (comma-separated):").split(",")
# retrieved_responses.extend(retrieved_context)  # Add extracted text to the context


if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore()

vector_store = FAISSVectorStore()

# st.write(texts)

if st.button("Generate Response"):
    if user_query.strip():
        with st.spinner("Generating response..."):
            vector_store.add_texts(texts)
            context = vector_store.retrieve(user_query)
            # st.write(context)
            response = llm.query(context, user_query, llm_name, temperature=temperature)
            st.subheader("💡 Response")
            st.write(response)
    else:
        st.warning("Please enter a question before generating a response.")
