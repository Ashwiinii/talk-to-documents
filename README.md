# talk-to-documents

A very simple project to run upload documents and "talk" to them using locally hosted models. The motivation of this project is to keep your documents safe and offline, and still get valuable insights from them. 
Local hosting of models is done using Ollama and the project currently supports only one locally hosted model -- phi4. You could use any model that you see fit --  as long as Ollama supports it. 
Here is a wonderful [blog post](https://www.ralgar.one/ollama-on-windows-a-beginners-guide/) that walks you through setting up Ollama on Windows, pulling models and running them locally. 

This project works using local RAG (Retrieval-Augmented Generation) powered by FAISS. Your documents are vectorized and indexed using `IndexFlatL2` in FAISS. A look-up (similarity search) is performed for every question you ask to return top 3 similar, contextually relevant information. This is then injected into the prompt for the LLM to understand the question, along with the "hints and context" for it to generate a crisp, informative response.

To start the Streamlit app on your localhost, run:
`streamlit run streamlit_demo.py`

## ✨ Future Work

This is an ongoing project -- just like the ever-evolving world of LLMs, and I intend to integrate the following functionalities:
- Support for multiple models
- PDF parsing improvements
- Improve file handling to address memory issues 

