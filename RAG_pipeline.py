import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from prompt import RAG_PROMPT_TEMPLATE

class FAISSVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", store_path="vector_store.pkl"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_data = []
        self.store_path = store_path
        # if os.path.exists(store_path):
        #     self.load_store()
        # else:
        self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())

    def add_texts(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if self.index.is_trained:
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
        self.text_data.extend(texts)
        # self.save_store()

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.text_data[i] for i in indices[0] if i < len(self.text_data)]

    # def save_store(self):
    #     with open(self.store_path, "wb") as f:
    #         pickle.dump((self.index, self.text_data), f)

    # def load_store(self):
    #     with open(self.store_path, "rb") as f:
    #         self.index, self.text_data = pickle.load(f)


def generate_response(vector_store, llm, query, model_name, temperature=0.5):
    retrieved_contexts = vector_store.retrieve(query)
    formatted_prompt = RAG_PROMPT_TEMPLATE.format(context="\n".join(retrieved_contexts), question=query)
    return llm.query(retrieved_contexts, query, model_name, temperature=temperature)

# if __name__ == "__main__":
#     vector_store = FAISSVectorStore()
#     texts = ["Example document text 1", "Example document text 2"]  # Replace with extracted text
#     vector_store.add_texts(texts)
    
#     query = "What is in the document?"
#     response = vector_store.retrieve(query)
#     print("Retrieved Context:", response)
