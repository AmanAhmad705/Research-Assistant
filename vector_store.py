import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, dim, store_dir="vector_store"):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def add(self, embeddings, chunks):
        """
        Add new embeddings and associated chunks to the vector store.
        """
        self.index.add(np.array(embeddings).astype("float32"))
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, top_k=5):
        """
        Search for the most similar text chunks.
        """
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), top_k)
        return [self.text_chunks[i] for i in I[0]]

    def save(self):
        """
        Save the FAISS index and text chunks.
        """
        faiss.write_index(self.index, f"{self.store_dir}/index.faiss")
        with open(f"{self.store_dir}/chunks.pkl", "wb") as f:
            pickle.dump(self.text_chunks, f)

    def load(self):
        """
        Load the FAISS index and text chunks.
        """
        self.index = faiss.read_index(f"{self.store_dir}/index.faiss")
        with open(f"{self.store_dir}/chunks.pkl", "rb") as f:
            self.text_chunks = pickle.load(f)
