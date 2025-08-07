from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )

    def chunk_text(self, text):
        """
        Split raw text into overlapping chunks.
        """
        return self.splitter.split_text(text)

    def embed_chunks(self, chunks):
        """
        Generate embeddings for each chunk.
        """
        return self.model.encode(chunks)
