import os
from paper_search import search_papers
from pdf_handler import download_pdf, extract_text_from_pdf
from vector_store import VectorStore
from embedding_handler import EmbeddingGenerator
from vector_store import VectorStore
from qa_engine import generate_answer



SAVE_DIR = "pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":
    topic = input("Enter a research topic: ")
    results = search_papers(topic)

    for i, paper in enumerate(results, 1):
        print(f"\nPaper {i}")
        print("Title:", paper["title"])
        print("Authors:", ", ".join(paper["authors"]))
        print("Year:", paper["year"])
        print("Citations:", paper["citationCount"])
        print("PDF URL:", paper["pdf_url"])
        print("Abstract:", paper["abstract"])

        if paper["pdf_url"]:
            filename = f"{SAVE_DIR}/paper_{i}.pdf"
            if download_pdf(paper["pdf_url"], filename):
                print("✔️ PDF downloaded.")
                full_text = extract_text_from_pdf(filename)
                print(f"Extracted {len(full_text)} characters of text.")
            else:
                print("❌ Failed to download.")
        else:
            print("⚠️ No open-access PDF available.")

from embedding_handler import EmbeddingGenerator

# Initialize embedding handler
embedder = EmbeddingGenerator()

for i, paper in enumerate(results, 1):
    print(f"\n--- Processing Paper {i} ---")

    if paper["pdf_url"]:
        filename = f"{SAVE_DIR}/paper_{i}.pdf"
        if download_pdf(paper["pdf_url"], filename):
            print("✔️ PDF downloaded.")
            full_text = extract_text_from_pdf(filename)

            # Chunking
            chunks = embedder.chunk_text(full_text)
            print(f"📄 Split into {len(chunks)} chunks.")

            # Embedding
            vectors = embedder.embed_chunks(chunks)
            print(f"✅ Generated {len(vectors)} embeddings.")
        else:
            print("❌ Failed to download.")
    else:
        print("⚠️ No PDF available.")

# Initialize vector store
store = VectorStore(dim=vectors[0].shape[0])

# Store chunks and embeddings
store.add(vectors, chunks)
store.save()

print("💾 Stored in vector DB.")

# Load vector store
store.load()
embedder = EmbeddingGenerator()

# Ask a question
question = input("\nAsk a question about the papers: ")
query_vec = embedder.model.encode(question)
relevant_chunks = store.search(query_vec, top_k=5)

# Generate answer using LLM
answer = generate_answer(question, relevant_chunks)
print("\n🤖 Answer:")
print(answer)

