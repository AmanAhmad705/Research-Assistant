import streamlit as st
from paper_search import search_papers
from pdf_handler import download_pdf, extract_text_from_pdf
from embedding_handler import EmbeddingGenerator
from vector_store import VectorStore
from qa_engine import generate_answer
import os
#from dotenv import load_dotenv

# Load .env file
#load_dotenv()


SAVE_DIR = "pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

embedder = EmbeddingGenerator()
store = None  # Will initialize after papers are loaded

st.set_page_config(page_title="PhD Research Assistant", layout="wide")
st.title("🧠 PhD Research Assistant (RAG System)")

# Step 1: Search Papers
topic = st.text_input("🔍 Enter a research topic:")
if st.button("Search & Process Papers"):
    with st.spinner("Searching and downloading papers..."):
        results = search_papers(topic)
        all_chunks = []
        all_embeddings = []

        for i, paper in enumerate(results, 1):
            st.subheader(f"📄 Paper {i}: {paper['title']}")
            st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
            st.markdown(f"**Year:** {paper['year']}")
            st.markdown(f"**Citations:** {paper['citationCount']}")
            st.markdown(f"**Abstract:** {paper['abstract']}")

            pdf_url = paper["pdf_url"]
            if pdf_url:
                filename = f"{SAVE_DIR}/paper_{i}.pdf"
                if download_pdf(pdf_url, filename):
                    full_text = extract_text_from_pdf(filename)
                    chunks = embedder.chunk_text(full_text)
                    vectors = embedder.embed_chunks(chunks)

                    all_chunks.extend(chunks)
                    all_embeddings.extend(vectors)
                else:
                    st.warning("❌ Failed to download PDF.")
            else:
                st.warning("⚠️ No open-access PDF available.")

        # Save to FAISS vector DB
        if all_embeddings:
            store = VectorStore(dim=len(all_embeddings[0]))
            store.add(all_embeddings, all_chunks)
            store.save()
            st.success("✅ Papers processed and stored!")

# Step 2: Ask Questions
st.markdown("---")
st.subheader("❓ Ask a question about the loaded papers:")
user_question = st.text_area("Your Question:")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    elif not os.path.exists("vector_store/index.faiss"):
        st.warning("No papers loaded yet. Please search and process papers first.")
    else:
        with st.spinner("Thinking..."):
            if store is None:
                store = VectorStore(dim=384)
                store.load()
            query_vec = embedder.model.encode(user_question)
            retrieved = store.search(query_vec, top_k=5)
            answer = generate_answer(user_question, retrieved)
            st.success("💡 Answer:")
            st.write(answer)

            with st.expander("🔍 See retrieved context chunks"):
                for chunk in retrieved:
                    st.markdown(f"> {chunk.strip()}")
