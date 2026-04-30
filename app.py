import streamlit as st
import tempfile, os, shutil

from pipeline.ingestion import ingest_file
from pipeline.retrieval import HybridRetriever
from pipeline.generation import get_pipeline
from pipeline.guardrails import check_hallucination
from pipeline.config import LLM_PROVIDER
from vectorstore.chroma_manager import ChromaManager

st.set_page_config(page_title="Enterprise RAG", page_icon="🔍", layout="wide")

@st.cache_resource
def get_chroma():
    return ChromaManager()

@st.cache_resource
def get_retriever(_chroma):
    return HybridRetriever(_chroma)

chroma = get_chroma()
retriever = get_retriever(chroma)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📁 Document Upload")
    st.caption(f"LLM: **{LLM_PROVIDER.upper()}** — This demo uses free-tier LLMs and frameworks. For full performance, replace the model with a higher-capacity option such as Claude or GPT.")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, MD)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Index Documents", type="primary"):
        # Clear existing index so only the current upload is queryable
        chroma.delete_collection()
        retriever.invalidate_bm25_cache()
        st.session_state.messages = []   # reset chat — old answers belong to old docs

        progress = st.progress(0)
        for i, f in enumerate(uploaded_files):
            suffix = os.path.splitext(f.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(f, tmp)
                tmp_path = tmp.name
            try:
                chunks = ingest_file(tmp_path)
                for c in chunks:
                    c.metadata["source_file"] = f.name
                chroma.add_documents(chunks)
                retriever.invalidate_bm25_cache()
                st.success(f"✅ {f.name} — {len(chunks)} chunks")
            except Exception as e:
                st.error(f"❌ {f.name}: {e}")
            finally:
                os.unlink(tmp_path)
            progress.progress((i + 1) / len(uploaded_files))
        st.rerun()

    st.divider()
    st.metric("Documents indexed", chroma.count())
    if st.button("Clear all documents", type="secondary"):
        chroma.delete_collection()
        retriever.invalidate_bm25_cache()
        st.success("Cleared.")
        st.rerun()

# ── Main ──────────────────────────────────────────────────────
st.title("🔍 Enterprise RAG Knowledge Base")
st.caption("Ask questions about your uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if chroma.count() == 0:
            st.warning("No documents indexed yet. Upload documents in the sidebar first.")
        else:
            with st.spinner("Searching and generating answer..."):
                hits = retriever.search(question)
                pipeline = get_pipeline()
                result = pipeline.generate(question, hits)
                hallucination = check_hallucination(result["answer"], hits)

            st.markdown(result["answer"])

            with st.expander("📄 Sources"):
                for i, h in enumerate(hits, 1):
                    meta = h.get("metadata", {})
                    st.markdown(f"**[{i}]** `{meta.get('source_file','?')}` — page {meta.get('page','?')}")
                    st.caption(h["text"][:200] + "...")

            if hallucination["hallucination_detected"]:
                st.warning(
                    f"⚠️ Potential hallucination detected "
                    f"(entailment score: {hallucination['score']}). "
                    f"Verify answer against source documents."
                )
            else:
                st.success(f"✅ Answer grounded (score: {hallucination['score']})")

            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
