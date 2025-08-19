# app.py — RAG-Based BHB Chatbot (FAISS on Google Drive)
import os
import zipfile
from typing import List

import streamlit as st
import numpy as np

# Vector store + LLM
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Utilities
import gdown

# -----------------------------
# Page setup & header
# -----------------------------
st.set_page_config(page_title="RAG-Based BHB Chatbot", page_icon="🔎", layout="wide")
st.title("🔎 RAG-Based BHB Chatbot")
st.caption(
    "This is an LLM model trained exclusively with scientific literature about BHB. "
    "It should give you more precise answer about BHB than standard LLMs like ChatGPT. "
    "It is also programmed to give references for all of its claim. "
    "The model is also very unlikely to hallucinate"
)

# -----------------------------
# Configuration via st.secrets
# -----------------------------
# REQUIRED:
#   - OPENAI_API_KEY
#   - DRIVE_FILE_ID  (Google Drive file ID pointing to a ZIP of your FAISS folder)
#   - EMBEDDING_MODEL (must match the model used to build the FAISS index)
#
# OPTIONAL:
#   - OPENAI_MODEL (default: gpt-4o-mini)
#   - INDEX_SUBDIR (subfolder name inside the ZIP; default: faiss_index)
#   - SYSTEM_PROMPT (override the default system prompt)
# -----------------------------

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
DRIVE_FILE_ID = st.secrets.get("DRIVE_FILE_ID", None)
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
INDEX_SUBDIR = st.secrets.get("INDEX_SUBDIR", "faiss_index")

# Your custom system prompt from Secrets (recommended)
CUSTOM_SYSTEM_PROMPT = st.secrets.get("SYSTEM_PROMPT", "").strip()

if not OPENAI_API_KEY or not DRIVE_FILE_ID or not EMBEDDING_MODEL:
    st.error("Missing required secrets. Please set OPENAI_API_KEY, DRIVE_FILE_ID, and EMBEDDING_MODEL in Secrets.")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def _download_from_drive(file_id: str, dest_path: str) -> str:
    """Download a file from Google Drive using gdown and return the local path."""
    url = f"https://drive.google.com/uc?id={file_id}"
    with st.spinner("Downloading FAISS index from Google Drive..."):
        out = gdown.download(url, dest_path, quiet=False)
    if out is None:
        raise RuntimeError("Failed to download FAISS index from Google Drive. Check sharing permissions & file ID.")
    return out

def _ensure_index_ready(cache_dir: str = ".cache") -> str:
    """
    Ensure the FAISS index folder exists locally. If not, download a ZIP from Drive and extract it.
    Returns the local folder path containing the FAISS index files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_zip = os.path.join(cache_dir, "faiss_index.zip")
    local_index_dir = os.path.join(cache_dir, INDEX_SUBDIR)

    # Already present? reuse
    index_files = ["index.faiss", "index.pkl"]
    if os.path.isdir(local_index_dir) and all(os.path.exists(os.path.join(local_index_dir, f)) for f in index_files):
        return local_index_dir

    # Not present: download and extract
    _download_from_drive(DRIVE_FILE_ID, local_zip)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(cache_dir)

    # If the zip didn't contain a folder, but the files are at root, move them into INDEX_SUBDIR
    if not os.path.isdir(local_index_dir):
        os.makedirs(local_index_dir, exist_ok=True)
        for member in os.listdir(cache_dir):
            if member.endswith(".faiss") or member.endswith(".pkl") or member.endswith(".json"):
                os.replace(os.path.join(cache_dir, member), os.path.join(local_index_dir, member))

    if not all(os.path.exists(os.path.join(local_index_dir, f)) for f in index_files):
        raise FileNotFoundError("Extracted index folder is missing required files (index.faiss and index.pkl).")

    return local_index_dir

def docs_to_context(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        pmid = d.metadata.get("pmid") or d.metadata.get("PMID") or d.metadata.get("id") or "NA"
        title = d.metadata.get("title") or d.metadata.get("Title") or ""
        snippet = (d.page_content or "")[:800].replace("\n", " ")
        lines.append(f"[PMID:{pmid}] {title} :: {snippet}")
    return "\n\n".join(lines)

# -----------------------------
# Load heavy resources (cached)
# -----------------------------
@st.cache_resource(show_spinner="Loading embeddings, vector store, and LLM...")
def load_resources():
    """
    Cache the heavy resources:
      - OpenAI Embeddings (MUST match the model used to build your FAISS index)
      - FAISS vector store loaded from the downloaded folder
      - ChatOpenAI LLM and RAG chain
    """
    # Environment for OpenAI
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Ensure index exists locally
    index_dir = _ensure_index_ready()

    # Embeddings used for querying the FAISS index
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Load FAISS (pickle-based metadata requires this flag)
    vectorstore = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)

    # Sanity check: embedding dimension must match index dimension
    index_dim = vectorstore.index.d
    test_vec = embeddings.embed_query("ping")
    query_dim = len(test_vec)
    if query_dim != index_dim:
        raise ValueError(
            f"Embedding mismatch: query_dim={query_dim}, index_dim={index_dim}. "
            f"Set EMBEDDING_MODEL to the one used to build the index."
        )

    # LLM
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    # System prompt
    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful scientific assistant. Answer the user's question using ONLY the provided context. "
        "If the context lacks the answer, say you don't know. Keep the answer concise and cite sources as [PMID:xxxxx]."
    )
    SYSTEM_PROMPT = CUSTOM_SYSTEM_PROMPT if CUSTOM_SYSTEM_PROMPT else DEFAULT_SYSTEM_PROMPT

    # Prompt: expects documents under the variable name "context"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nGive a direct answer first, then cite PMIDs."),
        ]
    )

    # Build a simple "stuff" chain (concat docs into {context})
    doc_chain = create_stuff_documents_chain(llm, prompt)

    return vectorstore, doc_chain, embeddings, index_dim, query_dim

# -----------------------------
# UI Controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Number of retrieved abstracts",
        min_value=1,
        max_value=50,
        value=5,
        help=(
            "Based on your prompt, the RAG system selects the most relevant abstracts to answer your prompt. "
            "With this slider, you can select the number of abstracts it uses as overall context."
        ),
    )
    search_type = st.selectbox("Retrieval mode", ["similarity", "mmr"], help="MMR reduces redundancy among results.")
    show_context = st.checkbox("Show retrieved context", value=False)
    st.divider()
    if st.button("Clear cached resources"):
        load_resources.clear()
        st.success("Cleared! Resources will be reloaded on next query.")

query = st.text_input("Ask a question about BHB", placeholder="e.g., What does BHB do to histone acetylation?")
submit = st.button("Run")

# Load resources once (cached)
vectorstore, doc_chain, embeddings, index_dim, query_dim = load_resources()
st.sidebar.caption(f"Index dim: {index_dim} | Query dim: {query_dim}")

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query: str, k: int, mode: str) -> List[Document]:
    if mode == "mmr":
        # Larger candidate pool to allow better diversity; capped at 100 to keep it snappy.
        fetch_k = max(k * 3, 10)
        fetch_k = min(fetch_k, 100)
        return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    # Pure similarity (no diversity penalty)
    return vectorstore.similarity_search(query, k=k)

# -----------------------------
# Main action
# -----------------------------
if submit and query.strip():
    with st.spinner("Retrieving and generating..."):
        docs = retrieve(query, top_k, search_type)

        if not docs:
            st.warning("No documents retrieved. Check your query or embedding/model compatibility.")
        else:
            # Optional: show the raw context for debugging/transparency
            if show_context:
                st.markdown("### Retrieved context")
                st.code(docs_to_context(docs))

            # IMPORTANT: create_stuff_documents_chain expects docs under "context"
            result = doc_chain.invoke({"context": docs, "question": query})

            st.markdown("### Answer")
            # result can be a string or an object with .content (LangChain versions vary)
            answer_text = getattr(result, "content", result)
            st.write(answer_text)

            st.markdown("### Sources")
            for i, d in enumerate(docs, start=1):
                pmid = d.metadata.get("pmid") or d.metadata.get("PMID") or d.metadata.get("id") or "NA"
                title = d.metadata.get("title") or d.metadata.get("Title") or "(no title)"
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if str(pmid).isdigit() else None

                with st.expander(f"Source {i}: PMID {pmid} — {title}"):
                    st.write(d.page_content)
                    if url:
                        st.markdown(f"[Open on PubMed]({url})")
else:
    st.info("Enter a question and click **Run**.")
