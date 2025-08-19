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
import re
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Page setup & header
# -----------------------------
st.set_page_config(page_title="RAG-Based BHB Chatbot", page_icon="🔎", layout="wide")
st.title("Beta-hydroxybutyrate Chatbot")
st.caption(
    "This is an LLM model trained exclusively with scientific literature about BHB. "
    "It should give you a more precise and in-depth answer about BHB than standard LLMs like ChatGPT. "
    "It is also programmed to give references for all of its claims. ")

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


import re
import requests
from bs4 import BeautifulSoup

def extract_pmid_from_content(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"^---\s*PUBMED\s+ABSTRACT\s*\(\s*(?:PMID[: ]*)?(\d{5,9})\s*\)",
                  text, flags=re.IGNORECASE | re.MULTILINE)
    if m: return m.group(1)
    m = re.search(r"\bPMID\s*[:#]?\s*(\d{5,9})\b", text, flags=re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r"\((\d{5,9})\)", text[:200])
    return m.group(1) if m else None

def extract_pmid(doc) -> str | None:
    md = (getattr(doc, "metadata", None) or {})
    for k in ("pmid", "PMID", "source_article_id", "id"):
        v = md.get(k)
        if v not in (None, ""):
            return str(v)
    return extract_pmid_from_content(getattr(doc, "page_content", "") or "")

@st.cache_data(show_spinner=False, ttl=7*24*3600)  # ✅ add the decorator
def fetch_pubmed_title(pmid: str) -> str | None:
    if not pmid or not str(pmid).isdigit():
        return None
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Preferred signals
        m = soup.find("meta", attrs={"name": "citation_title"})
        if m and m.get("content"):
            return m["content"].strip()

        m = soup.find("meta", attrs={"property": "og:title"})
        if m and m.get("content"):
            return m["content"].strip()

        # Fallback: <title> minus trailing " - PubMed"
        t = soup.title.string if soup.title and soup.title.string else ""
        t = re.sub(r"\s*[-|]\s*PubMed.*$", "", t).strip()
        return t or None
    except Exception:
        return None

def extract_title(doc, pmid: str | None) -> str:
    # 1) Try metadata first
    md = (getattr(doc, "metadata", None) or {})
    for k in ("title", "Title", "paper_title", "document_title", "name"):
        v = md.get(k)
        if v:
            return str(v)

    # 2) Try PubMed by PMID
    t = fetch_pubmed_title(pmid) if pmid and str(pmid).isdigit() else None
    if t:
        return t

    # 3) Fallback: first non-banner line from the content
    content = (getattr(doc, "page_content", "") or "")
    for ln in content.splitlines():
        ln = ln.strip()
        if ln and not ln.upper().startswith("--- PUBMED ABSTRACT"):
            return ln if len(ln) <= 150 else ln[:150] + "…"

    return "(no title)"

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
        pmid = extract_pmid(d) or "NA"
        title = extract_title(d, pmid)
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
    search_type = st.selectbox("Retrieval mode", ["similarity", "mmr"], help="Similarity: returns the closest matching abstracts. Good for depth; may repeat similar studies.MMR (Diverse): returns a mix of relevant abstracts from different angles/models. Fewer repeats; broader view.")
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
            # Optional: show the raw context for transparency
            if show_context:
                st.markdown("### Retrieved context")
                st.code(docs_to_context(docs))

            # create_stuff_documents_chain expects docs under "context"
            result = doc_chain.invoke({"context": docs, "question": query})

            st.markdown("### Answer")
            answer_text = getattr(result, "content", result)
            st.write(answer_text)

            st.markdown("### Sources")
            for i, d in enumerate(docs, start=1):
                pmid = extract_pmid(d) or "NA"
                title = extract_title(d, pmid)
                pmid_str = str(pmid)
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/" if pmid_str.isdigit() else None

                # Row with expander on the left and a PubMed button on the right
                left, right = st.columns([0.80, 0.20])
                with left:
                    header = f"{i}. {title} — PMID {pmid_str}"
                    with st.expander(header):
                        st.write(d.page_content)
                with right:
                    if url:
                        st.link_button("View on PubMed", url, use_container_width=True)
                    else:
                        st.caption("No PubMed link")

                st.divider()
else:
    st.info("Enter a question and click **Run**.")
