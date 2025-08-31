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
    "It is also programmed to give references for all of its claims. "
)

# -----------------------------
# Configuration via st.secrets
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
DRIVE_FILE_ID = st.secrets.get("DRIVE_FILE_ID", None)
EMBEDDING_MODEL = st.secrets.get("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
INDEX_SUBDIR = st.secrets.get("INDEX_SUBDIR", "faiss_index")
CUSTOM_SYSTEM_PROMPT = st.secrets.get("SYSTEM_PROMPT", "").strip()

if not OPENAI_API_KEY or not DRIVE_FILE_ID or not EMBEDDING_MODEL:
    st.error("Missing required secrets. Please set OPENAI_API_KEY, DRIVE_FILE_ID, and EMBEDDING_MODEL in Secrets.")
    st.stop()

# -----------------------------
# Example prompts (≤15 words each)
# -----------------------------
EXAMPLE_PROMPTS = [
    "Does BHB alter chromatin structure in humans. If so, via which mechanisms?",
    "Do exogenous BHB interventions improve adult cognition?",
    "Does BHB modulate NLRP3, list potential molecular mechanisms involved in the regulation of NLRP3",
    "Does exogenous BHB improve insulin sensitivity in IR/T2D?",
    "How do ketone esters affect endurance performance/recovery; subgroup by study population?",
    "Via which mechanisms might BHB alleviate non-alcoholic liver disease?",
    "Via which mechanisms might BHB alleviate traumatic brain injury?",
    "Summarize the safety of the ketone monoester, only focus on human studies"
]

# -----------------------------
# PubMed helpers
# -----------------------------
def extract_pmid_from_content(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"^---\s*PUBMED\s+ABSTRACT\s*\(\s*(?:PMID[: ]*)?(\d{5,9})\s*\)", text, flags=re.IGNORECASE | re.MULTILINE)
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

@st.cache_data(show_spinner=False, ttl=7*24*3600)
def fetch_pubmed_title(pmid: str) -> str | None:
    if not pmid or not str(pmid).isdigit():
        return None
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        m = soup.find("meta", attrs={"name": "citation_title"})
        if m and m.get("content"):
            return m["content"].strip()
        m = soup.find("meta", attrs={"property": "og:title"})
        if m and m.get("content"):
            return m["content"].strip()
        t = soup.title.string if soup.title and soup.title.string else ""
        t = re.sub(r"\s*[-|]\s*PubMed.*$", "", t).strip()
        return t or None
    except Exception:
        return None

def extract_title(doc, pmid: str | None) -> str:
    md = (getattr(doc, "metadata", None) or {})
    for k in ("title", "Title", "paper_title", "document_title", "name"):
        v = md.get(k)
        if v:
            return str(v)
    t = fetch_pubmed_title(pmid) if pmid and str(pmid).isdigit() else None
    if t:
        return t
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
    url = f"https://drive.google.com/uc?id={file_id}"
    with st.spinner("Downloading FAISS index from Google Drive..."):
        out = gdown.download(url, dest_path, quiet=False)
    if out is None:
        raise RuntimeError("Failed to download FAISS index from Google Drive. Check sharing permissions & file ID.")
    return out

def _ensure_index_ready(cache_dir: str = ".cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local_zip = os.path.join(cache_dir, "faiss_index.zip")
    local_index_dir = os.path.join(cache_dir, INDEX_SUBDIR)
    index_files = ["index.faiss", "index.pkl"]
    if os.path.isdir(local_index_dir) and all(os.path.exists(os.path.join(local_index_dir, f)) for f in index_files):
        return local_index_dir
    _download_from_drive(DRIVE_FILE_ID, local_zip)
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(cache_dir)
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

def linkify_pmids_md(text: str) -> str:
    if not text:
        return ""

    def num_to_link(m: re.Match) -> str:
        n = m.group(1)
        return f"[{n}](https://pubmed.ncbi.nlm.nih.gov/{n}/)"

    # 1) [PMID: ...] — linkify 5–9 digit numbers inside the brackets
    def repl_bracket(m: re.Match) -> str:
        inner = m.group(1)
        linked_inner = re.sub(r"\b(\d{5,9})\b", num_to_link, inner)
        return f"[PMID:{linked_inner}]"

    text = re.sub(r"\[PMID:\s*(.*?)\]", repl_bracket, text, flags=re.IGNORECASE)

    # 2) Plain "PMID: 12345, 67890" — linkify numbers after the label
    def repl_plain(m: re.Match) -> str:
        nums = m.group(1)
        linked = re.sub(r"\b(\d{5,9})\b", num_to_link, nums)
        return f"PMID:{linked}"

    # ✅ add the missing third argument: `text`
    text = re.sub(r"(?i)\bPMID[:\s]+\s*([0-9][0-9,\s]{4,})", repl_plain, text)

    return text


# -----------------------------
# Load heavy resources (cached)
# -----------------------------
@st.cache_resource(show_spinner="Loading embeddings, vector store, and LLM...")
def load_resources():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    index_dir = _ensure_index_ready()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    index_dim = vectorstore.index.d
    test_vec = embeddings.embed_query("ping")
    query_dim = len(test_vec)
    if query_dim != index_dim:
        raise ValueError(
            f"Embedding mismatch: query_dim={query_dim}, index_dim={index_dim}. "
            f"Set EMBEDDING_MODEL to the one used to build the index."
        )
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful scientific assistant. Answer the user's question using ONLY the provided context. "
        "If the context lacks the answer, say you don't know. Keep the answer concise and cite sources as [PMID:xxxxx]."
    )
    SYSTEM_PROMPT = CUSTOM_SYSTEM_PROMPT if CUSTOM_SYSTEM_PROMPT else DEFAULT_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nGive a direct answer first, then cite PMIDs."),
        ]
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return vectorstore, doc_chain, embeddings, index_dim, query_dim

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider(
        "Number of retrieved abstracts",
        min_value=1,
        max_value=100,
        value=30,
        help=(
            "Based on your prompt, the RAG system selects the most relevant abstracts to answer your prompt. "
            "With this slider, you can select the number of abstracts it uses as overall context."
        ),
    )
    search_type = st.selectbox(
        "Retrieval mode",
        ["similarity", "mmr"],
        help=(
            "Similarity: returns the closest matching abstracts. Good for depth; may repeat similar studies."
            "MMR (Diverse): returns a mix of relevant abstracts from different angles/models. Fewer repeats; broader view."
        ),
    )
    show_context = st.checkbox("Show retrieved context", value=False)
    st.divider()
    if st.button("Clear cached resources"):
        load_resources.clear()
        st.success("Cleared! Resources will be reloaded on next query.")

# Load resources (cached once)
vectorstore, doc_chain, embeddings, index_dim, query_dim = load_resources()
st.sidebar.caption(f"Index dim: {index_dim} | Query dim: {query_dim}")

# -----------------------------
# CSS spinner INSIDE text input (centered; toggled by a hidden flag)
# -----------------------------


# -----------------------------
# Pre-run handler for example buttons (before text_input uses key='query')
# -----------------------------
pending = st.session_state.pop("pending_query", None)
if pending is not None:
    st.session_state["query"] = pending
    st.session_state["auto_submit"] = True

# -----------------------------
# Query input + Run button
# -----------------------------
query = st.text_input(
    "Ask a question about BHB",
    placeholder="e.g., What does BHB do to histone acetylation?",
    key="query",
)
submit = st.button("Run") or st.session_state.pop("auto_submit", False)

# ---- NEW: placeholder right under the text box for the spinner text ----
spinner_slot = st.empty()
results_slot = st.container()  # persistent area for results


# -----------------------------
# One-click example prompts (shown BELOW the textbox)
# -----------------------------
with st.expander("View example prompts", expanded=False):
    cols = st.columns(2)
    picked = None
    for i, p in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(p, key=f"ex_{i}"):
            picked = p
    if picked is not None:
        st.session_state["pending_query"] = picked
        st.session_state["auto_submit"] = True
        st.rerun()

# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query: str, k: int, mode: str) -> List[Document]:
    if mode == "mmr":
        fetch_k = max(k * 3, 10)
        fetch_k = min(fetch_k, 100)
        return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    return vectorstore.similarity_search(query, k=k)

# -----------------------------
# Main action
# -----------------------------
docs = []
answer_md = ""

if submit and query.strip():
    # Turn ON in-input spinner
    flag_slot.markdown('<div id="loading-flag" style="display:none"></div>', unsafe_allow_html=True)

    # Compute inside the spinner, but DO NOT render results here
    with spinner_slot.container():
        with st.spinner("Retrieving and generating..."):
            docs = retrieve(query, top_k, search_type)
            if docs:
                result = doc_chain.invoke({"context": docs, "question": query})
                answer_text = getattr(result, "content", result)
                answer_md = linkify_pmids_md(str(answer_text))

    # Turn OFF spinner(s)
    spinner_slot.empty()
    flag_slot.empty()

    # --- Render persistently outside the spinner ---
    with results_slot:
        if not docs:
            st.warning("No documents retrieved. Check your query or embedding/model compatibility.")
        else:
            if show_context:
                st.markdown("### Retrieved context")
                st.code(docs_to_context(docs))

            st.markdown("### Answer")
            st.markdown(answer_md)

            st.markdown("### Sources")
            for i, d in enumerate(docs, start=1):
                pmid = extract_pmid(d) or "NA"
                title = extract_title(d, pmid)
                pmid_str = str(pmid)
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_str}/" if pmid_str.isdigit() else None

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
    st.info("Enter a question and click **Run**, or choose an example below.")

