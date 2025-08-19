# Streamlit RAG with FAISS (Google Drive-hosted index)

This template shows how to deploy a **RAG app** on **Streamlit Community Cloud** while hosting a prebuilt **FAISS** index on **Google Drive**.

## Quick Start

1) **Prepare your FAISS index (LangChain format)**

```python
# Example when you originally build the index:
vectorstore.save_local("faiss_index")
# This creates a folder containing: index.faiss, index.pkl (and possibly docstore.json)
```

Zip the folder so you have `faiss_index.zip`. Upload that zip to Google Drive and set sharing to:
**Anyone with the link → Viewer**.

Copy the file ID from the sharing link. Example link:
`https://drive.google.com/file/d/1AbCDeFgHiJKLmNoPQ/view?usp=sharing`  
File ID is the long string after `/d/` (here: `1AbCDeFgHiJKLmNoPQ`).

2) **Fork/clone this repo or upload the files to a new GitHub repo.**

3) **Set Streamlit secrets**

On Streamlit Community Cloud, open **App → Settings → Secrets** and paste:

```toml
OPENAI_API_KEY = "sk-..."
DRIVE_FILE_ID = "your_google_drive_file_id_here"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_MODEL = "gpt-4o-mini"
INDEX_SUBDIR = "faiss_index"  # (only change if your zip contains a differently named folder)
```

> **IMPORTANT:** `EMBEDDING_MODEL` **must match** the model used to build the FAISS index (dimension must be identical).

4) **Deploy**

Point Streamlit Cloud at `app.py` in your repo. First run will **download** the zip from Drive, **extract** the FAISS folder into a local cache, and **load** it. Subsequent runs reuse the cached files.

## Notes

- Large indexes may exceed Streamlit's memory limits. Consider sharding or moving to a hosted vector DB if that happens.
- `allow_dangerous_deserialization=True` is required by LangChain FAISS loader because it unpickles metadata. Only load indexes you trust.
- The app shows references as `[PMID:xxxxx]` and links to PubMed when `pmid` metadata exists.
