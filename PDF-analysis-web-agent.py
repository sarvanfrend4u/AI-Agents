import hashlib
from io import BytesIO
import streamlit as st  
from pypdf import PdfReader
from dotenv import load_dotenv

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    SEMANTIC_DEPS_AVAILABLE = True
    SEMANTIC_IMPORT_ERROR = ""
except Exception as semantic_import_error:
    SEMANTIC_DEPS_AVAILABLE = False
    SEMANTIC_IMPORT_ERROR = str(semantic_import_error)

# Essential 2026 ADK Imports
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

# 1. Setup Environment
st.set_page_config(page_title="PDF Analysis Agent", page_icon="PDF")
st.title("PDF Analysis Agent")
load_dotenv()

# 2. PDF utilities
@st.cache_data(show_spinner=False)
def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF byte stream."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages).strip()


@st.cache_data(show_spinner=False)
def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200):
    """Split text into overlapping chunks."""
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == length:
            break
    return chunks


@st.cache_resource(show_spinner=False)
def _get_embedding_model():
    if not SEMANTIC_DEPS_AVAILABLE:
        return None
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _build_faiss_index(chunks):
    if not SEMANTIC_DEPS_AVAILABLE:
        return None, "Semantic dependencies are not available."

    model = _get_embedding_model()
    if model is None:
        return None, "Embedding model is unavailable."

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.size == 0:
        return None, "Unable to build embeddings for this PDF."

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, ""


def _ensure_pdf_loaded() -> bool:
    return bool(st.session_state.get("pdf_uploaded", False))


def read_uploaded_pdf() -> str:
    """Reads the uploaded PDF content for the agent."""
    if not _ensure_pdf_loaded():
        return "No PDF uploaded. Please upload a PDF to begin."
    text = st.session_state.get("pdf_text", "")
    return text if text else "The uploaded PDF has no extractable text."


def search_uploaded_pdf(query: str, top_k: int = 3) -> str:
    """Return the most relevant chunks using semantic retrieval from FAISS."""
    if not _ensure_pdf_loaded():
        return "No PDF uploaded. Please upload a PDF to begin."

    if not query.strip():
        return "Please provide a more specific query."

    chunks = st.session_state.get("pdf_chunks", [])
    if not chunks:
        return "The uploaded PDF has no extractable text."

    if not SEMANTIC_DEPS_AVAILABLE:
        return (
            "Semantic retrieval dependencies are missing. Install: "
            "'faiss-cpu', 'sentence-transformers', and 'numpy'. "
            f"Import error: {SEMANTIC_IMPORT_ERROR}"
        )

    faiss_index = st.session_state.get("pdf_faiss_index")
    if faiss_index is None:
        index_error = st.session_state.get("pdf_index_error", "")
        if index_error:
            return f"Semantic index is unavailable: {index_error}"
        return "Semantic index is not ready yet. Please re-upload the PDF."

    model = _get_embedding_model()
    if model is None:
        return "Embedding model is unavailable."

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    k = max(1, min(top_k, len(chunks)))
    scores, indices = faiss_index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append(f"[Section {idx + 1} | Similarity {score:.3f}]\n{chunks[idx]}")

    if not results:
        return "No relevant sections found."
    return "\n\n".join(results)


# 3. Initialize ADK Components
@st.cache_resource
def setup_runtime():
    agent = Agent(
        name="pdf_analysis_agent",
        model="gemini-2.0-flash",
        instruction=(
            "You are a PDF Analysis Agent. Use 'search_uploaded_pdf' to retrieve "
            "the most relevant sections, then answer the question. Use "
            "'read_uploaded_pdf' only if you need broader context."
        ),
        tools=[search_uploaded_pdf, read_uploaded_pdf],
    )
    return Runner(
        app_name="pdf_app",
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )


runner = setup_runtime()

# 4. Streamlit Chat State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. PDF Upload UI
uploaded_file = st.file_uploader("Drop a PDF here", type=["pdf"], key="pdf_uploader")
if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    st.session_state["pdf_bytes"] = pdf_bytes
    st.session_state["pdf_name"] = uploaded_file.name
    st.session_state["pdf_uploaded"] = True
else:
    pdf_bytes = st.session_state.get("pdf_bytes")

if pdf_bytes:
    pdf_text = _extract_pdf_text(pdf_bytes)
    chunks = _chunk_text(pdf_text)

    st.session_state["pdf_text"] = pdf_text
    st.session_state["pdf_chunks"] = chunks

    # Build semantic index only when the uploaded document changes.
    doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
    if st.session_state.get("pdf_doc_hash") != doc_hash:
        st.session_state["pdf_doc_hash"] = doc_hash
        st.session_state["pdf_faiss_index"] = None
        st.session_state["pdf_index_error"] = ""

        if chunks and SEMANTIC_DEPS_AVAILABLE:
            with st.spinner("Building semantic index..."):
                faiss_index, index_error = _build_faiss_index(chunks)
            st.session_state["pdf_faiss_index"] = faiss_index
            st.session_state["pdf_index_error"] = index_error
        elif not SEMANTIC_DEPS_AVAILABLE:
            st.session_state["pdf_index_error"] = (
                "Missing dependencies for semantic retrieval."
            )

    if pdf_text:
        st.success(f"Loaded: {st.session_state.get('pdf_name', 'uploaded PDF')}")
        if st.session_state.get("pdf_index_error"):
            st.warning(f"Semantic retrieval issue: {st.session_state['pdf_index_error']}")
    else:
        st.warning("Uploaded PDF has no extractable text.")

# Redraw Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 6. Execution Loop (Streamlit-safe)
def _extract_text_from_event(event) -> str:
    """Extracts model text from an ADK Event."""
    if not hasattr(event, "content") or not event.content or not event.content.parts:
        return ""
    chunks = []
    for part in event.content.parts:
        if getattr(part, "text", None):
            chunks.append(part.text)
    return "".join(chunks)


def get_ai_response(prompt):
    user_input = types.Content(role="user", parts=[types.Part(text=prompt)])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        try:
            events = runner.run(
                user_id="default_user",
                session_id="pdf_session",
                new_message=user_input,
            )

            for event in events:
                # Surface model errors if present
                if getattr(event, "error_message", None):
                    st.error(f"Model error: {event.error_message}")

                # Stream text as it arrives
                chunk = _extract_text_from_event(event)
                if chunk:
                    full_text += chunk
                    placeholder.markdown(full_text + "|")

                # Show tool activity
                if event.get_function_calls():
                    st.caption("Checking document...")

            placeholder.markdown(full_text or "(No response from model)")
            st.session_state.messages.append({"role": "assistant", "content": full_text})

        except Exception as e:
            st.error(f"ADK Error: {e}")


# 7. User Input
if prompt := st.chat_input("Ask a question about the PDF"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    get_ai_response(prompt)
