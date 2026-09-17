import hashlib
import os
import re

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide",
)


GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5


def get_groq_api_key():
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    return api_key


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""

        if text.strip():
            pages.append(text)

    return "\n".join(pages)


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def build_faiss_index(
    chunks,
    embedding_model,
):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


def retrieve_chunks(
    question,
    chunks,
    index,
    embedding_model,
):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32,
    )

    number_of_results = min(
        TOP_K,
        len(chunks),
    )

    scores, indices = index.search(
        question_embedding,
        number_of_results,
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0],
    ):
        if index_number < 0:
            continue

        results.append(
            {
                "text": chunks[int(index_number)],
                "score": float(score),
            }
        )

    return results


def generate_answer(
    client,
    question,
    retrieved_chunks,
):
    context = "\n\n".join(
        f"Source {number}:\n{item['text']}"
        for number, item in enumerate(
            retrieved_chunks,
            start=1,
        )
    )

    system_prompt = (
        "You are a helpful PDF question-answering assistant. "
        "Answer the user's question using only the supplied "
        "document context. Do not invent facts and do not use "
        "outside knowledge. If the answer is not contained "
        "in the context, say that you could not find the "
        "information in the uploaded PDF. Keep answers clear "
        "and concise."
    )

    user_prompt = (
        "Document context:\n\n"
        + context
        + "\n\nUser question:\n"
        + question
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def reset_document():
    keys = [
        "document_hash",
        "document_name",
        "chunks",
        "index",
        "messages",
    ]

    for key in keys:
        st.session_state.pop(
            key,
            None,
        )


st.title("📚 RAG PDF Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)


api_key = get_groq_api_key()


if not api_key:
    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY to Streamlit Secrets."
    )

    st.stop()


client = Groq(
    api_key=api_key
)


with st.sidebar:

    st.header("RAG Configuration")

    st.write(
        f"LLM: {GROQ_MODEL}"
    )

    st.write(
        "Vector database: FAISS"
    )

    st.write(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    st.write(
        f"Retrieved chunks: {TOP_K}"
    )

    st.divider()

    if st.button("Clear document"):
        reset_document()
        st.rerun()


uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
)


if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    document_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    if (
        st.session_state.get("document_hash")
        != document_hash
    ):

        with st.spinner(
            "Reading and processing the PDF..."
        ):

            try:

                text = extract_pdf_text(
                    uploaded_file
                )

                text = clean_text(
                    text
                )

                if not text:

                    st.error(
                        "No readable text was found "
                        "in this PDF."
                    )

                    st.stop()

                chunks = create_chunks(
                    text
                )

                if not chunks:

                    st.error(
                        "No text chunks were created."
                    )

                    st.stop()

                embedding_model = (
                    load_embedding_model()
                )

                index = build_faiss_index(
                    chunks,
                    embedding_model
                )

                st.session_state.document_hash = (
                    document_hash
                )

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.index = (
                    index
                )

                st.session_state.messages = []

            except Exception as error:

                st.error(
                    "The PDF could not be processed."
                )

                st.exception(error)

                st.stop()


if "chunks" in st.session_state:

    st.success(
        f"Ready: "
        f"{st.session_state.document_name} "
        f"({len(st.session_state.chunks)} chunks)"
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


if "chunks" not in st.session_state:

    st.info(
        "Upload a PDF to start."
    )

else:

    question = st.chat_input(
        "Ask a question about your PDF"
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(
                question
            )

        with st.chat_message("assistant"):

            try:

                embedding_model = (
                    load_embedding_model()
                )

                with st.spinner(
                    "Searching the PDF..."
                ):

                    retrieved_chunks = (
                        retrieve_chunks(
                            question,
                            st.session_state.chunks,
                            st.session_state.index,
                            embedding_model,
                        )
                    )

                with st.spinner(
                    "Generating the answer..."
                ):

                    answer = generate_answer(
                        client,
                        question,
                        retrieved_chunks,
                    )

                st.markdown(
                    answer
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                with st.expander(
                    "Retrieved document context"
                ):

                    for number, item in enumerate(
                        retrieved_chunks,
                        start=1,
                    ):

                        st.markdown(
                            f"Source {number} | "
                            f"similarity: "
                            f"{item['score']:.3f}"
                        )

                        st.write(
                            item["text"]
                        )

            except Exception as error:

                st.error(
                    "An error occurred while "
                    "answering the question."
                )

                st.exception(error)
