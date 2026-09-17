import os
import re
import tempfile

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# --------------------------------------------------

# PAGE CONFIG

# --------------------------------------------------

st.set_page_config(
page_title="RAG PDF Assistant",
page_icon="📚",
layout="wide"
)

# --------------------------------------------------

# CUSTOM CSS

# --------------------------------------------------

st.markdown(
""" <style>
.title {
text-align: center;
font-size: 42px;
font-weight: bold;
margin-bottom: 5px;
}

```
.subtitle {
    text-align: center;
    color: #777777;
    font-size: 18px;
    margin-bottom: 30px;
}

.source-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f5f5;
    margin-top: 10px;
}
</style>
""",
unsafe_allow_html=True
```

)

# --------------------------------------------------

# TITLE

# --------------------------------------------------

st.markdown(
'<div class="title">📚 RAG PDF Assistant</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">'
"Upload a PDF and ask questions about its content"
"</div>",
unsafe_allow_html=True
)

# --------------------------------------------------

# SETTINGS

# --------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-120b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# --------------------------------------------------

# GROQ API KEY

# --------------------------------------------------

def get_api_key():

```
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None

if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

return api_key
```

api_key = get_api_key()

if not api_key:

```
st.error("GROQ_API_KEY was not found.")

st.info(
    "Add GROQ_API_KEY to Streamlit Cloud Secrets."
)

st.stop()
```

# --------------------------------------------------

# GROQ CLIENT

# --------------------------------------------------

client = Groq(
api_key=api_key
)

# --------------------------------------------------

# EMBEDDING MODEL

# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

```
return SentenceTransformer(
    EMBEDDING_MODEL
)
```

# --------------------------------------------------

# PDF TEXT EXTRACTION

# --------------------------------------------------

def extract_text_from_pdf(uploaded_file):

```
pdf_bytes = uploaded_file.getvalue()

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".pdf"
) as temp_file:

    temp_file.write(pdf_bytes)

    temp_path = temp_file.name

try:

    reader = PdfReader(temp_path)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)

finally:

    if os.path.exists(temp_path):
        os.remove(temp_path)
```

# --------------------------------------------------

# TEXT CLEANING

# --------------------------------------------------

def clean_text(text):

```
text = re.sub(
    r"\s+",
    " ",
    text
)

return text.strip()
```

# --------------------------------------------------

# TEXT CHUNKING

# --------------------------------------------------

def create_chunks(
text,
chunk_size=CHUNK_SIZE,
overlap=CHUNK_OVERLAP
):

```
if not text:
    return []

chunks = []

start = 0
text_length = len(text)

while start < text_length:

    end = start + chunk_size

    chunk = text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    if end >= text_length:
        break

    start = end - overlap

return chunks
```

# --------------------------------------------------

# CREATE FAISS DATABASE

# --------------------------------------------------

def create_vector_database(
chunks,
embedding_model
):

```
embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(embeddings)

return index
```

# --------------------------------------------------

# SEARCH VECTOR DATABASE

# --------------------------------------------------

def search_documents(
question,
chunks,
index,
embedding_model
):

```
question_embedding = embedding_model.encode(
    [question],
    convert_to_numpy=True,
    normalize_embeddings=True
)

question_embedding = np.asarray(
    question_embedding,
    dtype=np.float32
)

k = min(
    TOP_K,
    len(chunks)
)

scores, indices = index.search(
    question_embedding,
    k
)

results = []

for score, index_number in zip(
    scores[0],
    indices[0]
):

    if index_number < 0:
        continue

    results.append(
        {
            "text": chunks[index_number],
            "score": float(score)
        }
    )

return results
```

# --------------------------------------------------

# GENERATE ANSWER

# --------------------------------------------------

def generate_answer(
question,
retrieved_documents
):

```
context = "\n\n---\n\n".join(
    document["text"]
    for document in retrieved_documents
)

system_prompt = """
```

You are a helpful RAG document assistant.

Answer the user's question using only the
information provided in the document context.

Rules:

* Do not invent information.
* Do not use outside knowledge.
* If the answer cannot be found in the context,
  say that the information was not found in
  the uploaded document.
* Give a clear and direct answer.
* Use bullet points when useful.
  """

  user_prompt = f"""
  DOCUMENT CONTEXT:

{context}

QUESTION:

{question}
"""

```
response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    temperature=0.2
)

return response.choices[0].message.content
```

# --------------------------------------------------

# SIDEBAR

# --------------------------------------------------

with st.sidebar:

```
st.header("⚙️ RAG Settings")

st.write(
    f"**LLM:** {GROQ_MODEL}"
)

st.write(
    "**Vector DB:** FAISS"
)

st.write(
    f"**Embedding:** {EMBEDDING_MODEL}"
)

st.write(
    f"**Top K:** {TOP_K}"
)

st.divider()

st.write(
    "📌 Upload a PDF and ask questions "
    "about its content."
)
```

# --------------------------------------------------

# PDF UPLOAD

# --------------------------------------------------

uploaded_file = st.file_uploader(
"📄 Upload your PDF",
type=["pdf"]
)

# --------------------------------------------------

# PROCESS PDF

# --------------------------------------------------

if uploaded_file:

```
file_id = (
    uploaded_file.name,
    uploaded_file.size
)

if st.session_state.get(
    "file_id"
) != file_id:

    with st.spinner(
        "Processing PDF..."
    ):

        try:

            # Extract text
            text = extract_text_from_pdf(
                uploaded_file
            )

            if not text.strip():

                st.error(
                    "No readable text was found "
                    "in this PDF."
                )

                st.stop()

            # Clean text
            text = clean_text(text)

            # Create chunks
            chunks = create_chunks(text)

            if not chunks:

                st.error(
                    "No text chunks could be created."
                )

                st.stop()

            # Load embedding model
            embedding_model = (
                load_embedding_model()
            )

            # Create FAISS index
            index = create_vector_database(
                chunks,
                embedding_model
            )

            # Save in session
            st.session_state.file_id = file_id

            st.session_state.chunks = chunks

            st.session_state.index = index

            st.session_state.embedding_model = (
                embedding_model
            )

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.messages = []

        except Exception as error:

            st.error(
                "Something went wrong while "
                "processing the PDF."
            )

            st.exception(error)

            st.stop()
```

# --------------------------------------------------

# DOCUMENT STATUS

# --------------------------------------------------

if "chunks" in st.session_state:

```
st.success(
    f"✅ {st.session_state.document_name} "
    "is ready!"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Text Chunks",
        len(st.session_state.chunks)
    )

with col2:

    st.metric(
        "Vector Database",
        "FAISS"
    )
```

# --------------------------------------------------

# CHAT HISTORY

# --------------------------------------------------

if "messages" not in st.session_state:

```
st.session_state.messages = []
```

for message in st.session_state.messages:

```
with st.chat_message(
    message["role"]
):

    st.markdown(
        message["content"]
    )
```

# --------------------------------------------------

# CHAT

# --------------------------------------------------

if "chunks" in st.session_state:

```
question = st.chat_input(
    "Ask something about your PDF..."
)

if question:

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Retrieve
    with st.spinner(
        "🔎 Searching your document..."
    ):

        documents = search_documents(
            question,
            st.session_state.chunks,
            st.session_state.index,
            st.session_state.embedding_model
        )

    # Generate
    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    documents
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:

                st.error(
                    "Could not generate an answer."
                )

                st.exception(error)

    # Sources
    with st.expander(
        "🔎 View retrieved sources"
    ):

        for number, document in enumerate(
            documents,
            start=1
        ):

            st.markdown(
                f"**Source {number}** "
                f"(similarity: "
                f"{document['score']:.3f})"
            )

            st.write(
                document["text"]
            )

            st.divider()
```

else:

```
st.info(
    "👆 Upload a PDF to start."
)
```

````

### One important correction

I would **not** add random package versions just to make the file look complete. The two files above intentionally use a small dependency set:

**PDF:** `pypdf`  
**Chunking:** Python  
**Embeddings:** `sentence-transformers`  
**Vector DB:** `faiss-cpu`  
**Frontend:** Streamlit

Your Groq key should go in **Streamlit Cloud → App Settings → Secrets**, not in GitHub.

For local testing, you can use:

```toml
GROQ_API_KEY = "your_groq_api_key"
````

