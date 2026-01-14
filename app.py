import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from ingestion import load_documents
import os

st.set_page_config(page_title="Intelligent Document Q&A", layout="wide")
st.title("📄 Intelligent Document Q&A System")

# ---------------------------
# Initialize LLM
# ---------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    text_gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return text_gen

text_gen = load_llm()

# ---------------------------
# Load or create FAISS vectorstore
# ---------------------------
@st.cache_resource
def get_vectorstore(pdf_file=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    if pdf_file:
        # Save uploaded PDF to data folder
        os.makedirs("data", exist_ok=True)
        pdf_path = os.path.join("data", pdf_file.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        # Load PDF chunks
        docs = load_documents(pdf_path)

        # Create FAISS vectorstore
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index")
    else:
        # Load existing FAISS index
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    return db

# ---------------------------
# Upload PDF
# ---------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    st.info(f"Processing `{uploaded_file.name}` ...")
    db = get_vectorstore(uploaded_file)
    st.success("Vectorstore created successfully!")
else:
    if os.path.exists("faiss_index"):
        db = get_vectorstore()
        st.success("Loaded existing FAISS index")
    else:
        st.warning("Upload a PDF to start")
        st.stop()

# ---------------------------
# Ask a question
# ---------------------------
question = st.text_input("Ask a question about your documents:")

if question:
    # Retrieve top 3 chunks
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Prompt for LLM
    prompt = f"""
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    # Generate answer
    with st.spinner("Generating answer..."):
        result = text_gen(prompt, max_new_tokens=256)

    st.subheader("Answer:")
    st.write(result[0]['generated_text'])

    st.subheader("Source Chunks:")
    for i, doc in enumerate(docs, 1):
        st.markdown(f"**Chunk {i}:** {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
