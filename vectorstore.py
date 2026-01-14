from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingestion import load_documents

def create_vectorstore():
    # Load PDF chunks
    docs = load_documents("data/Reinforcement_Learning.pdf")

    # Create embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create FAISS index
    db = FAISS.from_documents(docs, embeddings)

    # Save locally
    db.save_local("faiss_index")

    print(f"✅ Vectorstore created with {len(docs)} chunks")

if __name__ == "__main__":
    create_vectorstore()
