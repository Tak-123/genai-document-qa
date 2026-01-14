📄 GenAI Document Question Answering System (RAG)

This project is a Retrieval-Augmented Generation (RAG) based Document Question Answering system that allows users to ask natural language questions over PDF documents. The application processes documents by splitting them into meaningful chunks, converts them into vector embeddings, and stores them in a FAISS vector database for efficient semantic search.

When a user asks a question, the system retrieves the most relevant document chunks using similarity search and passes them as context to a language model to generate accurate, context-aware answers. The project uses LangChain, FAISS, and HuggingFace’s FLAN-T5 model, enabling the entire pipeline to run locally without relying on paid APIs.

This project demonstrates a complete end-to-end GenAI workflow, including document ingestion, vector indexing, retrieval, and answer generation. It is well-suited for showcasing practical skills in Generative AI, NLP, and Large Language Models, and can be highlighted as a strong portfolio project for AI/ML or Data-focused roles.
