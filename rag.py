from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load local LLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
text_gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def answer_question(question):
    # Load FAISS vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Retrieve top 3 chunks
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Prompt for the local LLM
    prompt = f"""
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    # Generate answer
    result = text_gen(prompt, max_length=300)
    return result[0]['generated_text']

if __name__ == "__main__":
    q = input("Ask a question: ")
    print(answer_question(q))
