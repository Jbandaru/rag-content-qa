"""Step 2: retrieve the most relevant chunks and ask the LLM to answer from them."""
import os
import sys

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

DB_PATH = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4
# Check console.groq.com for currently available models and override with GROQ_MODEL if needed
LLM_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context excerpts. "
    "If the answer is not in the context, say you don't know. "
    "Cite the source file names you used in square brackets."
)


def retrieve(question: str):
    model = SentenceTransformer(EMBED_MODEL)
    collection = chromadb.PersistentClient(path=DB_PATH).get_collection("articles")
    q_emb = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=TOP_K)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def answer(question: str) -> str:
    hits = retrieve(question)
    context = "\n\n".join(f"[{m['source']}]\n{doc}" for doc, m in hits)

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],  # set in your shell, never hard-code it
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Question: ")
    print(answer(q))
