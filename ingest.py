"""Step 1: load articles, split into chunks, embed locally, store in Chroma."""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ARTICLES_DIR = Path("data/articles")
DB_PATH = "chroma_db"
CHUNK_WORDS = 200      # words per chunk
OVERLAP_WORDS = 40     # words shared between neighbouring chunks
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, free, runs on CPU


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS):
    """Split text into overlapping word windows so context isn't cut mid-idea."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)
            if words[i:i + size]]


def main():
    files = sorted(ARTICLES_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {ARTICLES_DIR}/. Add some articles first.")

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)
    # Recreate the collection so re-running ingest doesn't duplicate chunks
    try:
        client.delete_collection("articles")
    except Exception:
        pass
    collection = client.create_collection("articles")

    total = 0
    for f in files:
        chunks = chunk_text(f.read_text(encoding="utf-8", errors="ignore"))
        if not chunks:
            continue
        embeddings = model.encode(chunks).tolist()
        collection.add(
            ids=[f"{f.stem}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": f.name, "chunk": i} for i in range(len(chunks))],
        )
        total += len(chunks)
        print(f"{f.name}: {len(chunks)} chunks")

    print(f"\nDone. {len(files)} files, {total} chunks stored in {DB_PATH}/")


if __name__ == "__main__":
    main()
