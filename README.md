# Content Q&A Assistant (RAG)

Ask questions over a folder of articles. Retrieval-Augmented Generation in Python.

**Stack:** sentence-transformers (local embeddings) · ChromaDB (vector store) · Groq API (LLM, OpenAI-compatible)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key"                       # Windows: setx GROQ_API_KEY "your-key"
```

## Run
1. Save 20-30 public articles as `.txt` files in `data/articles/`
2. Index them: `python ingest.py`
3. Ask: `python ask.py "What did the article say about ...?"`

## How it works
1. **Chunk**: each article is split into ~200-word overlapping chunks.
2. **Embed**: chunks become vectors using `all-MiniLM-L6-v2` (runs locally, free).
3. **Retrieve**: the question is embedded and the 4 closest chunks are pulled from Chroma.
4. **Generate**: the chunks plus the question go to the LLM, told to answer only from that context and cite sources.

## Sample runs
Question: python ask.py "how many ballondor's does messi have?"
Answer: Lionel Messi has won **eight Ballon d'Or** awards【Lionel_Messi.txt】.

Question: python ask.py "how many ballondor's does cristiano have?"
Answer: I don’t know. 【Cristiano_Ronaldo.txt】

Question: python ask.py "how many ballondor's does ronaldo have?"
Answer: Cristiano Ronaldo has won **five** Ballon d’Or awards【Lionel_Messi.txt】, which notes “Ronaldo's five” in the comparison of their individual achievements. 【Cristiano_Ronaldo.txt】 also lists his wins in 2008, 2013, 2014, 2016 and 2017, confirming the total of five.