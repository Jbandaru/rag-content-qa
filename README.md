# Content Q&A Assistant (RAG)

A Python Retrieval-Augmented Generation (RAG) app that answers questions over a folder of articles and cites the source files it used. Demo corpus: 25 Wikipedia articles on soccer.

**Stack:** Python · sentence-transformers (local embeddings) · ChromaDB (vector store) · Groq API (Llama, via the OpenAI-compatible client)

## How it works
1. **Ingest** (`ingest.py`): split each article into ~200-word chunks with 40-word overlap, embed them locally with `all-MiniLM-L6-v2`, store in Chroma.
2. **Retrieve** (`ask.py`): embed the question and fetch the 4 nearest chunks.
3. **Generate**: send the chunks and question to the LLM with instructions to answer only from the context, cite sources, and say "I don't know" when the answer isn't there.

## Project structure
```
rag-content-qa/
├── ask.py              # retrieve chunks and ask the LLM
├── ingest.py           # chunk, embed and index the articles
├── fetch_articles.py   # optional: download the pages listed in urls.txt
├── urls.txt            # source URLs, one per line
├── data/articles/      # plain-text articles that get indexed (.txt)
└── requirements.txt
```

## Data
- **`urls.txt`** lists the source pages, one URL per line. The demo uses 25 Wikipedia pages on soccer. Replace them with any public pages on a topic of your choice.
- **`fetch_articles.py`** reads `urls.txt`, extracts the main text of each page, and saves it as a `.txt` file in `data/articles/`. Any URL that fails is printed as "failed" and skipped.
- **`data/articles/`** is the folder `ingest.py` reads. It holds one `.txt` file per article. You can skip `urls.txt` and add your own `.txt` files here directly.
- The downloaded articles are not committed to the repo (see `.gitignore`), so run `fetch_articles.py` after cloning. Re-run `ingest.py` whenever the articles change.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key"                       # Windows: setx GROQ_API_KEY "your-key"
python fetch_articles.py                             # downloads the pages in urls.txt into data/articles/
python ingest.py
python ask.py "How does the offside rule work?"
```
Set `GROQ_MODEL` to override the default model.

## Example
<!-- Paste one real question and answer from your own run here -->

## Design choices
- **Local embeddings:** free, no second API, easy to swap for a hosted model.
- **Overlapping chunks:** avoids cutting an idea in half at a chunk boundary.
- **Grounded prompt:** the model must answer from retrieved context and cite files, which reduces made-up answers.

## Limitations and next steps
- Fixed-size chunking ignores article structure; section-aware chunking would help.
- No re-ranking or answer evaluation yet.
- Possible extensions: auto-tagging and summarizing new articles, a simple agent that chooses between search and summarize.

## Sample runs
Question: python ask.py "how many ballondor's does messi have?"
Answer: Lionel Messi has won **eight Ballon d'Or** awards【Lionel_Messi.txt】.

Question: python ask.py "how many ballondor's does cristiano have?"
Answer: I don’t know. 【Cristiano_Ronaldo.txt】

Question: python ask.py "how many ballondor's does ronaldo have?"
Answer: Cristiano Ronaldo has won **five** Ballon d’Or awards【Lionel_Messi.txt】, which notes “Ronaldo's five” in the comparison of their individual achievements. 【Cristiano_Ronaldo.txt】 also lists his wins in 2008, 2013, 2014, 2016 and 2017, confirming the total of five.