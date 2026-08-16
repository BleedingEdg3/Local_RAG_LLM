# Local RAG Chatbot — Ollama + Gemma 4 (E2B) + ChromaDB

A fully local, retrieval-augmented chatbot. Instead of relying on a model's memorized knowledge, it retrieves relevant chunks from your own documents at query time and grounds every answer in that context — with source citations, running entirely offline.

## Why this exists

Fine-tuning (see my [QLoRA Qwen2.5 project](https://github.com/BleedingEdg3/fine_tuning_qwen_0.5B)) is great for teaching a model a *skill* or *style*. RAG is the better tool when you need a model to answer questions about *facts* that change often or that you don't want baked into the weights — like personal notes, internal docs, or anything updated after the model's training cutoff. This project implements that pattern end-to-end, from scratch.

## Features

- **Fully offline** — generation, embeddings, and vector search all run locally via [Ollama](https://ollama.com) and ChromaDB; no external API calls
- **Multi-format ingestion** — loads `.pdf`, `.md`, and `.txt` documents from a local folder
- **From-scratch RAG pipeline** — chunking, embedding, retrieval, and prompt construction are all implemented directly (no LangChain), so every step is transparent and easy to modify or debug
- **Grounded, cited answers** — the system prompt constrains the model to answer only from retrieved context and cite which source file it used
- **Multi-turn chat** — conversation history is preserved across turns within a session
- **Zero-setup demo** — auto-seeds a couple of sample documents if the knowledge base folder is empty, so the pipeline is testable immediately after cloning

## Architecture

```
                     ┌─────────────────────┐
                     │ data/knowledge_base │
                     │ (.pdf / .md / .txt) │
                     └──────────┬──────────┘
                                │  load + chunk
                                ▼
                     ┌─────────────────────┐
                     │  nomic-embed-text   │  (via Ollama)
                     │  (embedding model)  │
                     └──────────┬──────────┘
                                │  vectors
                                ▼
                     ┌─────────────────────┐
                     │      ChromaDB       │  (persistent, local)
                     └──────────┬──────────┘
                                │  top-k retrieval
                                │
User query ──────────►  build grounded prompt
                                │
                                ▼
                     ┌─────────────────────┐
                     │     gemma4:e2b      │  (via Ollama)
                     │  (generation model) │
                     └──────────┬──────────┘
                                ▼
                          Cited answer
```

## Tech stack

| Component | Choice |
|---|---|
| Generation model | `gemma4:e2b` (served via Ollama) |
| Embedding model | `nomic-embed-text` (served via Ollama) |
| Vector store | ChromaDB (persistent local client) |
| Document parsing | `pypdf` for PDFs, native I/O for Markdown/text |
| Orchestration | Plain Python — no LangChain or other RAG framework |

## Setup

**Prerequisites:** [Ollama](https://ollama.com/download) installed locally.

```bash
# 1. Start the Ollama server (keep running in a separate terminal)
ollama serve

# 2. Pull the required models
ollama pull gemma4:e2b
ollama pull nomic-embed-text

# 3. Install Python dependencies
pip install ollama chromadb pypdf tqdm
```

## Usage

1. Clone the repo and open `Local_RAG_Chatbot_Ollama_Gemma4.ipynb` in Jupyter
2. (Optional) Drop your own `.pdf` / `.md` / `.txt` files into `data/knowledge_base/`. If left empty, the notebook seeds it with two sample documents so you can test the full pipeline immediately.
3. Run all cells top to bottom:
   - Sections 1–2 configure and verify the local models
   - Sections 3–5 load, chunk, embed, and index your documents into ChromaDB
   - Sections 6–7 implement retrieval and grounded generation
   - Section 9 launches an interactive chat loop directly in the notebook (type `exit` to quit)

To re-index after changing your documents, delete `data/chroma_db/` and re-run the indexing cells.

## Project structure

```
.
├── Local_RAG_Chatbot_Ollama_Gemma4.ipynb # full pipeline + chat loop
│
├── data/
│   ├── knowledge_base/                   # source documents (.pdf/.md/.txt)
│   └── chroma_db/                        # persisted ChromaDB collection 
└── README.md
```

## Configuration

Key parameters are exposed at the top of the notebook and easy to tune:

| Parameter | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 800 chars | Size of each text chunk |
| `CHUNK_OVERLAP` | 120 chars | Overlap between consecutive chunks |
| `TOP_K` | 4 | Number of chunks retrieved per query |

## Verified behavior

Tested end-to-end against the seeded sample knowledge base:

- Both local models (`gemma4:e2b`, `nomic-embed-text`) pull and serve correctly via Ollama
- Retrieval correctly ranks the relevant source chunk first — for a QLoRA-related query, the QLoRA notes chunk returned the lowest vector distance versus unrelated chunks
- The grounding constraint holds under test: when asked a question outside the indexed knowledge base, the bot declines to answer rather than hallucinating a response

These checks were run against the small seed corpus (2 documents). Retrieval quality on a larger, real-world document set is expected to vary — see the evaluation harness below.

## Roadmap

- [ ] Retrieval evaluation harness (precision/recall against a labeled Q&A set, similar to the BERTScore evaluation used in the QLoRA fine-tuning project)
- [ ] Sentence/token-aware chunking instead of fixed-size windows
- [ ] Incremental indexing (upsert) instead of full collection rebuild
- [ ] Side-by-side comparison across different local generation models (e.g. `gemma4:e2b` vs `gemma3n:e2b` vs `qwen2.5:0.5b-instruct`)

## License

MIT
