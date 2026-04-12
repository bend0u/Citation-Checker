# 🔬 Sci-Verify: Verified Generative Information Retrieval for Science

**Sci-Verify** is a SaaS platform that addresses the **Provenance Problem** in AI-assisted research. It uses an Adversarial RAG (Retrieval-Augmented Generation) pipeline powered by the **DeepTRACE** auditing framework to ensure every AI-generated scientific claim is factually supported by peer-reviewed literature.

> Built for NTU SC4052 — Topic 5: Generative Information Retrieval for Science

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Deep Search** | Enter a scientific question → get a synthesized, fully audited report with real citations from OpenAlex |
| 🔎 **Hallucination Check** | Upload a ChatGPT export (`.json` / `.txt`) or paste text → detect "shadow citations" and hallucinated evidence |
| 🗺️ **Factual Support Matrix** | Visual heatmap showing Entailment / Neutral / Contradiction for every claim × source pair |
| 📊 **Trust Metrics** | Real-time Citation Accuracy and Citation Thoroughness scores |
| 📄 **Provenance Trace** | Click any claim to see the exact source quote, DOI, and author metadata |
| 🤖 **"I Don't Know"** | The AI explicitly states when evidence is insufficient instead of hallucinating |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Generator   │────▶│  Decomposer  │────▶│ Retrieve & Rerank│
│(Llama 70b)   │     │(Llama Scout)  │    │  (OpenAlex+FAISS)│
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
                                         ┌─────────────────┐
                                         │   NLI Auditor    │
                                         │ (Rate-Limited)   │
                                         └────────┬────────┘
                                                   │
                                                   ▼
                                         ┌─────────────────┐
                                         │ Metrics & Display│
                                         │  (Streamlit UI)  │
                                         └─────────────────┘
```

**Key Design Decisions:**
- **Abstract-First Retrieval**: Uses paper abstracts by default for speed and reliability. Full-text PDF parsing is attempted only for confirmed Open Access papers.
- **Adversarial RAG**: The Auditor independently verifies the Generator's claims — it does not trust the Generator's internal knowledge.
- **Annotate, Don't Rewrite**: Unsupported claims are visually highlighted (🔴 red / 🟡 yellow / 🟢 green), not silently deleted: showing the user exactly what failed.
- **Rate-Limited NLI**: Sequential LLM calls with a 2.5s delay between requests to respect Groq's free tier (30 RPM).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **Logic** | Python 3.11+ |
| **Agents** | LangGraph (State Machine) |
| **LLM Inference** | Groq (free tier) — Llama 3.3 70B, Llama 4 Scout 17B, Llama 3.1 8B |
| **Data Source** | OpenAlex API (free, no key required) |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Vector Search** | FAISS |
| **PDF Parsing** | PyMuPDF4LLM (fallback) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- A free Groq API key ([get one here](https://console.groq.com/keys))

### 1. Clone the repository
```bash
git clone https://github.com/bend0u/Citation-Checker.git
cd Citation-Checker
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Then edit `.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_your-groq-key-here
OPENALEX_EMAIL=your-email@example.com
```
> **Note:** Groq is **100% free** — no credit card required. `OPENALEX_EMAIL` is optional but gives faster rate limits.

### 5. Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

---

## 💰 Cost

**$0.00** — All LLM inference runs on **Groq's free tier**. No credit card needed.

| Node | Model | Free Tier Limits |
|---|---|---|
| Generator | `llama-3.3-70b-versatile` | 1K RPD, 12K TPM |
| Decomposer | `llama-4-scout-17b-16e` | 1K RPD, 30K TPM |
| NLI Auditor | `llama-3.1-8b-instant` | 14.4K RPD, 6K TPM |

> OpenAlex is also free. The only external service with a cost is optional Streamlit Cloud hosting (free tier available).

---

## 📁 Project Structure

```
citationChecker/
├── app.py              # Streamlit SaaS frontend
├── agent.py            # LangGraph orchestrator (5-node pipeline)
├── retrieval.py        # OpenAlex search + FAISS reranking
├── models.py           # Pydantic data models & provenance tracking
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
├── prompt.md           # Project overview document
├── verifstatements.md  # DeepTRACE Auditor technical details
├── userstory.md        # User stories & acceptance criteria
└── project_prof_instructions.md  # Professor's project brief
```

---

## 📚 References

1. Gibney, E. "Open-source AI tool beats giant LLMs in literature reviews — and gets citations right." *Nature*, Feb. 2026. [Link](https://www.nature.com/articles/d41586-026-00347-9)
2. Earp, B.D. et al. "LLM use in scholarly writing poses a provenance problem." *Nat Mach Intell* 7, 1889–1890, 2025. [Link](https://www.nature.com/articles/s42256-025-01159-8)
3. Venkit, P.N. et al. "DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence." 2026. [Link](https://arxiv.org/abs/2509.04499)

---

## 📝 License

This project was developed for academic purposes at Nanyang Technological University (NTU).
