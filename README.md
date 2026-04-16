# 🔬 Sci-Verify: Forensic SaaS for Scientific Information Retrieval and Citation Provenance


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
python -m streamlit run app.py
```
The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
citationChecker/
├── app.py              # Streamlit SaaS frontend
├── agent.py            # LangGraph orchestrator (Deep Search + Verify pipelines)
├── retrieval.py        # OpenAlex search, Progressive Relaxation, FAISS indexing
├── models.py           # Pydantic data models & provenance tracking
├── logging_config.py   # Structured file logging (sci_verify.log)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── .gitignore          # Git ignore rules
```
