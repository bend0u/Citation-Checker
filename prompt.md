# Sci-Verify: Verified Generative Information Retrieval for Science

## 1. Project Overview
**Sci-Verify** is a SaaS platform designed to address the "Provenance Problem" in AI research. It utilizes a multi-agent RAG (Retrieval-Augmented Generation) pipeline that integrates the **DeepTRACE** auditing framework to ensure every claim is factually supported by peer-reviewed literature.

---

## 2. System Architecture & Components

### A. Frontend / SaaS Layer
* **Purpose:** User interaction and data visualization.
* **Key Features:**
    * **Deep Search Interface:** Natural language query input for scientific topics.
    * **Audit Dashboard:** Visual "Heatmap" showing the Factual Support Matrix.
    * **External Verifier:** File uploader for ChatGPT `.json` or `.txt` exports.
    * **Trust Metrics:** Real-time display of Citation Accuracy and Source Necessity.
* **Technology:** `Streamlit` (Python-based web framework).

### B. Knowledge Retrieval Layer
* **Purpose:** Sourcing "Ground Truth" scientific data.
* **Key Features:**
    * **OpenAlex Orchestrator:** Connects to the OpenAlex API to find Open Access (OA) papers.
    * **Scientific Parser:** Converts complex multi-column PDFs and LaTeX formulas into clean Markdown.
* **Technology:** `OpenAlex API`, `PyMuPDF4LLM` (or `Marker-PDF`), `Requests`.

### C. Generation & Orchestration Layer
* **Purpose:** Synthesizing information while maintaining a chain of custody.
* **Key Features:**
    * **LangGraph Workflow:** Manages the state machine between "Searching," "Drafting," and "Auditing."
    * **Citation Mapper:** Tags every generated sentence with a metadata ID linked to the original DOI.
* **Technology:** `LangGraph`, `OpenAI API (GPT-4o)`.

### D. The DeepTRACE Auditor (Verification Layer)
* **Purpose:** The "Judge" that enforces citation accuracy.
* **Key Tasks:**
    1.  **Atomic Decomposition:** Breaking the draft into testable scientific claims.
    2.  **NLI Evaluation:** Performing Natural Language Inference between claims and source text.
    3.  **Metric Calculation:** Computing Citation Accuracy and Thoroughness scores.
* **Technology:** `GPT-4o-mini` (Cost-effective NLI Judge), `NumPy` (for Matrix calculations).

---

## 3. Technology Stack Summary

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **UI** | **Streamlit** | Rapid SaaS development and hosting. |
| **Logic** | **Python 3.11+** | Primary programming language. |
| **Agents** | **LangGraph** | Cyclic agentic workflows (Draft -> Audit -> Revise). |
| **LLMs** | **GPT-4o / GPT-4o-mini** | Synthesis and Fact-checking / NLI. |
| **Data Source**| **OpenAlex API** | Reliable, free scientific paper database. |
| **Database** | **FAISS / Local Storage**| Vector indexing for retrieved paper snippets. |
| **Parsing** | **PyMuPDF4LLM** | Scientific PDF extraction. |
| **Deployment** | **Streamlit Cloud** | Free cloud hosting for the project demo. |

---

## 4. Solving the Provenance Problem
The app solves the breakdown in scholarly acknowledgment through:
1.  **Identity Persistence:** Retaining the DOI and Author metadata through the entire generation loop.
2.  **Statement-Level Attribution:** Users can click any sentence to see the exact paragraph in the source paper that proves the claim.
3.  **Adversarial Verification:** The Auditor agent acts as an independent "critic" of the Generator agent, ensuring no claim is shown without a verified "Support" score of 1.0.





