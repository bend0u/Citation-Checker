# Technical Detail: The DeepTRACE Audit Agent

This document outlines the systematic process for building the **Auditor Agent**, the core "Judge" of the Sci-Verify system. This agent ensures the reliability of scientific information retrieval and solves the provenance problem.

---

## 1. Atomic Statement Decomposition
Standard sentence splitting (like `.split('.')`) fails in scientific contexts because a single sentence often contains multiple independent claims (e.g., "Compound X inhibits protein Y but shows high toxicity in mice").

### The Process
The agent takes the initial LLM response and breaks it into **verifiable nuggets**.

* **Logic:** Transform complex prose into a list of "Atomic Claims."
* **Technology:** GPT-4o-mini using **Structured Output (JSON mode)**.
* **The Prompt:**
    > "Break the following paragraph into a list of atomic scientific claims. Each claim should be a single standalone sentence. If a sentence contains two ideas, split them into two separate claims. Do not lose any technical detail."

---

## 2. The Support Matrix (The Cross-Examination)
The Auditor compares every atomic claim against every source document retrieved from OpenAlex/ArXiv. This creates a Bipartite Graph of evidence.

### The Matrix Logic
| | Source 1 (P1) | Source 2 (P2) | ... | Source M (Pm) |
|---|:---:|:---:|:---:|:---:|
| **Claim 1 (S1)** | 1.0 | 0.0 | ... | 0.5 |
| **Claim 2 (S2)** | 0.0 | 1.0 | ... | 0.0 |
| **...** | ... | ... | ... | ... |

* **Row:** Atomic Claims ($S_n$).
* **Column:** Sources ($P_m$).
* **Evaluation:** For every cell $(S_i, P_j)$, an LLM performs **Natural Language Inference (NLI)**.

### The "NLI Judge" Prompt
> "Given the claim: '{{claim}}' and the source text snippet: '{{source_text}}', does the source factually support the claim? 
> Respond with:
> - **SUPPORTED**: The evidence is explicit and direct.
> - **NOT_SUPPORTED**: The evidence is missing or contradicts the claim.
> - **PARTIAL**: The source supports the core idea but lacks specific details mentioned in the claim."

---

## 3. Citation Scoring (Mathematical Metrics)
After populating the matrix, the system calculates the reliability of the "Generator" AI.

### A. Citation Accuracy
Measures "Truthfulness": Did the AI correctly attribute the fact to the source it mentioned?
$$Citation Accuracy = \\frac{\\text{Number of Accurate Citations}}{\\text{Total Citations Provided by AI}}$$

### B. Citation Thoroughness
Measures "Exhaustiveness": Did the AI miss other sources in the retrieval set that also support the claim?
$$Citation Thoroughness = \\frac{\\text{Number of Accurate Citations}}{\\text{Total Possible Factual Supports Found by Auditor}}$$

---

## 4. Implementation Strategy: The "One-Week" Shortcut
To manage the state and transitions between these steps, use **LangGraph**.

### The State Machine Workflow
1.  **State Object:** A dictionary containing `draft_answer`, `list_of_claims`, `source_contents`, and `final_scores`.
2.  **Node 1 (Decomposition):** Breaks the answer into JSON-formatted claims.
3.  **Node 2 (Search/Retrieve):** Finds the most relevant paragraph in the source PDFs for each claim (using vector similarity).
4.  **Node 3 (Audit):** Loops through the claims and sources to fill the Factual Support Matrix.
5.  **Node 4 (Correction):** * If a claim is **NOT_SUPPORTED**, the agent deletes it or replaces it with a warning.
    * If a claim is **PARTIAL**, the agent appends a "Citation Needed" tag.

---

## 5. Solving the Provenance Problem
The **Provenance Problem** is a breakdown in the chain of scholarly acknowledgment. Sci-Verify solves this by:

* **Restoring the Chain:** Providing a direct, unalterable link between a generated sentence and a specific paragraph in a peer-reviewed paper.
* **Auditing the "Black Box":** The Auditor does not rely on the Generator's internal knowledge. It forces the system to look only at the "Ground Truth" text from external databases (OpenAlex).

### SaaS UI Implementation
In your **Streamlit** dashboard, visualize the matrix as a **Heatmap**. 
* **Green:** Full Support.
* **Yellow:** Partial Support.
* **Red:** Hallucination/No Support.
This visual proof allows the user to click any cell to see the source text side-by-side with the AI claim.