"""
Sci-Verify AI Orchestrator
===========================
Central LangGraph pipeline that powers both the "Deep Search" and "Hallucination Check"
workflows.  All LLM inference is routed through Groq's free tier.

Workflow 1 — Deep Search:
    User query  ➜  Query Reformulation  ➜  OpenAlex multi-loop fetch  ➜  LLM relevance scoring
                ➜  PDF extraction + FAISS RAG  ➜  LLM synthesis (executive summary + verbatim quotes)

Workflow 2 — External Verify (Hallucination Check):
    User text  ➜  Decomposer (extract citations)  ➜  Progressive OpenAlex search
               ➜  Dual-engine verification (fuzzy match OR NLI)  ➜  Metrics report
"""

import os
import json
import asyncio
import logging
import time
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from models import (
    ExtractedCitation, CitationTarget, VerificationStatus, VerificationResult
)
from retrieval import (
    search_openalex, build_snippet_index, retrieve_relevant_snippets,
    PaperSource,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# Each model is chosen for a specific role to balance quality vs. rate limits.
#
# Groq free-tier limits (as of 2026):
#   llama-3.3-70b-versatile  : 30 RPM, 1K RPD, 12K TPM  → best quality
#   llama-4-scout-17b-16e    : 30 RPM, 1K RPD, 30K TPM  → structured JSON
#   llama-3.1-8b-instant     : 30 RPM, 14.4K RPD, 6K TPM → fastest, bulk calls

MODEL_GENERATOR = "llama-3.3-70b-versatile"    # Deep Search synthesis (best quality)
MODEL_DECOMPOSER = "meta-llama/llama-4-scout-17b-16e-instruct"  # Citation extraction (good structured output)
MODEL_NLI = "llama-3.1-8b-instant"             # NLI auditor + relevance scoring (highest RPD)

# Rate-limit safety: pause between sequential NLI calls
# 30 RPM = 1 call every 2 seconds, plus buffer
NLI_DELAY_BETWEEN_CALLS = 2.5


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
# This TypedDict is the shared state container passed between all LangGraph nodes.
# Each node reads from it and returns a partial dict to update specific keys.

class SciVerifyState(TypedDict):
    # ── Core fields (shared across both workflows) ──
    query: str                          # The user's research question
    mode: str                           # "deep_search" | "external_verify"
    input_text: Optional[str]           # Raw user text (Verify workflow only)
    papers: list                        # list[dict] — serialized PaperSource objects
    draft: str                          # Generated text draft (or pasted input)
    citations: list                     # list[dict] — serialized ExtractedCitation objects
    verification_results: list          # list[dict] — serialized VerificationResult objects
    metrics: dict                       # Final report: accuracy, verified count, etc.
    confidence: float                   # Generator confidence (0–1)
    status: str                         # Pipeline stage tracker
    search_strategy: dict               # Audit log: {authors, keywords, loops, fallback_triggered}

    # ── Deep Search specific fields ──
    is_oa_only: bool                    # If True, only fetch Open Access papers
    max_papers: int                     # Maximum number of papers to include in synthesis
    ui_callback: Optional[callable]     # For real-time Streamlit status updates

    # ── Deep Search results ──
    deep_search_results: dict           # LLM output: {introduction, exact_citations}


# ══════════════════════════════════════════════════════════════════════════════
# RESILIENCY UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def retry_with_backoff(retries=3, backoff_in_seconds=2):
    """Decorator: retries a function with exponential backoff on any exception.
    
    Used to handle Groq rate limits (HTTP 429) gracefully.
    Example: @retry_with_backoff(retries=3, backoff_in_seconds=2)
    → Waits 2s, 4s, 8s between retries.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"[BACKOFF] Failed after {retries} retries: {e}")
                        raise e
                    sleep_time = (backoff_in_seconds * 2 ** x)
                    logger.warning(f"[BACKOFF] Exception: {e}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def get_llm(model: str = MODEL_NLI, temperature: float = 0.0) -> ChatGroq:
    """Create a ChatGroq instance for the given model.
    Temperature 0.0 is used everywhere for deterministic, reproducible outputs."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUERY REFORMULATOR
# ══════════════════════════════════════════════════════════════════════════════
# Transforms a natural-language research question into structured search parameters:
#   - Authors: up to 3 names mentioned or inferred
#   - Keywords: ordered from most specific to broadest (for cascading searches)
#
# This is critical for Deep Search — it lets us run targeted Author+Keyword
# combinations against OpenAlex rather than a single naive query.

REFORMULATE_SYSTEM = (
    "You are an academic search query extractor.\n"
    "Given a user's research question, extract the relevant author names and a list of precise, minimalist academic search queries/keywords.\n\n"
    "Rules:\n"
    "1. Extract authors explicitly mentioned. You MAY also use your knowledge to identify the authors of a specific famous paper if the user asks about it, but you MUST return a MAXIMUM of 3 authors.\n"
    "2. If the user mentions a specific famous paper (e.g., 'Attention is All You Need'), you MUST use that exact title as your MOST specific keyword.\n"
    "3. Keywords must go from MOST specific (covering the core concept, exact titles) to LEAST specific (broader topic).\n"
    "4. Keywords should be noun phrases only. Do NOT use question words (what, how).\n"
    "5. Do NOT include author names inside the keywords list.\n"
    "6. Be as minimal as possible. Do not use too many words.\n"
    'Respond ONLY with JSON:\n'
    '{"authors": ["Author1", "Author2"], "keywords": ["specific keyword combo", "broader keyword", "broadest keyword"]}\n\n'
    "Example Query: 'Describe the Attention is All You Need architecture.'\n"
    'Response: {"authors": ["Vaswani", "Shazeer", "Parmar"],"keywords":["Attention Is All You Need", "transformer architecture", "attention mechanism"]}\n\n'
    "Example Query: 'What are the three main properties of the Raft consensus algorithm as defined by Ongaro and Ousterhout'\n"
    'Response: {"authors": ["Ongaro", "Ousterhout"],"keywords":["Raft consensus algorithm", "Raft algorithm", "consensus algorithm"]}'
)


def reformulate_query(user_query: str) -> dict:
    """Use the LLM to extract authors and keyword cascade from a user question.
    
    Returns: {"authors": [...], "keywords": [...]}
    Falls back to the raw query as a single keyword on failure.
    """
    logger.info(f"[REFORMULATOR] Decoding query: '{user_query[:80]}...'")
    llm = get_llm(MODEL_NLI, temperature=0.0)  # Use the fast/cheap model for this
    try:
        resp = llm.invoke([
            SystemMessage(content=REFORMULATE_SYSTEM),
            HumanMessage(content=user_query),
        ])
        text = resp.content.strip()

        # Strip markdown code fences if the LLM wraps its response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        
        authors = parsed.get("authors", [])
        keywords = parsed.get("keywords", [user_query])
        if not keywords: keywords = [user_query]  # Ensure at least one keyword
            
        logger.info(f"[REFORMULATOR] Authors: {authors}")
        logger.info(f"[REFORMULATOR] Keywords: {keywords}")
        return {"authors": authors, "keywords": keywords}
    except Exception as e:
        logger.warning(f"[REFORMULATOR] Failed ({e}), falling back to original query.")
        return {"authors": [], "keywords": [user_query]}


# ══════════════════════════════════════════════════════════════════════════════
# RELEVANCE SCORING
# ══════════════════════════════════════════════════════════════════════════════
# After fetching papers from OpenAlex, we ask the LLM to score each abstract
# against the user query (0–100). Papers scoring below 50 are discarded.
# This prevents false positives from keyword-matching unrelated papers.

@retry_with_backoff(retries=3, backoff_in_seconds=2)
def score_papers_batch(query: str, abstracts_list: list[str]) -> list[int]:
    """Score a batch of abstracts 0-100 for relevance to the query.
    
    Returns a list of integer scores in the same order as the input abstracts.
    On parse failure, defaults every score to 50 (uncertain).
    """
    if not abstracts_list:
        return []
        
    llm = get_llm(MODEL_NLI, temperature=0.0)
    sys_prompt = (
        "You are an academic relevance scoring engine.\n"
        "Given a research QUERY and a list of ABSTRACTS, score each abstract from 0 to 100 based on how well it helps answer the query.\n"
        "0 = Completely irrelevant.\n"
        "100 = Directly answers the core question.\n"
        "Respond ONLY with a valid JSON array of integers corresponding EXACTLY to the order of the abstracts.\n"
        "Example output for 3 abstracts: [85, 10, 95]"
    )
    
    # Build the user prompt with each abstract truncated to 1500 chars
    user_prompt = f"QUERY: {query}\n\n"
    for i, abstract in enumerate(abstracts_list):
        user_prompt += f"ABSTRACT {i}:\n{abstract[:1500]}\n\n"
        
    resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    text = resp.content.strip()

    # Extract JSON array from response (LLM may wrap it in text)
    import re
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match: text = match.group(0)
    
    try:
        scores = json.loads(text)
        if not isinstance(scores, list) or len(scores) != len(abstracts_list):
            raise ValueError("LLM returned malformed score list.")
        return [int(s) for s in scores]
    except Exception as e:
        logger.error(f"[SCORING] Failed to parse scores: {e}. Defaulting to 50.")
        return [50 for _ in abstracts_list]


@retry_with_backoff(retries=3, backoff_in_seconds=2)
def evaluate_paper_relevance(user_query: str, paper_title: str, paper_abstract: str) -> bool:
    """Binary LLM check: is this paper relevant to the user's query?
    
    Used as a secondary filter in the Verify workflow.
    Returns True if relevant, False if off-topic.
    Defaults to True on LLM failure (conservative — keep the paper).
    """
    llm = get_llm(MODEL_NLI, temperature=0.0)
    sys = (
        "You are a strict academic relevance filter. Does the following paper CLEARLY relate "
        "to the user's research query? "
        "If it is relevant or plausibly helpful, answer YES. If it is completely off-topic, answer NO.\n"
        "Output ONLY the word YES or NO."
    )
    prompt = f"Query: {user_query}\n\nPaper Title: {paper_title}\nPaper Abstract: {paper_abstract[:1500]}"
    try:
        resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=prompt)])
        return "YES" in resp.content.upper()
    except Exception as e:
        logger.error(f"[RELEVANCE] LLM failure: {e}")
        return True  # Default to keeping the paper if the LLM crashes


# ══════════════════════════════════════════════════════════════════════════════
# DEEP SEARCH WORKFLOW  (Tab 1: "Deep Search")
# ══════════════════════════════════════════════════════════════════════════════
# This is a single-node LangGraph workflow that handles the entire pipeline:
#   1. Reformulate the query → authors + keyword cascade
#   2. Run Author+Keyword search loops against OpenAlex
#   3. Score and filter candidates by LLM relevance (threshold: 50/100)
#   4. Attempt PDF download + full-text extraction for OA papers
#   5. Build a FAISS index over extracted paragraphs
#   6. Retrieve top-15 relevant paragraphs via semantic search
#   7. Synthesize an executive summary + exact verbatim quotes via Llama 70B

def execute_deep_search(state: SciVerifyState) -> dict:
    """Main Deep Search node: search, extract, index, and synthesize."""
    from retrieval import try_extract_full_text
    
    query = state["query"]
    is_oa = state.get("is_oa_only", False)
    max_papers = state.get("max_papers", 3)
    ui_callback = state.get("ui_callback")
    
    def log_ui(msg: str):
        """Log to both the file logger and the Streamlit UI status widget."""
        logger.info(msg)
        if ui_callback: ui_callback(msg)
        
    log_ui(f"[DEEP SEARCH] Starting workflow: querying OpenAlex...")

    # Track the search strategy for the UI's "Search Strategy" expander
    search_strategy = {
        "authors": [], "keywords": [], "loops": [], "fallback_triggered": False
    }

    # ── Step 1: Decode the user query into structured search parameters ──
    strategy = reformulate_query(query)
    authors = strategy["authors"][:3]  # Hard cap at 3 authors
    keywords = strategy["keywords"]
    
    search_strategy["authors"] = authors
    search_strategy["keywords"] = keywords

    # Deduplication set: track papers by DOI (or title if no DOI)
    seen_dois: set[str] = set()
    candidate_papers = []
    
    def process_and_add(search_results, loop_name):
        """Deduplicate, score, and add qualifying papers to the candidate pool."""
        new_papers = []
        for p in search_results:
            key = p.doi or p.title
            if not key or key in seen_dois: continue
            seen_dois.add(key)
            new_papers.append(p)
            
        if not new_papers: return
        
        # Ask the LLM to score each abstract for relevance
        log_ui(f"  ↳ Scoring {len(new_papers)} fetched abstracts...")
        abstracts = [p.abstract or (p.title + " (No abstract available)") for p in new_papers]
        scores = score_papers_batch(query, abstracts)
        
        # Only keep papers with relevance score >= 50
        added = 0
        for p, score in zip(new_papers, scores):
            if score >= 50:
                p.relevance_score = score
                candidate_papers.append(p)
                added += 1
                if len(candidate_papers) >= 15: break  # Global cap at 15 candidates
                
        if added > 0:
            search_strategy["loops"].append(f"{loop_name} -> pulled {added} valid candidates (score >= 50)")

    # ── Step 2: Phase 1 — Author+Keyword search loops ──
    # Cross every author with every keyword for maximum coverage
    if authors:
        for a in authors:
            for k in keywords:
                if len(candidate_papers) >= 15: break
                loop_name = f"Search(Keyword: '{k}' + Author API: '{a}')"
                log_ui(f"Fetching from OpenAlex: Author='{a}', Keyword='{k}'...")
                results = search_openalex(query=k, author_name=a, is_oa=is_oa, max_results=5)
                process_and_add(results, loop_name)
    
    # ── Step 3: Phase 2 — Keyword-only fallback (when no authors were detected) ──
    if not authors and len(candidate_papers) < 15:
        for k in keywords:
            if len(candidate_papers) >= 15: break
            loop_name = f"Search(Keyword: '{k}')"
            log_ui(f"Fetching from OpenAlex fallback: Keyword='{k}'...")
            results = search_openalex(query=k, author_name=None, is_oa=is_oa, max_results=8)
            process_and_add(results, loop_name)

    # Early exit if nothing was found
    if not candidate_papers:
        log_ui("[DEEP SEARCH] No candidate papers found in OpenAlex.")
        return {
            "papers": [], "deep_search_results": {"introduction": "No relevant papers found.", "exact_citations": []},
            "status": "no_papers_found", "search_strategy": search_strategy
        }

    # ── Step 3.5: Sort by relevance and take the top N ──
    candidate_papers.sort(key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)
    all_papers = candidate_papers[:max_papers]
    
    # Re-index source_ids to be 0-based for the final output
    for i, p in enumerate(all_papers):
        p.source_id = i
    
    log_ui(f"Successfully selected top {len(all_papers)} highly relevant papers.")

    # ── Step 4: Extract full text from OA PDFs and build a FAISS index ──
    import faiss
    from sentence_transformers import SentenceTransformer
    from retrieval import get_embedding_model
    
    log_ui("Downloading Open Access PDFs and parsing full text...")
    all_paragraphs = []
    for p in all_papers:
        if p.oa_pdf_url: log_ui(f"  ↳ Attempting PDF download for: {p.title[:40]}...")
        text = try_extract_full_text(p)
        if text:
            # Split by double newlines → pragraph-level chunks (min 50 chars each)
            paras = [pr.strip() for pr in text.split("\n\n") if len(pr.strip()) > 50]
            for para in paras:
                all_paragraphs.append({"text": para, "source": f"{p.authors[0] if p.authors else 'Unknown'} ({p.publication_date[:4] if p.publication_date else 'N/A'}) - {p.title}", "paper": p})
            p.full_text = "Retrieved"  # Flag for the UI badge ("PDF Extracted" vs "Abstract Only")
        else:
            p.full_text = None

    # Build the RAG context: either full-text paragraphs or abstract fallback
    context = ""
    if all_paragraphs:
        # Build FAISS index over all extracted paragraphs
        log_ui(f"Creating semantic FAISS index over {len(all_paragraphs)} paragraphs...")
        model = get_embedding_model()
        v = model.encode([pr["text"] for pr in all_paragraphs], convert_to_numpy=True)
        index = faiss.IndexFlatIP(v.shape[1])  # Inner product = cosine similarity after normalization
        faiss.normalize_L2(v)
        index.add(v)
        
        # Retrieve the top 15 most relevant paragraphs for the query
        q_v = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_v)
        D, I = index.search(q_v, min(15, len(all_paragraphs)))
        
        ctx_parts = []
        for idx in I[0]:
            p_data = all_paragraphs[idx]
            ctx_parts.append(f"--- PAPER: {p_data['source']} ---\n{p_data['text']}")
        context = "\n\n".join(ctx_parts)
    else:
        # No PDFs were downloaded — use abstracts as the context instead
        log_ui("No full text successfully extracted. Falling back to structured abstracts.")
        ctx_parts = []
        for p in all_papers:
            ctx_parts.append(f"--- PAPER: {p.title} ---\nAbstract: {p.abstract}")
        context = "\n\n".join(ctx_parts)

    # ── Step 5: LLM Synthesis — generate executive summary + verbatim quotes ──
    log_ui("Synthesizing final executive summary and verbatim quotes...")
    SYS_PROMPT = (
        "You are an expert scientific researcher answering the user's question by extracting EXACT citations from the provided context.\n"
        "Rules:\n"
        "1. Write a short 'introduction' summarizing the findings.\n"
        "2. Extract EXACT verbatim quotes from the context that directly answer the query. Do NOT paraphrase the quotes.\n"
        "3. Provide the full 'paper_title' and 'authors' exactly as shown in the source headers.\n"
        "Respond ONLY with valid JSON exactly matching this schema:\n"
        "{\n"
        '  "introduction": "Brief 2-3 sentence answer",\n'
        '  "exact_citations": [\n'
        '    {"quote": "...", "paper_title": "...", "authors": "..."}\n'
        '  ]\n'
        "}"
    )
    USER_PROMPT = f"USER QUERY: {query}\n\nCONTEXT EXTRACTS:\n{context}"
    
    llm = get_llm(MODEL_GENERATOR, temperature=0.0)
    
    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def generate_json_with_backoff():
        return llm.invoke([SystemMessage(content=SYS_PROMPT), HumanMessage(content=USER_PROMPT)])

    try:
        resp = generate_json_with_backoff()
        text = resp.content.strip()
        # Extract JSON object from the response (LLM may add surrounding text)
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: text = match.group(0)
        
        parsed_results = json.loads(text)
    except Exception as e:
        logger.error(f"[DEEP SEARCH] JSON extraction failed: {e}")
        parsed_results = {"introduction": "Error extracting citations. See related papers.", "exact_citations": []}

    return {
        "papers": [p.model_dump() for p in all_papers],
        "deep_search_results": parsed_results,
        "status": "deep_search_complete",
        "search_strategy": search_strategy,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY WORKFLOW — NODE 2: DECOMPOSER
# ══════════════════════════════════════════════════════════════════════════════
# Extracts every citation/claim from the user's input text into structured objects.
# Each citation gets a CitationTarget with {title, authors, year, core_topic, domain}.
# Explicit quotes (inside quotation marks) are flagged separately for fuzzy matching.

def decompose_claims(state: SciVerifyState) -> dict:
    """Node 2: Extract explicit quotes and semantic citations into structured targets."""
    draft = state.get("input_text") or state.get("draft", "")
    if not draft:
        logger.warning("[DECOMPOSER] No draft text to decompose.")
        return {"citations": [], "status": "no_draft"}

    logger.info(f"[DECOMPOSER] Parsing citations from text ({len(draft)} chars) using {MODEL_DECOMPOSER}...")
    llm = get_llm(MODEL_DECOMPOSER, temperature=0.0)

    # System prompt instructs the LLM to extract citation metadata with strict rules:
    # - Only use full title if explicitly stated (prevents LLM from hallucinating titles)
    # - Never truncate fields — full value or null
    # - core_topic is a 2-5 word fallback keyword for searching when title is unknown
    sys = (
        "You are an expert scientific hallucination auditor.\n"
        "Your task is to extract every single claim or quote that refers to an external paper.\n"
        "If they use quotation marks, mark 'is_explicit_quote' as true.\n"
        "If there are ZERO external papers cited or mentioned, return an empty list [].\n\n"
        "CRITICAL RULES:\n"
        "- For 'title': ONLY use the EXACT, FULL paper title if it is explicitly and completely stated in the text.\n"
        "  If only a short name, abbreviation, or subject keyword is mentioned (e.g., 'the Raft paper',\n"
        "  'their work on CRISPR'), set title to null and put the short name in 'core_topic' instead.\n"
        "- NEVER truncate any field. If a title appears cut off with '...', try to reconstruct\n"
        "  the complete title from context. Output the full value or null — never a partial string.\n"
        "- 'core_topic' should be 2-5 words capturing the specific mechanism, algorithm, or subject\n"
        "  (e.g., 'Raft consensus algorithm', 'CRISPR off-target effects'). Include the author-specific\n"
        "  version or signature if possible.\n"
        "- 'domain' should be the broad academic field: 'computer science', 'biology', 'medicine',\n"
        "  'physics', 'chemistry', 'mathematics', etc. Use lowercase.\n\n"
        "Respond ONLY with valid JSON exactly matching the following schema:\n"
        "{\n"
        '  "citations": [\n'
        '    {\n'
        '      "text": "<The exact sentence or literal quote from the text>",\n'
        '      "is_explicit_quote": <boolean>,\n'
        '      "target_metadata": {\n'
        '          "title": "<The EXACT FULL paper title if explicitly stated, otherwise null>",\n'
        '          "authors": ["<Extracted author 1>", ...],\n'
        '          "year": <Integer. If none, null>,\n'
        '          "core_topic": "<2-5 words: specific mechanism or algorithm name>",\n'
        '          "domain": "<lowercase academic field, e.g. computer science>"\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        "}"
    )
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=draft)])

    try:
        text = resp.content.strip()
        # Extract JSON from the response (handle markdown wrapping)
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: text = match.group(0)
        parsed = json.loads(text)
        raw_citations = parsed.get("citations", [])
        
        # Validate each citation through the Pydantic model
        valid_citations = [
            ExtractedCitation(claim_id=i, **c).model_dump() 
            for i, c in enumerate(raw_citations)
        ]
        logger.info(f"[DECOMPOSER] Successfully extracted {len(valid_citations)} citation blocks.")
    except Exception as e:
        logger.error(f"[DECOMPOSER] JSON parse/validation failed: {e}")
        valid_citations = []

    for c in valid_citations:
        logger.info(f"[DECOMPOSER] Extracted (Quote: {c['is_explicit_quote']}): {c['text'][:60]}... -> {c['target_metadata']}")

    return {"citations": valid_citations, "status": "citations_decomposed"}


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY WORKFLOW — NODE 3: PROGRESSIVE OPENALEX SEARCH
# ══════════════════════════════════════════════════════════════════════════════
# For each extracted citation, tries to find the real paper in OpenAlex using
# the Progressive Relaxation Search (in retrieval.py).
#
# If a paper is found → temporary VERIFIED_QUOTE status (Phase 3 will refine).
# If no paper is found → HALLUCINATED_PAPER (skipped in Phase 3).
#
# NOTE: Even paywalled papers proceed to Phase 3 for abstract-based verification.

def search_literature(state: SciVerifyState) -> dict:
    """Node 3: Verify citations exist via cascading OpenAlex search."""
    from retrieval import verify_openalex_citation
    citations = state.get("citations", [])
    logger.info(f"[SEARCH] Verifying {len(citations)} extracted citations against OpenAlex...")
    
    results = []
    
    for c_dict in citations:
        ext_cit = ExtractedCitation(**c_dict)
        logger.info(f"[SEARCH] Validating target for claim {ext_cit.claim_id}: {ext_cit.target_metadata.title or ext_cit.target_metadata.core_topic}")

        # Run the 4-level cascading search (see retrieval.py: verify_openalex_citation)
        matched_paper, reasoning, delta = verify_openalex_citation(ext_cit.target_metadata)
        
        # Default: paper not found = hallucinated
        status = VerificationStatus.HALLUCINATED_PAPER
        if matched_paper:
            # Temporary status — Phase 3 will refine using full-text or abstract fallback
            status = VerificationStatus.VERIFIED_QUOTE
                
        res = VerificationResult(
            claim_id=ext_cit.claim_id,
            status=status,
            matched_paper=matched_paper,
            reasoning=reasoning,
            metadata_delta=delta
        )
        results.append(res.model_dump())
        
    return {"verification_results": results, "status": "search_complete"}


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY WORKFLOW — NODE 4: DUAL-ENGINE CONTENT VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
# This is the core forensic engine. For each citation that passed Phase 2:
#
# Engine A — Fuzzy String Match (for explicit quotes):
#   Build a FAISS index over the paper's text (full PDF or abstract fallback),
#   retrieve the top-5 most similar chunks, and run token_set_ratio fuzzy matching.
#   Score >= 85% → VERIFIED_QUOTE. Otherwise → HALLUCINATED_QUOTE.
#
# Engine B — LLM NLI Judge (for semantic/paraphrased claims):
#   Same FAISS retrieval, but each chunk is evaluated by the LLM:
#   "Does this source text support this claim?" → yes/no with reasoning.
#   Any chunk supporting → SUPPORTED_SUMMARY. All fail → UNSUPPORTED_SUMMARY.
#
# Fallback: If full-text extraction fails, the FAISS index is built on the abstract.
#           If abstract-only verification also fails → UNKNOWN_PAYWALLED.

NLI_SUPPORT_SYSTEM = (
    "You are a semantic evaluator. Does the SOURCE TEXT support the CLAIM?\n"
    "Respond ONLY with JSON:\n"
    '{"supported": true|false, "reasoning": "<1 sentence>"}'
)

def verify_quotes(state: SciVerifyState) -> dict:
    """Node 4: Run fuzzy string matching (quotes) or NLI semantic validation (summaries)."""
    citations_data = state.get("citations", [])
    results_data = state.get("verification_results", [])
    
    # Build lookup dicts keyed by claim_id for easy access
    citations = {c['claim_id']: ExtractedCitation(**c) for c in citations_data}
    results = {r['claim_id']: VerificationResult(**r) for r in results_data}
    
    llm = get_llm(MODEL_NLI, temperature=0.0)
    import re
    from thefuzz import fuzz
    from retrieval import build_snippet_index, retrieve_relevant_snippets

    for cid, res in results.items():
        # Skip claims where no paper was found in OpenAlex (nothing to verify against)
        if res.status == VerificationStatus.HALLUCINATED_PAPER:
            logger.info(f"[VERIFY] Claim {cid} skipped (Paper not found in OpenAlex)")
            continue
            
        cit = citations[cid]
        paper = res.matched_paper
        
        # Build a FAISS index for just this single paper.
        # attempt_full_text=True → tries PDF download first, falls back to abstract
        index, snippets, sids = build_snippet_index([paper], attempt_full_text=True)

        # If even the abstract is empty, we can't verify anything
        if index.ntotal == 0:
            res.status = VerificationStatus.UNKNOWN_PAYWALLED
            res.reasoning = "Document text could not be extracted."
            continue
            
        # Retrieve top 5 most semantically similar chunks to the claim
        top_chunks = retrieve_relevant_snippets(cit.text, index, snippets, sids, top_k=5)
        
        # ── Engine A: Explicit Quote → Fuzzy String Matching ──
        if cit.is_explicit_quote:
            best_score = 0
            for chunk_str, _, _ in top_chunks:
                # token_set_ratio handles word reordering and partial matches
                score = fuzz.token_set_ratio(cit.text.lower(), chunk_str.lower())
                if score > best_score:
                    best_score = score
            
            res.similarity_score = float(best_score)

            if best_score >= 85:
                # Quote found in the paper's text
                res.status = VerificationStatus.VERIFIED_QUOTE
                res.reasoning = f"Fuzzy sequence match succeeded (Score: {best_score}%)"
            else:
                # Quote not found — determine why (paywalled vs genuinely hallucinated)
                if paper.oa_pdf_url and not paper.full_text:
                    # Had an OA URL but PDF download/parse failed → checked abstract only
                    res.status = VerificationStatus.UNKNOWN_PAYWALLED
                    res.reasoning = f"Could not download full PDF to verify quote. Checked abstract only (Max match: {best_score}%)."
                elif not paper.oa_pdf_url:
                    # No OA URL → paywalled, could only check abstract
                    res.status = VerificationStatus.UNKNOWN_PAYWALLED
                    res.reasoning = f"Paper is paywalled. Checked abstract only (Max match: {best_score}%)."
                else:
                    # Full text was available and the quote still wasn't found
                    res.status = VerificationStatus.HALLUCINATED_QUOTE
                    res.reasoning = f"Fuzzy sequence match failed across full text (Highest similarity: {best_score}%)"
                
        # ── Engine B: Semantic Claim → LLM NLI Judge ──
        else:
            best_support = False
            reasons = []

            # Try each chunk until one supports the claim (or all fail)
            for chunk_str, _, _ in top_chunks:
                try:
                    resp = llm.invoke([
                        SystemMessage(content=NLI_SUPPORT_SYSTEM),
                        HumanMessage(content=f"CLAIM: {cit.text}\n\nSOURCE TEXT: {chunk_str[:1200]}")
                    ])
                    # Parse the NLI JSON response
                    txt = resp.content.strip()
                    match = re.search(r'\{.*\}', txt, re.DOTALL)
                    if match: txt = match.group(0)
                    p = json.loads(txt)
                    
                    if p.get("supported", False):
                        best_support = True
                        res.reasoning = p.get("reasoning", "Supported by context.")
                        break  # One supporting chunk is enough
                    else:
                        reasons.append(p.get("reasoning", "Not supported."))
                except Exception as e:
                    logger.error(f"[VERIFY] NLI JSON loop error: {e}")
            
            if best_support:
                res.status = VerificationStatus.SUPPORTED_SUMMARY
            else:
                # Determine why it failed (same paywalled vs hallucinated logic)
                if paper.oa_pdf_url and not paper.full_text:
                    res.status = VerificationStatus.UNKNOWN_PAYWALLED
                    res.reasoning = "Could not download full PDF. Abstract alone lacked sufficient evidence."
                elif not paper.oa_pdf_url:
                    res.status = VerificationStatus.UNKNOWN_PAYWALLED
                    res.reasoning = "Paper is paywalled. Abstract alone lacked sufficient evidence."
                else:
                    res.status = VerificationStatus.UNSUPPORTED_SUMMARY
                    res.reasoning = reasons[0] if reasons else "No semantic support found in full text context."

    # Serialize results back, sorted by claim_id for consistent ordering
    final_results = [results[cid].model_dump() for cid in sorted(results.keys())]
    return {"verification_results": final_results, "status": "verification_complete"}


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY WORKFLOW — NODE 5: REPORT COMPILATION
# ══════════════════════════════════════════════════════════════════════════════
# Aggregates the individual verification results into summary metrics
# for the Streamlit UI's metric cards.

def compile_report(state: SciVerifyState) -> dict:
    """Node 5: Calculate aggregate verification metrics."""
    results = state.get("verification_results", [])
    
    total = len(results)
    verified = sum(1 for r in results if r['status'] in (VerificationStatus.VERIFIED_QUOTE, VerificationStatus.SUPPORTED_SUMMARY))
    hallucinated = sum(1 for r in results if r['status'] in (VerificationStatus.HALLUCINATED_QUOTE, VerificationStatus.HALLUCINATED_PAPER, VerificationStatus.UNSUPPORTED_SUMMARY))
    
    accuracy = verified / total if total > 0 else 0.0
    
    logger.info(f"[METRICS] Final Verification completed. Total citations: {total}. Verified: {verified}. Hallucinatory: {hallucinated}")
    
    return {
        "metrics": {
            "citation_accuracy": round(accuracy, 4),
            "total_citations": total,
            "verified_claims": verified,
            "hallucinated": hallucinated
        },
        "status": "complete"
    }


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH WORKFLOW BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
# Two separate compiled workflows, each invoked from the Streamlit UI.

def build_deep_search_workflow():
    """Build the single-node Deep Search workflow (Tab 1)."""
    wf = StateGraph(SciVerifyState)
    wf.add_node("execute", execute_deep_search)
    wf.set_entry_point("execute")
    wf.add_edge("execute", END)
    return wf.compile()

def build_verify_workflow():
    """Build the 4-node External Verify pipeline (Tab 2).
    
    Flow: Decompose → Search → Verify → Metrics
    """
    wf = StateGraph(SciVerifyState)
    wf.add_node("decompose", decompose_claims)
    wf.add_node("search", search_literature)
    wf.add_node("verify", verify_quotes)
    wf.add_node("metrics", compile_report)
    wf.set_entry_point("decompose")
    wf.add_edge("decompose", "search")
    wf.add_edge("search", "verify")
    wf.add_edge("verify", "metrics")
    wf.add_edge("metrics", END)
    return wf.compile()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════
# These are the functions called by app.py to run each workflow.

def run_deep_search(query: str, max_papers: int, is_oa_only: bool, ui_callback: Optional[callable] = None) -> dict:
    """Run the Deep Search workflow. Called from Streamlit Tab 1."""
    logger.info(f"{'='*40} DEEP SEARCH START {'='*40}")
    logger.info(f"Query: {query} | Max: {max_papers} | OA: {is_oa_only}")
    app = build_deep_search_workflow()
    result = app.invoke({
        "query": query, "mode": "deep_search", "input_text": None,
        "papers": [], "draft": "", "claims": [], "evidence": {},
        "support_matrix": [], "metrics": {}, "confidence": 0.0,
        "status": "starting", "search_strategy": {},
        "max_papers": max_papers, "is_oa_only": is_oa_only,
        "deep_search_results": {},
        "ui_callback": ui_callback
    })
    logger.info(f"{'='*40} DEEP SEARCH END {'='*40}")
    return result


def run_external_verify(text: str) -> dict:
    """Run the External Verify workflow. Called from Streamlit Tab 2."""
    logger.info(f"{'='*40} EXTERNAL VERIFY START {'='*40}")
    logger.info(f"Input text: {len(text)} chars")
    app = build_verify_workflow()
    result = app.invoke({
        "query": "", "mode": "external_verify", "input_text": text,
        "papers": [], "draft": text, "claims": [], "evidence": {},
        "support_matrix": [], "metrics": {}, "confidence": 1.0,
        "status": "starting", "search_strategy": {},
        "max_papers": 5, "is_oa_only": False,
        "deep_search_results": {}
    })
    logger.info(f"{'='*40} EXTERNAL VERIFY END {'='*40}")
    return result
