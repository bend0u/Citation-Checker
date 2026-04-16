"""
Sci-Verify AI Orchestrator
LangGraph Adversarial RAG workflow with DeepTRACE auditing.
Uses Groq (free tier) for all LLM inference.
Async-batched NLI Judge with rate-limit-aware throttling.
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

# ── Groq Model Configuration ──────────────────────────────────────────────
# Free tier limits (as of 2026):
#   llama-3.3-70b-versatile  : 30 RPM, 1K RPD, 12K TPM
#   llama-4-scout-17b-16e    : 30 RPM, 1K RPD, 30K TPM
#   llama-3.1-8b-instant     : 30 RPM, 14.4K RPD, 6K TPM
#
# Strategy:
#   Generator  → 70b (best quality for synthesis)
#   Decomposer → scout-17b (good structured output, 30K TPM)
#   NLI Auditor → 8b-instant (highest RPD: 14.4K, fast)

MODEL_GENERATOR = "llama-3.3-70b-versatile"
MODEL_DECOMPOSER = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_NLI = "llama-3.1-8b-instant"

# Rate-limit safety: pause between async NLI calls (seconds)
NLI_DELAY_BETWEEN_CALLS = 2.5  # 30 RPM ≈ 1 call/2s, with buffer


# ── LangGraph State ────────────────────────────────────────────────────────
class SciVerifyState(TypedDict):
    query: str
    mode: str                       # "deep_search" | "external_verify"
    input_text: Optional[str]
    papers: list                    # list[dict]  (PaperSource dicts)
    draft: str
    citations: list                   # list[dict]  (ExtractedCitation dicts)
    verification_results: list        # list[dict]  (VerificationResult dicts)
    metrics: dict
    confidence: float
    status: str
    search_strategy: dict           # Logs {authors: [], keywords: [], loops: []}
    
    # UI Filters for Deep Search
    is_oa_only: bool
    max_papers: int
    ui_callback: Optional[callable] # For real-time streamlit status logging
    
    # Results for Deep Search
    deep_search_results: dict       # Stores the parsed JSON from LLM: intro + exact quotes

# ── Resiliency ─────────────────────────────────────────────────────────────
def retry_with_backoff(retries=3, backoff_in_seconds=2):
    """Exponential backoff decorator for hitting LLM rate limits."""
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


# ── LLM factory ────────────────────────────────────────────────────────────
def get_llm(model: str = MODEL_NLI, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Query Reformulator ─────────────────────────────────────────────────
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
    """Extract authors and specific->general keywords."""
    logger.info(f"[REFORMULATOR] Decoding query: '{user_query[:80]}...'")
    llm = get_llm(MODEL_NLI, temperature=0.0)  # fast, cheap model
    try:
        resp = llm.invoke([
            SystemMessage(content=REFORMULATE_SYSTEM),
            HumanMessage(content=user_query),
        ])
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
        
        authors = parsed.get("authors", [])
        keywords = parsed.get("keywords", [user_query])
        # Ensure at least one keyword
        if not keywords: keywords = [user_query]
            
        logger.info(f"[REFORMULATOR] Authors: {authors}")
        logger.info(f"[REFORMULATOR] Keywords: {keywords}")
        return {"authors": authors, "keywords": keywords}
    except Exception as e:
        logger.warning(f"[REFORMULATOR] Failed ({e}), falling back to original query.")
        return {"authors": [], "keywords": [user_query]}


# ── Relevance Checker ─────────────────────────────────────────────────────
@retry_with_backoff(retries=3, backoff_in_seconds=2)
def score_papers_batch(query: str, abstracts_list: list[str]) -> list[int]:
    """Score a batch of abstracts accurately 0-100 to handle hallucinated keyword matches."""
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
    
    user_prompt = f"QUERY: {query}\n\n"
    for i, abstract in enumerate(abstracts_list):
        user_prompt += f"ABSTRACT {i}:\n{abstract[:1500]}\n\n"
        
    resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
    text = resp.content.strip()
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
    """Uses LLM to evaluate if a retrieved paper is relevant to the search (used by Verifier)."""
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
        return True  # default to keeping it if the LLM crashes


# ── Deep Search Workflow ──────────────────────────────────────────────────
def execute_deep_search(state: SciVerifyState) -> dict:
    """Intelligent nested search, PDF extraction, RAG, and exact quoting."""
    from retrieval import try_extract_full_text
    
    query = state["query"]
    is_oa = state.get("is_oa_only", False)
    max_papers = state.get("max_papers", 3)
    ui_callback = state.get("ui_callback")
    
    def log_ui(msg: str):
        logger.info(msg)
        if ui_callback: ui_callback(msg)
        
    log_ui(f"[DEEP SEARCH] Starting workflow: querying OpenAlex...")

    search_strategy = {
        "authors": [], "keywords": [], "loops": [], "fallback_triggered": False
    }

    # Step 1: Decode the query
    strategy = reformulate_query(query)
    authors = strategy["authors"][:3]  # Enforce max 3 limit in code
    keywords = strategy["keywords"]
    
    search_strategy["authors"] = authors
    search_strategy["keywords"] = keywords

    seen_dois: set[str] = set()
    candidate_papers = []
    
    def process_and_add(search_results, loop_name):
        new_papers = []
        for p in search_results:
            key = p.doi or p.title
            if not key or key in seen_dois: continue
            seen_dois.add(key)
            new_papers.append(p)
            
        if not new_papers: return
        
        log_ui(f"  ↳ Scoring {len(new_papers)} fetched abstracts...")
        abstracts = [p.abstract or (p.title + " (No abstract available)") for p in new_papers]
        scores = score_papers_batch(query, abstracts)
        
        added = 0
        for p, score in zip(new_papers, scores):
            if score >= 50:
                p.relevance_score = score
                candidate_papers.append(p)
                added += 1
                if len(candidate_papers) >= 15: break
                
        if added > 0:
            search_strategy["loops"].append(f"{loop_name} -> pulled {added} valid candidates (score >= 50)")

    # Step 2: Phase 1 (Author)
    if authors:
        for a in authors:
            for k in keywords:
                if len(candidate_papers) >= 15: break
                loop_name = f"Search(Keyword: '{k}' + Author API: '{a}')"
                log_ui(f"Fetching from OpenAlex: Author='{a}', Keyword='{k}'...")
                results = search_openalex(query=k, author_name=a, is_oa=is_oa, max_results=5)
                process_and_add(results, loop_name)
    
    # Step 3: Phase 2 Fallback (Keyword only if no authors found)
    if not authors and len(candidate_papers) < 15:
        for k in keywords:
            if len(candidate_papers) >= 15: break
            loop_name = f"Search(Keyword: '{k}')"
            log_ui(f"Fetching from OpenAlex fallback: Keyword='{k}'...")
            results = search_openalex(query=k, author_name=None, is_oa=is_oa, max_results=8)
            process_and_add(results, loop_name)

    if not candidate_papers:
        log_ui("[DEEP SEARCH] No candidate papers found in OpenAlex.")
        return {
            "papers": [], "deep_search_results": {"introduction": "No relevant papers found.", "exact_citations": []},
            "status": "no_papers_found", "search_strategy": search_strategy
        }


    # Sort and slice
    candidate_papers.sort(key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)
    all_papers = candidate_papers[:max_papers]
    
    for i, p in enumerate(all_papers):
        p.source_id = i
    
    log_ui(f"Successfully selected top {len(all_papers)} highly relevant papers.")

    # Step 4: Extract Full Text and Build FAISS
    import faiss
    from sentence_transformers import SentenceTransformer
    from retrieval import get_embedding_model
    
    log_ui("Downloading Open Access PDFs and parsing full text...")
    all_paragraphs = []
    for p in all_papers:
        if p.oa_pdf_url: log_ui(f"  ↳ Attempting PDF download for: {p.title[:40]}...")
        text = try_extract_full_text(p)
        if text:
            # Simple paragraph split
            paras = [pr.strip() for pr in text.split("\n\n") if len(pr.strip()) > 50]
            for para in paras:
                all_paragraphs.append({"text": para, "source": f"{p.authors[0] if p.authors else 'Unknown'} ({p.publication_date[:4] if p.publication_date else 'N/A'}) - {p.title}", "paper": p})
            p.full_text = "Retrieved"
        else:
            p.full_text = None

    context = ""
    if all_paragraphs:
        log_ui(f"Creating semantic FAISS index over {len(all_paragraphs)} paragraphs...")
        model = get_embedding_model()
        v = model.encode([pr["text"] for pr in all_paragraphs], convert_to_numpy=True)
        index = faiss.IndexFlatIP(v.shape[1])
        faiss.normalize_L2(v)
        index.add(v)
        
        q_v = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_v)
        D, I = index.search(q_v, min(15, len(all_paragraphs)))
        
        ctx_parts = []
        for idx in I[0]:
            p_data = all_paragraphs[idx]
            ctx_parts.append(f"--- PAPER: {p_data['source']} ---\n{p_data['text']}")
        context = "\n\n".join(ctx_parts)
    else:
        log_ui("No full text successfully extracted. Falling back to structured abstracts.")
        ctx_parts = []
        for p in all_papers:
            ctx_parts.append(f"--- PAPER: {p.title} ---\nAbstract: {p.abstract}")
        context = "\n\n".join(ctx_parts)

    # Step 5: LLM Synthesis and pure citation extraction
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

# ── Node 2: Decomposer ────────────────────────────────────────────────────
def decompose_claims(state: SciVerifyState) -> dict:
    """Extract explicit quotes and semantic citations into structured targets."""
    draft = state.get("input_text") or state.get("draft", "")
    if not draft:
        logger.warning("[DECOMPOSER] No draft text to decompose.")
        return {"citations": [], "status": "no_draft"}

    logger.info(f"[DECOMPOSER] Parsing citations from text ({len(draft)} chars) using {MODEL_DECOMPOSER}...")
    llm = get_llm(MODEL_DECOMPOSER, temperature=0.0)
    sys = (
        "You are an expert scientific hallucination auditor.\n"
        "Your task is to extract every single claim or quote that refers to an external paper.\n"
        "If they use quotation marks, mark 'is_explicit_quote' as true.\n"
        "If there are ZERO external papers cited or mentioned, return an empty list [].\n\n"
        "Respond ONLY with valid JSON exactly matching the following schema:\n"
        "{\n"
        '  "citations": [\n'
        '    {\n'
        '      "text": "<The exact sentence or literal quote from the text>",\n'
        '      "is_explicit_quote": <boolean>,\n'
        '      "target_metadata": {\n'
        '          "title": "<The inferred or explicit paper title. If none, null>",\n'
        '          "authors": ["<Extracted author 1>", ...],\n'
        '          "year": <Integer. If none, null>,\n'
        '          "core_topic": "<Max 3 words capturing the specific mechanism>"\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        "}"
    )
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=draft)])

    try:
        text = resp.content.strip()
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match: text = match.group(0)
        parsed = json.loads(text)
        raw_citations = parsed.get("citations", [])
        
        # Cast to strictly validated models
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


# ── Node 3: OpenAlex Verification (Phase 2) ──────────────────────────────────
def search_literature(state: SciVerifyState) -> dict:
    """Verify citations exist via abstract metadata and cascading OpenAlex search."""
    from retrieval import verify_openalex_citation
    citations = state.get("citations", [])
    logger.info(f"[SEARCH] Verifying {len(citations)} extracted citations against OpenAlex...")
    
    results = []
    
    for c_dict in citations:
        ext_cit = ExtractedCitation(**c_dict)
        logger.info(f"[SEARCH] Validating target for claim {ext_cit.claim_id}: {ext_cit.target_metadata.title or ext_cit.target_metadata.core_topic}")
        matched_paper, reasoning = verify_openalex_citation(ext_cit.target_metadata)
        
        status = VerificationStatus.HALLUCINATED_PAPER
        if matched_paper:
            if matched_paper.oa_pdf_url:
                status = VerificationStatus.VERIFIED_QUOTE # Temporary status, to be updated in Phase 3
            else:
                status = VerificationStatus.UNKNOWN_PAYWALLED
                
        res = VerificationResult(
            claim_id=ext_cit.claim_id,
            status=status,
            matched_paper=matched_paper,
            reasoning=reasoning
        )
        results.append(res.model_dump())
        
    return {"verification_results": results, "status": "search_complete"}


# ── Node 4: Dual-Engine Verification (Phase 3) ─────────────────────────────
NLI_SUPPORT_SYSTEM = (
    "You are a semantic evaluator. Does the SOURCE TEXT support the CLAIM?\n"
    "Respond ONLY with JSON:\n"
    '{"supported": true|false, "reasoning": "<1 sentence>"}'
)

def verify_quotes(state: SciVerifyState) -> dict:
    """Run fuzzy string matching or NLI semantic validation."""
    citations_data = state.get("citations", [])
    results_data = state.get("verification_results", [])
    
    citations = {c['claim_id']: ExtractedCitation(**c) for c in citations_data}
    results = {r['claim_id']: VerificationResult(**r) for r in results_data}
    
    llm = get_llm(MODEL_NLI, temperature=0.0)
    import re
    from thefuzz import fuzz
    from retrieval import build_snippet_index, retrieve_relevant_snippets

    for cid, res in results.items():
        if res.status in (VerificationStatus.HALLUCINATED_PAPER, VerificationStatus.UNKNOWN_PAYWALLED):
            logger.info(f"[VERIFY] Claim {cid} skipped (Paper missing or paywalled)")
            continue
            
        cit = citations[cid]
        paper = res.matched_paper
        
        # Build index for just this paper (with full text extraction attempt)
        index, snippets, sids = build_snippet_index([paper], attempt_full_text=True)
        if index.ntotal == 0:
            res.status = VerificationStatus.UNKNOWN_PAYWALLED
            res.reasoning = "Document text could not be extracted."
            continue
            
        # Retrieve top 5 most relevant overlapping chunks
        top_chunks = retrieve_relevant_snippets(cit.text, index, snippets, sids, top_k=5)
        
        if cit.is_explicit_quote:
            # Mathematical Fuzz Match
            best_score = 0
            for chunk_str, _, _ in top_chunks:
                score = fuzz.token_set_ratio(cit.text.lower(), chunk_str.lower())
                if score > best_score:
                    best_score = score
            
            res.similarity_score = float(best_score)
            if best_score >= 85:
                res.status = VerificationStatus.VERIFIED_QUOTE
                res.reasoning = f"Fuzzy sequence match succeeded (Score: {best_score}%)"
            else:
                res.status = VerificationStatus.HALLUCINATED_QUOTE
                res.reasoning = f"Fuzzy sequence match failed (Highest similarity: {best_score}%)"
                
        else:
            # Semantic NLI Match
            best_support = False
            reasons = []
            for chunk_str, _, _ in top_chunks:
                try:
                    resp = llm.invoke([
                        SystemMessage(content=NLI_SUPPORT_SYSTEM),
                        HumanMessage(content=f"CLAIM: {cit.text}\n\nSOURCE TEXT: {chunk_str[:1200]}")
                    ])
                    # Parse JSON safely
                    txt = resp.content.strip()
                    match = re.search(r'\{.*\}', txt, re.DOTALL)
                    if match: txt = match.group(0)
                    p = json.loads(txt)
                    
                    if p.get("supported", False):
                        best_support = True
                        res.reasoning = p.get("reasoning", "Supported by context.")
                        break
                    else:
                        reasons.append(p.get("reasoning", "Not supported."))
                except Exception as e:
                    logger.error(f"[VERIFY] NLI JSON loop error: {e}")
            
            if best_support:
                res.status = VerificationStatus.SUPPORTED_SUMMARY
            else:
                res.status = VerificationStatus.UNSUPPORTED_SUMMARY
                res.reasoning = reasons[0] if reasons else "No semantic support found in context."

    final_results = [results[cid].model_dump() for cid in sorted(results.keys())]
    return {"verification_results": final_results, "status": "verification_complete"}


# ── Node 5: Report Compilation ─────────────────────────────────────────────
def compile_report(state: SciVerifyState) -> dict:
    """Calculate aggregate success metrics."""
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


# ── Routing ────────────────────────────────────────────────────────────────
# ── Build LangGraph Workflows ──────────────────────────────────────────────
def build_deep_search_workflow():
    wf = StateGraph(SciVerifyState)
    wf.add_node("execute", execute_deep_search)
    wf.set_entry_point("execute")
    wf.add_edge("execute", END)
    return wf.compile()

def build_verify_workflow():
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


# ── Public helpers ─────────────────────────────────────────────────────────
def run_deep_search(query: str, max_papers: int, is_oa_only: bool, ui_callback: Optional[callable] = None) -> dict:
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
