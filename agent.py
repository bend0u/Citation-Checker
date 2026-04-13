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
    AtomicClaim, NLILabel, NLIResult, SupportMatrixEntry,
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
    claims: list                    # list[dict]  (AtomicClaim dicts)
    evidence: dict                  # str(claim_id) -> [{snippet, source_id, score}]
    support_matrix: list            # list[dict]  (SupportMatrixEntry dicts)
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
    """Break the draft into atomic, verifiable claims (JSON mode)."""
    draft = state.get("input_text") or state.get("draft", "")
    if not draft:
        logger.warning("[DECOMPOSER] No draft text to decompose.")
        return {"claims": [], "status": "no_draft"}

    logger.info(f"[DECOMPOSER] Decomposing draft ({len(draft)} chars) using {MODEL_DECOMPOSER}...")
    llm = get_llm(MODEL_DECOMPOSER, temperature=0.0)
    sys = (
        "Break the following text into atomic scientific claims.\n"
        "Each claim = one standalone verifiable sentence.\n"
        "Split sentences with two ideas into two claims.\n"
        "Remove meta-statements. Keep all technical detail.\n"
        'Respond ONLY with JSON: {"claims": ["...", "..."]}'
    )
    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=draft)])

    try:
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
        claim_texts = parsed.get("claims", [])
        logger.info(f"[DECOMPOSER] Successfully parsed {len(claim_texts)} atomic claims.")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"[DECOMPOSER] JSON parse failed ({e}), falling back to sentence splitting.")
        import re
        claim_texts = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", draft)
            if len(s.strip()) > 20
        ]
        logger.info(f"[DECOMPOSER] Fallback produced {len(claim_texts)} claims.")

    claims = [
        AtomicClaim(claim_id=i, text=t).model_dump()
        for i, t in enumerate(claim_texts)
    ]
    for c in claims:
        logger.info(f"[DECOMPOSER]   Claim {c['claim_id']}: '{c['text'][:80]}...'")
    return {"claims": claims, "status": "claims_decomposed"}


# ── Node 3: Retrieve & Rerank ─────────────────────────────────────────────
def retrieve_evidence(state: SciVerifyState) -> dict:
    """Retrieve and rerank relevant snippets for each claim."""
    claims = state.get("claims", [])
    papers_dicts = state.get("papers", [])
    logger.info(f"[RETRIEVE] Starting evidence retrieval for {len(claims)} claims...")

    if not claims:
        logger.warning("[RETRIEVE] No claims to retrieve evidence for.")
        return {"evidence": {}, "status": "no_claims"}

    if not papers_dicts:
        logger.info("[RETRIEVE] No papers in state — reformulating claims for OpenAlex search...")
        combined = " ".join(c["text"] for c in claims[:3])
        strategy = reformulate_query(combined)
        keywords = strategy["keywords"]
        seen_dois: set[str] = set()
        all_papers = []
        for sq in keywords:
            results = search_openalex(sq, max_results=3)
            for p in results:
                key = p.doi or p.title
                if not key or key in seen_dois:
                    continue
                seen_dois.add(key)
                if evaluate_paper_relevance(combined, p.title, p.abstract or ""):
                    p.source_id = len(all_papers)
                    all_papers.append(p)
            if len(all_papers) >= 5:
                break
        papers = all_papers[:5]
        papers_dicts = [p.model_dump() for p in papers]
    else:
        papers = [PaperSource(**p) for p in papers_dicts]

    if not papers:
        logger.warning("[RETRIEVE] No sources found for evidence retrieval.")
        return {"papers": papers_dicts, "evidence": {}, "status": "no_sources"}

    index, snippets, sids = build_snippet_index(papers)

    evidence: dict = {}
    for claim in claims:
        cid = claim["claim_id"]
        results = retrieve_relevant_snippets(
            claim["text"], index, snippets, sids, top_k=3,
        )
        evidence[str(cid)] = [
            {"snippet": r[0], "source_id": r[1], "score": r[2]} for r in results
        ]

    total_pairs = sum(len(v) for v in evidence.values())
    logger.info(f"[RETRIEVE] Evidence retrieved: {total_pairs} claim-source pairs across {len(claims)} claims.")
    return {"papers": papers_dicts, "evidence": evidence, "status": "evidence_retrieved"}


# ── Node 4: NLI Auditor (rate-limit-aware) ─────────────────────────────────
NLI_SYSTEM = (
    "You are an NLI judge for scientific claims.\n"
    "Given a CLAIM and SOURCE TEXT, classify support.\n"
    "Respond ONLY with JSON:\n"
    '{"label":"Entailment"|"Neutral"|"Contradiction",'
    '"confidence":<0-1>,'
    '"source_quote":"<exact relevant quote from source>",'
    '"reasoning":"<one sentence>"}'
)


def audit_claims(state: SciVerifyState) -> dict:
    """Run NLI on each claim-evidence pair with Groq rate-limit throttling."""
    claims = state.get("claims", [])
    evidence = state.get("evidence", {})
    if not claims or not evidence:
        logger.warning("[AUDITOR] Nothing to audit — claims or evidence empty.")
        return {"support_matrix": [], "status": "nothing_to_audit"}

    llm = get_llm(MODEL_NLI, temperature=0.0)

    tasks_list = []
    for claim in claims:
        cid = claim["claim_id"]
        for ev in evidence.get(str(cid), []):
            tasks_list.append((claim["text"], ev["snippet"], cid, ev["source_id"]))

    logger.info(f"[AUDITOR] Starting NLI audit: {len(tasks_list)} evaluations using {MODEL_NLI}")
    logger.info(f"[AUDITOR] Estimated time: ~{len(tasks_list) * NLI_DELAY_BETWEEN_CALLS:.0f}s (rate-limited at {NLI_DELAY_BETWEEN_CALLS}s/call)")

    def eval_one_sync(claim_text, snippet, cid, sid):
        try:
            resp = llm.invoke([
                SystemMessage(content=NLI_SYSTEM),
                HumanMessage(
                    content=f"CLAIM: {claim_text}\n\n"
                            f"SOURCE TEXT: {snippet[:800]}"
                ),
            ])
            txt = resp.content.strip()
            if txt.startswith("```"):
                txt = txt.split("```")[1]
                if txt.startswith("json"):
                    txt = txt[4:]
                txt = txt.strip()
            p = json.loads(txt)
            result = NLIResult(
                label=NLILabel(p.get("label", "Neutral")),
                confidence=max(0.0, min(1.0, float(p.get("confidence", 0.5)))),
                source_quote=p.get("source_quote", ""),
                reasoning=p.get("reasoning", ""),
            )
        except Exception as e:
            logger.error(f"[AUDITOR] NLI error claim {cid} src {sid}: {e}")
            result = NLIResult(
                label=NLILabel.NEUTRAL, confidence=0.0,
                source_quote="", reasoning=f"Error: {e}",
            )
        return SupportMatrixEntry(
            claim_id=cid, source_id=sid, result=result,
        ).model_dump()

    matrix_entries = []
    for i, (claim_text, snippet, cid, sid) in enumerate(tasks_list):
        logger.info(f"[AUDITOR]   [{i+1}/{len(tasks_list)}] Evaluating claim {cid} vs source {sid}...")
        entry = eval_one_sync(claim_text, snippet, cid, sid)
        label = entry['result']['label']
        conf = entry['result']['confidence']
        logger.info(f"[AUDITOR]   [{i+1}/{len(tasks_list)}] Result: {label} (conf={conf:.2f})")
        matrix_entries.append(entry)

        if i < len(tasks_list) - 1:
            time.sleep(NLI_DELAY_BETWEEN_CALLS)

    logger.info(f"[AUDITOR] Audit complete: {len(matrix_entries)} evaluations finished.")
    return {"support_matrix": matrix_entries, "status": "audit_complete"}


# ── Node 5: Metrics Calculator ─────────────────────────────────────────────
def calculate_metrics(state: SciVerifyState) -> dict:
    """Compute Citation Accuracy and Citation Thoroughness."""
    claims = state.get("claims", [])
    matrix = state.get("support_matrix", [])
    logger.info(f"[METRICS] Calculating metrics from {len(matrix)} evaluations across {len(claims)} claims...")

    if not matrix:
        logger.warning("[METRICS] No matrix entries — returning zero metrics.")
        return {
            "metrics": {
                "citation_accuracy": 0.0, "citation_thoroughness": 0.0,
                "total_claims": len(claims), "verified_claims": 0,
                "total_evaluations": 0, "entailments": 0,
                "neutrals": 0, "contradictions": 0,
            },
            "status": "metrics_calculated",
        }

    ent = neu = con = 0
    claim_ok: dict[int, bool] = {}
    for entry in matrix:
        lab = entry["result"]["label"]
        cid = entry["claim_id"]
        if lab == "Entailment":
            ent += 1; claim_ok[cid] = True
        elif lab == "Neutral":
            neu += 1
        else:
            con += 1

    total = len(claims)
    verified = len(claim_ok)
    accuracy = verified / total if total else 0.0
    thoroughness = ent / len(matrix) if matrix else 0.0

    logger.info(
        f"[METRICS] Results: Accuracy={accuracy:.2%}, Thoroughness={thoroughness:.2%} | "
        f"Entailments={ent}, Neutrals={neu}, Contradictions={con} | "
        f"Verified={verified}/{total} claims"
    )

    return {
        "metrics": {
            "citation_accuracy": round(accuracy, 4),
            "citation_thoroughness": round(thoroughness, 4),
            "total_claims": total,
            "verified_claims": verified,
            "total_evaluations": len(matrix),
            "entailments": ent, "neutrals": neu, "contradictions": con,
        },
        "status": "complete",
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
    wf.add_node("retrieve", retrieve_evidence)
    wf.add_node("audit", audit_claims)
    wf.add_node("metrics", calculate_metrics)
    wf.set_entry_point("decompose")
    wf.add_edge("decompose", "retrieve")
    wf.add_edge("retrieve", "audit")
    wf.add_edge("audit", "metrics")
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
