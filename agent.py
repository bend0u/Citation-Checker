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


# ── LLM factory ────────────────────────────────────────────────────────────
def get_llm(model: str = MODEL_NLI, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Query Reformulator ─────────────────────────────────────────────────
REFORMULATE_SYSTEM = (
    "You are a scientific search query optimizer. "
    "Given a user's research question, generate 2-3 short, targeted academic "
    "search queries optimized for a scientific paper database.\n"
    "Rules:\n"
    "1. Each query should be 3-6 words, using precise academic terminology.\n"
    "2. Include author names if mentioned.\n"
    "3. Cover different angles of the question (core topic, specific method, key authors).\n"
    "4. Do NOT use question words (what, how, why). Use noun phrases only.\n"
    'Respond ONLY with JSON: {"queries": ["query1", "query2", "query3"]}'
)


def reformulate_query(user_query: str) -> list[str]:
    """Convert a natural language question into optimized academic search queries."""
    logger.info(f"[REFORMULATOR] Reformulating query: '{user_query[:80]}...'")
    llm = get_llm(MODEL_NLI, temperature=0.0)  # cheapest model
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
        queries = parsed.get("queries", [user_query])
        logger.info(f"[REFORMULATOR] Generated {len(queries)} search queries: {queries}")
        return queries if queries else [user_query]
    except Exception as e:
        logger.warning(f"[REFORMULATOR] Failed ({e}), falling back to original query.")
        return [user_query]


# ── Node 1: Generator ─────────────────────────────────────────────────────
def generate_draft(state: SciVerifyState) -> dict:
    """Search OpenAlex and draft a cited response (or say 'I don't know')."""
    query = state["query"]
    logger.info(f"[GENERATOR] Starting draft generation for query: '{query[:80]}...'")

    # Step 1: Reformulate the user query into targeted academic searches
    search_queries = reformulate_query(query)

    # Step 2: Search OpenAlex with each reformulated query, deduplicate by DOI
    seen_dois: set[str] = set()
    all_papers = []
    for sq in search_queries:
        results = search_openalex(sq, max_results=3)
        for p in results:
            key = p.doi or p.title  # fallback to title if no DOI
            if key not in seen_dois:
                seen_dois.add(key)
                p.source_id = len(all_papers)
                all_papers.append(p)

    # Limit to top 5 papers
    papers = all_papers[:5]
    papers_dicts = [p.model_dump() for p in papers]
    logger.info(f"[GENERATOR] Retrieved {len(papers)} unique papers from {len(search_queries)} queries.")

    if not papers:
        logger.warning("[GENERATOR] No papers found — returning insufficient evidence.")
        return {
            "papers": [],
            "draft": "Insufficient evidence: no relevant papers found in OpenAlex.",
            "confidence": 0.0,
            "status": "no_papers_found",
        }

    logger.info(f"[GENERATOR] Building context from {len(papers)} paper abstracts...")
    ctx_parts = []
    for p in papers:
        auths = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            auths += " et al."
        ctx_parts.append(
            f"[Source {p.source_id}] {p.title}\n"
            f"DOI: {p.doi}\nAuthors: {auths}\nAbstract: {p.abstract}\n"
        )
    context = "\n---\n".join(ctx_parts)

    logger.info(f"[GENERATOR] Calling {MODEL_GENERATOR} for synthesis...")
    llm = get_llm(MODEL_GENERATOR, temperature=0.2)
    sys = (
        "You are a scientific research synthesizer. Given REAL papers, "
        "write a concise, well-cited summary answering the user's query.\n"
        "RULES:\n"
        "1. ONLY use information from the provided sources.\n"
        "2. Cite using [Source N] after each claim.\n"
        "3. If the sources are insufficient, explicitly state:\n"
        '   "Insufficient evidence found in the retrieved literature."\n'
        "4. Include the DOI for each cited source.\n"
        "5. Be precise and scientific.\n\n"
        "On the LAST line output ONLY:  CONFIDENCE: <float 0-1>"
    )
    resp = llm.invoke([
        SystemMessage(content=sys),
        HumanMessage(content=f"Query: {query}\n\nSources:\n{context}"),
    ])

    lines = resp.content.strip().split("\n")
    draft_lines, confidence = [], 0.5
    confidence_found = False
    for ln in lines:
        if ln.strip().startswith("CONFIDENCE:"):
            try:
                confidence = float(ln.strip().split(":")[-1])
                confidence = max(0.0, min(1.0, confidence))
                confidence_found = True
            except ValueError:
                confidence = 0.5
        else:
            draft_lines.append(ln)

    if not confidence_found:
        logger.warning("[GENERATOR] No CONFIDENCE line found in LLM response — defaulting to 0.5")
        confidence = 0.5

    draft = "\n".join(draft_lines).strip()
    logger.info(f"[GENERATOR] Draft generated: {len(draft)} chars, confidence={confidence:.2f}")

    return {
        "papers": papers_dicts,
        "draft": draft,
        "confidence": confidence,
        "status": "draft_generated",
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
        search_queries = reformulate_query(combined)
        seen_dois: set[str] = set()
        all_papers = []
        for sq in search_queries:
            results = search_openalex(sq, max_results=3)
            for p in results:
                key = p.doi or p.title
                if key not in seen_dois:
                    seen_dois.add(key)
                    p.source_id = len(all_papers)
                    all_papers.append(p)
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
def _route_entry(state: SciVerifyState) -> str:
    return "decompose" if state.get("mode") == "external_verify" else "generate"

def _check_confidence(state: SciVerifyState) -> str:
    conf = state.get("confidence", 0)
    status = state.get("status", "")
    if status == "no_papers_found":
        logger.warning(f"[ROUTER] Stopping pipeline: no papers found.")
        return "end"
    # Only stop if confidence is essentially zero (generator explicitly said IDK)
    if conf <= 0.0:
        logger.warning(f"[ROUTER] Stopping pipeline: confidence={conf:.2f} (generator said 'I don't know').")
        return "end"
    logger.info(f"[ROUTER] Continuing pipeline: confidence={conf:.2f}")
    return "decompose"


# ── Build LangGraph Workflow ───────────────────────────────────────────────
def build_workflow():
    wf = StateGraph(SciVerifyState)

    wf.add_node("router", lambda s: s)
    wf.add_node("generate", generate_draft)
    wf.add_node("decompose", decompose_claims)
    wf.add_node("retrieve", retrieve_evidence)
    wf.add_node("audit", audit_claims)
    wf.add_node("metrics", calculate_metrics)

    wf.set_entry_point("router")
    wf.add_conditional_edges("router", _route_entry, {
        "generate": "generate", "decompose": "decompose",
    })
    wf.add_conditional_edges("generate", _check_confidence, {
        "decompose": "decompose", "end": END,
    })
    wf.add_edge("decompose", "retrieve")
    wf.add_edge("retrieve", "audit")
    wf.add_edge("audit", "metrics")
    wf.add_edge("metrics", END)

    return wf.compile()


# ── Public helpers ─────────────────────────────────────────────────────────
def run_deep_search(query: str) -> dict:
    logger.info(f"{'='*40} DEEP SEARCH START {'='*40}")
    logger.info(f"Query: {query}")
    app = build_workflow()
    result = app.invoke({
        "query": query, "mode": "deep_search", "input_text": None,
        "papers": [], "draft": "", "claims": [], "evidence": {},
        "support_matrix": [], "metrics": {}, "confidence": 0.0,
        "status": "starting",
    })
    logger.info(f"{'='*40} DEEP SEARCH END {'='*40}")
    return result


def run_external_verify(text: str) -> dict:
    logger.info(f"{'='*40} EXTERNAL VERIFY START {'='*40}")
    logger.info(f"Input text: {len(text)} chars")
    app = build_workflow()
    result = app.invoke({
        "query": "", "mode": "external_verify", "input_text": text,
        "papers": [], "draft": text, "claims": [], "evidence": {},
        "support_matrix": [], "metrics": {}, "confidence": 1.0,
        "status": "starting",
    })
    logger.info(f"{'='*40} EXTERNAL VERIFY END {'='*40}")
    return result
