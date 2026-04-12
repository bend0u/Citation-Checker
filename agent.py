"""
Sci-Verify AI Orchestrator
LangGraph Adversarial RAG workflow with DeepTRACE auditing.
Async-batched NLI Judge with Entailment / Neutral / Contradiction labels.
"""

import os
import json
import asyncio
import logging
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
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
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ── Node 1: Generator ─────────────────────────────────────────────────────
def generate_draft(state: SciVerifyState) -> dict:
    """Search OpenAlex and draft a cited response (or say 'I don't know')."""
    query = state["query"]
    papers = search_openalex(query, max_results=5)
    papers_dicts = [p.model_dump() for p in papers]

    if not papers:
        return {
            "papers": [],
            "draft": "Insufficient evidence: no relevant papers found in OpenAlex.",
            "confidence": 0.0,
            "status": "no_papers_found",
        }

    # Build context from abstracts
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

    llm = get_llm("gpt-4o-mini", temperature=0.2)
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
    for ln in lines:
        if ln.strip().startswith("CONFIDENCE:"):
            try:
                confidence = float(ln.strip().split(":")[-1])
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5
        else:
            draft_lines.append(ln)

    return {
        "papers": papers_dicts,
        "draft": "\n".join(draft_lines).strip(),
        "confidence": confidence,
        "status": "draft_generated",
    }


# ── Node 2: Decomposer ────────────────────────────────────────────────────
def decompose_claims(state: SciVerifyState) -> dict:
    """Break the draft into atomic, verifiable claims (JSON mode)."""
    draft = state.get("input_text") or state.get("draft", "")
    if not draft:
        return {"claims": [], "status": "no_draft"}

    llm = get_llm("gpt-4o-mini", temperature=0.0)
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
    except (json.JSONDecodeError, KeyError):
        import re
        claim_texts = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", draft)
            if len(s.strip()) > 20
        ]

    claims = [
        AtomicClaim(claim_id=i, text=t).model_dump()
        for i, t in enumerate(claim_texts)
    ]
    return {"claims": claims, "status": "claims_decomposed"}


# ── Node 3: Retrieve & Rerank ─────────────────────────────────────────────
def retrieve_evidence(state: SciVerifyState) -> dict:
    """Retrieve and rerank relevant snippets for each claim."""
    claims = state.get("claims", [])
    papers_dicts = state.get("papers", [])

    if not claims:
        return {"evidence": {}, "status": "no_claims"}

    # For external-verify mode: search OpenAlex if we have no papers yet
    if not papers_dicts:
        combined = " ".join(c["text"] for c in claims[:3])
        papers = search_openalex(combined, max_results=5)
        papers_dicts = [p.model_dump() for p in papers]
    else:
        papers = [PaperSource(**p) for p in papers_dicts]

    if not papers:
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

    return {"papers": papers_dicts, "evidence": evidence, "status": "evidence_retrieved"}


# ── Node 4: NLI Auditor (async-batched) ────────────────────────────────────
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
    """Run NLI on each claim-evidence pair using async batching."""
    claims = state.get("claims", [])
    evidence = state.get("evidence", {})
    if not claims or not evidence:
        return {"support_matrix": [], "status": "nothing_to_audit"}

    llm = get_llm("gpt-4o-mini", temperature=0.0)

    async def eval_one(claim_text, snippet, cid, sid):
        try:
            resp = await llm.ainvoke([
                SystemMessage(content=NLI_SYSTEM),
                HumanMessage(content=f"CLAIM: {claim_text}\n\nSOURCE TEXT: {snippet}"),
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
            logger.error(f"NLI error claim {cid} src {sid}: {e}")
            result = NLIResult(
                label=NLILabel.NEUTRAL, confidence=0.0,
                source_quote="", reasoning=f"Error: {e}",
            )
        return SupportMatrixEntry(
            claim_id=cid, source_id=sid, result=result,
        ).model_dump()

    async def run_all():
        sem = asyncio.Semaphore(5)
        async def bounded(coro):
            async with sem:
                return await coro
        tasks = []
        for claim in claims:
            cid = claim["claim_id"]
            for ev in evidence.get(str(cid), []):
                tasks.append(bounded(
                    eval_one(claim["text"], ev["snippet"], cid, ev["source_id"])
                ))
        return await asyncio.gather(*tasks) if tasks else []

    # Handle Streamlit's running event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                matrix = pool.submit(asyncio.run, run_all()).result()
        else:
            matrix = asyncio.run(run_all())
    except RuntimeError:
        matrix = asyncio.run(run_all())

    return {"support_matrix": list(matrix), "status": "audit_complete"}


# ── Node 5: Metrics Calculator ─────────────────────────────────────────────
def calculate_metrics(state: SciVerifyState) -> dict:
    """Compute Citation Accuracy and Citation Thoroughness."""
    claims = state.get("claims", [])
    matrix = state.get("support_matrix", [])

    if not matrix:
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
    if state.get("confidence", 0) < 0.15 or state.get("status") == "no_papers_found":
        return "end"
    return "decompose"


# ── Build LangGraph Workflow ───────────────────────────────────────────────
def build_workflow():
    wf = StateGraph(SciVerifyState)

    # router node (pass-through)
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
    app = build_workflow()
    return app.invoke({
        "query": query, "mode": "deep_search", "input_text": None,
        "papers": [], "draft": "", "claims": [], "evidence": {},
        "support_matrix": [], "metrics": {}, "confidence": 0.0,
        "status": "starting",
    })


def run_external_verify(text: str) -> dict:
    app = build_workflow()
    return app.invoke({
        "query": "", "mode": "external_verify", "input_text": text,
        "papers": [], "draft": text, "claims": [], "evidence": {},
        "support_matrix": [], "metrics": {}, "confidence": 1.0,
        "status": "starting",
    })
