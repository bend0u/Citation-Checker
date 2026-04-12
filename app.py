"""
Sci-Verify: Verified Generative Information Retrieval for Science
Streamlit SaaS Application
"""

import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize file logging — all logs go to sci_verify.log
from logging_config import setup_logging
setup_logging()

# ── Cost Safety: max input length for External Verifier ────────────────────
MAX_INPUT_CHARS = 5_000  # ~1,250 tokens — keeps NLI auditing under budget

st.set_page_config(
    page_title="Sci-Verify | Verified AI for Science",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.main, .stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
}

.hero-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(120deg, #00d2ff, #7b2ff7, #ff0080);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-size: 1.1rem; color: #8892b0;
    text-align: center; margin-bottom: 2rem;
}
.glass-card {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1rem;
}
.metric-card {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15); border-radius: 12px;
    padding: 1.2rem; text-align: center;
}
.metric-value { font-size: 2.2rem; font-weight: 700; margin: 0.3rem 0; }
.metric-label {
    font-size: 0.85rem; color: #8892b0;
    text-transform: uppercase; letter-spacing: 1px;
}
.claim-entailment {
    background: rgba(0,255,136,0.15); border-left: 3px solid #00ff88;
    padding: 0.5rem 1rem; margin: 0.5rem 0;
    border-radius: 0 8px 8px 0; color: #e0e0e0;
}
.claim-neutral {
    background: rgba(255,193,7,0.15); border-left: 3px solid #ffc107;
    padding: 0.5rem 1rem; margin: 0.5rem 0;
    border-radius: 0 8px 8px 0; color: #e0e0e0;
}
.claim-contradiction {
    background: rgba(255,23,68,0.15); border-left: 3px solid #ff1744;
    padding: 0.5rem 1rem; margin: 0.5rem 0;
    border-radius: 0 8px 8px 0; color: #e0e0e0;
}
.source-quote {
    background: rgba(123,47,247,0.1);
    border: 1px solid rgba(123,47,247,0.3);
    border-radius: 8px; padding: 0.8rem;
    font-size: 0.85rem; color: #b0b0d0; margin-top: 0.3rem;
}
.heatmap-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
.heatmap-table th {
    background: rgba(255,255,255,0.1); color: #8892b0;
    padding: 0.6rem; font-size: 0.8rem;
    border: 1px solid rgba(255,255,255,0.1);
}
.heatmap-table td {
    padding: 0.6rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
    font-weight: 600; font-size: 0.85rem;
}
.cell-entailment { background: rgba(0,255,136,0.3); color: #00ff88; }
.cell-neutral { background: rgba(255,193,7,0.3); color: #ffc107; }
.cell-contradiction { background: rgba(255,23,68,0.3); color: #ff1744; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05); border-radius: 8px;
    padding: 0.5rem 1.5rem; color: #8892b0;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7b2ff7, #00d2ff) !important;
    color: white !important;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e0e0e0 !important; border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Render helpers ─────────────────────────────────────────────────────────
def render_header():
    st.markdown('<h1 class="hero-title">🔬 Sci-Verify</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Verified Generative Information Retrieval for Science — Powered by DeepTRACE'
        '</p>', unsafe_allow_html=True,
    )


def render_metrics(metrics: dict):
    cols = st.columns(4)
    items = [
        ("Citation Accuracy", metrics.get("citation_accuracy", 0)),
        ("Thoroughness", metrics.get("citation_thoroughness", 0)),
        ("Total Claims", metrics.get("total_claims", 0)),
        ("Verified Claims", metrics.get("verified_claims", 0)),
    ]
    for col, (label, val) in zip(cols, items):
        if isinstance(val, float):
            color = "#00ff88" if val >= 0.7 else ("#ffc107" if val >= 0.4 else "#ff1744")
            display = f"{val:.0%}"
        else:
            color = "#00d2ff"
            display = str(val)
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{display}</div>
        </div>""", unsafe_allow_html=True)


def render_annotated_claims(claims, support_matrix, sources_dict):
    st.markdown("### 📝 Annotated Claims")
    # Best result per claim
    best: dict = {}
    for entry in support_matrix:
        cid = entry["claim_id"]
        conf = entry["result"]["confidence"]
        if cid not in best or conf > best[cid]["confidence"]:
            best[cid] = {**entry["result"], "source_id": entry["source_id"]}

    for claim in claims:
        cid = claim["claim_id"]
        r = best.get(cid)
        if not r:
            css, tag = "claim-neutral", "⚪ Not Evaluated"
        elif r["label"] == "Entailment":
            css, tag = "claim-entailment", f"✅ Entailment ({r['confidence']:.0%})"
        elif r["label"] == "Neutral":
            css, tag = "claim-neutral", f"⚠️ Neutral ({r['confidence']:.0%})"
        else:
            css, tag = "claim-contradiction", f"❌ Contradiction ({r['confidence']:.0%})"

        st.markdown(
            f'<div class="{css}"><strong>{tag}</strong><br/>{claim["text"]}</div>',
            unsafe_allow_html=True,
        )
        if r and r.get("source_quote"):
            src = sources_dict.get(r["source_id"], {})
            with st.expander(f"📄 Source: {src.get('title','Unknown')[:60]}"):
                st.markdown(f"""
                <div class="source-quote">
                <strong>Direct Quote:</strong> "{r['source_quote']}"<br/><br/>
                <strong>Reasoning:</strong> {r['reasoning']}<br/>
                <strong>DOI:</strong> <a href="{src.get('doi','')}" target="_blank"
                    style="color:#7b2ff7;">{src.get('doi','')}</a>
                </div>""", unsafe_allow_html=True)


def render_heatmap(claims, sources, support_matrix):
    st.markdown("### 🗺️ Factual Support Matrix")
    if not claims or not sources or not support_matrix:
        st.info("No data available for the heatmap.")
        return

    sids = sorted({s.get("source_id", i) for i, s in enumerate(sources)})
    lookup = {(e["claim_id"], e["source_id"]): e["result"] for e in support_matrix}

    hdr = "<th>Claim</th>"
    for sid in sids:
        src = next((s for s in sources if s.get("source_id") == sid), {})
        hdr += f'<th title="{src.get("title","")}">P{sid}</th>'

    rows = ""
    for c in claims:
        cid = c["claim_id"]
        row = (
            f'<td style="text-align:left;color:#e0e0e0;font-size:0.8rem;" '
            f'title="{c["text"]}">S{cid}</td>'
        )
        for sid in sids:
            r = lookup.get((cid, sid))
            if r:
                lab = r["label"]
                cls = (
                    "cell-entailment" if lab == "Entailment"
                    else "cell-neutral" if lab == "Neutral"
                    else "cell-contradiction"
                )
                row += f'<td class="{cls}">{r["confidence"]:.0%}</td>'
            else:
                row += "<td>—</td>"
        rows += f"<tr>{row}</tr>"

    st.markdown(f"""
    <div class="glass-card">
        <table class="heatmap-table">
            <thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody>
        </table>
        <div style="margin-top:.8rem;display:flex;gap:1.5rem;justify-content:center;">
            <span><span style="color:#00ff88;">●</span> Entailment</span>
            <span><span style="color:#ffc107;">●</span> Neutral</span>
            <span><span style="color:#ff1744;">●</span> Contradiction</span>
        </div>
    </div>""", unsafe_allow_html=True)


def render_sources(sources):
    st.markdown("### 📚 Retrieved Sources")
    for src in sources:
        auths = ", ".join(src.get("authors", [])[:3])
        if len(src.get("authors", [])) > 3:
            auths += " et al."
        doi = src.get("doi", "")
        st.markdown(f"""
        <div class="glass-card" style="padding:1rem;">
            <strong style="color:#00d2ff;">[P{src.get("source_id","?")}]</strong>
            <span style="color:#e0e0e0;">{src.get("title","Untitled")}</span><br/>
            <span style="color:#8892b0;font-size:0.85rem;">{auths} • {src.get("publication_date","N/A")}</span><br/>
            <a href="{doi}" target="_blank" style="color:#7b2ff7;font-size:0.85rem;">{doi}</a>
        </div>""", unsafe_allow_html=True)


def parse_chatgpt_export(content: str) -> str:
    """Parse ChatGPT JSON export or plain text."""
    try:
        data = json.loads(content)
        texts = []
        iterable = data if isinstance(data, list) else [data]
        for item in iterable:
            if not isinstance(item, dict):
                continue
            for node in item.get("mapping", {}).values():
                msg = node.get("message")
                if msg and msg.get("author", {}).get("role") == "assistant":
                    parts = msg.get("content", {}).get("parts", [])
                    texts.extend(p for p in parts if isinstance(p, str))
        return "\n\n".join(texts) if texts else content
    except (json.JSONDecodeError, TypeError, KeyError):
        return content


def render_results(result: dict):
    """Render the full pipeline output."""
    metrics = result.get("metrics", {})
    claims = result.get("claims", [])
    sources = result.get("papers", [])
    matrix = result.get("support_matrix", [])
    draft = result.get("draft", "")

    st.markdown("---")
    render_metrics(metrics)
    st.markdown("")

    col_left, col_right = st.columns([3, 2])
    sources_dict = {s.get("source_id", i): s for i, s in enumerate(sources)}

    with col_left:
        if draft:
            st.markdown("### 📄 Generated Draft")
            st.markdown(f'<div class="glass-card">{draft}</div>', unsafe_allow_html=True)
        render_annotated_claims(claims, matrix, sources_dict)

    with col_right:
        render_heatmap(claims, sources, matrix)
        render_sources(sources)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    render_header()

    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ Please set `GROQ_API_KEY` in your `.env` file to continue.")
        st.stop()

    tab1, tab2 = st.tabs(["🔍 Deep Search", "🔎 Hallucination Check"])

    # ── Tab 1: Deep Search ──
    with tab1:
        st.markdown(
            '<div class="glass-card">'
            "<h4 style='color:#e0e0e0;margin:0 0 .8rem 0;'>"
            "Ask a Scientific Question</h4></div>",
            unsafe_allow_html=True,
        )
        query = st.text_input(
            "Enter your research question",
            placeholder="e.g. Does CRISPR-Cas9 show off-target effects in hematopoietic stem cells?",
            label_visibility="collapsed",
            key="deep_search_input",
        )
        if st.button("🚀 Run Deep Search", key="btn_deep", use_container_width=True):
            if not query.strip():
                st.error("Please enter a query.")
            else:
                from agent import run_deep_search
                with st.spinner("🔬 Searching OpenAlex, drafting, and auditing…"):
                    result = run_deep_search(query.strip())
                st.session_state["last_result"] = result

        if "last_result" in st.session_state:
            render_results(st.session_state["last_result"])

    # ── Tab 2: External Verifier ──
    with tab2:
        st.markdown(
            '<div class="glass-card">'
            "<h4 style='color:#e0e0e0;margin:0 0 .8rem 0;'>"
            "Upload or Paste Text to Verify</h4></div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload ChatGPT export (.json) or plain text (.txt)",
            type=["json", "txt"],
            key="file_upload",
        )
        pasted = st.text_area(
            "Or paste text directly",
            height=200,
            placeholder="Paste AI-generated scientific text here…",
            key="paste_area",
        )

        if st.button("🔎 Verify", key="btn_verify", use_container_width=True):
            text = ""
            if uploaded:
                raw = uploaded.read().decode("utf-8", errors="ignore")
                text = parse_chatgpt_export(raw)
            elif pasted.strip():
                text = pasted.strip()

            if not text:
                st.error("Please upload a file or paste text.")
            else:
                # Cost guard: truncate massive inputs
                if len(text) > MAX_INPUT_CHARS:
                    st.warning(
                        f"⚠️ Input truncated from {len(text):,} to {MAX_INPUT_CHARS:,} "
                        f"characters to keep API costs low."
                    )
                    text = text[:MAX_INPUT_CHARS]

                from agent import run_external_verify
                with st.spinner("🔬 Decomposing claims and auditing…"):
                    result = run_external_verify(text)
                st.session_state["verify_result"] = result

        if "verify_result" in st.session_state:
            render_results(st.session_state["verify_result"])


if __name__ == "__main__":
    main()
