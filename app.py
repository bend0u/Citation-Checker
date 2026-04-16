"""
Sci-Verify: Verified Generative Information Retrieval for Science
==================================================================
Streamlit SaaS Application — the frontend layer.

Two tabs:
  Tab 1 — Deep Search:       User enters a question → synthesized report with verbatim quotes
  Tab 2 — Hallucination Check: User pastes/uploads AI text → forensic citation audit

UI Design:
  - Dark glassmorphism theme with gradient accents
  - Color-coded verification badges: 🟢 green (verified) / 🟡 yellow (unknown) / 🔴 red (hallucinated)
  - Glass cards, metric cards, and expandable audit details
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
# Limits the text size to prevent excessive LLM calls (and Groq rate limit hits)
MAX_INPUT_CHARS = 5_000  # ~1,250 tokens — keeps NLI auditing under budget

st.set_page_config(
    page_title="Sci-Verify | Verified AI for Science",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark Glassmorphism Theme
# ══════════════════════════════════════════════════════════════════════════════
# All visual styling is defined here as inline CSS injected via st.markdown.
# Uses Inter font, dark gradient backgrounds, frosted-glass cards, and
# color-coded claim badges (green/yellow/red).

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.main, .stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
}

/* Hero banner text */
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

/* Frosted-glass card (used for summaries, quotes, sources) */
.glass-card {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1rem;
}

/* Metric display cards (accuracy %, counts) */
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

/* Color-coded claim badges — left-border accent */
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

/* Source quote display box */
.source-quote {
    background: rgba(123,47,247,0.1);
    border: 1px solid rgba(123,47,247,0.3);
    border-radius: 8px; padding: 0.8rem;
    font-size: 0.85rem; color: #b0b0d0; margin-top: 0.3rem;
}

/* Heatmap table styling (for potential factual support matrix) */
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

/* Tab styling — gradient active tab */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05); border-radius: 8px;
    padding: 0.5rem 1.5rem; color: #8892b0;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7b2ff7, #00d2ff) !important;
    color: white !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Input field styling */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e0e0e0 !important; border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    """Render the hero banner with gradient title and subtitle."""
    st.markdown('<h1 class="hero-title">🔬 Sci-Verify</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Verified Generative Information Retrieval for Science — Powered by DeepTRACE'
        '</p>', unsafe_allow_html=True,
    )


def render_metrics(metrics: dict):
    """Render the 3-column metric cards (accuracy, verified, hallucinated)."""
    cols = st.columns(3)
    items = [
        ("Verification Acc.", metrics.get("citation_accuracy", 0)),
        ("Verified Claims", metrics.get("verified_claims", 0)),
        ("Hallucinated / Error", metrics.get("hallucinated", 0)),
    ]
    for col, (label, val) in zip(cols, items):
        # Color-code based on value: green (>=70%), yellow (>=40%), red (<40%)
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


def render_verification_report(citations_data: list, verification_results: list):
    """Render the detailed Forensic Audit Report for the Hallucination Check tab.
    
    Each citation is displayed with:
      - Color-coded status badge (green/yellow/red)
      - Expandable audit details (engine type, reasoning, similarity score)
      - Claimed vs Found metadata comparison (with mismatch warnings)
      - Retrieved paper provenance card (authors, DOI, OA status)
      - Corrected APA citation suggestion
    """
    st.markdown("### 📝 Forensic Audit Report")
    
    if not citations_data or not verification_results:
        st.info("No citations were extracted from the provided text.")
        return

    # Build claim_id → citation lookup
    citations = {c["claim_id"]: c for c in citations_data}
    
    for res in verification_results:
        cid = res["claim_id"]
        cit = citations.get(cid, {})
        status = res["status"]
        
        # Choose CSS class and color based on verification status
        if "Verified Exact Quote" in status or "Supported Semantic Claim" in status:
            css, color = "claim-entailment", "#00ff88"     # Green: verified
        elif "Unknown" in status:
            css, color = "claim-neutral", "#ffc107"         # Yellow: inconclusive
        else:
            css, color = "claim-contradiction", "#ff1744"   # Red: hallucinated
            
        # Render the status badge + claim text
        st.markdown(
            f'<div class="{css}"><strong><span style="color:{color}">{status}</span></strong><br/>{cit.get("text", "")}</div>',
            unsafe_allow_html=True,
        )
        
        # Expandable audit details
        with st.expander(f"📄 Audit Details: Claim {cid}"):
            mode = "Exact Quote Search" if cit.get("is_explicit_quote") else "Semantic Summary Search"
            
            st.markdown(f"**Verification Engine**: `{mode}`")
            st.markdown(f"**Target Citation**: `{cit.get('target_metadata', {}).get('title', 'Unknown Title')} by {cit.get('target_metadata', {}).get('authors', ['Unknown'])[0]}`")
            st.markdown(f"**Auditor Reasoning**: {res.get('reasoning', '')}")
            if cit.get("is_explicit_quote"):
                st.markdown(f"**Maximum Fuzzy Match**: `{res.get('similarity_score', 0):.1f}%`")
                
            paper = res.get("matched_paper")
            delta = res.get("metadata_delta")
            
            # ── Claimed vs Found metadata comparison ──
            if delta and paper:
                st.markdown("---")
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("##### 📝 Claimed Metadata")
                    st.markdown(f"**Title**: {delta.get('claimed_title', '')}")
                    st.markdown(f"**Year**: {delta.get('claimed_year', 'N/A')}")
                with cols[1]:
                    st.markdown("##### 🔎 Found in OpenAlex")
                    title_warn = " ⚠️ (Alias/Truncated)" if delta.get('is_title_alias') else ""
                    year_warn = " ⚠️ (Mismatch)" if delta.get('is_year_mismatch') else ""
                    st.markdown(f"**Title**: {delta.get('found_title', '')}{title_warn}")
                    st.markdown(f"**Year**: {delta.get('found_year', 'N/A')}{year_warn}")

            # ── Retrieved paper provenance card ──
            if paper:
                doi = paper.get("doi", "")
                authors_list = paper.get('authors', [])
                st.markdown(f"""
                <hr>
                <div class="source-quote" style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;">
                <strong>Retrieved Source:</strong> {paper.get("title", "")}<br/>
                <strong>Authors:</strong> {', '.join(authors_list)}<br/>
                <strong>Open Access PDF:</strong> {"✅ Yes" if paper.get("oa_pdf_url") else "❌ No (Paywalled)"}<br/>
                <strong>DOI:</strong> <a href="{doi}" target="_blank" style="color:#7b2ff7;">{doi}</a>
                </div>""", unsafe_allow_html=True)
                
                # Generate a corrected APA-style citation for the user
                apa_authors = " & ".join(authors_list) if len(authors_list) <= 2 else f"{authors_list[0]} et al." if authors_list else "Unknown"
                apa_year = delta.get('found_year', 'n.d.') if delta else 'n.d.'
                apa_citation = f"{apa_authors} ({apa_year}). {paper.get('title', 'Unknown Title')}."
                st.markdown("**📋 Corrected Citation (APA):**")
                st.code(apa_citation, language="text")


def render_sources(sources):
    """Render the list of retrieved sources as glass cards (used in Deep Search)."""
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


def render_deep_search_results(result: dict):
    """Render the full Deep Search output: strategy, summary, quotes, and papers.
    
    Layout:
      - Collapsible "Search Strategy" expander (shows how the query was decoded)
      - Two-column layout:
        Left: Executive summary + exact verbatim quotes
        Right: Evaluated papers with PDF/Abstract badges
    """
    search_strategy = result.get("search_strategy", {})
    sources = result.get("papers", [])
    deep_results = result.get("deep_search_results", {})
    
    st.markdown("---")
    
    # ── Search Strategy expander (shows query reformulation details) ──
    if search_strategy and (search_strategy.get("authors") or search_strategy.get("keywords")):
        with st.expander("🔍 Intelligient Search Strategy", expanded=False):
            st.markdown("**(LLM Query Reformulator)** Decoded the query to maximize OpenAlex hits.")
            if search_strategy.get("authors"):
                st.markdown(f"**Detected Authors:** `{'`, `'.join(search_strategy['authors'])}`")
            if search_strategy.get("keywords"):
                kw = " ➔ ".join(search_strategy["keywords"])
                st.markdown(f"**Keywords (Specific to General):** `{kw}`")
            if search_strategy.get("loops"):
                st.markdown("**Search Paths Executed ⚡:**")
                for loop in search_strategy["loops"]:
                    st.code(loop, language="text")
            if search_strategy.get("fallback_triggered"):
                st.warning("Phase 1 yielded insufficient relevant papers. Triggered Phase 2: Keyword-only Fallback.")

    st.markdown("")

    # ── Two-column layout: Summary (left) + Papers (right) ──
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Executive summary from LLM synthesis
        st.markdown("### 🔬 Executive Summary")
        intro = deep_results.get("introduction", "Error: No introduction returned.")
        st.markdown(f'<div class="glass-card" style="font-size:1.1rem; line-height:1.6;">{intro}</div>', unsafe_allow_html=True)
        
        # Verbatim quotes extracted from paper text
        st.markdown("<br>### 📌 Exact Quotes", unsafe_allow_html=True)
        quotes = deep_results.get("exact_citations", [])
        if not quotes:
            st.info("No exact quotes could be extracted from the retrieved text.")
        else:
            for q in quotes:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid #7b2ff7; padding: 1rem; margin-bottom: 1rem;">
                    <p style="font-style: italic; color: #e0e0e0; font-size: 1.05rem;">"{q.get('quote', '')}"</p>
                    <p style="color: #8892b0; font-size: 0.85rem; text-align: right; margin: 0;">— <strong>{q.get('authors', 'Unknown')}</strong>, <em>{q.get('paper_title', 'Unknown')}</em></p>
                </div>
                """, unsafe_allow_html=True)

    with col_right:
        # List of evaluated papers with PDF extraction status badges
        st.markdown("### 📚 Evaluated Papers")
        for src in sources:
            auths = ", ".join(src.get("authors", [])[:3])
            if len(src.get("authors", [])) > 3: auths += " et al."
            doi = src.get("doi", "")
            
            # Badge: green "PDF Extracted" or yellow "Abstract Only"
            ft_badge = '<span style="color:#00ff88; font-size:0.75rem; border:1px solid #00ff88; padding:2px 6px; border-radius:10px; margin-bottom:10px; display:inline-block;">PDF Extracted</span>' if src.get("full_text") == "Retrieved" else '<span style="color:#ffc107; font-size:0.75rem; border:1px solid #ffc107; padding:2px 6px; border-radius:10px; margin-bottom:10px; display:inline-block;">Abstract Only</span>'

            st.markdown(f"""
            <div class="glass-card" style="padding:1rem;">
                {ft_badge}<br>
                <strong style="color:#00d2ff;">[P{src.get("source_id","?")}]</strong>
                <span style="color:#e0e0e0;">{src.get("title","Untitled")}</span><br/>
                <span style="color:#8892b0;font-size:0.85rem;">{auths} • {src.get("publication_date","N/A")}</span><br/>
                <a href="{doi}" target="_blank" style="color:#7b2ff7;font-size:0.85rem;">{doi}</a>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHATGPT EXPORT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_chatgpt_export(content: str) -> str:
    """Parse a ChatGPT JSON export file and extract assistant messages.
    
    ChatGPT exports use a nested structure:
      [{mapping: {node_id: {message: {author: {role}, content: {parts: [...]}}}}}]
    
    If the content isn't valid JSON, returns it as-is (assumes plain text).
    """
    try:
        data = json.loads(content)
        texts = []
        iterable = data if isinstance(data, list) else [data]
        for item in iterable:
            if not isinstance(item, dict):
                continue
            # Walk the conversation tree and extract assistant messages
            for node in item.get("mapping", {}).values():
                msg = node.get("message")
                if msg and msg.get("author", {}).get("role") == "assistant":
                    parts = msg.get("content", {}).get("parts", [])
                    texts.extend(p for p in parts if isinstance(p, str))
        return "\n\n".join(texts) if texts else content
    except (json.JSONDecodeError, TypeError, KeyError):
        return content  # Not JSON — treat as plain text


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS RENDERER (Hallucination Check tab)
# ══════════════════════════════════════════════════════════════════════════════

def render_results(result: dict):
    """Render the full verification pipeline output (metrics + audit report)."""
    metrics = result.get("metrics", {})
    citations = result.get("citations", [])
    verification_results = result.get("verification_results", [])

    st.markdown("---")
    render_metrics(metrics)
    st.markdown("")

    st.markdown("### 🔬 Verification Details")
    render_verification_report(citations, verification_results)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_header()

    # Guard: require Groq API key
    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ Please set `GROQ_API_KEY` in your `.env` file to continue.")
        st.stop()

    tab1, tab2 = st.tabs(["🔍 Deep Search", "🔎 Hallucination Check"])

    # ── Tab 1: Deep Search ──────────────────────────────────────────────
    # User enters a research question → synthesized report with exact quotes
    with tab1:
        st.markdown(
            '<div class="glass-card">'
            "<h4 style='color:#e0e0e0;margin:0 0 .8rem 0;'>"
            "Deep Search Scientific Literature</h4></div>",
            unsafe_allow_html=True,
        )
        query = st.text_input(
            "Enter your research question",
            placeholder="e.g. Does CRISPR-Cas9 show off-target effects in hematopoietic stem cells?",
            label_visibility="collapsed",
            key="deep_search_input",
        )
        
        # Filter controls
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            is_oa_only = st.toggle("Prioritize Open Access (Forces Exact Quote Extraction)", value=False)
        with col_f2:
            max_papers = st.slider("Target Number of Papers", min_value=1, max_value=10, value=3)

        if st.button("🚀 Run Deep Search", key="btn_deep", use_container_width=True):
            if not query.strip():
                st.error("Please enter a query.")
            else:
                from agent import run_deep_search
                # Use st.status for real-time progress updates
                with st.status("🔬 Initializing Deep Search...", expanded=True) as status:
                    def log_to_ui(msg: str):
                        """Callback: updates the Streamlit status widget in real time."""
                        if len(msg) > 80: msg = msg[:77] + "..."
                        status.update(label=f"🔬 {msg}")
                        st.write(msg)
                    
                    result = run_deep_search(query.strip(), max_papers=max_papers, is_oa_only=is_oa_only, ui_callback=log_to_ui)
                    status.update(label="✅ Deep Search Complete!", state="complete", expanded=False)
                st.session_state["deep_result"] = result

        # Persist results across Streamlit reruns
        if "deep_result" in st.session_state:
            render_deep_search_results(st.session_state["deep_result"])

    # ── Tab 2: Hallucination Check (External Verifier) ──────────────────
    # User uploads/pastes AI-generated text → forensic citation audit
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
                # Cost guard: truncate massive inputs to stay under Groq limits
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

        # Persist results across Streamlit reruns
        if "verify_result" in st.session_state:
            render_results(st.session_state["verify_result"])


if __name__ == "__main__":
    main()
