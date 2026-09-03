"""
RAG Agent with Database Routing - Streamlit UI.

Chat over three routed Qdrant collections, with document ingestion,
execution telemetry, and evaluation views.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_agent import ConversationMemory, DocumentParser, PipelineResult, run_pipeline
from rag_agent.databases import add_documents, doc_count, reset_databases
from rag_agent.evaluator import EvaluationResult
from rag_agent.llm import build_client, get_active_model
from rag_agent.memory import ChatMessage
from rag_agent.retriever import RetrievedDoc
from rag_agent.router import RoutingDecision
from rag_agent.telemetry import ExecutionTrace

load_dotenv()

DB_LABELS = {
    "products": ("🛍️", "Products DB", "#0ea5e9"),
    "support": ("🎧", "Support DB", "#10b981"),
    "financial": ("💰", "Financial DB", "#f59e0b"),
}

EXAMPLE_QUERIES = [
    "What are the specs and price of the TechPro X1 laptop?",
    "How do I reset my password?",
    "What pricing plans are available?",
    "What were the Q1 2025 revenue figures?",
    "What is the return policy for physical products?",
]

# -- Local persistence ---------------------------------------------------------
# One workspace on disk: chunks.json (per-DB chunk text, the source of truth for
# index rebuilds) and chat.json (messages + serialized pipeline metadata).

WORKSPACE = Path(__file__).resolve().parent / ".workspace"
CHUNKS_FILE = WORKSPACE / "chunks.json"
CHAT_FILE = WORKSPACE / "chat.json"


def _write_json(path: Path, data) -> None:
    """Atomic JSON write (temp file + rename) so a crash never corrupts state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _load_chunks() -> dict[str, list[str]]:
    return _read_json(CHUNKS_FILE, {db: [] for db in DB_LABELS})


def _snapshot_metadata(result: PipelineResult) -> dict:
    """Serialize a PipelineResult into a JSON-safe dict."""
    return {
        "routing": {"database": result.routing.database, "reasoning": result.routing.reasoning},
        "used_fallback": result.used_fallback,
        "retrieval_attempts": result.retrieval_attempts,
        "contextualized_query": result.contextualized_query,
        "rewritten_query": result.rewritten_query,
        "evaluation": {
            "groundedness_score": result.evaluation.groundedness_score,
            "is_faithful": result.evaluation.is_faithful,
            "status_label": result.evaluation.status_label,
        },
        "docs": [{"text": d.text, "score": d.score, "source": d.source} for d in result.docs],
        "trace": {
            "total_latency_ms": result.trace.total_latency_ms,
            "steps": [
                {"step_name": s.step_name, "latency_ms": s.latency_ms, "details": s.details}
                for s in result.trace.steps
            ],
        },
    }


def _restore_metadata(snap: dict) -> PipelineResult:
    """Rebuild a PipelineResult from a chat.json snapshot."""
    trace = ExecutionTrace()
    for s in snap.get("trace", {}).get("steps", []):
        trace.add_step(s.get("step_name", ""), s.get("latency_ms", 0.0), s.get("details", ""))
    trace.total_latency_ms = snap.get("trace", {}).get("total_latency_ms", 0.0)
    ev = snap.get("evaluation", {})
    return PipelineResult(
        answer="",
        routing=RoutingDecision(
            database=snap.get("routing", {}).get("database", "support"),
            reasoning=snap.get("routing", {}).get("reasoning", ""),
        ),
        docs=[
            RetrievedDoc(text=d["text"], score=d["score"], source=d.get("source", ""))
            for d in snap.get("docs", [])
        ],
        used_fallback=snap.get("used_fallback", False),
        retrieval_attempts=snap.get("retrieval_attempts", 1),
        contextualized_query=snap.get("contextualized_query"),
        rewritten_query=snap.get("rewritten_query"),
        evaluation=EvaluationResult(
            groundedness_score=ev.get("groundedness_score", 1.0),
            is_faithful=ev.get("is_faithful", True),
            status_label=ev.get("status_label", "Grounded"),
        ),
        trace=trace,
    )


def _save_chat() -> None:
    _write_json(CHAT_FILE, {
        "messages": st.session_state.messages,
        "metadata": {str(i): _snapshot_metadata(r) for i, r in st.session_state.metadata.items()},
        "total_queries": st.session_state.total_queries,
        "fallback_count": st.session_state.fallback_count,
    })


def _load_chat_into_session() -> None:
    chat = _read_json(CHAT_FILE, {})
    st.session_state.messages = chat.get("messages", [])
    st.session_state.metadata = {
        int(i): _restore_metadata(s) for i, s in chat.get("metadata", {}).items()
    }
    st.session_state.total_queries = chat.get("total_queries", 0)
    st.session_state.fallback_count = chat.get("fallback_count", 0)
    memory = ConversationMemory()
    msgs = [ChatMessage(role=m["role"], content=m["content"]) for m in st.session_state.messages]
    memory.messages = msgs[-memory.max_turns * 2:]
    st.session_state.memory = memory
    chunks = _load_chunks()
    st.session_state.doc_counts = {db: len(chunks.get(db, [])) for db in DB_LABELS}


def _ensure_index() -> None:
    """Rebuild the Qdrant index from saved chunks (local FastEmbed, no API cost)."""
    if not st.session_state.index_dirty or st.session_state.pipeline is None:
        return
    client, embeddings, _ = st.session_state.pipeline
    chunks_by_db = _load_chunks()
    with st.spinner("Rebuilding vector index (local embeddings)..."):
        reset_databases(client, embeddings)
        for db_key in DB_LABELS:
            if chunks_by_db.get(db_key):
                add_documents(client, embeddings, db_key, chunks_by_db[db_key])
        for db_key in DB_LABELS:
            st.session_state.doc_counts[db_key] = doc_count(client, db_key)
    st.session_state.index_dirty = False


# -- Page config & theme -------------------------------------------------------

st.set_page_config(
    page_title="RAG Agent with Database Routing",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}
.badge-route     { background: rgba(14,165,233,0.12);  border: 1px solid #0ea5e9; color: #38bdf8; }
.badge-support   { background: rgba(16,185,129,0.12);  border: 1px solid #10b981; color: #34d399; }
.badge-financial { background: rgba(245,158,11,0.12);  border: 1px solid #f59e0b; color: #fbbf24; }
.badge-fallback  { background: rgba(168,85,247,0.12);  border: 1px solid #a855f7; color: #c084fc; }
.badge-rewrite   { background: rgba(56,189,248,0.08);  border: 1px solid rgba(56,189,248,0.35); color: #7dd3fc; }
.badge-eval-high { background: rgba(34,197,94,0.12);   border: 1px solid #22c55e; color: #4ade80; }
.badge-eval-warn { background: rgba(245,158,11,0.12);  border: 1px solid #f59e0b; color: #fbbf24; }

.metric-box {
    border: 1px solid rgba(148,163,184,0.2);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.metric-val { font-size: 22px; font-weight: 700; }
.metric-lbl { font-size: 11.5px; color: #94a3b8; margin-top: 2px; }

.empty-card {
    border: 1px dashed rgba(148,163,184,0.3);
    border-radius: 12px;
    padding: 28px 20px;
    text-align: center;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)


# -- Session state -------------------------------------------------------------

for key, default in [
    ("messages", []),
    ("metadata", {}),
    ("pipeline", None),
    ("pending_query", None),
    ("total_queries", 0),
    ("fallback_count", 0),
    ("doc_counts", {"products": 0, "support": 0, "financial": 0}),
    ("doc_previews", {}),
    ("index_dirty", True),
    ("loaded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if not st.session_state.loaded:
    _load_chat_into_session()
    st.session_state.loaded = True


@st.cache_resource(show_spinner=False)
def _shared_vector_engine():
    """One Qdrant + embeddings engine shared by every browser session."""
    from rag_agent.databases import build_databases

    return build_databases()


@st.cache_resource(show_spinner=False)
def _shared_llm_client(api_key: str):
    return build_client(api_key)


def _get_pipeline(api_key: str):
    """(qdrant, embeddings, llm) tuple, rebuilt when the key changes."""
    if st.session_state.pipeline is None or st.session_state.get("llm_key") != api_key:
        client, embeddings = _shared_vector_engine()
        st.session_state.pipeline = (client, embeddings, _shared_llm_client(api_key))
        st.session_state.llm_key = api_key
    _ensure_index()
    return st.session_state.pipeline


# -- Sidebar -------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚡ RAG Agent")
    st.caption("Routed retrieval across three knowledge bases")

    st.divider()
    st.markdown("**Groq API key**")
    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_...",
        label_visibility="collapsed",
    ).strip()
    if api_key:
        st.caption(f"● Connected — `{get_active_model()}`")
    else:
        st.caption("○ Paste a key to start")

    st.divider()
    st.markdown("**Knowledge base**")
    total_chunks = max(sum(st.session_state.doc_counts.values()), 1)
    for db_key, (icon, label, color) in DB_LABELS.items():
        cnt = st.session_state.doc_counts[db_key]
        pct = cnt * 100 // total_chunks if cnt else 0
        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
                <span>{icon} {label}</span><span style="color:#94a3b8;font-weight:700;">{cnt}</span>
            </div>
            <div style="height:5px;border-radius:999px;background:rgba(148,163,184,0.12);overflow:hidden;">
                <div style="height:100%;width:{max(pct, 2 if cnt else 0)}%;background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.caption("Add documents in the **Knowledge Base** tab.")

    st.divider()
    st.markdown("**Session**")
    turns = len(st.session_state.memory.messages) // 2
    st.caption(f"Memory: {turns} / {st.session_state.memory.max_turns} turns")
    if st.button("🧹 New chat (keep documents)", use_container_width=True):
        st.session_state.messages = []
        st.session_state.metadata = {}
        st.session_state.memory.clear()
        st.session_state.total_queries = 0
        st.session_state.fallback_count = 0
        _save_chat()
        st.rerun()

    st.divider()
    st.markdown("**Demo queries**")
    for q in EXAMPLE_QUERIES:
        if st.button(q[:48] + ("..." if len(q) > 48 else ""), use_container_width=True, key=f"ex_{q}"):
            st.session_state.pending_query = q


# -- Header --------------------------------------------------------------------

total_docs = sum(st.session_state.doc_counts.values())
st.markdown("## RAG Agent with Database Routing")
st.caption(
    "Guardrails → LLM router → hybrid retrieval (dense + BM25) → relevance grading "
    "with adaptive query rewrite → grounded generation → faithfulness scoring."
)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Indexed chunks", total_docs)
h2.metric("Queries", st.session_state.total_queries)
h3.metric("Web fallbacks", st.session_state.fallback_count)
h4.metric("Memory turns", f"{len(st.session_state.memory.messages) // 2}/{st.session_state.memory.max_turns}")

tab_chat, tab_kb, tab_telemetry, tab_eval = st.tabs(
    ["💬 Chat", "📚 Knowledge Base", "📊 Telemetry", "🧪 Evaluation"]
)


def _render_badges(result: PipelineResult) -> None:
    icon, label, _ = DB_LABELS.get(result.routing.database, ("🔀", "Unknown", "#64748b"))
    if result.used_fallback:
        st.markdown('<div class="badge badge-fallback">🌐 Web Fallback</div>', unsafe_allow_html=True)
    else:
        badge_class = f"badge-{result.routing.database}" if result.routing.database != "products" else "badge-route"
        st.markdown(f'<div class="badge {badge_class}">{icon} Routed to {label}</div>', unsafe_allow_html=True)
    if result.retrieval_attempts > 1:
        st.markdown(
            f'<div class="badge badge-rewrite">🔄 Query rewritten & retried ({result.retrieval_attempts}x)</div>',
            unsafe_allow_html=True,
        )
    eval_class = "badge-eval-high" if result.evaluation.is_faithful else "badge-eval-warn"
    st.markdown(
        f'<div class="badge {eval_class}">🎯 {result.evaluation.groundedness_score:.2f} ({result.evaluation.status_label})</div>',
        unsafe_allow_html=True,
    )


def _render_details(result: PipelineResult) -> None:
    if result.docs:
        with st.expander(f"📑 Grounded sources ({len(result.docs)})"):
            for j, doc in enumerate(result.docs):
                st.markdown(f"**[{j+1}]** *(score `{doc.score:.2f}`)*\n\n{doc.text}")
    with st.expander("⚡ Execution trace"):
        st.markdown(f"**Routing:** *{result.routing.reasoning}*")
        if result.contextualized_query:
            st.markdown(f"**Resolved follow-up:** `{html.escape(result.contextualized_query)}`")
        if result.rewritten_query:
            st.markdown(f"**Rewritten query:** `{html.escape(result.rewritten_query)}`")
        st.markdown(f"**Total latency:** `{result.trace.total_latency_ms:.1f} ms`")
        for s in result.trace.steps:
            st.markdown(f"- `{s.step_name}`: **{s.latency_ms:.1f} ms** — {s.details}")


# ==============================================================================
# TAB 1: CHAT
# ==============================================================================

with tab_chat:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            result = st.session_state.metadata.get(i) if message["role"] == "assistant" else None
            if result:
                _render_badges(result)
            st.markdown(message["content"])
            if result:
                _render_details(result)

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-card">
            <h4>Ready</h4>
            <p>Ask a question about your indexed documents, or try a demo query from the sidebar.<br>
            Out-of-domain questions fall back to a web search.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None
    else:
        prompt = st.chat_input("Ask a question...", disabled=(not api_key))

    if prompt:
        if not api_key:
            st.error("Paste a Groq API key in the sidebar to proceed.")
            st.stop()

        with st.spinner("Initializing..."):
            client, embeddings, groq_client = _get_pipeline(api_key)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Guardrails → route → retrieve → grade → generate..."):
                try:
                    result = run_pipeline(
                        query=prompt,
                        client=client,
                        embeddings=embeddings,
                        groq_client=groq_client,
                        memory=st.session_state.memory,
                    )
                    _render_badges(result)
                    st.markdown(result.answer)
                    _render_details(result)

                    if result.used_fallback:
                        st.session_state.fallback_count += 1
                    st.session_state.metadata[len(st.session_state.messages)] = result
                    st.session_state.messages.append({"role": "assistant", "content": result.answer})
                    st.session_state.total_queries += 1

                except Exception as e:
                    if "try again in" in str(e).lower() or "rate limit" in str(e).lower():
                        error_msg = (
                            "**⏳ Groq free-tier quota exhausted.** The request waited for the token "
                            "window to free up but timed out. Retry shortly, or start the server with "
                            "`RAG_MODEL=openai/gpt-oss-20b` for the higher-quota model."
                        )
                    else:
                        error_msg = f"**Error:** {e}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

        _save_chat()
        st.rerun()


# ==============================================================================
# TAB 2: KNOWLEDGE BASE
# ==============================================================================

with tab_kb:
    if not api_key:
        st.info("Paste a Groq API key in the sidebar to manage the knowledge bases.")
    else:
        ingest_tab, explore_tab = st.tabs(["📥 Ingest", "🔍 Chunks"])

        with ingest_tab:
            target_db = st.selectbox(
                "Target knowledge base",
                options=list(DB_LABELS),
                format_func=lambda k: f"{DB_LABELS[k][0]} {DB_LABELS[k][1]}",
            )
            uploaded = st.file_uploader(
                "Upload a document (PDF, DOCX, XLSX, PPTX, image, TXT, MD)",
                type=["pdf", "docx", "xlsx", "pptx", "png", "jpg", "jpeg", "webp", "txt", "md"],
            )
            pasted = st.text_area("Or paste text / markdown", height=110)

            sig = (uploaded.name if uploaded else None, pasted)
            prev = st.session_state.doc_previews.get(target_db)
            disabled = uploaded is None and not pasted.strip()

            def _parse_inputs(groq_client) -> tuple[list[str], str]:
                parsed: list[str] = []
                engine = "Text Parser"
                if uploaded is not None:
                    chunks, engine = DocumentParser.parse_file(uploaded, uploaded.name, groq_client=groq_client)
                    parsed.extend(chunks)
                if pasted.strip():
                    chunks, _ = DocumentParser.parse_file(
                        pasted.encode("utf-8"), f"{target_db}_pasted.txt", groq_client=groq_client
                    )
                    parsed.extend(chunks)
                return parsed, engine

            c1, c2 = st.columns(2)
            if c1.button("👁 Preview", use_container_width=True, disabled=disabled):
                _, _, groq_client = _get_pipeline(api_key)
                with st.spinner("Parsing (no indexing yet)..."):
                    chunks, engine_used = _parse_inputs(groq_client)
                if chunks:
                    st.session_state.doc_previews[target_db] = {
                        "name": uploaded.name if uploaded else f"{target_db}_pasted.txt",
                        "engine": engine_used,
                        "chunks": chunks,
                        "sig": sig,
                    }
                else:
                    st.session_state.doc_previews.pop(target_db, None)
                    st.warning("No document content found.")
                st.rerun()

            if c2.button("⚡ Ingest & index", use_container_width=True, disabled=disabled):
                client, embeddings, groq_client = _get_pipeline(api_key)
                if prev is not None and prev.get("sig") == sig:
                    chunks, engine_used = list(prev["chunks"]), prev["engine"]
                else:
                    chunks, engine_used = _parse_inputs(groq_client)
                if chunks:
                    with st.spinner(f"Embedding {len(chunks)} chunk(s) via {engine_used}..."):
                        added = add_documents(client, embeddings, target_db, chunks)
                    st.session_state.doc_counts[target_db] += added
                    saved = _load_chunks()
                    saved.setdefault(target_db, []).extend(chunks)
                    _write_json(CHUNKS_FILE, saved)
                    st.session_state.doc_previews.pop(target_db, None)
                    st.success(f"Indexed {added} chunk(s) via {engine_used}.")
                    st.rerun()
                else:
                    st.warning("No document content found.")

            if prev and prev.get("sig") != sig:
                st.caption("⚠️ Input changed since the last preview — it will be re-parsed on ingest.")
            if prev:
                n = len(prev["chunks"])
                with st.expander(f"👁 Preview — {prev['name']} · {n} chunk(s)", expanded=True):
                    st.caption(f"Engine: **{prev['engine']}** · showing {min(n, 10)} of {n}")
                    for i, chunk in enumerate(prev["chunks"][:10]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.markdown(chunk if len(chunk) <= 1000 else chunk[:1000] + "\n… (truncated)")

        with explore_tab:
            inspect_db = st.selectbox(
                "Inspect database",
                options=list(DB_LABELS),
                format_func=lambda k: f"{DB_LABELS[k][0]} {DB_LABELS[k][1]} ({st.session_state.doc_counts.get(k, 0)})",
                key="inspect_db",
            )
            saved_chunks = _load_chunks()
            current = saved_chunks.get(inspect_db, [])
            search = st.text_input("Filter by keyword", placeholder="Type to filter...")
            filtered = [c for c in current if not search.strip() or search.lower() in c.lower()]
            st.caption(f"{len(filtered)} of {len(current)} chunks")

            for idx, chunk in enumerate(filtered[:20]):
                with st.expander(f"Chunk #{idx+1} · {len(chunk)} chars · {chunk[:60].strip()}..."):
                    st.text(chunk)
            if len(filtered) > 20:
                st.caption(f"… and {len(filtered) - 20} more.")

            st.divider()
            m1, m2 = st.columns(2)
            if m1.button(f"🧹 Clear {DB_LABELS[inspect_db][1]}", use_container_width=True):
                saved_chunks[inspect_db] = []
                _write_json(CHUNKS_FILE, saved_chunks)
                st.session_state.doc_counts[inspect_db] = 0
                st.session_state.index_dirty = True
                _ensure_index()
                st.success(f"Cleared {DB_LABELS[inspect_db][1]}.")
                st.rerun()
            if m2.button("🔄 Rebuild index", use_container_width=True):
                st.session_state.index_dirty = True
                _ensure_index()
                st.success("Index rebuilt from saved chunks.")
                st.rerun()

            st.caption(
                "Embeddings: `BAAI/bge-small-en-v1.5` (384d cosine) + `Qdrant/bm25` sparse, "
                "fused with Reciprocal Rank Fusion (k=60)."
            )


# ==============================================================================
# TAB 3: TELEMETRY
# ==============================================================================

with tab_telemetry:
    latencies = [
        r.trace.total_latency_ms for r in st.session_state.metadata.values()
        if r.trace and r.trace.total_latency_ms > 0
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    rewrites = sum(1 for r in st.session_state.metadata.values() if r.retrieval_attempts > 1)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Queries", st.session_state.total_queries)
    t2.metric("Avg latency", f"{avg_latency:.0f} ms")
    t3.metric("Fallback rate", f"{(st.session_state.fallback_count / max(st.session_state.total_queries, 1)) * 100:.0f}%")
    t4.metric("Rewrite retries", rewrites)

    st.divider()
    st.markdown("#### Latest execution trace")
    last_key = max(st.session_state.metadata) if st.session_state.metadata else None
    last = st.session_state.metadata.get(last_key) if last_key is not None else None

    if last and last.trace.steps:
        st.caption(
            f"Total **{last.trace.total_latency_ms:.1f} ms** · routed to "
            f"**{last.routing.database.upper()}** · {last.retrieval_attempts} retrieval attempt(s)"
        )
        st.bar_chart({s.step_name: round(s.latency_ms, 1) for s in last.trace.steps}, height=220)
        st.dataframe(
            [
                {"Phase": s.step_name, "Latency (ms)": f"{s.latency_ms:.1f}", "Details": s.details}
                for s in last.trace.steps
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Run a query in the Chat tab to capture a trace.")

    st.divider()
    c_route, c_mem = st.columns(2)
    with c_route:
        st.markdown("#### Routing distribution")
        counts = {"products": 0, "support": 0, "financial": 0, "web_fallback": 0}
        for r in st.session_state.metadata.values():
            if r.used_fallback:
                counts["web_fallback"] += 1
            elif r.routing.database in counts:
                counts[r.routing.database] += 1
        st.bar_chart(counts, height=200)

    with c_mem:
        st.markdown("#### Conversation memory")
        mem_msgs = st.session_state.memory.messages
        if mem_msgs:
            st.caption(f"{len(mem_msgs)} messages ({len(mem_msgs) // 2} turns) in the sliding window")
            for m in mem_msgs[-6:]:
                st.markdown(f"**{'👤' if m.role == 'user' else '🤖'} {m.role}:** `{m.content[:80]}...`")
        else:
            st.info("Memory buffer is empty.")


# ==============================================================================
# TAB 4: EVALUATION
# ==============================================================================

with tab_eval:
    st.markdown("#### Live session groundedness")
    st.caption(
        "Every answer is scored in-pipeline: documents are LLM-graded for relevance before "
        "generation, numeric claims are verified against the retrieved context, and the final "
        "answer gets a deterministic groundedness score."
    )

    if st.session_state.metadata:
        rows = []
        for idx, r in st.session_state.metadata.items():
            question = (
                st.session_state.messages[idx - 1]["content"]
                if 0 < idx <= len(st.session_state.messages) else "Query"
            )
            rows.append({
                "Query": question[:60] + ("..." if len(question) > 60 else ""),
                "Routed DB": r.routing.database,
                "Attempts": r.retrieval_attempts,
                "Fallback": "Yes" if r.used_fallback else "No",
                "Groundedness": f"{r.evaluation.groundedness_score:.2f}",
                "Verdict": r.evaluation.status_label,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        scores = [r.evaluation.groundedness_score for r in st.session_state.metadata.values()]
        e1, e2, e3 = st.columns(3)
        e1.metric("Answers scored", len(scores))
        e2.metric("Mean groundedness", f"{sum(scores) / len(scores):.2f}")
        e3.metric("Flagged low", sum(1 for r in st.session_state.metadata.values() if not r.evaluation.is_faithful))
    else:
        st.info("No queries in this session yet.")

    st.divider()
    st.markdown("#### Offline benchmarks")
    st.markdown(
        """
Benchmark results are produced by the evaluation harnesses, not by this page — run them to
regenerate the reports rather than reading numbers from the UI:

```bash
# 100-question LLM-judged accuracy benchmark
uv run python evaluation/evaluate_pdf_module.py --label run1 --judge-model openai/gpt-oss-20b

# Adversarial benchmark (numeric traps, multi-hop, unanswerable)
uv run python evaluation/evaluate_hard.py --label hard1 --judge-model openai/gpt-oss-20b

# RAGAS metrics over a saved run
uv run python evaluation/evaluate_ragas.py --source run1 --judge-model openai/gpt-oss-20b
```

Reports and raw results land in `evaluation/artifacts/`.
        """
    )
