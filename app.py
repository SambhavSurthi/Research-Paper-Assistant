import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.btw_handler import handle_btw
from backend.content_guard import (
    ContentRejected,
    screen_arxiv_query,
    screen_upload_bytes,
    screen_url,
    screen_user_message,
)
from backend.graph_store import (
    check_connection as neo4j_check_connection,
    get_session_graph_stats,
    get_session_hierarchy,
    get_session_semantic_provenance,
    get_session_subgraph,
    is_neo4j_configured,
    run_readonly_cypher,
)
from backend.graph_visualizer import build_graph_html
from backend.library import index_all_sources, index_saved_source
from backend.models import ResearchToolOutput
from backend.rag_graph import build_graph
from backend.research_tools import (
    RESEARCH_TOOLS,
    extract_pdf_full_context,
    format_tool_output_markdown,
    run_research_tool,
)
from backend.session_store import (
    EVAL_SESSION_ID,
    GLOBAL_SESSION_ID,
    add_source,
    list_sources,
    load_record,
    remove_source,
    save_chats,
    save_tool_history,
    snapshot_file,
    source_file,
)
from backend.vector_store import list_papers

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern UI Styling ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
  .hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.96));
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }
  .hero-title { font-size: 1.7rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.3rem; }
  .hero-sub { font-size: 0.94rem; color: #cbd5e1; margin: 0; }
  .pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
  .pill {
    font-size: 0.78rem; padding: 0.22rem 0.65rem; border-radius: 999px;
    background: rgba(51, 65, 85, 0.7); border: 1px solid rgba(148, 163, 184, 0.28); color: #e2e8f0;
  }
  .section-card {
    border-radius: 12px; padding: 1rem 1.2rem; margin: 0.75rem 0;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background: rgba(15, 23, 42, 0.55);
  }
  .evidence-card {
    border-left: 4px solid #22d3ee;
    background: rgba(14, 116, 144, 0.12);
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.6rem;
  }
  .ai-card {
    border-left: 4px solid #a78bfa;
    background: rgba(91, 33, 182, 0.12);
    border-radius: 8px; padding: 0.9rem 1.1rem; margin: 0.75rem 0;
  }
  .ideas-card {
    border-left: 4px solid #34d399;
    background: rgba(6, 95, 70, 0.14);
    border-radius: 8px; padding: 0.9rem 1.1rem; margin: 0.75rem 0;
  }
  .badge {
    display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em;
    padding: 0.15rem 0.55rem; border-radius: 999px; text-transform: uppercase; margin-right: 0.45rem;
  }
  .badge-evidence { background: #0891b2; color: #f8fafc; }
  .badge-ai { background: #7c3aed; color: #f8fafc; }
  .badge-ideas { background: #059669; color: #f8fafc; }
</style>
""",
    unsafe_allow_html=True,
)


def _graph_code_token() -> str:
    """Changes whenever the answer graph source changes, so Streamlit drops the cached graph."""
    parts = []
    for rel in ("backend/rag_graph.py", "backend/vector_store.py"):
        path = Path(rel)
        parts.append(f"{rel}:{path.stat().st_mtime_ns}" if path.exists() else rel)
    return "|".join(parts)


@st.cache_resource
def get_graph(_code_token: str):
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = ChatOpenAI(model="gpt-5-mini")


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


def generate_session_name(first_message: str) -> str:
    verdict = screen_user_message(first_message)
    if not verdict.allowed:
        return "New Session"
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
        "kind": "research",
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    st.session_state.tool_histories[sid] = []
    save_chats(sid, [])
    save_tool_history(sid, [])
    return sid


def ensure_evaluation_session() -> None:
    if EVAL_SESSION_ID in st.session_state.sessions_meta:
        st.session_state.sessions_meta[EVAL_SESSION_ID]["kind"] = "evaluation"
        st.session_state.sessions_meta[EVAL_SESSION_ID]["name"] = "Evaluation"
        save_sessions(st.session_state.sessions_meta)
        return
    st.session_state.sessions_meta[EVAL_SESSION_ID] = {
        "id": EVAL_SESSION_ID,
        "name": "Evaluation",
        "created_at": "1970-01-01T00:00:00",
        "is_named": True,
        "kind": "evaluation",
    }
    save_sessions(st.session_state.sessions_meta)


def _persist_chats(session_id: str) -> None:
    if session_id == EVAL_SESSION_ID:
        return
    save_chats(session_id, st.session_state.chats.get(session_id) or [])


def _persist_tools(session_id: str) -> None:
    if session_id == EVAL_SESSION_ID:
        return
    raw_items = []
    for item in st.session_state.tool_histories.get(session_id) or []:
        output = item.get("output")
        dumped = output.model_dump() if hasattr(output, "model_dump") else output
        raw_items.append({**item, "output": dumped})
    save_tool_history(session_id, raw_items)


def _payload_from_source(source: dict) -> dict | None:
    label = source.get("label") or "Untitled"
    stored = source_file(source)
    if source.get("kind") == "pdf" and stored is not None:
        return extract_pdf_full_context(stored.read_bytes(), label)
    snap = snapshot_file(source)
    if snap is not None:
        text = snap.read_text(encoding="utf-8")
        return {
            "title": label,
            "filename": label,
            "num_pages": 1,
            "text": text,
            "truncated": False,
            "b64": "",
            "kind": source.get("kind") or "text",
        }
    return None


def _ingest_saved_source(session_id: str, source: dict) -> str:
    try:
        result = index_saved_source(session_id, source, also_global=True)
    except ContentRejected:
        remove_source(session_id, source["id"])
        raise
    return _format_ingest_summary(source.get("label") or "source", result)


def _flash(level: str, text: str) -> None:
    st.session_state.setdefault("ingest_notices", []).append({"level": level, "text": text})


def _show_flash() -> None:
    for notice in st.session_state.pop("ingest_notices", []) or []:
        level = notice.get("level") or "info"
        text = notice.get("text") or ""
        if level == "error":
            st.error(text)
        elif level == "success":
            st.success(text)
        elif level == "warning":
            st.warning(text)
        else:
            st.info(text)


def _hydrate_tool_history(items: list[dict]) -> list[dict]:
    hydrated = []
    for item in items or []:
        raw = item.get("output")
        if isinstance(raw, dict):
            try:
                item = {**item, "output": ResearchToolOutput.model_validate(raw)}
            except Exception:
                continue
        elif not isinstance(raw, ResearchToolOutput):
            continue
        hydrated.append(item)
    return hydrated


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if type_name == "HumanMessage":
                chats.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                turn += 1
                chats.append({"role": "assistant", "content": content, "turn": turn, "graph_state": {}})
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    record = load_record(session_id)
    if session_id not in st.session_state.chats:
        saved_chats = record.get("chats") or []
        st.session_state.chats[session_id] = saved_chats or load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(1 for m in st.session_state.chats[session_id] if m["role"] == "assistant")
        st.session_state.turns[session_id] = turn_count
    if session_id not in st.session_state.tool_histories:
        st.session_state.tool_histories[session_id] = _hydrate_tool_history(record.get("tool_history") or [])


def _format_ingest_summary(label: str, result: dict) -> str:
    g = result.get("graph")
    if g:
        return (
            f"Added **{label}** — L1: {g['sections']} sections, {g['chunks']} embedded chunks | "
            f"L2: {g['entities']} semantic nodes, {g['relations']} relations | {g['similar_links']} semantic links."
        )
    if result.get("graph_error"):
        return f"Added **{label}** to vector store (Neo4j graph skipped: {result['graph_error']})."
    return f"Added: {label}"


def _render_retrieved_context_summary(graph_state: dict) -> None:
    docs = graph_state.get("retrieved_docs") or []
    if not docs:
        return
    reason_labels = {
        "vector_seed": "Vector Seed (Neo4j)",
        "kg_entity_seed": "Level-2 Semantic Match",
        "sequential_neighbor": "Level-1 NEXT_CHUNK",
        "hierarchy_section_sibling": "Level-1 Section Sibling",
        "graph_shared_entity": "Shared Concept Hop",
        "semantic_chunk_link": "SIMILAR_TO Link",
        "knowledge_graph_triples": "Semantic Triples + Provenance",
    }
    counts: dict[str, int] = {}
    for d in docs:
        meta = d.get("metadata") or {}
        reason = meta.get("retrieval_reason") or "vector_store"
        label = reason_labels.get(reason, reason)
        counts[label] = counts.get(label, 0) + 1
    summary_str = " · ".join(f"{k}: {v}" for k, v in counts.items())
    st.caption(f"🕸️ Two-Level Retrieval Breakdown — {summary_str}")


graph = get_graph(_graph_code_token())

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
ensure_evaluation_session()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "tool_histories" not in st.session_state:
    st.session_state.tool_histories = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id
if active_sid not in st.session_state.chats or active_sid not in st.session_state.tool_histories:
    switch_session(active_sid)
neo4j_ok, neo4j_msg = neo4j_check_connection() if is_neo4j_configured() else (False, "Neo4j credentials missing in .env")
neo4j_browser_url = os.environ.get("NEO4J_HTTP_URL", "http://localhost:7476").rstrip("/")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()
    st.divider()
    st.markdown("## 💬 Sessions")

    research_sessions = sorted(
        [s for s in st.session_state.sessions_meta.values() if s.get("kind") != "evaluation"],
        key=lambda s: s["created_at"],
        reverse=True,
    )
    eval_sessions = [s for s in st.session_state.sessions_meta.values() if s.get("kind") == "evaluation"]
    sorted_sessions = eval_sessions + research_sessions
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
    if active_sid == EVAL_SESSION_ID:
        st.markdown("## Evaluation")
        st.caption("This session shows saved project evaluation results. Papers are added from a research session.")
    else:
        st.markdown("## 📄 Index Papers")
        st.caption(
            "This is the only place papers are added. Only research papers are accepted. "
            "Other documents and instruction-override text are rejected before they are saved or sent."
        )
        _show_flash()

        st.markdown("**Upload Files**")
        uploaded_files = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key=f"uploader_{active_sid}",
            label_visibility="collapsed",
        )
        if st.button("Add Files", use_container_width=True, key="btn_add_files"):
            if uploaded_files:
                with st.spinner("Checking files, then saving and indexing this session…"):
                    for f in uploaded_files:
                        suffix = Path(f.name).suffix.lower()
                        kind = "pdf" if suffix == ".pdf" else "text"
                        raw = f.getvalue()
                        verdict = screen_upload_bytes(f.name, raw)
                        if not verdict.allowed:
                            _flash("error", f"{f.name}: {verdict.reason}")
                            continue
                        try:
                            source, created = add_source(
                                active_sid,
                                kind=kind,
                                label=f.name,
                                raw=raw,
                                suffix=suffix or ".bin",
                            )
                            if not created:
                                _flash("info", f"Already saved: {f.name}")
                                continue
                            _flash("success", _ingest_saved_source(active_sid, source))
                        except Exception as e:
                            _flash("error", f"Failed: {f.name} — {e}")
                st.rerun()
            else:
                st.warning("No files selected.")

        st.markdown("**Web Pages**")
        url_input = st.text_area(
            "URLs (one per line)",
            key=f"url_area_{active_sid}",
            height=72,
            label_visibility="collapsed",
            placeholder="https://example.com/paper",
        )
        if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
            urls = [u.strip() for u in url_input.splitlines() if u.strip()]
            if urls:
                with st.spinner("Checking links, then saving and indexing this session…"):
                    for url in urls:
                        verdict = screen_url(url)
                        if not verdict.allowed:
                            _flash("error", f"{url[:60]}: {verdict.reason}")
                            continue
                        try:
                            source, created = add_source(
                                active_sid,
                                kind="webpage",
                                label=url,
                                url=url,
                            )
                            if not created:
                                _flash("info", f"Already saved: {url[:60]}")
                                continue
                            _flash("success", _ingest_saved_source(active_sid, source))
                        except Exception as e:
                            _flash("error", f"Failed: {url[:60]} — {e}")
                st.rerun()
            else:
                st.warning("Enter at least one URL.")

        st.markdown("**ArXiv Papers**")
        arxiv_title = st.text_input(
            "Paper title or ArXiv ID",
            key=f"arxiv_input_{active_sid}",
            label_visibility="collapsed",
            placeholder="1706.03762  or  Attention Is All You Need",
        )
        if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
            if arxiv_title.strip():
                query = arxiv_title.strip()
                arxiv_verdict = screen_arxiv_query(query)
                if not arxiv_verdict.allowed:
                    _flash("error", arxiv_verdict.reason)
                else:
                    with st.spinner("Saving the arXiv paper and indexing this session…"):
                        try:
                            source, created = add_source(
                                active_sid,
                                kind="arxiv",
                                label=query,
                                query=query,
                            )
                            if not created:
                                _flash("info", "That arXiv paper is already saved in this session.")
                            else:
                                _flash("success", _ingest_saved_source(active_sid, source))
                        except Exception as e:
                            _flash("error", f"Failed: {e}")
                st.rerun()
            else:
                st.warning("Enter a paper title or ArXiv ID.")

        saved_sources = list_sources(active_sid)
        st.markdown("### Saved in this session")
        if not saved_sources:
            st.caption("Nothing saved yet. Add a PDF, web page, or arXiv paper above.")
        for source in saved_sources:
            label = source.get("label") or "Untitled"
            kind = source.get("kind") or "file"
            st.markdown(f"**{label}**")
            st.caption(f"{kind} · {source.get('added_at', '')}")
            stored = source_file(source)
            snap = snapshot_file(source)
            if stored is not None:
                st.download_button(
                    "Download file",
                    data=stored.read_bytes(),
                    file_name=Path(label).name,
                    key=f"dl_{active_sid}_{source['id']}",
                    use_container_width=True,
                )
            elif snap is not None:
                st.download_button(
                    "Download text",
                    data=snap.read_text(encoding="utf-8"),
                    file_name=f"{Path(label).stem}.txt",
                    mime="text/plain",
                    key=f"dl_txt_{active_sid}_{source['id']}",
                    use_container_width=True,
                )

    st.divider()
    st.markdown("### Indexed in this session")
    try:
        doc_titles = list_papers(active_sid) if active_sid != EVAL_SESSION_ID else []
    except Exception:
        doc_titles = None
    if doc_titles is None:
        st.caption("Could not load document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents indexed in this session yet.")

    st.divider()
    st.markdown("### 🕸️ Neo4j Knowledge Base")
    if neo4j_ok:
        st.success("Connected (Docker Neo4j)")
        st.markdown(f"[Open Neo4j Browser ↗]({neo4j_browser_url})")
    else:
        st.warning("Neo4j offline")
        st.caption(neo4j_msg)

# ── Modern Hero Header ─────────────────────────────────────────────────────────
neo4j_pill = "🟢 Neo4j Two-Level KG Online" if neo4j_ok else "🟡 Neo4j Offline"
st.markdown(
    f"""
<div class="hero-card">
  <div class="hero-title">📚 AI Research Paper Assistant</div>
  <p class="hero-sub">
    Document-Hierarchy + Semantic Provenance Knowledge Graph · 13 Academic Research Intelligence Tools · Strict Evidence vs. AI Separation
  </p>
  <div class="pill-row">
    <span class="pill">{neo4j_pill}</span>
    <span class="pill">🌲 Level 1: Paper → Section → Chunk</span>
    <span class="pill">🧬 Level 2: Concept · Method · Dataset · Metric · Finding + Provenance</span>
    <span class="pill">🔬 13 Direct-PDF Research Tools</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

def _render_evaluation() -> None:
    st.markdown("### Project evaluation")
    st.caption(
        "Saved DeepEval results for this project: contextual precision, contextual recall, "
        "contextual relevancy, answer relevancy, and faithfulness."
    )
    results_path = Path("eval_results.json")
    goldens_path = Path("goldens.json")
    if goldens_path.exists():
        try:
            golden_count = len(json.loads(goldens_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            golden_count = 0
        st.caption(f"{golden_count} golden questions in goldens.json.")
    if not results_path.exists():
        st.info("No evaluation results yet. Run `python evaluate.py` from the project folder, then reopen this session.")
        return
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.error("eval_results.json could not be read.")
        return
    if not isinstance(results, list) or not results:
        st.info("The evaluation file is empty.")
        return

    metric_scores: dict[str, list[float]] = {}
    passed_cases = 0
    for case in results:
        if case.get("success"):
            passed_cases += 1
        for metric in case.get("metrics") or []:
            name = metric.get("name") or "Metric"
            score = metric.get("score")
            if isinstance(score, (int, float)):
                metric_scores.setdefault(name, []).append(float(score))

    c1, c2, c3 = st.columns(3)
    c1.metric("Test cases", len(results))
    c2.metric("Cases passed", passed_cases)
    c3.metric("Cases failed", len(results) - passed_cases)
    if metric_scores:
        cols = st.columns(len(metric_scores))
        for col, (name, scores) in zip(cols, metric_scores.items()):
            avg = sum(scores) / len(scores)
            col.metric(name, f"{avg:.2f}")

    for index, case in enumerate(results, start=1):
        title = (case.get("input") or "Question")[:110]
        status = "Passed" if case.get("success") else "Failed"
        with st.expander(f"{index}. {status} — {title}", expanded=False):
            st.markdown("**Question**")
            st.write(case.get("input") or "")
            st.markdown("**Answer**")
            st.write(case.get("actual_output") or "")
            rows = []
            for metric in case.get("metrics") or []:
                rows.append({
                    "Metric": metric.get("name"),
                    "Score": metric.get("score"),
                    "Passed": metric.get("passed"),
                    "Reason": metric.get("reason"),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)


if active_sid == EVAL_SESSION_ID:
    _render_evaluation()
    st.stop()

tab_chat, tab_tools, tab_kb = st.tabs(
    ["💬 Research Chat", "🔬 Research Intelligence Tools", "🕸️ Knowledge Base Platform"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: RESEARCH CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    try:
        shared_titles = list_papers(GLOBAL_SESSION_ID)
    except Exception:
        shared_titles = []
    if shared_titles:
        st.caption(f"Chat answers use the shared index: {len(shared_titles)} paper(s) from every session.")
    else:
        st.caption("The shared index is empty. Add papers in the sidebar, or use Index everything on the Knowledge Base tab.")
    for msg in st.session_state.chats.get(active_sid, []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_retrieved_context_summary(msg.get("graph_state") or {})
                with st.expander(f"📊 Graph state · turn {msg.get('turn', '')}", expanded=False):
                    st.json(msg.get("graph_state") or {})

    if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
        is_btw = prompt.strip().lower().startswith("/btw")

        if is_btw:
            query = prompt.strip()[4:].strip()
            with st.chat_message("user"):
                st.markdown(prompt)
                st.caption("Side channel — not saved to session history.")
            with st.chat_message("assistant"):
                if not query:
                    st.markdown("Please add a question after `/btw`, e.g. `/btw What is attention?`")
                else:
                    placeholder = st.empty()
                    response_text = ""
                    for chunk in handle_btw(query):
                        response_text += chunk
                        placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
                st.caption("Side channel — not saved to session history.")
        else:
            if active_sid not in st.session_state.chats:
                st.session_state.chats[active_sid] = []
            if active_sid not in st.session_state.turns:
                st.session_state.turns[active_sid] = 0

            is_first_message = len(st.session_state.chats[active_sid]) == 0
            message_verdict = screen_user_message(prompt)
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
            st.session_state.turns[active_sid] += 1
            current_turn = st.session_state.turns[active_sid]

            if not message_verdict.allowed:
                with st.chat_message("assistant"):
                    st.markdown(message_verdict.reason)
                st.session_state.chats[active_sid].append(
                    {
                        "role": "assistant",
                        "content": message_verdict.reason,
                        "graph_state": {},
                        "turn": current_turn,
                    }
                )
                _persist_chats(active_sid)
            else:
                if is_first_message:
                    maybe_rename_session(active_sid, prompt)

                input_state = {
                    "messages": [HumanMessage(content=prompt)],
                    "session_id": active_sid,
                    "query": prompt,
                    "route": None,
                    "retrieved_docs": [],
                    "retrieval_attempts": 0,
                    "claim_verdict": None,
                    "claim_source": None,
                    "superseding_papers": [],
                    "answer": None,
                    "is_relevant": None,
                    "rewrite_count": 0,
                }
                config = {"configurable": {"thread_id": active_sid}}

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response_text = ""

                    for chunk, metadata in graph.stream(input_state, config, stream_mode="messages"):
                        if (
                            metadata.get("langgraph_node") == "generate_answer"
                            and hasattr(chunk, "content")
                            and chunk.content
                        ):
                            response_text += chunk.content
                            placeholder.markdown(response_text + "▌")

                    if not response_text:
                        final_values = graph.get_state(config).values
                        response_text = final_values.get("answer") or "No response generated."

                    placeholder.markdown(response_text)

                    final_values = graph.get_state(config).values
                    state_snapshot = _serialize_state(final_values)
                    _render_retrieved_context_summary(state_snapshot)

                    with st.expander(f"📊 Graph state · turn {current_turn}", expanded=False):
                        st.json(state_snapshot)

                st.session_state.chats[active_sid].append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "graph_state": state_snapshot,
                        "turn": current_turn,
                    }
                )
                _persist_chats(active_sid)
                if is_first_message:
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RESEARCH INTELLIGENCE TOOLS (DIRECT PDF + HARD SYSTEM PROMPTS)
# ══════════════════════════════════════════════════════════════════════════════
with tab_tools:
    st.markdown("### 🔬 Research Intelligence Studio")
    st.caption(
        "Research papers are sent to the model with specialized, strictly grounded system prompts across all 13 academic analysis tools. "
        "Non-research documents and instruction-override text are blocked before the model sees them. "
        "Every response explicitly separates **Verified Paper Evidence** from **AI-Generated Analysis & Ideas**."
    )

    with st.expander("🗺️ Research Intelligence Workflow Architecture", expanded=False):
        st.code(
            "Direct PDF(s) / Two-Level Knowledge Graph\n"
            "   ├── Single-Paper Analysis: Summarization · Abstract Extraction · Methodology · Algorithms & Equations\n"
            "   ├── Data & Results:        Datasets & Metrics · Results Analysis · Contributions, Limitations & Future Work\n"
            "   ├── Multi-Paper Analysis:  Paper Comparison · Cross-Paper Synthesis · Contradiction Detection\n"
            "   └── Ideation & Writing:    Research-Gap Discovery → Research-Question Generation → Literature-Review Draft",
            language="text",
        )

    src_col, cfg_col = st.columns([1.2, 1.8], gap="large")

    session_sources = list_sources(active_sid)
    source_labels = {
        source["id"]: f"{source.get('kind', 'file')}: {source.get('label', 'Untitled')}"
        for source in session_sources
    }
    with src_col:
        st.markdown("#### 1. Papers saved in this session")
        st.caption("Papers are added only from the sidebar. Pick which saved sources this tool should read.")
        selected_source_ids = st.multiselect(
            "Saved sources",
            options=list(source_labels.keys()),
            default=list(source_labels.keys()),
            format_func=lambda source_id: source_labels.get(source_id, source_id),
            key=f"rt_sources_{active_sid}",
        )
        include_indexed = st.checkbox(
            "Also include this session's knowledge graph",
            value=True,
            key=f"rt_include_kg_{active_sid}",
        )
        attach_native_pdf = st.checkbox(
            "Attach raw PDF binary directly to model (recommended)",
            value=True,
            key=f"rt_attach_pdf_{active_sid}",
        )
        if selected_source_ids:
            st.info(f"📑 **{len(selected_source_ids)} saved source(s)** ready.")
        elif include_indexed:
            st.info("Using this session's knowledge graph.")
        else:
            st.warning("Add a PDF, web page, or arXiv paper from the sidebar first.")

    with cfg_col:
        st.markdown("#### 2. Select Research Intelligence Tool")
        categories = [
            "All Tools (13)",
            "Single-Paper Analysis",
            "Data & Results",
            "Multi-Paper Intelligence",
            "Ideation & Writing",
        ]
        selected_cat = st.segmented_control(
            "Workflow Stage",
            options=categories,
            default="All Tools (13)",
            key="rt_category_filter",
        ) or "All Tools (13)"

        filtered_specs = [
            spec for spec in RESEARCH_TOOLS.values()
            if selected_cat == "All Tools (13)" or spec.category == selected_cat
        ]
        tool_labels = {f"{s.icon} {s.title}": s.id for s in filtered_specs}
        chosen_label = st.selectbox("Tool", list(tool_labels.keys()), key="rt_tool_select")
        chosen_spec = RESEARCH_TOOLS[tool_labels[chosen_label]]

        st.markdown(
            f"<div class='section-card'><strong>{chosen_spec.icon} {chosen_spec.title}</strong> "
            f"<span class='pill'>{chosen_spec.category}</span><br/>"
            f"<span style='color:#cbd5e1;font-size:0.9rem;'>{chosen_spec.description}</span></div>",
            unsafe_allow_html=True,
        )

        if chosen_spec.requires_multi_paper and len(selected_source_ids) < 2 and not include_indexed:
            st.caption("💡 Tip: This tool works best with **2+ PDFs** (or with the indexed multi-paper corpus enabled).")

        focus_prompt = st.text_area(
            "Focus Instruction / Custom Research Question (optional)",
            placeholder=chosen_spec.example_prompt,
            height=82,
            key="rt_focus_prompt",
        )

        run_clicked = st.button(
            f"Run {chosen_spec.title}",
            type="primary",
            use_container_width=True,
            key="rt_run_btn",
        )

    if run_clicked:
        chosen_sources = [s for s in session_sources if s["id"] in set(selected_source_ids)]
        pdf_payloads = []
        for source in chosen_sources:
            try:
                payload = _payload_from_source(source)
            except ContentRejected as exc:
                st.error(str(exc))
                continue
            except Exception as exc:
                st.warning(f"Could not read {source.get('label')}: {exc}")
                continue
            if payload:
                pdf_payloads.append(payload)

        if not pdf_payloads and not include_indexed:
            st.error("Select a saved source or include this session's knowledge graph.")
        else:
            with st.spinner(f"Running {chosen_spec.title} on this session's papers…"):
                try:
                    output = run_research_tool(
                        tool_id=chosen_spec.id,
                        pdf_payloads=pdf_payloads,
                        session_id=active_sid,
                        include_indexed_corpus=include_indexed,
                        user_focus_prompt=focus_prompt,
                        attach_raw_pdf=attach_native_pdf,
                    )
                    st.session_state.tool_histories.setdefault(active_sid, []).insert(
                        0,
                        {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "sources": [p["filename"] for p in pdf_payloads]
                            + (["Session knowledge graph"] if include_indexed else []),
                            "output": output,
                        },
                    )
                    _persist_tools(active_sid)
                except ContentRejected as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Tool execution failed: {exc}")

    # ── Render Latest & Past Research Intelligence Results ───────────────────────
    tool_history = st.session_state.tool_histories.get(active_sid) or []
    if tool_history:
        st.divider()
        latest = tool_history[0]
        out = latest["output"]

        hdr_l, hdr_r = st.columns([3, 1])
        with hdr_l:
            st.markdown(f"### {out.tool_title}")
            st.caption(
                f"Evidence Tier: `{out.evidence_tier}` · Sources: {', '.join(latest['sources']) or 'Corpus'} · Generated at {latest['timestamp']}"
            )
        with hdr_r:
            md_export = format_tool_output_markdown(out)
            st.download_button(
                "⬇ Export Markdown",
                data=md_export,
                file_name=f"{out.tool_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # 1. Verified Paper Evidence (Published Material / Retrieved Evidence / Extracted Information)
        st.markdown("#### 📘 Verified Paper Evidence & Extracted Information")
        if out.verified_evidence:
            for ev in out.verified_evidence:
                st.markdown(
                    f"<div class='evidence-card'>"
                    f"<span class='badge badge-evidence'>{ev.content_type}</span>"
                    f"<strong>{ev.paper_title}</strong> · <em>{ev.section_or_page}</em><br/>"
                    f"<div style='margin-top:0.35rem;'>{ev.statement}</div>"
                    f"<div style='margin-top:0.35rem;color:#94a3b8;font-size:0.85rem;'>"
                    f"Source quote: “{ev.source_quote}”</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No direct evidence items were returned.")

        # 2. Structured Tables
        if out.structured_tables:
            st.markdown("#### 📊 Extracted & Comparative Tables")
            for tbl in out.structured_tables:
                st.markdown(f"**{tbl.title}**")
                if tbl.headers and tbl.rows:
                    norm_rows = [
                        (r + [""] * len(tbl.headers))[: len(tbl.headers)]
                        for r in tbl.rows
                    ]
                    st.dataframe(pd.DataFrame(norm_rows, columns=tbl.headers), use_container_width=True)

        # 3. AI-Generated Analysis
        st.markdown("#### 🤖 AI-Generated Analysis")
        st.markdown(
            "<div class='ai-card'><span class='badge badge-ai'>AI-Generated Content</span>"
            "Synthesized explanation & interpretation grounded in the evidence above.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(out.ai_generated_analysis)

        # 4. AI-Generated Research Ideas / Drafts
        if out.ai_generated_ideas_or_drafts:
            st.markdown("#### 💡 AI-Generated Research Ideas / Literature Draft")
            st.markdown(
                "<div class='ideas-card'><span class='badge badge-ideas'>AI-Generated Ideas / Draft</span>"
                "Candidate opportunities, questions, or drafted prose — verify against broader literature before claiming novelty.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(out.ai_generated_ideas_or_drafts)

        # 5. Grounding Notes
        if out.grounding_notes:
            with st.expander("⚠️ Grounding & Scope Notes", expanded=True):
                for note in out.grounding_notes:
                    st.markdown(f"- {note}")

        if len(tool_history) > 1:
            with st.expander(f"🕘 Previous Tool Runs ({len(tool_history) - 1})", expanded=False):
                for item in tool_history[1:8]:
                    prev_out = item["output"]
                    st.markdown(f"**{prev_out.tool_title}** ({item['timestamp']}) — {', '.join(item['sources'])}")
                    st.caption(prev_out.ai_generated_analysis[:280] + "…")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: TWO-LEVEL KNOWLEDGE BASE PLATFORM
# ══════════════════════════════════════════════════════════════════════════════
with tab_kb:
    st.markdown("**Shared index for every chat**")
    st.caption(
        "Index every PDF, web page, and arXiv paper saved in any chat, plus papers already stored in those sessions. "
        "Research chat then answers from that shared index."
    )
    if st.button("Index everything from all chats", type="primary", key="btn_index_all"):
        session_ids = [
            sid for sid, meta in st.session_state.sessions_meta.items()
            if meta.get("kind") != "evaluation" and sid != EVAL_SESSION_ID
        ]
        with st.spinner("Indexing saved PDFs, web pages, and arXiv papers from every chat…"):
            report = index_all_sources(session_ids)
        indexed = report.get("indexed") or []
        skipped = report.get("skipped") or []
        errors = report.get("errors") or []
        st.success(f"Added {len(indexed)} paper(s) to the shared index. {len(skipped)} already present.")
        if indexed:
            st.write(", ".join(indexed))
        if errors:
            st.warning("Some sources could not be indexed: " + "; ".join(errors[:8]))
        st.rerun()

    col_info, col_link = st.columns([3, 1])
    with col_info:
        if neo4j_ok:
            st.success(f"🟢 {neo4j_msg}")
        else:
            st.error(f"🔴 Neo4j Offline: {neo4j_msg}")
            st.code("docker compose up -d neo4j", language="bash")
    with col_link:
        st.link_button("Open Neo4j Browser ↗", neo4j_browser_url, use_container_width=True)

    if neo4j_ok:
        stats = get_session_graph_stats(active_sid)
        sem = stats.get("semantic_breakdown") or {}

        st.markdown("**Level 1 — Document-Hierarchy Graph**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Papers", stats.get("papers", 0))
        m2.metric("Sections", stats.get("sections", 0))
        m3.metric("Chunks", stats.get("chunks", 0))
        m4.metric("Embedded Chunks (1536d)", stats.get("embedded_chunks", 0))

        st.markdown("**Level 2 — Semantic Graph (with Provenance)**")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Concepts", sem.get("Concept", 0))
        s2.metric("Methods", sem.get("Method", 0))
        s3.metric("Datasets", sem.get("Dataset", 0))
        s4.metric("Metrics", sem.get("Metric", 0))
        s5.metric("Findings", sem.get("Finding", 0))

        rels = stats.get("relationships") or {}
        if rels:
            rel_pills = " · ".join(f"`{k}`: {v}" for k, v in sorted(rels.items()))
            st.markdown(f"**Graph Edges & Provenance Links:** {rel_pills}")

        if stats.get("papers", 0) == 0:
            st.info(
                "No papers in this session's Two-Level Knowledge Graph yet. "
                "Upload a paper in the sidebar or index the included sample report below."
            )
            sample_pdf = Path("documents/Openclaw_Research_Report.pdf")
            if sample_pdf.exists() and st.button("Index Sample Report into Neo4j Two-Level Graph"):
                raw = sample_pdf.read_bytes()
                verdict = screen_upload_bytes(sample_pdf.name, raw)
                if not verdict.allowed:
                    st.error(f"{sample_pdf.name}: {verdict.reason}")
                else:
                    with st.spinner("Extracting Document-Hierarchy & Semantic Graph with provenance…"):
                        try:
                            source, created = add_source(
                                active_sid,
                                kind="pdf",
                                label=sample_pdf.name,
                                raw=raw,
                                suffix=".pdf",
                            )
                            if not created:
                                st.info("That sample report is already saved in this session.")
                            else:
                                st.success(_ingest_saved_source(active_sid, source))
                        except ContentRejected as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(f"Failed: {sample_pdf.name} — {exc}")
                    st.rerun()
        else:
            sub_graph, sub_tree, sub_prov, sub_cypher = st.tabs(
                [
                    "🌐 Interactive Two-Level Graph",
                    "🌲 Level 1: Document Hierarchy",
                    "🧬 Level 2: Semantic Provenance",
                    "⚡ Cypher Console",
                ]
            )

            with sub_graph:
                hierarchy = get_session_hierarchy(active_sid)
                paper_options = {"All Papers": None}
                for p in hierarchy:
                    paper_options[p["title"]] = p["id"]

                view_map = {
                    "Two-Level (Hierarchy + Semantic)": "two_level",
                    "Level 1: Document-Hierarchy Only": "document_hierarchy",
                    "Level 2: Semantic Graph + Provenance": "semantic_graph",
                }
                fc1, fc2, fc3, fc4 = st.columns([1.5, 1.5, 1, 1])
                selected_paper_label = fc1.selectbox("Filter by Paper", list(paper_options.keys()))
                selected_view_label = fc2.selectbox("Knowledge Layer", list(view_map.keys()))
                show_chunks = fc3.checkbox("Show Chunks", value=True)
                node_limit = fc4.slider("Max Nodes", min_value=40, max_value=300, value=160, step=20)

                subgraph = get_session_subgraph(
                    session_id=active_sid,
                    paper_id=paper_options[selected_paper_label],
                    include_chunks=show_chunks,
                    include_entities=True,
                    representation_view=view_map[selected_view_label],
                    limit=node_limit,
                )
                st.caption(
                    f"Showing **{len(subgraph['nodes'])} nodes** and **{len(subgraph['edges'])} edges** "
                    f"(click any node to inspect details & provenance, scroll to zoom, drag to rearrange)."
                )
                components.html(build_graph_html(subgraph, height=580), height=595)

            with sub_tree:
                hierarchy = get_session_hierarchy(active_sid)
                for paper in hierarchy:
                    with st.expander(f"📄 {paper['title']} ({paper.get('num_chunks', 0)} chunks)", expanded=True):
                        st.caption(f"Source: `{paper.get('source', '')}` · Type: `{paper.get('doc_type', '')}`")
                        for sec in paper.get("sections", []):
                            indent = "　" * max(0, int(sec.get("level", 1)) - 1)
                            sec_chunks = sec.get("chunks", [])
                            page_badge = f" · p.{sec['page']}" if sec.get("page") else ""
                            with st.expander(f"{indent}📂 {sec['title']}{page_badge} ({len(sec_chunks)} chunks)", expanded=False):
                                st.caption(f"Path: `{sec.get('path', '')}`")
                                for ch in sec_chunks:
                                    emb_badge = "✅ Embedded" if ch.get("has_embedding") else "❌ No Vector"
                                    ents = [e for e in (ch.get("entities") or []) if e]
                                    ent_str = f" · Semantic Nodes: {', '.join(ents[:8])}" if ents else ""
                                    st.markdown(f"**Chunk {ch['chunk_index']}** ({emb_badge}{ent_str})")
                                    st.text(ch["text"][:600] + ("…" if len(ch["text"]) > 600 else ""))

            with sub_prov:
                st.markdown("Level-2 Semantic Graph nodes (`Concept`, `Method`, `Dataset`, `Metric`, `Finding`) linked to their exact paper, section, page, and quote provenance.")
                prov_rows = get_session_semantic_provenance(active_sid, limit=150)
                if prov_rows:
                    df_prov = pd.DataFrame(prov_rows)
                    type_filter = st.multiselect(
                        "Filter Semantic Category",
                        options=["Concept", "Method", "Dataset", "Metric", "Finding"],
                        default=["Concept", "Method", "Dataset", "Metric", "Finding"],
                    )
                    if type_filter:
                        df_prov = df_prov[df_prov["type"].isin(type_filter)]
                    st.dataframe(df_prov, use_container_width=True, height=480)
                else:
                    st.info("No semantic provenance records extracted yet.")

            with sub_cypher:
                st.markdown(f"Run read-only Cypher queries against your Neo4j knowledge base (or open **Neo4j Browser** at `{neo4j_browser_url}`).")
                presets = {
                    "Level-2 Semantic Nodes & Provenance": (
                        "MATCH (c:Chunk {session_id: $session_id})-[m:MENTIONS]->(e:Entity)\n"
                        "RETURN e.type AS category, e.name AS entity, m.paper_title AS paper, m.section_path AS section, m.page AS page, m.evidence_quote AS provenance_quote\n"
                        "LIMIT 30"
                    ),
                    "Level-2 Semantic Relations with Provenance": (
                        "MATCH (e1:Entity {session_id: $session_id})-[r:RELATED_TO]->(e2:Entity)\n"
                        "RETURN e1.type + ':' + e1.name AS source, r.relation AS relation, e2.type + ':' + e2.name AS target, r.paper_title AS paper, r.section_path AS section, r.evidence_quote AS quote\n"
                        "LIMIT 30"
                    ),
                    "Level-1 Chunk Interconnections (NEXT_CHUNK & SIMILAR_TO)": (
                        "MATCH (a:Chunk {session_id: $session_id})-[r:NEXT_CHUNK|SIMILAR_TO]->(b:Chunk)\n"
                        "RETURN a.chunk_index AS from_chunk, type(r) AS rel, r.score AS score, b.chunk_index AS to_chunk, a.section_title AS section\n"
                        "LIMIT 25"
                    ),
                    "Level-1 Document Hierarchy (Paper -> Section -> Chunks)": (
                        "MATCH (p:Paper {session_id: $session_id})-[:HAS_SECTION]->(s:Section)\n"
                        "OPTIONAL MATCH (s)-[:SUBSECTION*0..2]->(sub:Section)-[:HAS_CHUNK]->(c:Chunk)\n"
                        "RETURN p.title AS paper, s.title AS top_section, sub.path AS section_path, count(DISTINCT c) AS chunks\n"
                        "LIMIT 30"
                    ),
                    "Verify Vector Embeddings on Chunks": (
                        "MATCH (c:Chunk {session_id: $session_id})\n"
                        "RETURN c.chunk_index AS idx, c.section_title AS section, size(c.embedding) AS embedding_dims, left(c.text, 80) AS preview\n"
                        "ORDER BY idx ASC LIMIT 20"
                    ),
                }
                chosen = st.selectbox("Preset Queries", list(presets.keys()))
                cypher_text = st.text_area("Cypher Query (read-only)", value=presets[chosen], height=120)
                if st.button("Run Cypher Query"):
                    try:
                        rows = run_readonly_cypher(cypher_text, {"session_id": active_sid})
                        if rows:
                            st.dataframe(rows, use_container_width=True)
                        else:
                            st.info("Query returned 0 rows.")
                    except Exception as exc:
                        st.error(f"Query failed: {exc}")
