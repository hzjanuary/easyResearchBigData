import json
import os
import shutil
import time
from pathlib import Path

import streamlit as st

from core.embedder import (
    add_to_vector_db,
    get_all_notebooks,
    delete_notebook,
    delete_file_from_notebook,
    get_notebook_stats,
)
from core.generator import query_rag_system
from core.summarizer import generate_notebook_summary
from core.cleaner_pro import discover_files
from core.ingestion_worker import run_pipeline_async, pipeline_status
from config import (
    UPLOAD_DIR,
    CHROMA_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)

CHAT_DIR = "database/chat_history"
os.makedirs(CHAT_DIR, exist_ok=True)


def _chat_path(name: str) -> str:
    return os.path.join(CHAT_DIR, f"{name}.json")


def save_chat(name: str, messages: list) -> None:
    with open(_chat_path(name), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_chat(name: str) -> list | None:
    p = _chat_path(name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def delete_chat(name: str) -> None:
    p = _chat_path(name)
    if os.path.exists(p):
        os.remove(p)


def get_recent_questions(name: str, limit: int = 5) -> list[str]:
    msgs = load_chat(name)
    if not msgs:
        return []
    return [m["content"] for m in msgs if m["role"] == "user"][-limit:][::-1]


st.set_page_config(
    page_title="easyResearch — AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body,p,div,h1,h2,h3,h4,h5,h6,label,input,textarea,a,li,td,th,
[data-testid="stMarkdownContainer"],[data-testid="stText"],[data-testid="stCaption"]
{ font-family:'Inter',sans-serif!important }
span[data-testid],.material-symbols-rounded,[class*="material-symbols"]
{ font-family:'Material Symbols Rounded'!important }

.stApp { background:#1c1c1f!important }
header[data-testid="stHeader"] { background:#1c1c1f!important; border-bottom:none!important }
section[data-testid="stSidebar"] { background:#111!important; border-right:1px solid #2d2d30!important }

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1
{ color:#fff!important; font-size:1.15rem!important; font-weight:700!important; text-align:center!important; margin:0 0 1rem 0!important }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4
{ color:#71717a!important; font-size:.7rem!important; font-weight:600!important; text-transform:uppercase!important; letter-spacing:.08em!important }

input[type="text"],input[type="password"],textarea
{ background:#27272a!important; border:1px solid #3f3f46!important; border-radius:8px!important; color:#e4e4e7!important }
input:focus,textarea:focus { border-color:#6366f1!important; box-shadow:0 0 0 1px #6366f1!important }
div[data-baseweb="select"]>div
{ background:#27272a!important; border:1px solid #3f3f46!important; border-radius:8px!important; color:#e4e4e7!important }

.stButton>button[kind="primary"]
{ background:#4f46e5!important; color:#fff!important; border:none!important; border-radius:8px!important; font-weight:600!important }
.stButton>button[kind="primary"]:hover { background:#4338ca!important }
.stButton>button:not([kind])
{ background:transparent!important; color:#a1a1aa!important; border:1px solid #3f3f46!important; border-radius:8px!important }
.stButton>button:not([kind]):hover { background:#27272a!important; color:#fff!important }

[data-testid="stFileUploader"]
{ border:1px dashed #3f3f46!important; border-radius:10px!important; background:#1f1f23!important }

[data-testid="stExpander"]>details
{ border:1px solid #27272a!important; border-radius:8px!important; background:#18181b!important }

hr { border:none!important; border-top:1px solid #27272a!important }
[data-testid="stProgress"]>div>div { background:#6366f1!important }

[data-testid="stChatMessage"] { background:transparent!important; border:none!important }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color:#d4d4d8!important; line-height:1.7 }
[data-testid="stChatInput"]>div,
[data-testid="stChatInput"]>div>div { background:#27272a!important; border:none!important; border-radius:14px!important; box-shadow:none!important; outline:none!important }
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus
{ background:#27272a!important; color:#e4e4e7!important; border:none!important; border-radius:10px!important; box-shadow:none!important; outline:none!important }
[data-testid="stChatInput"] textarea::placeholder { color:#71717a!important }
[data-testid="stChatInput"] button
{ color:#71717a!important; background:transparent!important; border:none!important }
[data-testid="stChatInput"] button:hover { color:#e4e4e7!important }
[data-testid="stBottomBlockContainer"] { background:#1c1c1f!important; border-top:none!important }

section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"]
{ gap:0!important; background:#1f1f23!important; border-radius:10px!important; padding:3px!important; border:1px solid #27272a!important; display:flex!important; width:100%!important }
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab"]
{ flex:1!important; border-radius:7px!important; color:#71717a!important; font-weight:500!important; font-size:.78rem!important; justify-content:center!important; padding:6px 0!important; white-space:nowrap!important; min-width:0!important }
section[data-testid="stSidebar"] .stTabs [aria-selected="true"]
{ background:#27272a!important; color:#fff!important }
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-highlight"],
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-border"]
{ display:none!important }
section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-panel"]
{ padding-top:.8rem!important }

::-webkit-scrollbar { width:6px; height:6px }
::-webkit-scrollbar-track { background:transparent }
::-webkit-scrollbar-thumb { background:#3f3f46; border-radius:3px }

.ws-badge
{ display:inline-flex; align-items:center; gap:6px; background:#27272a; border:1px solid #3f3f46; padding:6px 14px; border-radius:6px; color:#e4e4e7; font-weight:500; font-size:.85rem }

.stats-row { display:flex; gap:8px; margin:8px 0 }
.stat-card { flex:1; background:#1f1f23; border:1px solid #27272a; border-radius:8px; padding:10px 12px; text-align:center }
.stat-card .val { font-size:1.15rem; font-weight:700; color:#e4e4e7 }
.stat-card .lbl { font-size:.6rem; color:#71717a; text-transform:uppercase; letter-spacing:.05em; margin-top:2px }

.sidebar-footer { color:#3f3f46; font-size:.7rem; text-align:center; padding:.5rem 0 }

button[class*="st-key-del_file_"]
{ padding:0!important; min-height:0!important; width:24px!important; height:24px!important; font-size:13px!important; background:transparent!important; border:1px solid #3f3f46!important; color:#71717a!important; border-radius:4px!important; display:inline-flex!important; align-items:center!important; justify-content:center!important }
button[class*="st-key-del_file_"]:hover
{ background:#7f1d1d!important; border-color:#dc2626!important; color:#fca5a5!important }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("# 🧠 easyResearch")

    st.markdown("#### Workspaces")
    existing_notebooks = get_all_notebooks()
    options = ["➕ New workspace…"] + existing_notebooks
    selected_option = st.selectbox(
        "Workspace", options, label_visibility="collapsed",
        index=1 if existing_notebooks else 0,
    )

    final_notebook_name = "Default_Project"
    if selected_option == "➕ New workspace…":
        new_name = st.text_input(
            "Name", "New_Project", label_visibility="collapsed",
            placeholder="Enter workspace name…",
        )
        final_notebook_name = new_name.replace(" ", "_").strip()
        st.caption(f"Will create **{final_notebook_name}**")
    else:
        final_notebook_name = selected_option
        st.markdown(
            f'<div class="ws-badge">📂 {final_notebook_name}</div>',
            unsafe_allow_html=True,
        )
        stats = get_notebook_stats(final_notebook_name)
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card"><div class="val">{len(stats["files"])}</div><div class="lbl">Docs</div></div>
            <div class="stat-card"><div class="val">{stats["chunks"]}</div><div class="lbl">Vectors</div></div>
            <div class="stat-card"><div class="val">{stats["size_mb"]}</div><div class="lbl">MB</div></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    tab_ingest, tab_files, tab_cfg = st.tabs(["📥 Ingest", "📁 Files", "⚙️ Settings"])

    with tab_ingest:
        st.caption("Upload files or download a dataset → Start pipeline")

        uploaded = st.file_uploader(
            "Drop files", type=["pdf", "txt", "docx", "py"],
            accept_multiple_files=True, label_visibility="collapsed",
        )

        if uploaded:
            if st.button(f"💾 Save {len(uploaded)} file(s) to uploads/", use_container_width=True):
                for uf in uploaded:
                    (UPLOAD_DIR / uf.name).write_bytes(uf.getvalue())
                st.toast(f"Saved {len(uploaded)} file(s)")
                time.sleep(0.4)
                st.rerun()


        with st.expander("📦 Download dataset", expanded=False):
            dl_source = st.radio(
                "Source", ["🤗 HuggingFace", "📊 Kaggle"],
                horizontal=True, label_visibility="collapsed",
            )
            if "HuggingFace" in dl_source:
                ds_id = st.text_input(
                    "Dataset ID", placeholder="teyler/epstein-files-20k",
                    label_visibility="collapsed", key="hf_ds_id",
                )
                dl_btn = st.button("⬇ Download", key="dl_hf", use_container_width=True, type="primary")
                if dl_btn and ds_id and ds_id.strip():
                    with st.spinner(f"Downloading `{ds_id.strip()}` from HuggingFace…"):
                        try:
                            from datasets import load_dataset
                            dataset = load_dataset(ds_id.strip())
                            split = list(dataset.keys())[0]
                            ds_dir = UPLOAD_DIR / ds_id.strip().replace("/", "_")
                            ds_dir.mkdir(parents=True, exist_ok=True)
                            count = 0
                            for i, row in enumerate(dataset[split]):
                                # Try common text columns
                                text = None
                                for col in ("text", "content", "document", "page_content"):
                                    if col in row and row[col]: text = str(row[col]); break
                                if text is None:
                                    text = "\n".join(str(v) for v in row.values() if v)
                                if text.strip():
                                    (ds_dir / f"doc_{i:06d}.txt").write_text(text, encoding="utf-8")
                                    count += 1
                            st.success(f"✅ Saved {count} documents to `uploads/{ds_dir.name}/`")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Download failed: {e}")
            else:
                kg_id = st.text_input(
                    "Dataset ID", placeholder="jazivxt/the-epstein-files",
                    label_visibility="collapsed", key="kg_ds_id",
                )
                dl_btn = st.button("⬇ Download", key="dl_kg", use_container_width=True, type="primary")
                if dl_btn and kg_id and kg_id.strip():
                    with st.spinner(f"Downloading `{kg_id.strip()}` from Kaggle…"):
                        try:
                            import kagglehub
                            downloaded_path = kagglehub.dataset_download(kg_id.strip())
                            downloaded_path = Path(downloaded_path)
                            from config import SUPPORTED_EXTENSIONS
                            count = 0
                            dest_dir = UPLOAD_DIR / kg_id.strip().replace("/", "_")
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            for f in downloaded_path.rglob("*"):
                                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                                    shutil.copy2(f, dest_dir / f.name)
                                    count += 1
                            st.success(f"✅ Copied {count} files to `uploads/{dest_dir.name}/`")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Download failed: {e}")

        pending_files = discover_files(UPLOAD_DIR)
        st.caption(f"**{len(pending_files)}** file(s) ready in `uploads/`")

        with st.expander("⚙ Chunk settings", expanded=False):
            chunk_size = st.slider("Chunk size", 200, 2000, DEFAULT_CHUNK_SIZE, 50)
            chunk_overlap = st.slider("Overlap", 0, 500, DEFAULT_CHUNK_OVERLAP, 10)
            reset_db = st.checkbox("Reset DB before ingestion", value=False)

        col_go, col_reset = st.columns(2)
        with col_go:
            start_btn = st.button(
                "🚀 Start", key="start_pipeline",
                use_container_width=True, type="primary",
                disabled=len(pending_files) == 0,
            )
        with col_reset:
            if st.button("🗑 Reset DB", key="reset_db_btn", use_container_width=True):
                if Path(CHROMA_DIR).exists():
                    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
                    st.toast("Chroma DB reset.")
                    time.sleep(0.4)
                    st.rerun()

        if start_btn:
            run_pipeline_async(
                source_dir=UPLOAD_DIR,
                collection_name=final_notebook_name,
                reset_db=reset_db,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state["ingestion_running"] = True

        if st.session_state.get("ingestion_running", False):
            pbar = st.progress(0.0, text="Starting pipeline…")
            status_text = st.empty()

            while pipeline_status.stage not in ("done", "error", "idle"):
                pbar.progress(
                    min(pipeline_status.progress, 1.0),
                    text=f"**{pipeline_status.stage.upper()}** — {pipeline_status.message}",
                )
                status_text.caption(
                    f"Cleaned: {pipeline_status.docs_cleaned}  ·  "
                    f"Chunks: {pipeline_status.chunks_created}  ·  "
                    f"Embedded: {pipeline_status.chunks_embedded}"
                )
                time.sleep(0.5)

            if pipeline_status.stage == "done":
                pbar.progress(1.0, text="✅ Pipeline complete!")
                st.success(pipeline_status.message)
            elif pipeline_status.stage == "error":
                pbar.progress(1.0, text="❌ Pipeline failed.")
                st.error(pipeline_status.error)

            st.session_state["ingestion_running"] = False

    with tab_files:
        if selected_option == "➕ New workspace…":
            st.caption("Create a workspace first.")
        else:
            summary_path = os.path.join(CHROMA_DIR, f"{final_notebook_name}_summary.txt")
            if os.path.exists(summary_path):
                with st.expander("📝 Summary", expanded=False):
                    with open(summary_path, "r", encoding="utf-8") as f:
                        st.markdown(f.read())

            _stats = get_notebook_stats(final_notebook_name)
            if _stats["files"]:
                for idx, fname in enumerate(_stats["files"], 1):
                    c1, c2 = st.columns([0.85, 0.15])
                    display = fname if len(fname) <= 28 else fname[:25] + "…"
                    c1.caption(f"{idx}. {display}")
                    if c2.button("✕", key=f"del_file_{hash(fname)}", help=f"Delete {fname}"):
                        n = delete_file_from_notebook(final_notebook_name, fname)
                        if n:
                            st.toast(f"Deleted {fname} ({n} chunks)")
                            time.sleep(0.4)
                            st.rerun()
            else:
                st.caption("No documents yet.")

            recent = get_recent_questions(final_notebook_name)
            if recent:
                with st.expander(f"🔍 Recent ({len(recent)})", expanded=False):
                    for q in recent:
                        disp = q if len(q) <= 35 else q[:35] + "…"
                        if st.button(disp, key=f"hist_{hash(q)}", use_container_width=True):
                            st.session_state.setdefault("messages", [])
                            st.session_state.messages.append({"role": "user", "content": q})
                            st.rerun()

    with tab_cfg:
        llm_provider = st.selectbox("LLM Provider", ["Groq (LLaMA 3.3 70B)", "Google Gemini"])
        if "Groq" in llm_provider:
            user_key = st.text_input("API Key", type="password", placeholder="gsk_…")
            st.session_state.llm_provider = "groq"
        else:
            user_key = st.text_input("API Key", type="password", placeholder="AIza…")
            st.session_state.llm_provider = "gemini"
        st.session_state.user_api_key = user_key

        st.divider()

        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared. How can I help?"}
            ]
            delete_chat(final_notebook_name)
            st.rerun()

        if selected_option != "➕ New workspace…":
            if st.button("🗑 Delete workspace", use_container_width=True):
                if delete_notebook(final_notebook_name):
                    sp = os.path.join(CHROMA_DIR, f"{final_notebook_name}_summary.txt")
                    if os.path.exists(sp):
                        os.remove(sp)
                    delete_chat(final_notebook_name)
                    st.toast("Workspace deleted.")
                    time.sleep(0.4)
                    st.rerun()

    st.markdown('<div class="sidebar-footer">easyResearch · Big Data RAG</div>', unsafe_allow_html=True)


_default_welcome = [
    {
        "role": "assistant",
        "content": (
            "Welcome to your workspace.\n\n"
            "**Upload files** in the sidebar, run the **pipeline**, "
            "then ask me anything about your documents."
        ),
    }
]

if "current_notebook" not in st.session_state:
    st.session_state.current_notebook = final_notebook_name
    saved = load_chat(final_notebook_name)
    st.session_state.messages = saved if saved else list(_default_welcome)
elif st.session_state.current_notebook != final_notebook_name:
    save_chat(st.session_state.current_notebook, st.session_state.messages)
    saved = load_chat(final_notebook_name)
    st.session_state.messages = saved if saved else [
        {"role": "assistant", "content": f"Switched to **{final_notebook_name}**. Ask me anything!"}
    ]
    st.session_state.current_notebook = final_notebook_name
elif "messages" not in st.session_state:
    saved = load_chat(final_notebook_name)
    st.session_state.messages = saved if saved else list(_default_welcome)

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

prompt = st.chat_input("Send a message")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Searching documents…"):
            try:
                result = query_rag_system(
                    prompt,
                    collection_name=final_notebook_name,
                    chat_history=st.session_state.messages,
                    k_target=10,
                    user_api_key=st.session_state.get("user_api_key", ""),
                    llm_provider=st.session_state.get("llm_provider", "groq"),
                )

                answer = result["answer"]
                sources = result["sources"]
                standalone_q = result.get("standalone_question")
                info = result.get("pipeline_info", {})

                words = answer.split()
                for i, w in enumerate(words):
                    full_response += w + " "
                    if i % 3 == 0:
                        placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)
                placeholder.markdown(full_response)

                if standalone_q:
                    st.caption(f'🔍 Interpreted as: "{standalone_q}"')

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"{i}. `{src}`")

                if info:
                    with st.expander("🔬 Pipeline info", expanded=False):
                        cols = st.columns(3)
                        cols[0].metric("Retrieved", info.get("total_retrieved", 0))
                        cols[1].metric("Used", info.get("final_docs", 0))
                        cols[2].metric("Context", "✅" if info.get("contextualized") else "—")

            except Exception as e:
                st.error(str(e))
                full_response = "An error occurred. Please try again."
                placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat(final_notebook_name, st.session_state.messages)
