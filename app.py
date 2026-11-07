from __future__ import annotations

import html
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.rag_pipeline import (  # noqa: E402
    MODE_HYBRID,
    MODE_LABELS,
    SOURCE_MODES,
    RAGPipeline,
    RAGResponse,
    StorageStatus,
)
from retrieval.retriever import STORAGE_LABELS  # noqa: E402
from utils.config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from vectorstore.factory import STORAGE_DATABRICKS, STORAGE_LOCAL, STORAGES  # noqa: E402

logger = get_logger(__name__)


DISCLAIMER = (
    "This AI assistant provides answers based on uploaded documents. "
    "Responses may contain errors and should be independently verified. "
    "It is intended for informational purposes only."
)


MODE_ICONS = {STORAGE_LOCAL: "🖥️", STORAGE_DATABRICKS: "☁️", MODE_HYBRID: "🔀"}
MODE_HELP = {
    STORAGE_LOCAL: "Search documents stored on this machine (ChromaDB).",
    STORAGE_DATABRICKS: "Search documents stored in Databricks (Delta table + Vector Search).",
    MODE_HYBRID: "Search both sources together. Each citation shows where it came from.",
}
UPLOAD_BOTH = "both"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")

THEME_LIGHT = "light"
THEME_DARK = "dark"
THEME_LABELS = {THEME_LIGHT: "☀️ Light", THEME_DARK: "🌙 Dark"}


# ----------------------------------------------------------------------------
# Theme system
# ----------------------------------------------------------------------------
# Every colour used by the UI comes from a CSS variable defined here, so the
# whole interface (including Streamlit's native widgets) re-skins instantly.
_LIGHT_TOKENS: Dict[str, str] = {
    "color-scheme": "light",
    "bg": "#FFFFFF",
    "bg-glow": "rgba(109,40,217,.055)",
    "surface": "#FFFFFF",
    "surface-soft": "#FAFAFB",
    "surface-input": "#FFFFFF",
    "sidebar-top": "#FCFBFF",
    "sidebar-bottom": "#F8F6FC",
    "sidebar-border": "#E9E4EF",
    "header-from": "rgba(255,255,255,.98)",
    "header-to": "rgba(248,245,255,.96)",
    "header-border": "#E9E2F5",
    "empty-from": "#FFFFFF",
    "empty-to": "#F9F6FF",
    "empty-border": "#E8E0F2",
    "cite-from": "#FBF9FE",
    "cite-to": "#FFFFFF",
    "cite-border": "#E3D9F0",
    "cite-meta": "#4C1D75",
    "text-primary": "#18181B",
    "text-secondary": "#52525B",
    "text-muted": "#71717A",
    "border": "#E4E4E7",
    "border-soft": "#F1F1F3",
    "btn-border": "#DDD6E5",
    "dropzone-bg": "rgba(255,255,255,.75)",
    "dropzone-border": "#CFC2DF",
    "primary": "#6D28D9",
    "primary-text": "#6D28D9",
    "primary-dark": "#5521B5",
    "primary-soft": "#F3EEFF",
    "primary-border": "#DDD3F3",
    "success": "#15803D",
    "success-soft": "#F0FDF4",
    "success-border": "#CDEBD5",
    "warning": "#B45309",
    "warning-soft": "#FFFBEB",
    "warning-border": "#F3DFB0",
    "danger": "#B91C1C",
    "danger-soft": "#FEF2F2",
    "danger-border": "#F5C9C9",
    "info": "#1D4ED8",
    "info-soft": "#EFF6FF",
    "info-border": "#CFE0FB",
    "chip-bg": "#F4F4F5",
    "chip-text": "#3F3F46",
    "code-bg": "#F6F6F8",
    "shadow-sm": "0 1px 2px rgba(24,24,27,.04)",
    "shadow-md": "0 8px 24px rgba(24,24,27,.07)",
    "shadow-card": "0 3px 12px rgba(24,24,27,.035)",
    "shadow-primary": "0 8px 20px rgba(109,40,217,.18)",
}

_DARK_TOKENS: Dict[str, str] = {
    "color-scheme": "dark",
    "bg": "#0C0C14",
    "bg-glow": "rgba(139,92,246,.17)",
    "surface": "#151520",
    "surface-soft": "#121219",
    "surface-input": "#1A1A27",
    "sidebar-top": "#11111B",
    "sidebar-bottom": "#0D0D15",
    "sidebar-border": "#23233A",
    "header-from": "#171725",
    "header-to": "#1B1733",
    "header-border": "#2E2951",
    "empty-from": "#161622",
    "empty-to": "#1A1632",
    "empty-border": "#2B2748",
    "cite-from": "#191927",
    "cite-to": "#141420",
    "cite-border": "#2D2A46",
    "cite-meta": "#DDD6FE",
    "text-primary": "#F4F4F8",
    "text-secondary": "#B0B0C3",
    "text-muted": "#8A8AA0",
    "border": "#2A2A3D",
    "border-soft": "#222234",
    "btn-border": "#37374F",
    "dropzone-bg": "rgba(255,255,255,.03)",
    "dropzone-border": "#3B3659",
    "primary": "#8B5CF6",
    "primary-text": "#A78BFA",
    "primary-dark": "#C4B5FD",
    "primary-soft": "rgba(139,92,246,.15)",
    "primary-border": "rgba(139,92,246,.38)",
    "success": "#4ADE80",
    "success-soft": "rgba(34,197,94,.10)",
    "success-border": "rgba(34,197,94,.30)",
    "warning": "#FBBF24",
    "warning-soft": "rgba(245,158,11,.10)",
    "warning-border": "rgba(245,158,11,.32)",
    "danger": "#F87171",
    "danger-soft": "rgba(239,68,68,.10)",
    "danger-border": "rgba(239,68,68,.32)",
    "info": "#60A5FA",
    "info-soft": "rgba(59,130,246,.13)",
    "info-border": "rgba(59,130,246,.32)",
    "chip-bg": "#232335",
    "chip-text": "#CACADB",
    "code-bg": "#0F0F18",
    "shadow-sm": "0 1px 2px rgba(0,0,0,.35)",
    "shadow-md": "0 10px 30px rgba(0,0,0,.40)",
    "shadow-card": "0 3px 14px rgba(0,0,0,.30)",
    "shadow-primary": "0 8px 22px rgba(139,92,246,.28)",
}


def _resolve_theme() -> str:
    """Current theme: session value, else ?theme= query param, else light."""
    theme = st.session_state.get("theme")
    if theme not in THEME_LABELS:
        try:
            theme = st.query_params.get("theme")
        except Exception:  # noqa: BLE001
            theme = None
    return theme if theme in THEME_LABELS else THEME_LIGHT


def _on_theme_change() -> None:
    try:
        st.query_params["theme"] = st.session_state.theme
    except Exception:  # noqa: BLE001
        pass


def _theme_variables(theme: str) -> str:
    tokens = _DARK_TOKENS if theme == THEME_DARK else _LIGHT_TOKENS
    body = "".join(
        f"{'' if k == 'color-scheme' else '--'}{k}:{v};" for k, v in tokens.items()
    )
    return f":root{{{body}}}"


_BASE_CSS = """
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
.stApp { background: radial-gradient(circle at 75% 0%, var(--bg-glow), transparent 28rem), var(--bg); color: var(--text-primary); }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] button, [data-testid="stToolbar"] svg { color: var(--text-secondary); }
.block-container { max-width: 1260px; padding-top: 2rem; padding-bottom: 2rem; }
footer { visibility: hidden; }
hr { border-color: var(--border) !important; }

/* ---- Typography for native Streamlit content ---- */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4, [data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label, label, .stRadio label p { color: var(--text-primary); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--text-muted); }
[data-testid="stMarkdownContainer"] a { color: var(--primary-text); }
[data-testid="stMarkdownContainer"] code { background: var(--code-bg); color: var(--primary-dark); border-radius: 6px; }
[data-testid="stCode"] pre, pre { background: var(--code-bg) !important; border: 1px solid var(--border); border-radius: 10px; }
[data-testid="stCode"] code, pre code { color: var(--text-primary) !important; background: transparent !important; }

/* ---- Keyboard focus ---- */
button:focus-visible, [role="radio"]:focus-visible, [role="slider"]:focus-visible, a:focus-visible,
[data-baseweb="select"]:focus-within, [data-baseweb="input"]:focus-within {
    outline: 2px solid var(--primary); outline-offset: 2px; }

/* ---- App header ---- */
.app-header-shell { display:flex; align-items:center; justify-content:space-between; gap:1.5rem;
    padding:1.45rem 1.6rem; margin-bottom:1.25rem; border:1px solid var(--header-border); border-radius:20px;
    background:linear-gradient(135deg,var(--header-from),var(--header-to)); box-shadow:var(--shadow-md); }
.app-header-content { display:flex; align-items:center; gap:1rem; }
.app-logo { width:52px; height:52px; flex:none; display:flex; align-items:center; justify-content:center;
    border-radius:16px; color:#FFF; font-size:1.45rem; background:linear-gradient(145deg,#8B5CF6,#5B21B6);
    box-shadow:var(--shadow-primary); }
.app-eyebrow { margin-bottom:.2rem; color:var(--primary-text); font-size:.78rem; font-weight:700; letter-spacing:.02em; }
.app-title { margin:0; color:var(--text-primary); font-size:clamp(1.3rem,2.4vw,1.8rem); font-weight:750;
    letter-spacing:-.035em; line-height:1.2; }
.app-subtitle { max-width:640px; margin:.45rem 0 0; color:var(--text-secondary); font-size:.92rem; line-height:1.55; }
.header-pills { display:flex; flex-direction:column; align-items:flex-end; gap:.45rem; }
.secure-indicator, .source-indicator { display:inline-flex; align-items:center; gap:.45rem; padding:.45rem .8rem;
    border-radius:999px; font-size:.78rem; font-weight:650; white-space:nowrap; border:1px solid; }
.secure-indicator { border-color:var(--success-border); color:var(--success); background:var(--success-soft); }
.source-indicator { border-color:var(--primary-border); color:var(--primary-dark); background:var(--primary-soft); }
.secure-dot { width:7px; height:7px; border-radius:999px; background:#22C55E; box-shadow:0 0 0 4px rgba(34,197,94,.16); }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { width:350px !important;
    background:linear-gradient(180deg,var(--sidebar-top) 0%,var(--sidebar-bottom) 100%);
    border-right:1px solid var(--sidebar-border); }
section[data-testid="stSidebar"] > div { background: transparent; }
.sidebar-brand { display:flex; align-items:center; gap:.65rem; padding:.2rem 0 .9rem; }
.sidebar-brand .mark { width:34px; height:34px; display:flex; align-items:center; justify-content:center;
    border-radius:10px; color:#FFF; font-size:1rem; background:linear-gradient(145deg,#8B5CF6,#5B21B6); box-shadow:var(--shadow-primary); }
.sidebar-brand .name { color:var(--text-primary); font-weight:750; font-size:1rem; letter-spacing:-.02em; line-height:1.1; }
.sidebar-brand .tag { color:var(--text-muted); font-size:.72rem; }
.sidebar-section-label { margin:.4rem 0 .35rem; color:var(--text-primary); font-size:.84rem; font-weight:700; }
.sidebar-help { margin-bottom:.8rem; color:var(--text-secondary); font-size:.77rem; line-height:1.55; }
.sidebar-help b { color:var(--text-primary); }
.conn-row { display:flex; align-items:flex-start; gap:.55rem; padding:.55rem .7rem; margin:.35rem 0;
    border:1px solid; border-radius:10px; font-size:.76rem; line-height:1.4; }
.conn-row.ok { border-color:var(--success-border); background:var(--success-soft); }
.conn-row.down { border-color:var(--danger-border); background:var(--danger-soft); }
.conn-title { color:var(--text-primary); font-weight:700; }
.conn-detail { color:var(--text-secondary); word-break:break-word; }

/* ---- Inputs and widgets ---- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
    background-color: var(--surface-input) !important; border-color: var(--border) !important; color: var(--text-primary) !important; }
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea, [data-baseweb="select"] input,
[data-baseweb="select"] span, [data-baseweb="select"] div { color: var(--text-primary); }
[data-baseweb="select"] svg, [data-baseweb="input"] svg { fill: var(--text-secondary); }
[data-baseweb="popover"] > div, [data-baseweb="menu"], [data-baseweb="popover"] ul {
    background: var(--surface) !important; border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow-md); }
[data-baseweb="popover"] li, [data-baseweb="menu"] li { color: var(--text-primary); background: transparent; }
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover, [data-baseweb="menu"] li[aria-selected="true"] {
    background: var(--primary-soft) !important; }
[data-baseweb="tag"] { background: var(--primary-soft) !important; border: 1px solid var(--primary-border); border-radius: 8px; }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: var(--primary-dark) !important; fill: var(--primary-dark) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--primary) !important; border-color: var(--primary) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--text-secondary) !important; }
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child { border-color: var(--primary) !important; }
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed var(--dropzone-border); border-radius:14px; background:var(--dropzone-bg); }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploaderFile"] * { color: var(--text-secondary); }
[data-testid="stFileUploaderDropzone"] small { color: var(--text-muted); }

/* ---- Buttons ---- */
div.stButton > button, div.stDownloadButton > button, [data-testid="stFileUploaderDropzone"] button {
    min-height:2.6rem; border:1px solid var(--btn-border); border-radius:10px; font-weight:600;
    background: var(--surface); color: var(--text-primary); box-shadow:var(--shadow-sm);
    transition: border-color .15s ease, background-color .15s ease, transform .15s ease; }
div.stButton > button:hover, div.stDownloadButton > button:hover, [data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--primary); color: var(--primary-text); background: var(--primary-soft); }
div.stButton > button:active { transform: translateY(1px); }
div.stButton > button:disabled { opacity:.5; }
div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] {
    border-color:var(--primary); color:#FFF; background:linear-gradient(135deg,#8B5CF6,#6525C6); box-shadow:var(--shadow-primary); }
div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    color:#FFF; filter:brightness(1.08); background:linear-gradient(135deg,#8B5CF6,#6525C6); }
div.stButton > button[kind="primary"] p, div.stButton > button[data-testid="stBaseButton-primary"] p { color:#FFF; }
div.stButton > button p, div.stDownloadButton > button p { color: inherit; }

/* ---- Expanders, status, alerts, metrics ---- */
[data-testid="stExpander"] details { background: var(--surface); border:1px solid var(--border); border-radius:12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color: var(--text-primary); }
[data-testid="stExpander"] summary svg { color: var(--text-secondary); fill: var(--text-secondary); }
[data-testid="stStatus"] details, [data-testid="stStatusWidget"] { background: var(--surface); border:1px solid var(--border); border-radius:12px; }
[data-testid="stAlert"] { background: var(--surface-soft); border:1px solid var(--border); border-radius:12px; }
[data-testid="stAlert"] * { color: var(--text-primary); }
[data-testid="stMetric"] { background: var(--surface-soft); border:1px solid var(--border-soft); border-radius:10px; padding:.5rem .7rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: var(--text-muted) !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
[data-testid="stSpinner"] * { color: var(--text-secondary); }

/* ---- Documents / status cards ---- */
.document-status { display:flex; gap:.7rem; padding:.8rem; border:1px solid; border-radius:12px; margin:.75rem 0 .9rem; }
.document-status.ready { border-color:var(--success-border); background:var(--success-soft); }
.document-status.empty { border-color:var(--warning-border); background:var(--warning-soft); }
.document-status.ready > div:first-child { color:var(--success); font-weight:800; }
.document-status.empty > div:first-child { color:var(--warning); font-weight:800; }
.document-status-title { color:var(--text-primary); font-size:.79rem; font-weight:700; }
.document-status-description { color:var(--text-secondary); font-size:.72rem; line-height:1.4; }

/* ---- Empty state ---- */
.empty-state { max-width:820px; margin:2rem auto 1.2rem; padding:2.2rem 2rem; border:1px solid var(--empty-border);
    border-radius:22px; text-align:center; background:linear-gradient(145deg,var(--empty-from),var(--empty-to)); box-shadow:var(--shadow-md); }
.empty-state h2 { margin:0; color:var(--text-primary); font-size:1.4rem; font-weight:750; letter-spacing:-.02em; }
.empty-state p { max-width:600px; margin:.7rem auto 0; color:var(--text-secondary); font-size:.91rem; line-height:1.65; }
.empty-state p b { color:var(--text-primary); }
.feature-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin-top:1.5rem; text-align:left; }
.feature { padding:.9rem 1rem; border:1px solid var(--border); border-radius:14px; background:var(--surface); }
.feature .ico { font-size:1.15rem; }
.feature .ttl { margin-top:.35rem; color:var(--text-primary); font-size:.84rem; font-weight:700; }
.feature .dsc { margin-top:.15rem; color:var(--text-secondary); font-size:.76rem; line-height:1.5; }

/* ---- Chat ---- */
[data-testid="stChatMessage"] { max-width:950px; padding:1.05rem 1.15rem; margin:0 auto .85rem;
    border:1px solid var(--border-soft); border-radius:16px; background:var(--surface); box-shadow:var(--shadow-card); }
[data-testid="stChatMessage"] * { color: var(--text-primary); }
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] * { color: var(--text-muted); }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stChatInput"] { max-width:950px; margin-left:auto; margin-right:auto; }
[data-testid="stChatInput"] > div { background: var(--surface-input) !important; border:1px solid var(--border) !important;
    border-radius:14px !important; box-shadow: var(--shadow-sm); }
[data-testid="stChatInput"] > div:focus-within { border-color: var(--primary) !important; }
[data-testid="stChatInput"] textarea { color: var(--text-primary) !important; background: transparent !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stChatInput"] button { background: var(--primary) !important; border-radius: 10px; }
[data-testid="stChatInput"] button svg { fill:#FFF !important; color:#FFF !important; }

/* ---- Citations ---- */
.citation-card { position:relative; padding:.85rem 1rem .85rem 1.2rem; margin-bottom:.65rem;
    border:1px solid var(--cite-border); border-radius:12px; background:linear-gradient(135deg,var(--cite-from),var(--cite-to)); }
.citation-card::before { content:""; position:absolute; top:0; bottom:0; left:0; width:4px;
    border-radius:12px 0 0 12px; background:linear-gradient(180deg,#8B5CF6,#C4B5FD); }
.cite-label { display:inline-block; padding:.05rem .45rem; margin-right:.4rem; border-radius:6px;
    background:var(--primary-soft); color:var(--primary-dark); font-weight:700; font-size:.74rem; }
.cite-storage { display:inline-block; padding:.05rem .45rem; margin-left:.4rem; border-radius:6px;
    font-weight:650; font-size:.7rem; border:1px solid transparent; }
.cite-storage.local { background:var(--chip-bg); color:var(--chip-text); border-color:var(--border); }
.cite-storage.databricks { background:var(--info-soft); color:var(--info); border-color:var(--info-border); }
.cite-meta { color:var(--cite-meta); font-size:.79rem; font-weight:700; }
.cite-section { color:var(--text-muted); font-size:.74rem; margin-top:.15rem; }
.cite-preview { margin-top:.35rem; color:var(--text-secondary); font-size:.78rem; line-height:1.55; }
.cite-scores { margin-top:.3rem; color:var(--text-muted); font-size:.7rem; }
.badge { display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.72rem; font-weight:650; margin-right:.35rem; border:1px solid; }
.badge.ok { color:var(--success); background:var(--success-soft); border-color:var(--success-border); }
.badge.warn { color:var(--warning); background:var(--warning-soft); border-color:var(--warning-border); }
.badge.source { color:var(--primary-dark); background:var(--primary-soft); border-color:var(--primary-border); }


/* ---- Dark-mode fixes: buttons and expanders (override Streamlit's own light styles) ---- */
.stApp button[data-testid^="stBaseButton"]:not([data-testid="stBaseButton-primary"]),
.stApp [data-testid="stFileUploaderDropzone"] button {
    background: var(--surface) !important; color: var(--text-primary) !important;
    border: 1px solid var(--btn-border) !important; }
.stApp button[data-testid^="stBaseButton"]:not([data-testid="stBaseButton-primary"]):hover,
.stApp [data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--primary-soft) !important; border-color: var(--primary) !important; color: var(--primary-text) !important; }
.stApp button[data-testid^="stBaseButton"]:not([data-testid="stBaseButton-primary"]) *,
.stApp [data-testid="stFileUploaderDropzone"] button * { color: inherit !important; }
.stApp button[data-testid^="stBaseButton"]:disabled { opacity: .45; }
.stApp button[data-testid="stBaseButton-primary"] { background: linear-gradient(135deg,#8B5CF6,#6525C6) !important;
    border-color: var(--primary) !important; color: #FFF !important; }
.stApp button[data-testid="stBaseButton-primary"] * { color: #FFF !important; }

[data-testid="stExpander"], [data-testid="stExpander"] > details {
    background: var(--surface) !important; border-color: var(--border) !important; }
[data-testid="stExpander"] > details { border:1px solid var(--border); border-radius:12px; overflow:hidden; }
[data-testid="stExpander"] summary { background: var(--surface) !important; }
[data-testid="stExpander"] summary:hover { background: var(--primary-soft) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--text-primary) !important; }
[data-testid="stExpander"] summary svg { fill: var(--text-secondary) !important; color: var(--text-secondary) !important; }
[data-testid="stExpanderDetails"] { background: transparent !important; }
[data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"],
[data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"] * { color: var(--text-secondary) !important; }
[data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"] strong { color: var(--text-primary) !important; }

/* ---- Footer ---- */
.app-footer { max-width:950px; margin:2.5rem auto 0; padding:1rem 1.1rem; border:1px solid var(--border);
    border-radius:12px; color:var(--text-muted); background:var(--surface-soft); font-size:.74rem; line-height:1.55; }
.app-credit { max-width:950px; margin:.9rem auto 0; text-align:center; color:var(--text-muted); font-size:.78rem; }
.app-credit strong { color:var(--primary-text); font-weight:700; }
.app-footer strong { color:var(--text-secondary); }

/* ---- Scrollbars ---- */
* { scrollbar-width: thin; scrollbar-color: var(--border) transparent; }

@media (max-width: 900px) {
    .app-header-shell { flex-direction:column; align-items:flex-start; }
    .header-pills { align-items:flex-start; }
    .feature-grid { grid-template-columns:1fr; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
"""


def _inject_custom_css() -> None:
    theme = _resolve_theme()
    st.markdown(f"<style>{_theme_variables(theme)}{_BASE_CSS}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Pipeline (one instance per server process: models and DB clients are reused)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def _pipeline() -> RAGPipeline:
    return st.session_state.rag_pipeline


def _render_startup_failure(exc: Exception) -> None:
    logger.exception("Application startup failed")
    st.markdown("## ⚠️ The assistant could not start")
    st.write("There is a configuration issue that must be fixed before the application can run.")
    st.error(str(exc))
    st.caption("Check your .env file against .env.example, then restart the app.")
    st.stop()


def _friendly_error(user_message: str, exc: Exception) -> None:
    logger.exception(user_message)
    st.error(f"⚠️ {user_message}")
    with st.expander("Technical details for support"):
        st.code(f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------
# Token usage helpers
# ----------------------------------------------------------------------------
_TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens", "thinking_tokens", "cached_tokens")


def _default_token_usage() -> Dict[str, Any]:
    return {**{k: 0 for k in _TOKEN_KEYS}, "model": ""}


def _safe_token_value(usage: Optional[Dict[str, Any]], key: str) -> int:
    try:
        return int((usage or {}).get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _reset_token_usage() -> None:
    st.session_state.last_token_usage = _default_token_usage()
    for key in _TOKEN_KEYS:
        st.session_state[f"session_{key}"] = 0


def _update_token_usage(usage_metadata: Optional[Dict[str, Any]]) -> None:
    usage = usage_metadata or {}
    values = {k: _safe_token_value(usage, k) for k in _TOKEN_KEYS}
    if values["total_tokens"] == 0:
        values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
    st.session_state.last_token_usage = {**values, "model": usage.get("model", "")}
    for key, value in values.items():
        st.session_state[f"session_{key}"] += value


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def _init_session_state() -> None:
    if "rag_pipeline" not in st.session_state:
        with st.spinner("Starting up the QueryNexus AI..."):
            try:
                st.session_state.rag_pipeline = _get_pipeline()
            except Exception as exc:  # noqa: BLE001
                _render_startup_failure(exc)
    defaults = {
        "chat_history": [],
        "pending_question": None,
        "last_token_usage": _default_token_usage(),
        "document_filter": [],
        "top_k": settings.retrieval_top_k,
        "last_ingestion": None,
        "uploader_key": 0,
        "source_mode": _pipeline().default_mode,
        "upload_target": UPLOAD_BOTH,
        "confirm_clear": False,
        "theme": _resolve_theme(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.source_mode not in SOURCE_MODES:
        st.session_state.source_mode = _pipeline().default_mode
    for key in _TOKEN_KEYS:
        st.session_state.setdefault(f"session_{key}", 0)


def _mode() -> str:
    return st.session_state.source_mode


def _on_mode_change() -> None:
    st.session_state.document_filter = []
    st.session_state.confirm_clear = False
    st.session_state.last_ingestion = None


def _statuses_for_mode(mode: str) -> dict[str, StorageStatus]:
    pipeline = _pipeline()
    statuses: dict[str, StorageStatus] = {}
    for name in pipeline.storages_for_mode(mode):
        if name == STORAGE_DATABRICKS and not pipeline.storage_checked(name):
            with st.spinner("Connecting to Databricks…"):
                statuses[name] = pipeline.storage_status(name)
        else:
            statuses[name] = pipeline.storage_status(name)
    return statuses


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
def _render_header() -> None:
    mode = _mode()
    st.html(
        f"""
<div class="app-header-shell">
    <div class="app-header-content">
        <div class="app-logo">✦</div>
        <div>
            <div class="app-eyebrow">QueryNexus AI</div>
            <h1 class="app-title">Document Intelligence Assistant</h1>
            <p class="app-subtitle">Ask questions about your documents and get answers grounded in,
            and cited to, the exact document page and section.</p>
        </div>
    </div>
    <div class="header-pills">
        <div class="source-indicator">{MODE_ICONS[mode]} Source: {html.escape(MODE_LABELS[mode])}</div>
        <div class="secure-indicator"><span class="secure-dot"></span>Grounded answers only</div>
    </div>
</div>
"""
    )


# ----------------------------------------------------------------------------
# Sidebar: brand, appearance, knowledge source
# ----------------------------------------------------------------------------
def _render_sidebar_brand() -> None:
    st.markdown(
        '<div class="sidebar-brand"><div class="mark">✦</div><div>'
        '<div class="name">QueryNexus AI</div><div class="tag">Document intelligence</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def _render_theme_switch() -> None:
    st.markdown('<div class="sidebar-section-label">Appearance</div>', unsafe_allow_html=True)
    st.radio(
        "Theme",
        options=list(THEME_LABELS),
        format_func=lambda t: THEME_LABELS[t],
        key="theme",
        horizontal=True,
        on_change=_on_theme_change,
        label_visibility="collapsed",
    )


def _render_source_selector() -> dict[str, StorageStatus]:
    st.markdown('<div class="sidebar-section-label">Knowledge source</div>', unsafe_allow_html=True)
    st.radio(
        "Knowledge source",
        options=list(SOURCE_MODES),
        format_func=lambda m: f"{MODE_ICONS[m]} {MODE_LABELS[m]}",
        key="source_mode",
        on_change=_on_mode_change,
        label_visibility="collapsed",
    )
    st.caption(MODE_HELP[_mode()])

    statuses = _statuses_for_mode(_mode())
    for name, status in statuses.items():
        if status.available:
            detail = f"Connected · {status.chunk_count:,} chunks"
            css, icon = "ok", "🟢"
        else:
            detail = f"Not connected · {status.error}"
            css, icon = "down", "🔴"
        st.markdown(
            f'<div class="conn-row {css}"><div>{icon}</div><div><div class="conn-title">'
            f'{html.escape(status.label)}</div><div class="conn-detail">{html.escape(detail)}</div></div></div>',
            unsafe_allow_html=True,
        )
        if not status.available and name == STORAGE_DATABRICKS:
            if st.button("Retry Databricks connection", key="retry_databricks", use_container_width=True):
                _pipeline().reset_storage(STORAGE_DATABRICKS)
                st.rerun()
    if _mode() == MODE_HYBRID:
        down = [s.label for s in statuses.values() if not s.available]
        if down and len(down) < len(statuses):
            st.caption(f"⚠️ Hybrid is running on one source only until {', '.join(down)} is connected.")
    return statuses


# ----------------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------------
def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "").strip()
    cleaned = _SAFE_NAME_RE.sub("_", base).strip(" .") or "document"
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:150]


def _upload_targets() -> list[str]:
    mode = _mode()
    if mode != MODE_HYBRID:
        return [mode]
    choice = st.session_state.upload_target
    return list(STORAGES) if choice == UPLOAD_BOTH else [choice]


def _handle_ingestion(uploaded_files, targets: list[str]) -> None:
    if not uploaded_files:
        st.warning("Please select at least one PDF document.")
        return
    os.makedirs(settings.pdf_upload_dir, exist_ok=True)
    saved_paths: list[str] = []
    rejected: list[str] = []
    for uploaded_file in uploaded_files:
        name = _safe_filename(uploaded_file.name)
        if uploaded_file.size > settings.max_upload_bytes:
            rejected.append(f"{name}: larger than {settings.max_upload_mb} MB")
            continue
        path = os.path.join(settings.pdf_upload_dir, name)
        try:
            with open(path, "wb") as handle:
                handle.write(uploaded_file.getbuffer())
            saved_paths.append(path)
        except OSError as exc:
            _friendly_error(f"Could not save {name}.", exc)
            return

    report = None
    if saved_paths:
        target_text = " + ".join(STORAGE_LABELS[t] for t in targets)
        with st.status(f"Processing {len(saved_paths)} document(s) into {target_text}…", expanded=True) as status:
            st.write("Validating PDFs and extracting text…")
            if STORAGE_DATABRICKS in targets:
                st.write("Writing to Databricks and syncing the Vector Search index (this can take a few minutes)…")
            try:
                report = _pipeline().ingest_with_report(saved_paths, targets=targets)
            except Exception as exc:  # noqa: BLE001
                status.update(label="Processing failed", state="error")
                _friendly_error("Something went wrong while processing your documents.", exc)
                return
            failed = len(report.failed)
            status.update(
                label=f"Done: {report.chunks_added:,} new chunks" + (f", {failed} item(s) failed" if failed else ""),
                state="error" if failed == len(report.files) else "complete",
                expanded=False,
            )
    st.session_state.last_ingestion = {
        "files": [vars(f) for f in report.files] if report else [],
        "rejected": rejected,
    }
    st.session_state.uploader_key += 1
    st.rerun()


def _render_ingestion_results() -> None:
    result = st.session_state.last_ingestion
    if not result:
        return
    icons = {"indexed": "✅", "updated": "🔄", "unchanged": "⏭️", "failed": "❌"}
    with st.expander("Last upload results", expanded=True):
        for item in result["files"]:
            detail = f" · {item['chunks_added']} chunks" if item["chunks_added"] else ""
            where = f" → {STORAGE_LABELS.get(item.get('storage', ''), item.get('storage', ''))}" if item.get("storage") else ""
            message = f" — {item['message']}" if item["message"] else ""
            st.caption(
                f"{icons.get(item['status'], '•')} **{item['source_file']}**{where}: {item['status']}{detail}{message}"
            )
        for item in result["rejected"]:
            st.caption(f"❌ {item}")


def _render_upload_section(statuses: dict[str, StorageStatus]) -> None:
    st.markdown('<div class="sidebar-section-label">Upload documents</div>', unsafe_allow_html=True)
    if _mode() == MODE_HYBRID:
        st.radio(
            "Upload to",
            options=[STORAGE_LOCAL, STORAGE_DATABRICKS, UPLOAD_BOTH],
            format_func=lambda t: "Both" if t == UPLOAD_BOTH else STORAGE_LABELS[t],
            key="upload_target",
            horizontal=True,
        )
    targets = _upload_targets()
    unavailable = [STORAGE_LABELS[t] for t in targets if t in statuses and not statuses[t].available]
    st.markdown(
        '<div class="sidebar-help">Uploads go to <b>'
        + html.escape(" + ".join(STORAGE_LABELS[t] for t in targets))
        + f"</b> (max {settings.max_upload_mb} MB per PDF). An unchanged file is skipped; a changed file "
        "with the same name replaces the old version.</div>",
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if unavailable:
        st.caption(f"⚠️ Cannot upload: {', '.join(unavailable)} is not connected.")
    if uploaded_files and st.button(
        "Process documents", type="primary", use_container_width=True, disabled=bool(unavailable)
    ):
        _handle_ingestion(uploaded_files, targets)
    _render_ingestion_results()


# ----------------------------------------------------------------------------
# Documents, settings, actions
# ----------------------------------------------------------------------------
def _render_document_manager(documents) -> None:
    show_storage = _mode() == MODE_HYBRID
    with st.expander(f"Indexed documents ({len(documents)})", expanded=False):
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            where = f" · {STORAGE_LABELS.get(doc.storage, doc.storage)}" if show_storage else ""
            col1.caption(f"**{doc.source_file}**  \n{doc.page_count} pages · {doc.chunk_count} chunks{where}")
            if col2.button("🗑️", key=f"del_{doc.storage}_{doc.document_id}", help=f"Remove {doc.source_file}"):
                try:
                    _pipeline().delete_document(doc.document_id, storage=doc.storage)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    _friendly_error(f"Could not remove {doc.source_file}.", exc)


def _render_knowledge_base(statuses: dict[str, StorageStatus]) -> list:
    try:
        documents = _pipeline().list_documents(_mode()) if any(s.available for s in statuses.values()) else []
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not read document list")
        documents = []
        st.error(f"Could not read the knowledge base ({type(exc).__name__}).")
    chunk_count = sum(s.chunk_count for s in statuses.values() if s.available)
    st.session_state.documents_ingested = chunk_count > 0
    if chunk_count:
        st.markdown(
            f'<div class="document-status ready"><div>✓</div><div><div class="document-status-title">'
            f'Knowledge base ready</div><div class="document-status-description">{len(documents)} document(s), '
            f"{chunk_count:,} chunks in {html.escape(MODE_LABELS[_mode()])}.</div></div></div>",
            unsafe_allow_html=True,
        )
        _render_document_manager(documents)
    else:
        st.markdown(
            '<div class="document-status empty"><div>○</div><div><div class="document-status-title">'
            f'No documents in {html.escape(MODE_LABELS[_mode()])}</div><div class="document-status-description">'
            "Upload a PDF or switch the knowledge source to begin asking questions.</div></div></div>",
            unsafe_allow_html=True,
        )
    return documents


def _render_search_settings(documents) -> None:
    st.markdown('<div class="sidebar-section-label">Search settings</div>', unsafe_allow_html=True)
    names = sorted({d.source_file for d in documents})
    st.session_state.document_filter = [f for f in st.session_state.document_filter if f in names]
    st.multiselect(
        "Search only these documents",
        options=names,
        key="document_filter",
        placeholder="All documents",
        help="Leave empty to search every document in the selected source.",
    )
    st.slider("Sources per answer (Top-K)", 1, 15, key="top_k")


def _render_workspace_actions(statuses: dict[str, StorageStatus]) -> None:
    st.markdown('<div class="sidebar-section-label">Workspace actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("New chat", use_container_width=True, help="Start over without deleting documents."):
        st.session_state.chat_history = []
        st.session_state.pending_question = None
        _reset_token_usage()
        st.rerun()
    clearable = [name for name, s in statuses.items() if s.available and s.chunk_count]
    if col2.button(
        "Clear docs",
        use_container_width=True,
        disabled=not clearable,
        help="Delete all indexed content in the selected knowledge source.",
    ):
        st.session_state.confirm_clear = True
    if st.session_state.confirm_clear and clearable:
        labels = " + ".join(STORAGE_LABELS[n] for n in clearable)
        st.warning(f"Delete ALL documents in {labels}? This cannot be undone.")
        c1, c2 = st.columns(2)
        if c1.button("Yes, delete", type="primary", use_container_width=True, key="confirm_clear_yes"):
            try:
                for name in clearable:
                    _pipeline().clear_storage(name)
                st.session_state.chat_history = []
                st.session_state.last_ingestion = None
                _reset_token_usage()
            except Exception as exc:  # noqa: BLE001
                _friendly_error("Could not clear the documents.", exc)
            st.session_state.confirm_clear = False
            st.rerun()
        if c2.button("Cancel", use_container_width=True, key="confirm_clear_no"):
            st.session_state.confirm_clear = False
            st.rerun()

    if st.session_state.chat_history:
        st.download_button(
            "Export conversation",
            data=_transcript(),
            file_name=f"querynexus_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _render_sidebar_token_usage() -> None:
    usage = st.session_state.get("last_token_usage", {})
    with st.expander("Token usage", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("Last input", f"{_safe_token_value(usage, 'input_tokens'):,}")
        c2.metric("Last output", f"{_safe_token_value(usage, 'output_tokens'):,}")
        c3, c4 = st.columns(2)
        c3.metric("Last request", f"{_safe_token_value(usage, 'total_tokens'):,}")
        c4.metric("Session total", f"{int(st.session_state.get('session_total_tokens', 0)):,}")
        model = usage.get("model") or getattr(_pipeline(), "_chat_model_label", "Not available")
        st.caption(f"**Model:** {model}")
        st.caption("Session total is usage in this chat, not your remaining API quota.")


def _render_sidebar() -> None:
    with st.sidebar:
        _render_sidebar_brand()
        _render_theme_switch()
        st.divider()
        statuses = _render_source_selector()
        st.divider()
        _render_upload_section(statuses)
        documents = _render_knowledge_base(statuses)
        st.divider()
        _render_search_settings(documents)
        st.divider()
        _render_workspace_actions(statuses)
        with st.expander("System information"):
            st.caption(
                f"**Default source:** {MODE_LABELS[_pipeline().default_mode]}  \n"
                f"**Chat provider:** {settings.resolved_chat_provider}  \n"
                f"**Embedding provider:** {settings.embedding_provider}  \n"
                f"**Keyword + vector search:** {'on' if settings.hybrid_search_enabled else 'off'}  \n"
                f"**Reranker:** {settings.reranker}  \n"
                f"**Strict citations:** {'on' if settings.strict_citations else 'off'}"
            )
        _render_sidebar_token_usage()


def _transcript() -> str:
    lines = []
    for message in st.session_state.chat_history:
        role = "You" if message.get("role") == "user" else "Assistant"
        source = f" [{MODE_LABELS.get(message.get('mode', ''), '')}]" if message.get("mode") else ""
        lines.append(f"{role}{source}: {message.get('content', '')}")
        for cite in message.get("citations", []) or []:
            section = f", section: {cite['section_title']}" if cite.get("section_title") else ""
            storage = f" ({STORAGE_LABELS.get(cite.get('storage', ''), '')})" if cite.get("storage") else ""
            lines.append(f"    [{cite['label']}] {cite['source_file']}, page {cite['page_number']}{section}{storage}")
    return "\n\n".join(lines)


# ----------------------------------------------------------------------------
# Citations and messages
# ----------------------------------------------------------------------------
def _preview(text: str, limit: int = 450) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _render_citations(sources) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for index, chunk in enumerate(sources, start=1):
            label = html.escape(str(getattr(chunk, "citation_label", "") or f"S{index}"))
            source_file = html.escape(str(getattr(chunk, "source_file", "Unknown document")))
            page = html.escape(str(getattr(chunk, "page_number", "?")))
            storage = str(getattr(chunk, "storage", "") or "")
            storage_html = (
                f'<span class="cite-storage {html.escape(storage)}">'
                f"{MODE_ICONS.get(storage, '')} {html.escape(STORAGE_LABELS.get(storage, storage))}</span>"
                if storage
                else ""
            )
            section = str(getattr(chunk, "section_title", "") or "")
            section_html = (
                f'<div class="cite-section">Section: {html.escape(section)}</div>'
                if section and section != "Untitled Section"
                else ""
            )
            similarity = getattr(chunk, "similarity_score", None)
            rerank = getattr(chunk, "rerank_score", None)
            scores = []
            if isinstance(similarity, (int, float)) and similarity > 0:
                scores.append(f"semantic similarity {similarity:.2f}")
            if getattr(chunk, "keyword_coverage", 0):
                scores.append(f"keyword match {chunk.keyword_coverage:.0%}")
            if isinstance(rerank, (int, float)):
                scores.append(f"rerank {rerank:.2f}")
            scores_html = f'<div class="cite-scores">{" · ".join(scores)}</div>' if scores else ""
            st.markdown(
                f'<div class="citation-card"><div class="cite-meta"><span class="cite-label">{label}</span>'
                f"{source_file} · Page {page}{storage_html}</div>{section_html}"
                f'<div class="cite-preview">{html.escape(_preview(getattr(chunk, "text", "")))}</div>'
                f"{scores_html}</div>",
                unsafe_allow_html=True,
            )


def _render_answer_meta(message: Dict[str, Any]) -> None:
    badges = []
    if message.get("searched"):
        badges.append(f'<span class="badge source">Searched: {html.escape(" + ".join(message["searched"]))}</span>')
    if message.get("is_grounded"):
        badges.append('<span class="badge ok">✓ Verified citations</span>')
    elif not message.get("is_no_answer") and message.get("sources"):
        badges.append('<span class="badge warn">Review citations</span>')
    if badges:
        st.markdown("".join(badges), unsafe_allow_html=True)
    for warning in message.get("warnings", []) or []:
        st.caption(f"⚠️ {warning}")
    usage = message.get("usage_metadata") or {}
    total = _safe_token_value(usage, "total_tokens")
    if total:
        latency = message.get("latency_ms")
        st.caption(
            f"Tokens: {_safe_token_value(usage, 'input_tokens'):,} in · {_safe_token_value(usage, 'output_tokens'):,} out"
            + (f" · {latency / 1000:.1f}s" if latency else "")
        )


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        with st.chat_message(role, avatar="🤖" if role == "assistant" else None):
            st.markdown(str(message.get("content", "")))
            if role == "assistant":
                _render_citations(message.get("sources"))
                _render_answer_meta(message)


def _current_filters() -> Optional[Dict[str, Any]]:
    selected = st.session_state.get("document_filter") or []
    return {"source_file": list(selected)} if selected else None


def _answer_question(question: str) -> None:
    cleaned = question.strip()
    if not cleaned:
        return
    mode = _mode()
    st.session_state.chat_history.append({"role": "user", "content": cleaned, "mode": mode})
    with st.chat_message("user"):
        st.markdown(cleaned)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner(f"Searching {MODE_LABELS[mode]} and verifying sources…"):
            try:
                response: RAGResponse = _pipeline().ask(
                    cleaned, filters=_current_filters(), top_k=int(st.session_state.top_k), mode=mode
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("RAG pipeline failed to answer")
                answer = "⚠️ Sorry, something went wrong while generating an answer. Please try again in a moment."
                st.markdown(answer)
                with st.expander("Technical details (for support)"):
                    st.code(f"{type(exc).__name__}: {exc}")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": [], "mode": mode}
                )
                return
        _update_token_usage(response.usage_metadata)
        message = {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "citations": [vars(c) for c in response.citations],
            "usage_metadata": response.usage_metadata,
            "is_grounded": response.is_grounded,
            "is_no_answer": response.is_no_answer,
            "warnings": response.warnings,
            "latency_ms": response.latency_ms,
            "mode": mode,
            "searched": response.searched,
        }
        st.markdown(response.answer)
        _render_citations(response.sources)
        _render_answer_meta(message)
    st.session_state.chat_history.append(message)


def _render_chat_input() -> None:
    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = None
        _answer_question(pending)
        st.rerun()
    user_input = st.chat_input(
        f"Ask about documents in {MODE_LABELS[_mode()]} (one question per line)",
        disabled=not st.session_state.get("documents_ingested", False),
    )
    if not user_input:
        return
    questions = [q.strip() for q in user_input.splitlines() if q.strip()]
    for question in questions[:10]:
        _answer_question(question)
    if len(questions) > 10:
        st.warning("Only the first 10 questions were answered. Please submit the rest separately.")
    st.rerun()


def _render_empty_state() -> None:
    st.html(
        """
<div class="empty-state">
    <h2>Ask questions about your documents</h2>
    <p>Choose a knowledge source in the sidebar: <b>Local docs</b>, <b>Databricks</b>, or <b>Hybrid</b> to
    search both. Every answer cites the document, page, section and source it came from. If the
    documents don't contain the answer, the assistant will say so.</p>
    <div class="feature-grid">
        <div class="feature"><div class="ico">📄</div><div class="ttl">Cited to the page</div>
            <div class="dsc">Each answer links back to the document, page and section it came from.</div></div>
        <div class="feature"><div class="ico">🔀</div><div class="ttl">One or many sources</div>
            <div class="dsc">Search local files, Databricks, or both together.</div></div>
        <div class="feature"><div class="ico">🛡️</div><div class="ttl">Grounded answers only</div>
            <div class="dsc">If your documents don't contain the answer, the assistant says so.</div></div>
    </div>
</div>
"""
    )


def _render_footer() -> None:
    st.markdown(
        f'<div class="app-footer"><strong>Important information:</strong> {html.escape(DISCLAIMER)}</div>'
        '<div class="app-credit">Developed and designed by <strong>Gulam Mustafa</strong></div>',
        unsafe_allow_html=True,
    )


def run_app() -> None:
    st.set_page_config(
        page_title="QueryNexus AI",
        page_icon="💜",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_custom_css()
    _init_session_state()
    _render_sidebar()
    _render_header()
    if not st.session_state.chat_history:
        _render_empty_state()
    _render_chat_history()
    _render_chat_input()
    _render_footer()


if __name__ == "__main__":
    run_app()






# import os
# import sys

# ROOT = os.path.dirname(os.path.abspath(__file__))
# if os.path.basename(ROOT) == "frontend":
#     ROOT = os.path.dirname(ROOT)
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# from app import run_app

# run_app()