# # ==============================================================================
# # frontend/streamlit_ui.py
# # ------------------------------------------------------------------------------
# # STEP 1 & 10 of the RAG pipeline: "Upload PDF files" + "Display answer with
# # citations in Streamlit"
# #
# # WHAT THIS FILE DOES:
# #   Builds the entire user-facing chat application: a sidebar for uploading
# #   insurance PDFs, and a main chat window where the user asks questions and
# #   sees grounded answers with source citations underneath each response.
# #
# # HOW STREAMLIT WORKS (GenAI / web beginners):
# #   Streamlit re-runs this ENTIRE script from top to bottom every time the
# #   user interacts with a widget (clicks a button, types a message, etc.).
# #   That means any data we want to "remember" between interactions — like
# #   chat history, or whether the RAG pipeline has already been initialized —
# #   must be stored in `st.session_state`, a dictionary-like object Streamlit
# #   preserves across re-runs for a given browser session.
# # ==============================================================================

# import os
# import sys

# import streamlit as st

# # Allow "import rag.rag_pipeline" etc. to work when this file is run
# # directly via `streamlit run frontend/streamlit_ui.py` from the project
# # root, by adding the project root to Python's module search path.
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from rag.rag_pipeline import RAGPipeline, RAGResponse  # noqa: E402
# from utils.config import settings  # noqa: E402
# from utils.logger import get_logger  # noqa: E402

# logger = get_logger(__name__)


# def _init_session_state() -> None:
#     """
#     Set up st.session_state defaults exactly once per browser session.
#     Streamlit re-runs this whole script on every interaction, so we must
#     guard each key with an "if ... not in st.session_state" check —
#     otherwise things like chat history would reset on every message.
#     """
#     if "rag_pipeline" not in st.session_state:
#         # Building the RAGPipeline connects to OpenAI + opens ChromaDB, so
#         # we only want to do this ONCE per session, not on every re-run.
#         with st.spinner("Starting up the Aetna Virtual Assistant..."):
#             st.session_state.rag_pipeline: RAGPipeline = RAGPipeline()

#     if "chat_history" not in st.session_state:
#         # Each entry: {"role": "user"|"assistant", "content": str, "sources": [...]}
#         st.session_state.chat_history = []

#     if "documents_ingested" not in st.session_state:
#         st.session_state.documents_ingested = st.session_state.rag_pipeline.retriever.has_documents()


# def _render_sidebar() -> None:
#     """
#     Renders the sidebar: PDF upload widget + ingestion status + a "clear
#     knowledge base" utility button. This is where Step 1 (Upload PDF files)
#     happens from the user's point of view.
#     """
#     with st.sidebar:
#         st.header("📄 Upload Insurance Documents")
#         st.caption(
#             "Upload your Aetna Evidence of Coverage (EOC), Summary of Benefits "
#             "and Coverage (SBC), Medicare Managed Care Manual, or other plan "
#             "documents (PDF only)."
#         )

#         uploaded_files = st.file_uploader(
#             "Choose PDF file(s)",
#             type=["pdf"],
#             accept_multiple_files=True,
#             help="You can select multiple PDFs at once.",
#         )

#         if uploaded_files and st.button("📥 Ingest Documents", type="primary"):
#             _handle_ingestion(uploaded_files)

#         st.divider()

#         # Status indicator so the user always knows whether the chatbot has
#         # anything to search yet.
#         doc_count = st.session_state.rag_pipeline.chroma_manager.document_count()
#         if doc_count > 0:
#             st.success(f"✅ Knowledge base ready — {doc_count} chunks indexed")
#         else:
#             st.warning("⚠️ No documents indexed yet")

#         if st.button("🗑️ Clear knowledge base"):
#             st.session_state.rag_pipeline.chroma_manager.clear_collection()
#             st.session_state.documents_ingested = False
#             st.session_state.chat_history = []
#             st.success("Knowledge base cleared.")
#             st.rerun()

#         st.divider()
#         st.caption(
#             f"**Chat model:** {settings.chat_model}  \n"
#             f"**Embedding model:** {settings.embedding_model}  \n"
#             f"**Top-K retrieval:** {settings.retrieval_top_k}"
#         )


# def _handle_ingestion(uploaded_files) -> None:
#     """
#     Save each Streamlit UploadedFile to disk (data/pdfs/), then run the
#     RAGPipeline's ingest_documents() to extract, chunk, embed, and store
#     them in ChromaDB. This bridges Streamlit's in-memory upload objects to
#     the file-path-based ingestion pipeline (ingestion/pdf_loader.py expects
#     real file paths on disk, since PyMuPDF reads from disk).
#     """
#     os.makedirs(settings.pdf_upload_dir, exist_ok=True)
#     saved_paths = []

#     for uploaded_file in uploaded_files:
#         save_path = os.path.join(settings.pdf_upload_dir, uploaded_file.name)
#         with open(save_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         saved_paths.append(save_path)

#     with st.spinner(f"Processing {len(saved_paths)} document(s): extracting text, chunking, and generating embeddings..."):
#         try:
#             chunk_count = st.session_state.rag_pipeline.ingest_documents(saved_paths)
#             st.session_state.documents_ingested = True
#             st.success(f"✅ Successfully indexed {chunk_count} chunks from {len(saved_paths)} document(s)!")
#         except Exception as exc:
#             logger.exception("Ingestion failed")
#             st.error(f"❌ Something went wrong while processing your documents: {exc}")


# def _render_chat_history() -> None:
#     """
#     Redraw every past message in the conversation (Streamlit re-runs the
#     script on every interaction, so without this the chat would appear to
#     "forget" everything after each new message).
#     """
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             if message["role"] == "assistant" and message.get("sources"):
#                 _render_citations(message["sources"])


# def _render_citations(sources) -> None:
#     """
#     STEP 10: Display answer with citations.

#     Renders an expandable "Sources" panel underneath an assistant's answer,
#     listing exactly which document + page each retrieved chunk came from,
#     plus a preview of the chunk text itself — so the user can verify the
#     chatbot's answer against the original document.
#     """
#     with st.expander(f"📚 Sources ({len(sources)})"):
#         for i, chunk in enumerate(sources, start=1):
#             st.markdown(
#                 f"**{i}. {chunk.source_file} — page {chunk.page_number}** "
#                 f"_(similarity distance: {chunk.similarity_score:.3f}, lower = more relevant)_"
#             )
#             preview = chunk.text[:400] + ("..." if len(chunk.text) > 400 else "")
#             st.caption(preview)


# def _render_chat_input() -> None:
#     """
#     The main chat input box at the bottom of the page. When the user submits
#     a question, this triggers Steps 6-9 via RAGPipeline.ask(), then displays
#     the grounded answer + citations (Step 10).
#     """
#     question = st.chat_input("Ask a question about your insurance plan...")
#     if not question:
#         return

#     # 1. Show the user's own message immediately and save it to history.
#     st.session_state.chat_history.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     # 2. Run the RAG pipeline (Steps 6-9) and stream the "thinking" state.
#     with st.chat_message("assistant"):
#         with st.spinner("Searching your documents and drafting an answer..."):
#             try:
#                 response: RAGResponse = st.session_state.rag_pipeline.ask(question)
#             except Exception as exc:
#                 logger.exception("RAG pipeline failed to answer")
#                 response = RAGResponse(
#                     answer=f"⚠️ Sorry, something went wrong while generating an answer: {exc}",
#                     sources=[],
#                 )

#         st.markdown(response.answer)
#         if response.sources:
#             _render_citations(response.sources)

#     # 3. Save the assistant's turn to history so it survives the next re-run.
#     st.session_state.chat_history.append(
#         {"role": "assistant", "content": response.answer, "sources": response.sources}
#     )


# def run_app() -> None:
#     """Main entry point for the Streamlit application."""
#     st.set_page_config(
#         page_title="Aetna Insurance Virtual Assistant",
#         page_icon="🏥",
#         layout="wide",
#     )

#     st.title("🏥 Aetna Insurance Virtual Assistant")
#     st.caption(
#         "Ask questions about your plan's benefits, coverage, copays, deductibles, "
#         "claims, appeals, and more — answered strictly from your uploaded documents."
#     )

#     _init_session_state()
#     _render_sidebar()

#     if not st.session_state.documents_ingested:
#         st.info(
#             "👋 Welcome! To get started, upload your insurance documents "
#             "(EOC, SBC, etc.) using the sidebar on the left, then ask me anything."
#         )

#     _render_chat_history()
#     _render_chat_input()


# # Streamlit executes the script top-to-bottom, so we simply call run_app()
# # at import/run time rather than using an `if __name__ == "__main__":` guard
# # (Streamlit doesn't execute the file as "__main__" the way plain Python does).
# run_app()




"""Legacy entry point kept for compatibility: ``streamlit run frontend/streamlit_ui.py``.

The UI now lives in a single place (app.py) so the two front-ends can no longer drift apart.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import run_app  # noqa: E402

run_app()
