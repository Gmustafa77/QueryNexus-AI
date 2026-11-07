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
