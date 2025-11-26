"""
Chainlit application entry point.

Run this with:
    chainlit run chainlit_app.py
"""

from src.logger import setup_logging

# Setup logging
setup_logging()

# Import the Chainlit app (this will register the handlers)
from src import app  # noqa: E402, F401

