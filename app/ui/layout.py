"""Layout helpers for the Streamlit interface."""

from __future__ import annotations

import streamlit as st


def get_layout_name() -> str:
    """Return the default layout mode."""
    return "wide"


def render_header(title: str, description: str) -> None:
    """Render the main app header."""
    st.title(title)
    st.caption(description)

