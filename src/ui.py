"""Small shared Streamlit UI/data context helpers."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from .data import dataframe_hash, load_baseline
from .models import train_models

@st.cache_resource(show_spinner="Training models on refreshed data…")
def cached_models(_data: pd.DataFrame, data_hash: str):
    return train_models(_data)

def baseline_context():
    # Streamlit raises when no secrets.toml exists at all, even for `.get()`.
    # The API integration is optional, so absence must select the local fallback.
    try:
        key = st.secrets.get("DATA_GOV_API_KEY", None)
    except st.errors.StreamlitSecretNotFoundError:
        key = None
    baseline, source, notice = load_baseline(key)
    return baseline, source, notice, cached_models(baseline, dataframe_hash(baseline))

def page_header(title: str, source: str, notice: str | None = None) -> None:
    st.title(title)
    if notice: st.warning("Offline fallback — " + notice)
    st.caption(f"Last updated: data loaded at runtime | Source: {source}")
