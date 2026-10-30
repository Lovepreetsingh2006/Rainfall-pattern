from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from src.features import SEASONS
from src.models import predict
from src.ui import baseline_context, page_header

baseline, source, notice, models = baseline_context(); page_header("Rainfall Predictor", source, notice)
subdivision = st.selectbox("Subdivision", sorted(baseline.SUBDIVISION.unique()), index=sorted(baseline.SUBDIVISION.unique()).index("PUNJAB"))
recent = baseline[baseline.SUBDIVISION.eq(subdivision)].sort_values("YEAR").iloc[-1]
if st.button("Load example (Punjab)"):
    subdivision = "PUNJAB"; st.session_state["Jan-Feb"], st.session_state["Mar-May"], st.session_state["Jun-Sep"], st.session_state["Oct-Dec"] = 60., 90., 450., 35.
st.caption(f"Inputs start from {subdivision}'s latest complete IMD year ({int(recent.YEAR)}); override as needed.")
c1,c2,c3,c4 = st.columns(4)
values = {}
for col, season in zip([c1,c2,c3,c4], SEASONS): values[season] = col.number_input(season + " (mm)", min_value=0.0, value=float(recent[season]), key=season)
if any(v < 0 for v in values.values()): st.error("Rainfall inputs cannot be negative.")
else:
    results = predict(models, subdivision, values)
    for col, (name, value) in zip(st.columns(3), results.items()): col.metric(name, f"{value:,.1f} mm")
    chart = pd.DataFrame({"Model":list(results), "Forecast mm":list(results.values())})
    historical = baseline[baseline.SUBDIVISION.eq(subdivision)].ANNUAL.mean()
    fig = px.bar(chart, x="Model", y="Forecast mm", color="Model", title=f"Next-year rainfall forecast — {subdivision}")
    fig.add_hline(y=historical, line_dash="dash", annotation_text=f"Historical average {historical:.0f} mm")
    st.plotly_chart(fig, use_container_width=True)
