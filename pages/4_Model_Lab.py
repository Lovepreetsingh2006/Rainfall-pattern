from __future__ import annotations
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data import dataframe_hash
from src.ui import baseline_context, page_header

baseline, source, notice, models = baseline_context()
page_header("Model Lab", source, notice)
if st.button("Retrain models on latest data"):
    from src.ui import cached_models
    cached_models.clear(); st.rerun()
st.caption(f"Training date: {datetime.now().strftime('%d %b %Y %H:%M')} | Data hash: `{dataframe_hash(baseline)}` | Training rows: {models.trained_rows}")
st.dataframe(models.metrics.style.format({"MAE":"{:.2f}","MSE":"{:.2f}","RMSE":"{:.2f}","R²":"{:.2f}"}), use_container_width=True, hide_index=True)
st.plotly_chart(px.bar(models.metrics, x="Model", y="RMSE", color="Model", title="RMSE comparison (lower is better)"), use_container_width=True)
tabs = st.tabs(["Simple LR", "Multiple LR", "KNN"])
for tab, name in zip(tabs, ["Simple LR", "Multiple LR", "KNN"]):
    actual, pred = models.predictions[name]
    maximum = max(max(actual), max(pred)); fig = px.scatter(x=actual, y=pred, labels={"x":"Actual next-year rainfall (mm)","y":"Predicted (mm)"}, title=f"{name}: actual vs predicted")
    fig.add_shape(type="line", x0=0, y0=0, x1=maximum, y1=maximum, line={"dash":"dash", "color":"black"})
    with tab: st.plotly_chart(fig, use_container_width=True)
fig = px.line(models.cv, x="K", y="CV MSE", markers=True, title="KNN cross-validation MSE")
best = models.cv.loc[models.cv["CV MSE"].idxmin()]; fig.add_scatter(x=[best.K], y=[best["CV MSE"]], mode="markers", marker={"size":14,"symbol":"star"}, name="Best K")
st.plotly_chart(fig, use_container_width=True)
best_rmse = models.metrics.RMSE.min(); mlr = models.metrics.query("Model == 'Multiple LR'").RMSE.iloc[0]
recommended = "Multiple LR" if mlr <= best_rmse * 1.02 else models.metrics.loc[models.metrics.RMSE.idxmin(), "Model"]
st.success(f"Recommendation: **{recommended}**. It is the lowest-RMSE option or is within 2% of it while remaining simpler to explain.")
