from __future__ import annotations
import streamlit as st
from src.data import bias_check
from src.live import load_live_archive
from src.ui import baseline_context, page_header

baseline, source, notice, _ = baseline_context()
# A one-year 2015 archive request provides the explicit source-overlap needed
# for the bias diagnostic; the normal live layer itself begins in 2016.
live, live_notice = load_live_archive("2015-01-01")
page_header("About this project", f"{source} + Open-Meteo", notice or live_notice)
st.markdown("""### Methodology
The app follows the original project: numeric nulls receive column means; monthly totals are recomputed; annual rainfall is grouped into Low, Medium and High; then next-year annual rainfall is modelled by simple linear regression, multiple linear regression (seasons plus subdivision) and KNN. The live layer is never silently included in the baseline model training.

### IMD vs Open-Meteo bias check
IMD is area-weighted gauge data; Open-Meteo is a model/satellite-informed gridded estimate at a representative centroid. Values have different measurement characteristics and must not be treated as interchangeable.""")
calibrate = st.toggle("Apply bias calibration (display only)", value=False, help="Calibration is off by default and never modifies model-training data.")
bias = bias_check(baseline, live)
if bias.empty:
    st.info("No overlap is currently available for a numerical bias check; Open-Meteo archive loading begins in 2016 while the packaged IMD baseline ends in 2015.")
else:
    if calibrate:
        bias["calibrated_open_meteo_adjustment_mm"] = -bias["mean_bias_mm"]
        st.caption("Calibration is a display-only adjustment: add this value to an Open-Meteo annual total for the corresponding subdivision. Training data is unchanged.")
    st.dataframe(bias.style.format({"mean_bias_mm":"{:.1f}","correlation":"{:.2f}"}), use_container_width=True, hide_index=True)
st.markdown("""### Limitations
- The random (not chronological) train/test split can make true forecasting performance look slightly optimistic.
- Outlier removal means very wet years can be under-predicted.
- Missing values are filled with the column mean.
- IMD gauge and Open-Meteo model/satellite data are not directly comparable without calibration.
- The current year is partial and excluded from model training.

### Credits
Built for INT234 · Lovepreet Singh · Reg. No. 12411255 · Lovely Professional University.

### Tech stack
Python, Streamlit, pandas, scikit-learn, Plotly, requests and Open-Meteo / data.gov.in data services. Centroids in `data/subdivision_coords.csv` are sensible regional or administrative representative points, documented per row; they are not subdivision-area averages.""")
