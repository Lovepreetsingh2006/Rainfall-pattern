from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from src.live import load_forecast, load_live_archive, updated_label
from src.maps import aggregate_subdivisions_to_states, india_state_geojson
from src.ui import baseline_context, page_header

baseline, source, notice, _ = baseline_context()
page_header("Live Dashboard", "Open-Meteo representative subdivision centroids")
coords = pd.read_csv("data/subdivision_coords.csv")
subdivision = st.selectbox("Subdivision", coords.SUBDIVISION)
location = coords.loc[coords.SUBDIVISION.eq(subdivision)].iloc[0]

@st.fragment(run_every="1h")
def live_panel():
    try:
        recent, forecast = load_forecast(float(location.latitude), float(location.longitude))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rainfall today (mm)", f"{recent.precipitation_sum.iloc[-1]:.1f}")
        c2.metric("Last 30 days (mm)", f"{recent.precipitation_sum.sum():.1f}")
        c3.metric("Next 16 days (mm)", f"{forecast.precipitation_sum.sum():.1f}")
        st.plotly_chart(px.bar(forecast, x="date", y="precipitation_sum", title="16-day precipitation forecast", labels={"precipitation_sum":"mm"}), use_container_width=True)
        st.caption(updated_label("Open-Meteo forecast (auto-refresh hourly)"))
    except Exception:
        st.warning("Live forecast is temporarily unavailable. Please use the refresh button later.")
live_panel()

st.subheader("Last 30 days across subdivisions")
try:
    rows = []
    for _, point in coords.iterrows():
        recent, _ = load_forecast(float(point.latitude), float(point.longitude))
        rows.append({"SUBDIVISION": point.SUBDIVISION, "latitude": point.latitude, "longitude": point.longitude, "rainfall_mm": recent.precipitation_sum.sum()})
    points = pd.DataFrame(rows)
    # OpenStreetMap is key-free. CARTO's public tiles can show an API-key watermark.
    state_values = aggregate_subdivisions_to_states(points)
    fig = px.choropleth_mapbox(
        state_values, geojson=india_state_geojson(), locations="STATE", featureidkey="properties.NAME_1",
        color="rainfall_mm", color_continuous_scale="Blues", range_color=(0, float(state_values.rainfall_mm.max())),
        mapbox_style="open-street-map", zoom=3.3, center={"lat": 22.5, "lon": 79}, opacity=.72,
        hover_name="STATE", labels={"rainfall_mm": "Rainfall (mm)"}, title="Last-30-day rainfall by state area",
    )
    fig.update_layout(margin={"r": 0, "t": 45, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("State colours show a mean of their represented IMD meteorological subdivision readings. Telangana is included in the Andhra Pradesh boundary because the public boundary layer uses the pre-2014 state outline.")
except Exception:
    st.warning("The all-subdivision live map is unavailable right now.")
