"""State-area mapping helpers for the live rainfall dashboard."""
from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st

# IMD reports meteorological subdivisions. This crosswalk makes their readings
# viewable at state level; a multi-state subdivision contributes equally to each
# listed state and repeated contributing subdivisions are averaged.
SUBDIVISION_STATES: dict[str, list[str]] = {
    "ANDAMAN & NICOBAR ISLANDS": ["Andaman and Nicobar"], "ARUNACHAL PRADESH": ["Arunachal Pradesh"],
    "ASSAM & MEGHALAYA": ["Assam", "Meghalaya"], "NAGA MANI MIZO TRIPURA": ["Nagaland", "Manipur", "Mizoram", "Tripura"],
    "SUB HIMALAYAN WEST BENGAL & SIKKIM": ["West Bengal", "Sikkim"], "GANGETIC WEST BENGAL": ["West Bengal"],
    "ORISSA": ["Orissa"], "JHARKHAND": ["Jharkhand"], "BIHAR": ["Bihar"],
    "EAST UTTAR PRADESH": ["Uttar Pradesh"], "WEST UTTAR PRADESH": ["Uttar Pradesh"], "UTTARAKHAND": ["Uttaranchal"],
    "HARYANA DELHI & CHANDIGARH": ["Haryana", "Delhi", "Chandigarh"], "PUNJAB": ["Punjab"],
    "HIMACHAL PRADESH": ["Himachal Pradesh"], "JAMMU & KASHMIR": ["Jammu and Kashmir"],
    "WEST RAJASTHAN": ["Rajasthan"], "EAST RAJASTHAN": ["Rajasthan"],
    "WEST MADHYA PRADESH": ["Madhya Pradesh"], "EAST MADHYA PRADESH": ["Madhya Pradesh"],
    "GUJARAT REGION": ["Gujarat"], "SAURASHTRA & KUTCH": ["Gujarat"],
    "KONKAN & GOA": ["Maharashtra", "Goa"], "MADHYA MAHARASHTRA": ["Maharashtra"],
    "MATATHWADA": ["Maharashtra"], "VIDARBHA": ["Maharashtra"], "CHHATTISGARH": ["Chhattisgarh"],
    "COASTAL ANDHRA PRADESH": ["Andhra Pradesh"], "TELANGANA": ["Andhra Pradesh"], "RAYALSEEMA": ["Andhra Pradesh"],
    "TAMIL NADU": ["Tamil Nadu"], "COASTAL KARNATAKA": ["Karnataka"],
    "NORTH INTERIOR KARNATAKA": ["Karnataka"], "SOUTH INTERIOR KARNATAKA": ["Karnataka"],
    "KERALA": ["Kerala"], "LAKSHADWEEP": ["Lakshadweep"],
}

STATE_GEOJSON_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"


@st.cache_data(ttl=86400, show_spinner=False)
def india_state_geojson() -> dict[str, Any]:
    """Retrieve and cache the public state-boundary GeoJSON; no API key is needed."""
    response = requests.get(STATE_GEOJSON_URL, timeout=45)
    response.raise_for_status()
    return response.json()


def aggregate_subdivisions_to_states(subdivision_data: pd.DataFrame) -> pd.DataFrame:
    """Turn subdivision rainfall values into source-labelled state-area values."""
    state_rows = subdivision_data[["SUBDIVISION", "rainfall_mm"]].copy()
    state_rows["STATE"] = state_rows["SUBDIVISION"].map(SUBDIVISION_STATES)
    state_rows = state_rows.explode("STATE").dropna(subset=["STATE"])
    return state_rows.groupby("STATE", as_index=False).rainfall_mm.mean()
