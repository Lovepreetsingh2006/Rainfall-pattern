# Rainfall Patterns in India (1901–Today)

A multi-page Streamlit application for live rainfall exploration and next-year rainfall prediction.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m pytest
```

On Windows, use `python -m streamlit` rather than a bare `streamlit` command. This guarantees that Streamlit runs in the same Python environment where dependencies were installed, avoiding accidental use of a separate Anaconda installation.

For a data.gov.in refresh, add `DATA_GOV_API_KEY` and the dataset's `DATA_GOV_RESOURCE_ID` to `.streamlit/secrets.toml` (see `.env.example`). Without credentials or on an API outage, the app clearly uses the packaged IMD 1901–2015 fallback.

## Structure

- `app.py` and `pages/`: Streamlit overview, live dashboard, explorer, EDA, model lab, predictor and methodology pages.
- `src/features.py`, `src/pipeline.py`, `src/models.py`: the original cleaning/features and model methodology as testable pure functions.
- `src/data.py`, `src/live.py`: data.gov.in fallback and Open-Meteo archive/forecast retrieval (cached 24 hours and 1 hour respectively).
- `data/subdivision_coords.csv`: representative regional centroids, with selection notes.

## Sources and interpretation

IMD/data.gov.in is the historical, area-weighted gauge baseline. Open-Meteo provides daily gridded precipitation at representative subdivision centroids from 2016 to today plus a 16-day forecast. They are visibly source-labelled and are not silently merged for modelling; the About page presents an overlap bias check when an overlap source is configured. Current-year records are partial and excluded from training.

## Screenshots

_Add dashboard screenshot here._

_Add model lab screenshot here._
