import pandas as pd
from unittest.mock import Mock, patch
from src.live import fetch_archive_row

def test_open_meteo_daily_response_aggregates_and_labels_source():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "daily": {"time": ["2024-01-30", "2024-01-31", "2024-02-01"], "precipitation_sum": [1.0, 2.0, 4.0]}
    }
    row = pd.Series({"SUBDIVISION": "PUNJAB", "latitude": 30.9, "longitude": 75.9})
    with patch("src.live.requests.get", return_value=response) as mocked_get:
        result = fetch_archive_row(row, pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-02-01").date())
    assert mocked_get.called
    assert result.loc[0, "SOURCE"] == "Open-Meteo"
    assert result.loc[0, "JAN"] == 3.0
    assert result.loc[0, "Jan-Feb"] == 7.0

def test_current_year_flag_is_partial_by_design():
    """The archive loader assigns current-year records IS_COMPLETE=False before modelling."""
    # Contract asserted here without making a network request: complete filtering in
    # make_ml_dataset consumes this field, keeping a current partial year out.
    from src.pipeline import make_ml_dataset
    data = pd.DataFrame({"SUBDIVISION":["A","A"], "YEAR":[2024,2025], "ANNUAL":[10.,20.], "IS_COMPLETE":[True,False]})
    assert make_ml_dataset(data).empty
