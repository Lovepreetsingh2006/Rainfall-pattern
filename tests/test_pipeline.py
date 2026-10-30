from pathlib import Path
import numpy as np
import pandas as pd
from src.features import aggregate_daily, clean_and_engineer
from src.models import train_models
from src.pipeline import make_ml_dataset, multiple_features

ROOT = Path(__file__).resolve().parents[1]

def baseline(): return clean_and_engineer(pd.read_csv(ROOT / "data" / "rainfalldata.csv"))

def test_baseline_regression_metrics():
    result = train_models(baseline()).metrics.set_index("Model")
    expected = {"Simple LR": [225.86, 86596.39, 294.27, .82], "Multiple LR": [167.75, 51657.50, 227.28, .89], "KNN": [169.62, 52838.33, 229.87, .89]}
    for name, values in expected.items():
        assert np.allclose(result.loc[name, ["MAE", "MSE", "RMSE", "R²"]].astype(float), values, atol=.01)

def test_multiple_matrix_has_39_columns():
    assert multiple_features(make_ml_dataset(baseline())).shape[1] == 39

def test_daily_aggregation_marks_source_and_partial_ready():
    daily = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "precipitation_sum": [1., 2., 3.]})
    result = aggregate_daily(daily, "PUNJAB")
    assert result.loc[0, "SOURCE"] == "Open-Meteo"
    assert result.loc[0, "JAN"] == 6 and result.loc[0, "Jan-Feb"] == 6 and result.loc[0, "ANNUAL"] == 6
