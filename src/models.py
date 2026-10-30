"""Model fitting, metrics and prediction helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from .features import SEASONS
from .pipeline import make_ml_dataset, multiple_features


@dataclass
class ModelBundle:
    simple: Any; multiple: Any; knn: Any; simple_scaler: Any; knn_scaler: Any
    feature_columns: list[str]; metrics: pd.DataFrame; cv: pd.DataFrame; predictions: dict[str, tuple[np.ndarray, np.ndarray]]
    trained_rows: int


def _metric_row(name: str, actual: pd.Series, predicted: np.ndarray) -> dict[str, float | str]:
    mse = mean_squared_error(actual, predicted)
    return {"Model": name, "MAE": mean_absolute_error(actual, predicted), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R²": r2_score(actual, predicted)}


def train_models(data: pd.DataFrame) -> ModelBundle:
    """Fit the three reference models with the original split and KNN selection."""
    s1 = make_ml_dataset(data)
    y = s1["NEXT_YEAR_RAIN"]
    simple_scaler = MinMaxScaler()
    simple_x = simple_scaler.fit_transform(s1[["ANNUAL"]])
    sx_train, sx_test, sy_train, sy_test = train_test_split(simple_x, y, test_size=.2, random_state=34)
    simple = LinearRegression().fit(sx_train, sy_train)
    simple_pred = simple.predict(sx_test)

    x2 = multiple_features(s1)
    x_train, x_test, y_train, y_test = train_test_split(x2, y, test_size=.2, random_state=34)
    multiple = LinearRegression().fit(x_train, y_train)
    multiple_pred = multiple.predict(x_test)
    knn_scaler = MinMaxScaler()
    k_train, k_test = knn_scaler.fit_transform(x_train), knn_scaler.transform(x_test)
    rows, best_k, best_mse = [], 15, float("inf")
    for k in [15, 20, 25, 30, 40, 50]:
        mse = float(-cross_val_score(KNeighborsRegressor(n_neighbors=k), k_train, y_train, cv=5, scoring="neg_mean_squared_error").mean())
        rows.append({"K": k, "CV MSE": mse})
        if mse < best_mse: best_k, best_mse = k, mse
    knn = KNeighborsRegressor(n_neighbors=best_k).fit(k_train, y_train)
    knn_pred = knn.predict(k_test)
    metrics = pd.DataFrame([_metric_row("Simple LR", sy_test, simple_pred), _metric_row("Multiple LR", y_test, multiple_pred), _metric_row("KNN", y_test, knn_pred)])
    return ModelBundle(simple, multiple, knn, simple_scaler, knn_scaler, list(x2.columns), metrics, pd.DataFrame(rows),
                       {"Simple LR": (sy_test.to_numpy(), simple_pred), "Multiple LR": (y_test.to_numpy(), multiple_pred), "KNN": (y_test.to_numpy(), knn_pred)}, len(s1))


def predict(bundle: ModelBundle, subdivision: str, values: dict[str, float]) -> dict[str, float]:
    """Return three next-year predictions for seasonal input values."""
    annual = sum(values[s] for s in SEASONS)
    row = pd.DataFrame([{**values, "SUBDIVISION": subdivision}])
    encoded = pd.get_dummies(row, columns=["SUBDIVISION"], dtype=int).reindex(columns=bundle.feature_columns, fill_value=0)
    return {"Simple LR": float(bundle.simple.predict(bundle.simple_scaler.transform(pd.DataFrame({"ANNUAL": [annual]})))[0]),
            "Multiple LR": float(bundle.multiple.predict(encoded)[0]),
            "KNN": float(bundle.knn.predict(bundle.knn_scaler.transform(encoded))[0])}
