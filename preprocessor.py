import numpy as np
import pandas as pd


def build_chronos_input(
    df: pd.DataFrame,
    target_col: str = "close",
    covariate_cols: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if covariate_cols is None:
        covariate_cols = ["open", "high", "low", "volume"]

    close = df[target_col].astype(float).values

    past_covariates = {}
    for col in covariate_cols:
        if col in df.columns:
            past_covariates[col] = df[col].astype(float).values

    return close, past_covariates


def build_forecast_index(
    last_date: pd.Timestamp,
    prediction_length: int,
) -> pd.DatetimeIndex:
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=prediction_length)
