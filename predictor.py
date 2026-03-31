import os
import numpy as np
import torch
from chronos import Chronos2Pipeline

_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "chronos-2")


def _get_pipeline():
    global _MODEL
    if _MODEL is None:
        _MODEL = Chronos2Pipeline.from_pretrained(
            _MODEL_PATH,
            device_map="auto",
        )
    return _MODEL


def predict_close(
    close: np.ndarray,
    past_covariates: dict[str, np.ndarray] | None = None,
    prediction_length: int = 30,
    quantile_levels: list[float] | None = None,
) -> torch.Tensor:
    if quantile_levels is None:
        quantile_levels = [0.1, 0.5, 0.9]

    pipeline = _get_pipeline()

    inputs = {
        "target": close.astype(np.float32),
    }
    if past_covariates:
        inputs["past_covariates"] = {
            k: v.astype(np.float32) for k, v in past_covariates.items()
        }

    predictions = pipeline.predict(
        [inputs],
        prediction_length=prediction_length,
    )

    return predictions[0]
