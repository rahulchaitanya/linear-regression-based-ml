"""Reusable prediction helpers for the persisted fitted pipeline."""
import pandas as pd
from .preprocessing import FEATURES


def predict_one(pipeline, values):
    missing = set(FEATURES) - set(values)
    if missing:
        raise ValueError(f"Missing prediction features: {', '.join(sorted(missing))}")
    frame = pd.DataFrame([{name: values[name] for name in FEATURES}], columns=FEATURES)
    return float(pipeline.predict(frame)[0])
