from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .features import FEATURES


@dataclass
class FittedModel:
    estimator: GradientBoostingRegressor
    medians: pd.Series
    version: str
    cutoff: str
    last_label_end: str
    rows: int

    def predict(self, frame):
        return self.estimator.predict(frame[FEATURES])

    def explain(self, row):
        """Local median-replacement sensitivity; not SHAP or causal attribution."""
        sample = pd.DataFrame([{name: float(row[name]) for name in FEATURES}])
        prediction = float(self.predict(sample)[0])
        perturbed = pd.concat([sample] * len(FEATURES), ignore_index=True)
        for i, name in enumerate(FEATURES):
            perturbed.loc[i, name] = self.medians[name]
        effects = prediction - self.predict(perturbed)
        top = np.argsort(np.abs(effects))[-3:][::-1]
        return [{"feature": FEATURES[i], "value": float(row[FEATURES[i]]),
                 "sensitivity": float(effects[i])} for i in top]

    def audit(self):
        return {"version": self.version, "cutoff": self.cutoff,
                "last_label_end": self.last_label_end, "training_rows": self.rows}


def fit_model(features, cutoff, config):
    cutoff = pd.Timestamp(cutoff)
    train = features.loc[(features.date < cutoff) & (features.label_end < cutoff)].dropna(
        subset=FEATURES + ["target"])
    if train.date.nunique() < config.min_train_sessions:
        raise ValueError("Insufficient training history after feature warmup and label purging.")
    estimator = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.05,
                                         min_samples_leaf=20, random_state=config.seed, loss="huber")
    estimator.fit(train[FEATURES], train.target)
    digest = hashlib.sha256(pd.util.hash_pandas_object(
        train[["date", "symbol", *FEATURES, "target"]], index=False).values.tobytes()
        + str(config.seed).encode()).hexdigest()[:12]
    return FittedModel(estimator, train[FEATURES].median(), f"gbr-v1-{digest}",
                       str(cutoff.date()), str(train.label_end.max().date()), len(train))


def walk_forward(features, config):
    usable = features.dropna(subset=FEATURES)
    dates = pd.DatetimeIndex(sorted(features.date.unique()))[49:]
    counts = usable.groupby("date").symbol.nunique().reindex(dates, fill_value=0)
    if (counts != len(config.symbols)).any():
        raise ValueError("Invalid feature rows after the 50-session warmup; require complete features for every asset/session.")
    train_end = int(len(dates) * config.train_fraction)
    test_start = int(len(dates) * (config.train_fraction + config.validation_fraction))
    if train_end < config.min_train_sessions + 2 or test_start >= len(dates) - 2:
        raise ValueError("Dataset too short for the requested train/validation/test split.")
    pieces, audits = [], []
    model = None
    for i in range(train_end, len(dates)):
        # Expanding validation fits; one frozen fit at the final-test boundary.
        if model is None or i == test_start or (i < test_start and (i - train_end) % config.retrain_sessions == 0):
            model = fit_model(features, dates[i], config)
            audits.append(model.audit())
        day = usable.loc[usable.date == dates[i]].copy()
        day["prediction"] = model.predict(day)
        day["model_version"] = model.version
        day["model_cutoff"] = model.cutoff
        day["sensitivity"] = [json.dumps(model.explain(row)) for _, row in day.iterrows()]
        day["period"] = "validation" if i < test_start else "test"
        pieces.append(day)
    return pd.concat(pieces, ignore_index=True), {
        "validation_signal_start": str(dates[train_end].date()),
        "test_signal_start": str(dates[test_start].date()),
        "models": audits,
        "policy": "Expanding validation; purged labels; frozen final-test model; no parameter search.",
    }
