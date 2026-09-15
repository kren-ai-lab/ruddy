"""Multivariate anomaly-scoring methods that never mutate or remove observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ruddy.core.enums import ScalingMethod
from ruddy.data import FeatureMatrix
from ruddy.projections.preprocessing import prepare_features
from ruddy.results import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    scores: pd.DataFrame
    methods: pd.DataFrame
    exclusions: pd.DataFrame
    provenance: AnalysisProvenance


def analyze_anomalies(
    features: FeatureMatrix,
    *,
    methods: tuple[str, ...] = ("isolation_forest", "lof"),
    scaling: ScalingMethod | str = ScalingMethod.NONE,
    contamination: str | float = "auto",
    isolation_estimators: int = 200,
    lof_neighbors: int = 20,
    random_state: int = 0,
) -> AnomalyResult:
    """Score multivariate anomalies using explicitly requested ML diagnostics."""
    normalized = tuple(str(m).lower() for m in methods)
    allowed = {"isolation_forest", "lof"}
    if not normalized or any(m not in allowed for m in normalized):
        raise ValueError("methods must contain isolation_forest and/or lof.")
    prepared = prepare_features(features, scaling=scaling, minimum_observations=3)
    if prepared.is_sparse:
        x = prepared.matrix.tocsr()
    else:
        x = np.asarray(prepared.matrix, dtype=float)
    n = prepared.n_observations
    rows, method_rows = [], []
    if "isolation_forest" in normalized:
        model = IsolationForest(
            n_estimators=isolation_estimators, contamination=contamination, random_state=random_state
        )
        pred = model.fit_predict(x)
        anomaly = -model.score_samples(x)
        for idx, oid, score, flag in zip(
            prepared.source_row_indices, prepared.observation_ids, anomaly, pred == -1
        ):
            rows.append(
                {
                    "method": "isolation_forest",
                    "source_row_index": int(idx),
                    "observation_id": oid,
                    "anomaly_score": float(score),
                    "is_flagged": bool(flag),
                    "status": "ok",
                    "reason": None,
                }
            )
        method_rows.append(
            {
                "method": "isolation_forest",
                "n_observations": n,
                "n_flagged": int((pred == -1).sum()),
                "contamination": contamination,
                "parameter": f"n_estimators={isolation_estimators}",
                "status": "ok",
                "reason": None,
            }
        )
    if "lof" in normalized:
        if lof_neighbors < 2 or lof_neighbors >= n:
            method_rows.append(
                {
                    "method": "lof",
                    "n_observations": n,
                    "n_flagged": 0,
                    "contamination": contamination,
                    "parameter": f"n_neighbors={lof_neighbors}",
                    "status": "skipped",
                    "reason": "lof_neighbors_must_be_less_than_observations",
                }
            )
        else:
            model = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=contamination)
            pred = model.fit_predict(x)
            anomaly = -model.negative_outlier_factor_
            for idx, oid, score, flag in zip(
                prepared.source_row_indices, prepared.observation_ids, anomaly, pred == -1
            ):
                rows.append(
                    {
                        "method": "lof",
                        "source_row_index": int(idx),
                        "observation_id": oid,
                        "anomaly_score": float(score),
                        "is_flagged": bool(flag),
                        "status": "ok",
                        "reason": None,
                    }
                )
            method_rows.append(
                {
                    "method": "lof",
                    "n_observations": n,
                    "n_flagged": int((pred == -1).sum()),
                    "contamination": contamination,
                    "parameter": f"n_neighbors={lof_neighbors}",
                    "status": "ok",
                    "reason": None,
                }
            )
    provenance = AnalysisProvenance(
        analysis="anomaly",
        parameters={
            "methods": normalized,
            "scaling": str(scaling),
            "contamination": contamination,
            "isolation_estimators": isolation_estimators,
            "lof_neighbors": lof_neighbors,
        },
        input_summary={
            "n_source_observations": features.n_observations,
            "n_analyzed_observations": n,
            "n_features": features.n_features,
        },
        random_state=random_state,
    )
    return AnomalyResult(pd.DataFrame(rows), pd.DataFrame(method_rows), prepared.exclusions, provenance)


__all__ = ["AnomalyResult", "analyze_anomalies"]
