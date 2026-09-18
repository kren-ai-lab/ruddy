"""Multivariate anomaly-scoring methods that never mutate or remove observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ruddy.core.enums import ScalingMethod
from ruddy.core.frames import id_dtype
from ruddy.projections.preprocessing import prepare_features
from ruddy.results import AnalysisProvenance

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

    from ruddy.core.types import ObservationID
    from ruddy.data import FeatureMatrix

ANOMALY_SCORE_SCHEMA_BASE: dict[str, PolarsDataType] = {
    "source_row_index": pl.Int64,
    "method": pl.String,
    "anomaly_score": pl.Float64,
    "is_flagged": pl.Boolean,
    "status": pl.String,
    "reason": pl.String,
}

ANOMALY_SCORE_COLUMNS: tuple[str, ...] = (
    "observation_id",
    "source_row_index",
    "method",
    "anomaly_score",
    "is_flagged",
    "status",
    "reason",
)

ANOMALY_METHOD_SCHEMA: dict[str, PolarsDataType] = {
    "method": pl.String,
    "n_observations": pl.Int64,
    "n_flagged": pl.Int64,
    "contamination": pl.String,
    "parameter": pl.String,
    "status": pl.String,
    "reason": pl.String,
}


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    """Results of anomaly detection algorithms across observations."""

    scores: pl.DataFrame
    methods: pl.DataFrame
    exclusions: pl.DataFrame
    provenance: AnalysisProvenance


def _build_anomaly_scores(
    rows: list[dict[str, Any]],
    ids: tuple[ObservationID, ...],
) -> pl.DataFrame:
    id_dt = id_dtype(ids)
    if not rows:
        schema = {**ANOMALY_SCORE_SCHEMA_BASE, "observation_id": id_dt}
        return pl.DataFrame(schema={col: schema[col] for col in ANOMALY_SCORE_COLUMNS})
    frame = pl.DataFrame(rows, schema_overrides=ANOMALY_SCORE_SCHEMA_BASE)
    return frame.select(list(ANOMALY_SCORE_COLUMNS))


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
    x = prepared.matrix.tocsr() if prepared.is_sparse else np.asarray(prepared.matrix, dtype=float)
    n = prepared.n_observations
    rows, method_rows = [], []
    if "isolation_forest" in normalized:
        model = IsolationForest(
            n_estimators=isolation_estimators, contamination=contamination, random_state=random_state
        )
        pred = model.fit_predict(x)
        anomaly = -model.score_samples(x)
        for idx, oid, score, flag in zip(
            prepared.source_row_indices, prepared.observation_ids, anomaly, pred == -1, strict=False
        ):
            rows.append(
                {
                    "observation_id": oid,
                    "source_row_index": int(idx),
                    "method": "isolation_forest",
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
                "contamination": str(contamination),
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
                    "contamination": str(contamination),
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
                prepared.source_row_indices, prepared.observation_ids, anomaly, pred == -1, strict=False
            ):
                rows.append(
                    {
                        "observation_id": oid,
                        "source_row_index": int(idx),
                        "method": "lof",
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
                    "contamination": str(contamination),
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
    scores = _build_anomaly_scores(rows, prepared.observation_ids)
    methods_df = (
        pl.DataFrame(method_rows, schema_overrides=ANOMALY_METHOD_SCHEMA)
        if method_rows
        else pl.DataFrame(schema=ANOMALY_METHOD_SCHEMA)
    )
    return AnomalyResult(scores, methods_df, prepared.exclusions, provenance)


__all__ = ["AnomalyResult", "analyze_anomalies"]
