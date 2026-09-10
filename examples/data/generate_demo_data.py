"""Generate deterministic, domain-agnostic datasets used by Ruddy demos."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main(output_dir: str | Path | None = None) -> None:
    out = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260910)
    n_per_cell = 30
    group = np.repeat(["A", "B", "C"], 2 * n_per_cell)
    source = np.tile(np.repeat(["S1", "S2"], n_per_cell), 3)
    n = len(group)
    order = rng.permutation(n)
    group = group[order]
    source = source[order]
    ids = np.array([f"obs_{i:03d}" for i in range(n)])
    batch_raw = np.repeat([f"B{i:02d}" for i in range(12)], n // 12)
    batch = batch_raw[rng.permutation(n)]

    g_code = pd.Series(group).map({"A": 0.0, "B": 1.0, "C": 2.0}).to_numpy()
    s_code = (source == "S2").astype(float)
    length = rng.normal(50.0 + 2.0 * g_code, 7.0, n)
    charge = rng.normal(0.2 * g_code - 0.1 * s_code, 1.15, n)
    hydrophobicity = rng.normal(0.45 + 0.04 * g_code, 0.10, n)
    batch_levels = sorted(set(batch))
    batch_effects = {b: e for b, e in zip(batch_levels, rng.normal(0.0, 0.45, len(batch_levels)), strict=True)}
    random_batch = np.array([batch_effects[b] for b in batch])
    interaction = ((group == "C") & (source == "S2")).astype(float)

    activity = (
        1.0 + 0.85 * g_code + 0.55 * s_code + 0.70 * interaction
        + 0.035 * (length - 50.0) - 0.20 * charge + random_batch
        + rng.normal(0.0, 0.55, n)
    )
    stability = (
        2.4 - 0.35 * g_code + 0.30 * s_code + 0.025 * (length - 50.0)
        + 0.25 * hydrophobicity + 0.45 * random_batch + rng.normal(0.0, 0.45, n)
    )
    nonlinear = 0.8 * charge**2 + rng.normal(0.0, 0.35, n)
    phenotype = np.where(activity > np.quantile(activity, 0.66), "high", np.where(activity < np.quantile(activity, 0.33), "low", "mid"))
    flag = activity > np.median(activity)
    measurement = 0.6 * activity - 0.25 * stability + rng.normal(0.0, 0.4, n)
    measurement[[9, 77]] = np.inf
    hydrophobicity[[5, 44, 131]] = np.nan

    frame = pd.DataFrame({
        "id": ids,
        "activity": activity,
        "stability": stability,
        "length": length,
        "charge": charge,
        "hydrophobicity": hydrophobicity,
        "nonlinear_response": nonlinear,
        "measurement": measurement,
        "group": group,
        "source": source,
        "batch": batch,
        "phenotype": phenotype,
        "flag": flag,
        "recorded_at": pd.date_range("2026-01-01", periods=n, freq="12h"),
        "annotation": [f"sample_{i % 9}" for i in range(n)],
    })
    frame.to_csv(out / "tabular_demo.csv", index=False)

    latent = np.column_stack([
        (activity - activity.mean()) / activity.std(ddof=1),
        (stability - stability.mean()) / stability.std(ddof=1),
        g_code + rng.normal(0.0, 0.25, n),
        (nonlinear - nonlinear.mean()) / nonlinear.std(ddof=1),
    ])
    wa = rng.normal(size=(4, 12))
    wb = rng.normal(size=(4, 8))
    rep_a = latent @ wa + rng.normal(0.0, 0.28, size=(n, 12))
    rep_b = latent @ wb + rng.normal(0.0, 0.38, size=(n, 8))
    # Same-dimensional related space for Procrustes demonstration.
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    rep_c = rep_a @ q + rng.normal(0.0, 0.08, size=rep_a.shape)
    # Deliberate multivariate anomalies.
    rep_a[-3:] += np.array([6.0, -7.0, 8.0])[:, None]

    for name, matrix in [("representation_a", rep_a), ("representation_b", rep_b), ("representation_c", rep_c)]:
        cols = [f"f{i+1}" for i in range(matrix.shape[1])]
        pd.DataFrame(matrix, columns=cols).assign(id=ids).loc[:, ["id", *cols]].to_csv(out / f"{name}.csv", index=False)

    alpha_base = np.array([2.0, 3.0, 4.0, 2.5, 3.5, 1.8])
    comp = np.empty((n, len(alpha_base)), dtype=float)
    for i, code in enumerate(g_code):
        alpha = alpha_base.copy()
        alpha[0] += 0.8 * code
        alpha[3] += 0.5 * (2.0 - code)
        comp[i] = rng.dirichlet(alpha)
    comp[[2, 37, 115], [0, 2, 5]] = 0.0
    comp_cols = [f"part_{i+1}" for i in range(comp.shape[1])]
    pd.DataFrame(comp, columns=comp_cols).assign(id=ids).loc[:, ["id", *comp_cols]].to_csv(out / "compositional_demo.csv", index=False)

    # Separate mixed-effects demo chosen to yield a stable random-intercept fit.
    rng_mixed = np.random.default_rng(20260911)
    n_groups = 18
    per_group = 14
    n_mixed = n_groups * per_group
    mixed_batch = np.repeat(np.array([f"mb{i:02d}" for i in range(n_groups)], dtype=object), per_group)
    condition = np.tile(np.array(["A", "B"] * (per_group // 2), dtype=object), n_groups)
    mixed_x = rng_mixed.normal(size=n_mixed)
    random_intercept = rng_mixed.normal(scale=1.35, size=n_groups)
    mixed_y = 1.7 * (condition == "B") + 0.65 * mixed_x + np.repeat(random_intercept, per_group) + rng_mixed.normal(scale=0.5, size=n_mixed)
    mixed = pd.DataFrame({
        "id": [f"mix_{i:03d}" for i in range(n_mixed)],
        "condition": condition,
        "batch": mixed_batch,
        "x": mixed_x,
        "y": mixed_y,
    })
    mixed.to_csv(out / "mixed_demo.csv", index=False)


if __name__ == "__main__":
    main()
