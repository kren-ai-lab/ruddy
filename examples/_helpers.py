"""Visualization helpers for the Ruddy marimo examples.

These helpers deliberately live under ``examples/``.  They consume structured
Ruddy outputs but are not imported by the scientific core.
"""
from __future__ import annotations

from pathlib import Path
import inspect
from typing import Iterable, Sequence

import marimo as mo
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


DATA = Path(__file__).resolve().parent / "data"


def load_tabular_demo():
    from ruddy import TabularDataset

    frame = pd.read_csv(DATA / "tabular_demo.csv", parse_dates=["recorded_at"])
    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "activity": "response",
            "stability": "response",
            "group": "factor",
            "source": "factor",
            "batch": "factor",
            "length": "covariate",
            "charge": "covariate",
            "annotation": "annotation",
        },
    )
    return frame, dataset


def load_mixed_demo():
    from ruddy import TabularDataset

    frame = pd.read_csv(DATA / "mixed_demo.csv")
    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "y": "response",
            "condition": "factor",
            "batch": "factor",
            "x": "covariate",
        },
    )
    return frame, dataset


def load_feature_demo(name: str):
    from ruddy import FeatureMatrix

    path = DATA / f"{name}.csv"
    frame = pd.read_csv(path)
    feature_names = [column for column in frame.columns if column != "id"]
    return FeatureMatrix(
        frame[feature_names].to_numpy(dtype=float),
        observation_ids=frame["id"].tolist(),
        feature_names=feature_names,
    )


def load_compositional_demo():
    from ruddy import FeatureMatrix

    frame = pd.read_csv(DATA / "compositional_demo.csv")
    feature_names = [column for column in frame.columns if column != "id"]
    return frame, FeatureMatrix(
        frame[feature_names].to_numpy(dtype=float),
        observation_ids=frame["id"].tolist(),
        feature_names=feature_names,
    )


def display(obj) -> None:
    """Append a table or object to the current marimo cell output."""
    mo.output.append(obj)


def show() -> None:
    """Render the current Matplotlib figure into the cell output and close it."""
    fig = plt.gcf()
    mo.output.append(mo.as_html(fig))
    plt.close(fig)


def configure_plots() -> None:
    plt.rcParams.update({
        "figure.figsize": (7.6, 4.8),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 115,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "legend.frameon": False,
    })


def finish(fig, *, title: str | None = None):
    if title:
        fig.suptitle(title, y=1.01, fontsize=13, fontweight="semibold")
    fig.tight_layout()
    return fig


def matrix_heatmap(frame: pd.DataFrame, *, title: str, fmt: str = ".2f", annotate: bool = True):
    values = frame.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.62 * len(frame.columns) + 2), max(4.6, 0.52 * len(frame.index) + 2)))
    image = ax.imshow(values, aspect="auto")
    ax.set_xticks(np.arange(len(frame.columns)), frame.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(frame.index)), frame.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8)
    if annotate and values.size <= 144:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=7.5)
    return finish(fig), ax


def group_violin_box_scatter(frame: pd.DataFrame, *, value: str, group: str, title: str):
    data = frame[[group, value]].replace([np.inf, -np.inf], np.nan).dropna()
    levels = list(pd.unique(data[group]))
    arrays = [data.loc[data[group] == level, value].to_numpy(dtype=float) for level in levels]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.violinplot(arrays, positions=np.arange(1, len(levels) + 1), showmeans=False, showmedians=False, showextrema=False)
    ax.boxplot(arrays, positions=np.arange(1, len(levels) + 1), widths=0.18, showfliers=False)
    rng = np.random.default_rng(42)
    for i, arr in enumerate(arrays, start=1):
        jitter = rng.normal(i, 0.035, arr.size)
        ax.scatter(jitter, arr, s=16, alpha=0.55)
    ax.set_xticks(np.arange(1, len(levels) + 1), [str(x) for x in levels])
    ax.set_xlabel(group)
    ax.set_ylabel(value)
    ax.set_title(title)
    return finish(fig), ax


def density_by_group(frame: pd.DataFrame, *, value: str, group: str, title: str):
    data = frame[[group, value]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for level, subset in data.groupby(group, sort=False):
        x = subset[value].to_numpy(dtype=float)
        if len(np.unique(x)) < 2:
            continue
        grid = np.linspace(np.min(x), np.max(x), 220)
        kde = gaussian_kde(x)
        ax.plot(grid, kde(grid), label=str(level), linewidth=2)
        ax.fill_between(grid, 0, kde(grid), alpha=0.10)
    ax.set_xlabel(value)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(title=group)
    return finish(fig), ax


def ecdf_by_group(frame: pd.DataFrame, *, value: str, group: str, title: str):
    data = frame[[group, value]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for level, subset in data.groupby(group, sort=False):
        x = np.sort(subset[value].to_numpy(dtype=float))
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", label=str(level), linewidth=1.8)
    ax.set_xlabel(value)
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.legend(title=group)
    return finish(fig), ax


def scatter_by_group(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    title: str,
    fit_lines: bool = True,
    annotate_centroids: bool = True,
):
    data = frame[[x, y, group]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for level, subset in data.groupby(group, sort=False):
        xv = subset[x].to_numpy(dtype=float)
        yv = subset[y].to_numpy(dtype=float)
        ax.scatter(xv, yv, label=str(level), alpha=0.68, s=28)
        if fit_lines and len(xv) >= 3 and np.ptp(xv) > 0:
            slope, intercept = np.polyfit(xv, yv, 1)
            grid = np.linspace(np.min(xv), np.max(xv), 100)
            ax.plot(grid, slope * grid + intercept, linewidth=1.8)
        if annotate_centroids:
            cx, cy = float(np.mean(xv)), float(np.mean(yv))
            ax.scatter([cx], [cy], s=100, marker="X")
            ax.annotate(str(level), (cx, cy), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(title=group)
    return finish(fig), ax


def grouped_histograms(frame: pd.DataFrame, *, value: str, group: str, title: str, bins: int = 20):
    data = frame[[group, value]].replace([np.inf, -np.inf], np.nan).dropna()
    levels = list(pd.unique(data[group]))
    fig, axes = plt.subplots(len(levels), 1, figsize=(7.8, 2.25 * len(levels) + 1.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels, strict=True):
        arr = data.loc[data[group] == level, value].to_numpy(dtype=float)
        ax.hist(arr, bins=bins, alpha=0.75)
        ax.axvline(np.mean(arr), linestyle="--", linewidth=1.4)
        ax.set_ylabel(str(level))
    axes[-1].set_xlabel(value)
    axes[0].set_title(title)
    return finish(fig), axes


def errorbar_table(
    table: pd.DataFrame,
    *,
    label_col: str,
    estimate_col: str,
    low_col: str,
    high_col: str,
    title: str,
    xlabel: str,
    zero_line: bool = True,
):
    data = table.reset_index(drop=True).copy()
    y = np.arange(len(data))
    estimate = data[estimate_col].to_numpy(dtype=float)
    low = data[low_col].to_numpy(dtype=float)
    high = data[high_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, max(3.8, 0.48 * len(data) + 1.8)))
    ax.errorbar(estimate, y, xerr=np.vstack([estimate - low, high - estimate]), fmt="o", capsize=3)
    if zero_line:
        ax.axvline(0.0, linewidth=1, linestyle="--")
    ax.set_yticks(y, data[label_col].astype(str))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return finish(fig), ax


def bar_metric(table: pd.DataFrame, *, label: str, value: str, title: str, ylabel: str | None = None):
    data = table.copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, max(4.2, 0.38 * len(data) + 2)))
    y = np.arange(len(data))
    ax.barh(y, data[value].to_numpy(dtype=float))
    ax.set_yticks(y, data[label].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel(ylabel or value)
    ax.set_title(title)
    return finish(fig), ax


def confidence_ellipse(ax, x: np.ndarray, y: np.ndarray, *, n_std: float = 2.0):
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if np.any(vals <= 0):
        return None
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse((np.mean(x), np.mean(y)), width, height, angle=angle, fill=False, linewidth=1.4, alpha=0.75)
    ax.add_patch(ell)
    return ell


def score_plot(scores: pd.DataFrame, metadata: pd.DataFrame, *, x: str, y: str, group: str, title: str, ellipses: bool = False):
    merged = scores.merge(metadata[["id", group]], left_on="observation_id", right_on="id", how="left")
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for level, subset in merged.groupby(group, sort=False, dropna=False):
        xv = subset[x].to_numpy(dtype=float)
        yv = subset[y].to_numpy(dtype=float)
        ax.scatter(xv, yv, label=str(level), alpha=0.72, s=30)
        if ellipses:
            confidence_ellipse(ax, xv, yv)
    ax.axhline(0, linewidth=0.8, alpha=0.4)
    ax.axvline(0, linewidth=0.8, alpha=0.4)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(title=group)
    return finish(fig), ax


def biplot(scores: pd.DataFrame, loadings: pd.DataFrame, metadata: pd.DataFrame, *, group: str, x: str = "PC1", y: str = "PC2", top_n: int = 8, title: str = "PCA biplot"):
    merged = scores.merge(metadata[["id", group]], left_on="observation_id", right_on="id", how="left")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for level, subset in merged.groupby(group, sort=False):
        ax.scatter(subset[x], subset[y], label=str(level), alpha=0.48, s=22)
    importance = np.sqrt(loadings[x] ** 2 + loadings[y] ** 2)
    chosen = loadings.assign(_importance=importance).nlargest(top_n, "_importance")
    sx = max(np.nanstd(scores[x]), 1e-9) * 2.2
    sy = max(np.nanstd(scores[y]), 1e-9) * 2.2
    for _, row in chosen.iterrows():
        dx, dy = float(row[x]) * sx, float(row[y]) * sy
        ax.arrow(0, 0, dx, dy, width=0.006, head_width=0.08, length_includes_head=True, alpha=0.8)
        ax.text(dx * 1.08, dy * 1.08, str(row["feature"]), fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(title=group)
    return finish(fig), ax


def anomaly_agreement(scores: pd.DataFrame, *, title: str = "Anomaly-method agreement"):
    pivot = scores.pivot(index="observation_id", columns="method", values="is_flagged").fillna(False).astype(int)
    order = pivot.sum(axis=1).sort_values(ascending=False).index
    top = pivot.loc[order[: min(40, len(order))]]
    return matrix_heatmap(top, title=title, fmt=".0f", annotate=False)


def distance_scatter(x: np.ndarray, y: np.ndarray, *, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.3, 5.5))
    ax.scatter(x, y, s=17, alpha=0.40)
    if len(x) >= 3 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(grid, slope * grid + intercept, linewidth=1.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return finish(fig), ax


def procrustes_segments(x: np.ndarray, y: np.ndarray, *, labels: Sequence[str] | None = None, title: str = "Procrustes alignment"):
    if x.shape[1] < 2 or y.shape[1] < 2:
        raise ValueError("Procrustes plot requires at least two dimensions in each space.")
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(x[:, 0], x[:, 1], label="Space X", alpha=0.60, s=24)
    ax.scatter(y[:, 0], y[:, 1], label="Space Y", alpha=0.60, s=24, marker="x")
    for i in range(min(len(x), len(y))):
        ax.plot([x[i, 0], y[i, 0]], [x[i, 1], y[i, 1]], linewidth=0.45, alpha=0.25)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.legend()
    return finish(fig), ax


def stacked_composition(frame: pd.DataFrame, *, id_col: str = "id", n: int = 35, title: str = "Composition profiles"):
    data = frame.head(n).set_index(id_col)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bottom = np.zeros(len(data))
    for column in data.columns:
        vals = data[column].to_numpy(dtype=float)
        ax.bar(np.arange(len(data)), vals, bottom=bottom, width=0.95, label=column)
        bottom += vals
    ax.set_xticks([])
    ax.set_ylabel("Composition")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8)
    return finish(fig), ax


def labeled_boxplot(ax, data, labels, **kwargs):
    """Draw a boxplot with labels across Matplotlib API versions."""
    params = inspect.signature(ax.boxplot).parameters
    label_key = "tick_labels" if "tick_labels" in params else "labels"
    options = dict(kwargs)
    options[label_key] = labels
    return ax.boxplot(data, **options)
