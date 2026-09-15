"""``ruddy project`` — PCA and exploratory nonlinear feature-space projections."""

from __future__ import annotations

import typer

from ruddy.cli._enums import (
    Scaling,
)
from ruddy.cli._options import (
    CONTEXT_SETTINGS,
    FEATURE_COLUMN,
    FEATURES_ARGUMENT,
    ID_COLUMN,
    IDS_FILE,
    MAX_ITER,
    METRIC,
    MIN_DIST,
    N_COMPONENTS,
    N_NEIGHBORS,
    OUTPUT_DIR,
    PERPLEXITY,
    RANDOM_STATE,
    SCALING,
    CliArgs,
    bundle,
    dispatch,
)
from ruddy.cli.commands import run_project

app = typer.Typer(
    name="project",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    help="Run PCA or an exploratory nonlinear projection on a feature matrix.",
)


def _dispatch(args: CliArgs, method: str) -> None:
    """Pin the projection method the subcommand stands for, then run it."""
    args.method = method
    dispatch(run_project, args, command=f"project {method}", legacy="project")


@app.command("pca", context_settings=CONTEXT_SETTINGS)
def pca(
    input: str = FEATURES_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    ids_file: str | None = IDS_FILE,
    feature_column: list[str] = FEATURE_COLUMN,
    n_components: int = N_COMPONENTS,
    scaling: Scaling = SCALING,
    random_state: int = RANDOM_STATE,
    output_dir: str | None = OUTPUT_DIR,
) -> None:
    """Run a principal component analysis on a feature matrix."""
    _dispatch(bundle(locals()), "pca")


@app.command("tsne", context_settings=CONTEXT_SETTINGS)
def tsne(
    input: str = FEATURES_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    ids_file: str | None = IDS_FILE,
    feature_column: list[str] = FEATURE_COLUMN,
    n_components: int = N_COMPONENTS,
    scaling: Scaling = SCALING,
    metric: str = METRIC,
    perplexity: float = PERPLEXITY,
    max_iter: int = MAX_ITER,
    random_state: int = RANDOM_STATE,
    output_dir: str | None = OUTPUT_DIR,
) -> None:
    """Run an exploratory t-SNE projection on a feature matrix."""
    _dispatch(bundle(locals()), "tsne")


@app.command("umap", context_settings=CONTEXT_SETTINGS)
def umap(
    input: str = FEATURES_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    ids_file: str | None = IDS_FILE,
    feature_column: list[str] = FEATURE_COLUMN,
    n_components: int = N_COMPONENTS,
    scaling: Scaling = SCALING,
    metric: str = METRIC,
    n_neighbors: int = N_NEIGHBORS,
    min_dist: float = MIN_DIST,
    random_state: int = RANDOM_STATE,
    output_dir: str | None = OUTPUT_DIR,
) -> None:
    """Run an exploratory UMAP projection on a feature matrix."""
    _dispatch(bundle(locals()), "umap")
