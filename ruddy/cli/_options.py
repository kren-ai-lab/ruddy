"""Shared Typer options and the Typer -> commands-layer bridge.

Option declarations live here once, as module-level ``typer.Option`` constants,
and are reused by every command signature that needs them. ``ruddy.cli.commands``
never sees Typer or argparse: it receives a :class:`CliArgs` parameter bag.
"""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import TYPE_CHECKING

import typer

from ruddy._version import __version__
from ruddy.cli._console import fail, record_blocks, record_output, render
from ruddy.cli._enums import (
    Alignment,
    BootstrapMethod,
    CompositionalTransform,
    PAdjust,
    RobustCovariance,
    Scaling,
    SimilarityMethod,
    SSType,
)
from ruddy.core.exceptions import RuddyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

#: Exit code used for every user-facing (non-crash) CLI failure.
USER_ERROR_EXIT_CODE = 2


class CliArgs(SimpleNamespace):
    """Explicit parameter bag passed from the Typer layer to ``ruddy.cli.commands``."""


def bundle(params: dict[str, object]) -> CliArgs:
    """Turn a Typer command's parameters into a :class:`CliArgs` bag.

    Call as ``args = bundle(locals())`` on the first line of a command body,
    where ``locals()`` contains exactly the declared parameters.
    """
    return CliArgs(**{name: _plain(value) for name, value in params.items()})


def _plain(value: object) -> object:
    """Unwrap enum choices so ``ruddy.cli.commands`` only ever sees plain values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def dispatch(
    handler: Callable[[CliArgs], int],
    args: CliArgs,
    *,
    command: str,
    legacy: str | None = None,
) -> None:
    """Run a commands-layer handler, map user errors to a clean exit code.

    ``legacy`` is the flat pre-2.0 command name; ``build_config`` still keys a
    couple of defaults off it.
    """
    args.command = legacy or command.rsplit(" ", 1)[-1]
    record_output(getattr(args, "output_dir", None))
    try:
        code = handler(args)
    except (RuddyError, TypeError, ValueError, OSError) as exc:
        fail(exc)
        raise typer.Exit(USER_ERROR_EXIT_CODE) from exc
    if code:
        raise typer.Exit(int(code))
    if not _blocks_recorded():
        record_blocks([(args.command, True)])
    render(command)


def _blocks_recorded() -> bool:
    from ruddy.cli._console import STATE

    return bool(STATE.blocks)


def version_callback(value: bool) -> None:  # noqa: FBT001
    """Print the Ruddy version and exit."""
    if value:
        typer.echo(f"ruddy {__version__}")
        raise typer.Exit


# --------------------------------------------------------------------------
# Tabular input, column roles and kinds
# --------------------------------------------------------------------------
TABLE_ARGUMENT = typer.Argument(
    ..., metavar="INPUT", help="Input .csv, .tsv, .txt, or .parquet table."
)
FEATURES_ARGUMENT = typer.Argument(
    ...,
    metavar="INPUT",
    help="Feature matrix: CSV/TSV/TXT/Parquet, NPY, or sparse NPZ.",
)
ID_COLUMN = typer.Option(None, "--id-column", help="Observation identifier column.")
RESPONSE = typer.Option(None, "--response", help="Response column; repeatable.")
FACTOR = typer.Option(None, "--factor", help="Factor column; repeatable.")
COVARIATE = typer.Option(None, "--covariate", help="Covariate column; repeatable.")
ANNOTATION = typer.Option(None, "--annotation", help="Annotation column; repeatable.")
EXCLUDE = typer.Option(None, "--exclude", help="Excluded column; repeatable.")
NUMERIC = typer.Option(
    None, "--numeric", help="Force numeric kind without coercion; repeatable."
)
CATEGORICAL = typer.Option(
    None, "--categorical", help="Interpret as categorical; repeatable."
)
MIN_NUMERIC_N = typer.Option(3, "--min-numeric-n")
MAX_CATEGORY_LEVELS = typer.Option(50, "--max-category-levels")
MAX_MISSINGNESS_PATTERNS = typer.Option(20, "--max-missingness-patterns")
MAX_PAIRWISE_COLUMNS = typer.Option(200, "--max-pairwise-columns")
OUTPUT_DIR = typer.Option(
    None, "--output-dir", help="Write structured CSV/JSON outputs to this directory."
)

# --------------------------------------------------------------------------
# Shared inference / resampling controls
# --------------------------------------------------------------------------
RANDOM_STATE = typer.Option(0, "--random-state")
MAX_SHAPIRO_N = typer.Option(5000, "--max-shapiro-n")
PARTIAL_COVARIATE = typer.Option(
    None,
    "--partial-covariate",
    help="Numeric covariate for partial correlation; repeatable.",
)
DEPENDENCE_MIN_PAIRS = typer.Option(5, "--dependence-min-pairs")
DEPENDENCE_MAX_COLUMNS = typer.Option(50, "--dependence-max-columns")
DEPENDENCE_PERMUTATIONS = typer.Option(199, "--dependence-permutations")
MI_NEIGHBORS = typer.Option(3, "--mi-neighbors")
CONFIDENCE_LEVEL = typer.Option(0.95, "--confidence-level")
INTERVAL_CORRELATION = typer.Option(None, "--interval-correlation")
BOOTSTRAP_RESAMPLES = typer.Option(1000, "--bootstrap-resamples")
BOOTSTRAP_METHOD = typer.Option(BootstrapMethod.bca, "--bootstrap-method")

P_ADJUST = typer.Option(PAdjust.fdr_bh, "--p-adjust")
P_ADJUST_NONE = typer.Option(PAdjust.none, "--p-adjust")
GROUP = typer.Option(None, "--group", help="Grouping column; repeatable.")
MIN_GROUP_N = typer.Option(3, "--min-group-n")
MIN_GROUP_N_2 = typer.Option(2, "--min-group-n")
MAX_GROUP_LEVELS = typer.Option(20, "--max-group-levels")
MIN_CORRELATION_PAIRS = typer.Option(3, "--min-correlation-pairs")
MAX_CORRELATION_COLUMNS = typer.Option(100, "--max-correlation-columns")
PAIRWISE = typer.Option(
    False,  # noqa: FBT003
    "--pairwise",
    help="Enable pairwise post-hoc tests for multi-group comparisons.",
)
CORRELATION = typer.Option(
    None,
    "--correlation",
    help="Correlation method; repeatable. Defaults to all supported methods.",
)
COMPARISON_TEST = typer.Option(
    None,
    "--comparison-test",
    help="Inferential test; repeatable. Defaults to all supported tests.",
)

# --------------------------------------------------------------------------
# Feature-matrix inputs
# --------------------------------------------------------------------------
IDS_FILE = typer.Option(
    None, "--ids-file", help="One observation ID per line for NPY/NPZ inputs."
)
FEATURE_COLUMN = typer.Option(
    None, "--feature-column", help="Feature column for tabular inputs; repeatable."
)
FEATURE_INPUT = typer.Option(
    None,
    "--feature-input",
    help="Optional separate feature matrix for feature-space blocks.",
)
FEATURE_ID_COLUMN = typer.Option(None, "--feature-id-column")
FEATURE_IDS_FILE = typer.Option(None, "--feature-ids-file")
FEATURE_ALIGNMENT = typer.Option(Alignment.strict, "--feature-alignment")
SCALING = typer.Option(Scaling.none, "--scaling")

# --------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------
OUTLIER_METHOD = typer.Option(
    None,
    "--outlier-method",
    help="Outlier rule; repeatable. Defaults to IQR and robust Z.",
)
IQR_MULTIPLIER = typer.Option(1.5, "--iqr-multiplier")
ROBUST_Z_THRESHOLD = typer.Option(3.5, "--robust-z-threshold")
INCLUDE_FLAGS = typer.Option(
    False,  # noqa: FBT003
    "--include-flags",
    help="Persist observation-level flags in addition to variable summaries.",
)

# --------------------------------------------------------------------------
# Multivariate / Mahalanobis
# --------------------------------------------------------------------------
NO_SPEARMAN = typer.Option(False, "--no-spearman")  # noqa: FBT003
NO_ROBUST_MAHALANOBIS = typer.Option(False, "--no-robust-mahalanobis")  # noqa: FBT003
MAHALANOBIS_QUANTILE = typer.Option(0.975, "--mahalanobis-quantile")
ROBUST_SUPPORT_FRACTION = typer.Option(None, "--robust-support-fraction")
MAX_COVARIANCE_FEATURES = typer.Option(200, "--max-covariance-features")
MAX_COLLINEARITY_FEATURES = typer.Option(100, "--max-collinearity-features")
MAX_MAHALANOBIS_FEATURES = typer.Option(100, "--max-mahalanobis-features")

# --------------------------------------------------------------------------
# Factorial / model terms
# --------------------------------------------------------------------------
FORMULA = typer.Option(
    None,
    "--formula",
    help="Compact formula using direct column names and +, :, *, e.g. y ~ A * B + x.",
)
INTERACTION = typer.Option(
    None,
    "--interaction",
    help="Explicit interaction as colon-separated predictors; repeatable.",
)
SS_TYPE = typer.Option(SSType.two, "--ss-type")
ROBUST_COVARIANCE_OPTION = typer.Option(RobustCovariance.none, "--robust-covariance")
MIN_CELL_N = typer.Option(2, "--min-cell-n")
MAX_FACTOR_LEVELS = typer.Option(20, "--max-factor-levels")
MAX_DESIGN_CELLS = typer.Option(5000, "--max-design-cells")
MAX_DESIGN_COLUMNS = typer.Option(500, "--max-design-columns")
MAX_INTERACTION_ORDER = typer.Option(3, "--max-interaction-order")
DIAGNOSTIC_ALPHA = typer.Option(0.05, "--diagnostic-alpha")
CONDITION_NUMBER_THRESHOLD = typer.Option(30.0, "--condition-number-threshold")

# --------------------------------------------------------------------------
# PERMANOVA / post-hoc / marginal means / mixed effects
# --------------------------------------------------------------------------
PERMANOVA_FACTOR = typer.Option(None, "--permanova-factor")
PERMANOVA_METRIC = typer.Option("euclidean", "--permanova-metric")
PERMANOVA_PERMUTATIONS = typer.Option(999, "--permanova-permutations")
PERMANOVA_MIN_GROUP_N = typer.Option(3, "--permanova-min-group-n")
PERMANOVA_MAX_GROUP_LEVELS = typer.Option(20, "--permanova-max-group-levels")

POSTHOC_RESPONSE = typer.Option(None, "--posthoc-response")
POSTHOC_RESPONSE_REQUIRED = typer.Option(..., "--posthoc-response")
POSTHOC_FACTOR = typer.Option(None, "--posthoc-factor")
POSTHOC_FACTOR_REQUIRED = typer.Option(..., "--posthoc-factor")
POSTHOC_METHOD = typer.Option(None, "--posthoc-method")

MARGINAL_RESPONSE = typer.Option(None, "--marginal-response")
MARGINAL_FACTOR = typer.Option(None, "--marginal-factor")
MARGINAL_COVARIATE = typer.Option(None, "--marginal-covariate")
MARGINAL_INTERACTION = typer.Option(None, "--marginal-interaction")
MARGINAL_FORMULA = typer.Option(None, "--marginal-formula")
MARGINAL_TERM = typer.Option(None, "--marginal-term")
MARGINAL_P_ADJUST = typer.Option(PAdjust.fdr_bh, "--marginal-p-adjust")

MIXED_GROUP = typer.Option(None, "--mixed-group")
MIXED_GROUP_REQUIRED = typer.Option(..., "--mixed-group")
MIXED_RESPONSE = typer.Option(None, "--mixed-response")
MIXED_FACTOR = typer.Option(None, "--mixed-factor")
MIXED_COVARIATE = typer.Option(None, "--mixed-covariate")
MIXED_INTERACTION = typer.Option(None, "--mixed-interaction")
MIXED_FORMULA = typer.Option(None, "--mixed-formula")
MIXED_RANDOM_SLOPE = typer.Option(None, "--mixed-random-slope")
MIXED_ML = typer.Option(
    False,  # noqa: FBT003
    "--mixed-ml",
    help="Use ML instead of REML.",
)
MIXED_OPTIMIZER = typer.Option("lbfgs", "--mixed-optimizer")
MIXED_MAX_ITER = typer.Option(1000, "--mixed-max-iter")
MIXED_MIN_GROUPS = typer.Option(3, "--mixed-min-groups")
MIXED_MIN_GROUP_N = typer.Option(2, "--mixed-min-group-n")

# --------------------------------------------------------------------------
# Representation / compositional / bayesian / anomaly
# --------------------------------------------------------------------------
REPRESENTATION_ALIGNMENT = typer.Option(Alignment.strict, "--representation-alignment")
CCA_COMPONENTS = typer.Option(2, "--cca-components")
CCA_SCALING = typer.Option(Scaling.standard, "--cca-scaling")
CCA_MAX_ITER = typer.Option(1000, "--cca-max-iter")
CCA_TOL = typer.Option(1e-6, "--cca-tol")
DISTANCE_METRIC = typer.Option("euclidean", "--distance-metric")
DISTANCE_SIMILARITY_METHOD = typer.Option(
    SimilarityMethod.spearman, "--distance-similarity-method"
)
MANTEL_PERMUTATIONS = typer.Option(999, "--mantel-permutations")

COMPOSITIONAL_TRANSFORM = typer.Option(
    CompositionalTransform.clr, "--compositional-transform"
)
REPLACE_ZEROS = typer.Option(False, "--replace-zeros")  # noqa: FBT003
ZERO_REPLACEMENT_FRACTION = typer.Option(0.65, "--zero-replacement-fraction")
ALR_DENOMINATOR = typer.Option(-1, "--alr-denominator")

BAYESIAN_VARIABLE = typer.Option(None, "--bayesian-variable")
BAYESIAN_GROUP = typer.Option(None, "--bayesian-group")
BAYESIAN_CREDIBLE_LEVEL = typer.Option(0.95, "--bayesian-credible-level")
BAYESIAN_ROPE_LOW = typer.Option(-0.1, "--bayesian-rope-low")
BAYESIAN_ROPE_HIGH = typer.Option(0.1, "--bayesian-rope-high")
BAYESIAN_DRAWS = typer.Option(5000, "--bayesian-draws")
BAYESIAN_MIN_N = typer.Option(3, "--bayesian-min-n")

ANOMALY_METHOD = typer.Option(None, "--anomaly-method")
ANOMALY_SCALING = typer.Option(Scaling.none, "--anomaly-scaling")
CONTAMINATION = typer.Option("auto", "--contamination")
ISOLATION_ESTIMATORS = typer.Option(200, "--isolation-estimators")
LOF_NEIGHBORS = typer.Option(20, "--lof-neighbors")

# --------------------------------------------------------------------------
# Projections
# --------------------------------------------------------------------------
N_COMPONENTS = typer.Option(2, "--n-components")
METRIC = typer.Option("euclidean", "--metric")
PERPLEXITY = typer.Option(30.0, "--perplexity")
MAX_ITER = typer.Option(1000, "--max-iter")
N_NEIGHBORS = typer.Option(15, "--n-neighbors")
MIN_DIST = typer.Option(0.1, "--min-dist")

# --------------------------------------------------------------------------
# External annotation sources (``ruddy analyze groups``)
# --------------------------------------------------------------------------
ANNOTATION_FILE = typer.Option(
    None, "--annotation-file", help="External annotation table to align by ID."
)
ANNOTATION_NAME = typer.Option("annotations", "--annotation-name")
ANNOTATION_ID_COLUMN = typer.Option(None, "--annotation-id-column")
ANNOTATION_ALIGNMENT = typer.Option(Alignment.strict, "--annotation-alignment")
ANNOTATION_FACTOR = typer.Option(None, "--annotation-factor")
ANNOTATION_RESPONSE = typer.Option(None, "--annotation-response")
ANNOTATION_COVARIATE = typer.Option(None, "--annotation-covariate")
ANNOTATION_NUMERIC = typer.Option(None, "--annotation-numeric")
ANNOTATION_CATEGORICAL = typer.Option(None, "--annotation-categorical")

# --------------------------------------------------------------------------
# Paired representation spaces (``ruddy analyze representation``)
# --------------------------------------------------------------------------
REPRESENTATION_X_ARGUMENT = typer.Argument(
    ..., metavar="INPUT_X", help="First representation matrix."
)
REPRESENTATION_Y_ARGUMENT = typer.Argument(
    ..., metavar="INPUT_Y", help="Second representation matrix."
)
X_ID_COLUMN = typer.Option(None, "--x-id-column")
Y_ID_COLUMN = typer.Option(None, "--y-id-column")
X_IDS_FILE = typer.Option(None, "--x-ids-file")
Y_IDS_FILE = typer.Option(None, "--y-ids-file")
X_FEATURE_COLUMN = typer.Option(None, "--x-feature-column")
Y_FEATURE_COLUMN = typer.Option(None, "--y-feature-column")

# --------------------------------------------------------------------------
# ``ruddy analyze run`` block selection and block-scoped overrides
# --------------------------------------------------------------------------
ENABLE = typer.Option(
    None,
    "--enable",
    help=(
        "Top-level analysis block to execute; repeatable. "
        "Defaults to profiling + univariate."
    ),
)
PROJECTION_N_COMPONENTS = typer.Option(2, "--projection-n-components")
PROJECTION_SCALING = typer.Option(Scaling.none, "--projection-scaling")
PROJECTION_METRIC = typer.Option("euclidean", "--projection-metric")
TSNE_PERPLEXITY = typer.Option(30.0, "--tsne-perplexity")
TSNE_MAX_ITER = typer.Option(1000, "--tsne-max-iter")
UMAP_N_NEIGHBORS = typer.Option(15, "--umap-n-neighbors")
UMAP_MIN_DIST = typer.Option(0.1, "--umap-min-dist")
MULTIVARIATE_SCALING = typer.Option(Scaling.none, "--multivariate-scaling")
MANOVA_MAX_RESPONSES = typer.Option(20, "--manova-max-responses")
MANOVA_MAX_FACTOR_LEVELS = typer.Option(20, "--manova-max-factor-levels")
MANOVA_MIN_LEVEL_N = typer.Option(3, "--manova-min-level-n")
MAX_RESPONSES = typer.Option(20, "--max-responses")
MIN_LEVEL_N = typer.Option(3, "--min-level-n")
FACTORIAL_RESPONSE = typer.Option(None, "--factorial-response")
FACTORIAL_FORMULA = typer.Option(None, "--factorial-formula")
FACTORIAL_P_ADJUST = typer.Option(PAdjust.none, "--factorial-p-adjust")
COMPARISON_FEATURE_INPUT = typer.Option(None, "--comparison-feature-input")
COMPARISON_FEATURE_ID_COLUMN = typer.Option(None, "--comparison-feature-id-column")
COMPARISON_FEATURE_IDS_FILE = typer.Option(None, "--comparison-feature-ids-file")
COMPARISON_FEATURE_COLUMN = typer.Option(None, "--comparison-feature-column")
