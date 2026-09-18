"""Unified analysis orchestration engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import polars as pl

from ruddy.analysis.config import AnalysisConfig
from ruddy.analysis.intervals import analyze_confidence_intervals
from ruddy.analysis.results import UnifiedAnalysisResult
from ruddy.anomaly import analyze_anomalies
from ruddy.bayesian import analyze_bayesian_eda
from ruddy.bivariate import (
    analyze_bivariate,
    analyze_contingency_diagnostics,
    analyze_dependence,
    analyze_grouped_responses,
    analyze_posthoc,
)
from ruddy.compositional import analyze_composition
from ruddy.core.enums import AnalysisBlock, ColumnRole
from ruddy.data import AlignedAnnotations, FeatureMatrix, TabularDataset
from ruddy.data.validation import align_annotations
from ruddy.factorial import analyze_factorial, analyze_marginal_means, analyze_mixed_effects
from ruddy.multivariate import analyze_manova, analyze_multivariate, analyze_permutation_group_structure
from ruddy.profiling import profile_dataset
from ruddy.projections import analyze_pca, analyze_tsne, analyze_umap
from ruddy.representation import analyze_representation_similarity
from ruddy.results import AnalysisProvenance
from ruddy.univariate import analyze_distribution_diagnostics, analyze_outliers, analyze_univariate

if TYPE_CHECKING:
    from collections.abc import Iterable


def _factor_columns(dataset: TabularDataset, config: AnalysisConfig) -> tuple[str, ...]:
    if config.groups:
        return config.groups
    return dataset.columns_with_role(ColumnRole.FACTOR)


def _covariate_columns(dataset: TabularDataset) -> tuple[str, ...]:
    return dataset.columns_with_role(ColumnRole.COVARIATE)


def analyze(
    dataset: TabularDataset,
    *,
    config: AnalysisConfig | None = None,
    features: FeatureMatrix | None = None,
    comparison_features: FeatureMatrix | None = None,
    annotations: Iterable[AlignedAnnotations] = (),
) -> UnifiedAnalysisResult:
    """Execute only the top-level analysis blocks explicitly enabled in ``config``.

    The orchestrator contains no new statistical logic. Every component delegates to
    the same standalone public function used outside the unified pipeline.
    """
    if not isinstance(dataset, TabularDataset):
        msg = "dataset must be a TabularDataset."
        raise TypeError(msg)
    if features is not None and not isinstance(features, FeatureMatrix):
        msg = "features must be a FeatureMatrix or None."
        raise TypeError(msg)
    if comparison_features is not None and not isinstance(comparison_features, FeatureMatrix):
        msg = "comparison_features must be a FeatureMatrix or None."
        raise TypeError(msg)

    cfg = AnalysisConfig() if config is None else config
    # __post_init__ has already normalised every entry to an AnalysisBlock.
    blocks = cast("tuple[AnalysisBlock, ...]", cfg.enabled_blocks)
    annotation_sources = tuple(annotations)
    components: dict[str, Any] = {}

    feature_alignment = None
    if features is not None:
        _, feature_alignment = align_annotations(
            dataset.observation_ids,
            pl.DataFrame({"__id": features.observation_ids}),
            id_column="__id",
            mode=cfg.feature_alignment,
        )

    if AnalysisBlock.PROFILING in blocks:
        components["profiling"] = profile_dataset(dataset, **cfg.profiling_kwargs())

    if AnalysisBlock.UNIVARIATE in blocks:
        components["univariate"] = analyze_univariate(dataset, **cfg.univariate_kwargs())

    if AnalysisBlock.BIVARIATE in blocks:
        components["bivariate"] = analyze_bivariate(dataset, **cfg.bivariate_kwargs())

    if AnalysisBlock.GROUPS in blocks:
        components["groups"] = analyze_grouped_responses(
            dataset,
            annotations=annotation_sources,
            **cfg.grouped_kwargs(),
        )

    if AnalysisBlock.OUTLIERS in blocks:
        components["outliers"] = analyze_outliers(dataset, **cfg.outlier_kwargs())

    if AnalysisBlock.DIAGNOSTICS in blocks:
        components["diagnostics"] = analyze_distribution_diagnostics(
            dataset, **cfg.distribution_diagnostics_kwargs()
        )

    if AnalysisBlock.DEPENDENCE in blocks:
        components["dependence"] = analyze_dependence(dataset, **cfg.dependence_kwargs())

    if AnalysisBlock.CONTINGENCY in blocks:
        components["contingency"] = analyze_contingency_diagnostics(dataset, **cfg.contingency_kwargs())

    if AnalysisBlock.INTERVALS in blocks:
        components["intervals"] = analyze_confidence_intervals(dataset, **cfg.interval_kwargs())

    feature_blocks = {
        AnalysisBlock.PCA,
        AnalysisBlock.TSNE,
        AnalysisBlock.UMAP,
        AnalysisBlock.MULTIVARIATE,
        AnalysisBlock.PERMANOVA,
        AnalysisBlock.COMPOSITIONAL,
        AnalysisBlock.ANOMALY,
        AnalysisBlock.REPRESENTATION,
    }
    requested_feature_blocks = tuple(block for block in blocks if block in feature_blocks)
    if requested_feature_blocks and features is None:
        names = ", ".join(block.value for block in requested_feature_blocks)
        msg = f"FeatureMatrix is required for enabled feature-space blocks: {names}."
        raise ValueError(msg)
    # The guard above makes this non-None for every feature-space block below.
    feature_space = cast("FeatureMatrix", features)

    if AnalysisBlock.PCA in blocks:
        components["pca"] = analyze_pca(feature_space, **cfg.pca_kwargs())
    if AnalysisBlock.TSNE in blocks:
        components["tsne"] = analyze_tsne(feature_space, **cfg.tsne_kwargs())
    if AnalysisBlock.UMAP in blocks:
        components["umap"] = analyze_umap(feature_space, **cfg.umap_kwargs())
    if AnalysisBlock.MULTIVARIATE in blocks:
        components["multivariate"] = analyze_multivariate(
            feature_space,
            **cfg.multivariate_kwargs(),
        )
    if AnalysisBlock.COMPOSITIONAL in blocks:
        components["compositional"] = analyze_composition(
            feature_space,
            **cfg.compositional_kwargs(),
        )
    if AnalysisBlock.ANOMALY in blocks:
        components["anomaly"] = analyze_anomalies(
            feature_space,
            **cfg.anomaly_kwargs(),
        )
    if AnalysisBlock.REPRESENTATION in blocks:
        if comparison_features is None:
            msg = "Representation block requires comparison_features."
            raise ValueError(msg)
        components["representation"] = analyze_representation_similarity(
            feature_space,
            comparison_features,
            **cfg.representation_kwargs(),
        )

    factors = _factor_columns(dataset, cfg)
    covariates = _covariate_columns(dataset)

    if AnalysisBlock.PERMANOVA in blocks:
        factor = cfg.permanova_factor
        if factor is None:
            candidates = cfg.groups or factors
            if len(candidates) != 1:
                msg = "PERMANOVA block requires permanova_factor or exactly one configured group/factor."
                raise ValueError(msg)
            factor = candidates[0]
        components["permanova"] = analyze_permutation_group_structure(
            feature_space,
            dataset,
            factor=factor,
            **cfg.permanova_kwargs(),
        )

    if AnalysisBlock.MANOVA in blocks:
        responses = cfg.manova_responses or cfg.responses
        manova_factors = cfg.manova_factors or factors
        manova_covariates = cfg.manova_covariates or covariates
        if len(responses) < 2:
            msg = "MANOVA block requires at least two configured responses."
            raise ValueError(msg)
        if not manova_factors:
            msg = "MANOVA block requires at least one configured factor."
            raise ValueError(msg)
        components["manova"] = analyze_manova(
            dataset,
            responses=responses,
            factors=manova_factors,
            covariates=manova_covariates,
            **cfg.manova_kwargs(),
        )

    if AnalysisBlock.FACTORIAL in blocks:
        if cfg.factorial_formula is not None:
            components["factorial"] = analyze_factorial(
                dataset,
                formula=cfg.factorial_formula,
                **cfg.factorial_kwargs(),
            )
        else:
            response = cfg.factorial_response
            if response is None:
                if len(cfg.responses) != 1:
                    msg = (
                        "Factorial block requires factorial_response, factorial_formula, "
                        "or exactly one configured response."
                    )
                    raise ValueError(msg)
                response = cfg.responses[0]
            factorial_factors = cfg.factorial_factors or factors
            factorial_covariates = cfg.factorial_covariates or covariates
            if not factorial_factors and not factorial_covariates:
                msg = "Factorial block requires at least one factor or covariate."
                raise ValueError(msg)
            components["factorial"] = analyze_factorial(
                dataset,
                response=response,
                factors=factorial_factors,
                covariates=factorial_covariates,
                interactions=cfg.factorial_interactions,
                **cfg.factorial_kwargs(),
            )

    if AnalysisBlock.POSTHOC in blocks:
        response = cfg.posthoc_response
        if response is None:
            if len(cfg.responses) != 1:
                msg = "Posthoc block requires posthoc_response or exactly one configured response."
                raise ValueError(msg)
            response = cfg.responses[0]
        factor = cfg.posthoc_factor
        if factor is None:
            candidates = cfg.groups or factors
            if len(candidates) != 1:
                msg = "Posthoc block requires posthoc_factor or exactly one configured group/factor."
                raise ValueError(msg)
            factor = candidates[0]
        components["posthoc"] = analyze_posthoc(
            dataset, response=response, factor=factor, **cfg.posthoc_kwargs()
        )

    if AnalysisBlock.MARGINAL_MEANS in blocks:
        if cfg.marginal_formula is not None:
            components["marginal_means"] = analyze_marginal_means(
                dataset,
                formula=cfg.marginal_formula,
                terms=cfg.marginal_terms or None,
                **cfg.marginal_means_kwargs(),
            )
        else:
            response = cfg.marginal_response
            if response is None:
                if len(cfg.responses) != 1:
                    msg = (
                        "Marginal-means block requires marginal_response, marginal_formula, "
                        "or exactly one configured response."
                    )
                    raise ValueError(msg)
                response = cfg.responses[0]
            mm_factors = cfg.marginal_factors or factors
            mm_covariates = cfg.marginal_covariates or covariates
            if not mm_factors:
                msg = "Marginal-means block requires at least one factor."
                raise ValueError(msg)
            components["marginal_means"] = analyze_marginal_means(
                dataset,
                response=response,
                factors=mm_factors,
                covariates=mm_covariates,
                interactions=cfg.marginal_interactions,
                terms=cfg.marginal_terms or None,
                **cfg.marginal_means_kwargs(),
            )

    if AnalysisBlock.MIXED_EFFECTS in blocks:
        if cfg.mixed_group is None:
            msg = "Mixed-effects block requires mixed_group."
            raise ValueError(msg)
        if cfg.mixed_formula is not None:
            components["mixed_effects"] = analyze_mixed_effects(
                dataset, group=cfg.mixed_group, formula=cfg.mixed_formula, **cfg.mixed_effects_kwargs()
            )
        else:
            response = cfg.mixed_response
            if response is None:
                if len(cfg.responses) != 1:
                    msg = (
                        "Mixed-effects block requires mixed_response, mixed_formula, "
                        "or exactly one configured response."
                    )
                    raise ValueError(msg)
                response = cfg.responses[0]
            mixed_factors = cfg.mixed_factors or factors
            mixed_covariates = cfg.mixed_covariates or covariates
            if not mixed_factors and not mixed_covariates:
                msg = "Mixed-effects block requires at least one fixed factor or covariate."
                raise ValueError(msg)
            components["mixed_effects"] = analyze_mixed_effects(
                dataset,
                group=cfg.mixed_group,
                response=response,
                factors=mixed_factors,
                covariates=mixed_covariates,
                interactions=cfg.mixed_interactions,
                **cfg.mixed_effects_kwargs(),
            )

    if AnalysisBlock.BAYESIAN in blocks:
        components["bayesian"] = analyze_bayesian_eda(dataset, **cfg.bayesian_kwargs())

    provenance = AnalysisProvenance(
        analysis="unified",
        parameters={
            "enabled_blocks": tuple(block.value for block in blocks),
            "feature_matrix_supplied": features is not None,
            "comparison_feature_matrix_supplied": comparison_features is not None,
            "annotation_sources": len(annotation_sources),
            "random_state": cfg.random_state,
        },
        input_summary={
            "n_observations": dataset.frame.height,
            "n_columns": dataset.frame.width,
            "id_column": dataset.id_column,
            "feature_shape": None if features is None else features.shape,
            "comparison_feature_shape": None if comparison_features is None else comparison_features.shape,
        },
        random_state=cfg.random_state,
    )
    return UnifiedAnalysisResult(
        profiling=components.get("profiling"),
        univariate=components.get("univariate"),
        bivariate=components.get("bivariate"),
        groups=components.get("groups"),
        outliers=components.get("outliers"),
        pca=components.get("pca"),
        tsne=components.get("tsne"),
        umap=components.get("umap"),
        multivariate=components.get("multivariate"),
        manova=components.get("manova"),
        factorial=components.get("factorial"),
        diagnostics=components.get("diagnostics"),
        dependence=components.get("dependence"),
        contingency=components.get("contingency"),
        intervals=components.get("intervals"),
        permanova=components.get("permanova"),
        posthoc=components.get("posthoc"),
        marginal_means=components.get("marginal_means"),
        mixed_effects=components.get("mixed_effects"),
        representation=components.get("representation"),
        compositional=components.get("compositional"),
        bayesian=components.get("bayesian"),
        anomaly=components.get("anomaly"),
        feature_alignment=feature_alignment,
        executed_blocks=blocks,
        provenance=provenance,
    )
