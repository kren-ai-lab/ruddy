"""Factorial design specification, safe formula parsing, and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from ruddy.core.enums import ColumnKind, ColumnRole

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ruddy.data import TabularDataset

_SIMPLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _unique(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} cannot contain duplicates.")
    return result


def _normalize_ss_type(value: int | str) -> int:
    if isinstance(value, bool):
        raise TypeError("ss_type must be 2/II or 3/III.")
    if isinstance(value, int):
        if value in {2, 3}:
            return value
        raise ValueError("ss_type must be 2 or 3.")
    normalized = str(value).strip().lower().replace("type", "").replace("_", "").replace("-", "")
    mapping = {"2": 2, "ii": 2, "3": 3, "iii": 3}
    if normalized not in mapping:
        raise ValueError("ss_type must be 2/II or 3/III.")
    return mapping[normalized]


@dataclass(frozen=True, slots=True)
class FactorialTerm:
    """One main or interaction term in a hierarchical factorial design."""

    columns: tuple[str, ...]
    kinds: tuple[str, ...]

    @property
    def order(self) -> int:
        """Return the interaction order (number of interacting columns)."""
        return len(self.columns)

    @property
    def label(self) -> str:
        """Return the formula string representation of the term."""
        return ":".join(self.columns)

    @property
    def term_type(self) -> str:
        """Return the statistical classification of the term."""
        if self.order == 1:
            return self.kinds[0]
        return "interaction"


@dataclass(frozen=True, slots=True)
class FactorialDesign:
    """Resolved, domain-agnostic factorial model specification."""

    response: str
    factors: tuple[str, ...]
    covariates: tuple[str, ...]
    interactions: tuple[tuple[str, ...], ...]
    terms: tuple[FactorialTerm, ...]
    requested_formula: str | None
    resolved_formula: str
    ss_type: int

    @property
    def predictors(self) -> tuple[str, ...]:
        """Return all factor and covariate predictor column names."""
        return (*self.factors, *self.covariates)


def _infer_predictor_role(dataset: TabularDataset, column: str) -> str:
    role = dataset.role_of(column)
    kind = dataset.kind_of(column)
    if role is ColumnRole.FACTOR or kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}:
        return "factor"
    if kind is ColumnKind.NUMERIC:
        return "covariate"
    raise ValueError(
        f"Factorial predictor {column!r} must be numeric, categorical/boolean, "
        "or explicitly declared as a factor."
    )


def _parse_formula(
    dataset: TabularDataset,
    formula: str,
    *,
    max_interaction_order: int,
) -> tuple[str, tuple[str, ...], tuple[tuple[str, ...], ...]]:
    if formula.count("~") != 1:
        raise ValueError("factorial formula must contain exactly one '~'.")
    lhs, rhs = (part.strip() for part in formula.split("~", 1))
    if not lhs or not rhs:
        raise ValueError("factorial formula requires a response and at least one predictor.")
    if not _SIMPLE_NAME.fullmatch(lhs):
        raise ValueError(
            "Formula syntax accepts simple column names only; use the programmatic API "
            "for column names containing spaces or operators."
        )
    if lhs not in dataset.frame.columns:
        raise ValueError(f"Unknown factorial response column: {lhs!r}.")

    atomic: list[str] = []
    interactions: list[tuple[str, ...]] = []

    def add_atomic(name: str) -> None:
        if not _SIMPLE_NAME.fullmatch(name):
            raise ValueError(
                f"Formula syntax accepts only direct column names and '+', ':', '*'; invalid token={name!r}."
            )
        if name not in dataset.frame.columns:
            raise ValueError(f"Unknown factorial predictor column: {name!r}.")
        if name == lhs:
            raise ValueError("The response cannot also appear as a factorial predictor.")
        if name not in atomic:
            atomic.append(name)

    raw_terms = [term.strip() for term in rhs.split("+")]
    if any(not term for term in raw_terms):
        raise ValueError("factorial formula contains an empty RHS term.")
    for term in raw_terms:
        if "(" in term or ")" in term or "/" in term or "-" in term:
            raise ValueError(
                "Ruddy factorial formulas intentionally exclude transforms/functions; "
                "provide transformed variables explicitly as columns."
            )
        if "*" in term:
            if ":" in term:
                raise ValueError("Do not mix '*' and ':' inside one factorial formula term.")
            names = tuple(part.strip() for part in term.split("*"))
            if len(names) < 2 or any(not name for name in names):
                raise ValueError(f"Invalid factorial expansion term: {term!r}.")
            if len(names) > max_interaction_order:
                raise ValueError(
                    f"Interaction order {len(names)} exceeds max_interaction_order={max_interaction_order}."
                )
            for name in names:
                add_atomic(name)
            for order in range(2, len(names) + 1):
                for combo in combinations(names, order):
                    if combo not in interactions:
                        interactions.append(combo)
        elif ":" in term:
            names = tuple(part.strip() for part in term.split(":"))
            if len(names) < 2 or any(not name for name in names):
                raise ValueError(f"Invalid factorial interaction term: {term!r}.")
            if len(names) > max_interaction_order:
                raise ValueError(
                    f"Interaction order {len(names)} exceeds max_interaction_order={max_interaction_order}."
                )
            for name in names:
                add_atomic(name)
            if names not in interactions:
                interactions.append(names)
        else:
            add_atomic(term)

    return lhs, tuple(atomic), tuple(interactions)


def build_factorial_design(
    dataset: TabularDataset,
    *,
    response: str | None = None,
    factors: Iterable[str] = (),
    covariates: Iterable[str] = (),
    interactions: Iterable[Iterable[str]] = (),
    formula: str | None = None,
    ss_type: int | str = 2,
    max_interaction_order: int = 3,
) -> FactorialDesign:
    """Resolve a hierarchical ANOVA/ANCOVA design without evaluating arbitrary code.

    The compact formula syntax supports direct column names, ``+``, ``:`` and ``*``.
    Atomic predictors are always included as main effects, so interactions retain the
    usual hierarchical interpretation. For unusual column names, use the explicit
    ``response``/``factors``/``covariates``/``interactions`` arguments.
    """
    if max_interaction_order < 2:
        raise ValueError("max_interaction_order must be at least 2.")
    resolved_ss = _normalize_ss_type(ss_type)

    requested_formula: str | None = None
    if formula is not None:
        requested_formula = str(formula).strip()
        if response is not None or tuple(factors) or tuple(covariates) or tuple(interactions):
            raise ValueError(
                "Use either formula=... or explicit response/factors/covariates/interactions, not both."
            )
        response_name, predictors, parsed_interactions = _parse_formula(
            dataset, requested_formula, max_interaction_order=max_interaction_order
        )
        factor_names = tuple(name for name in predictors if _infer_predictor_role(dataset, name) == "factor")
        covariate_names = tuple(
            name for name in predictors if _infer_predictor_role(dataset, name) == "covariate"
        )
        interaction_names = parsed_interactions
    else:
        if response is None:
            responses = dataset.columns_with_role(ColumnRole.RESPONSE)
            if len(responses) != 1:
                raise ValueError("Specify response=... unless exactly one column has the response role.")
            response_name = responses[0]
        else:
            response_name = str(response)
        factor_names = _unique(factors, label="factors")
        covariate_names = _unique(covariates, label="covariates")
        if not factor_names and not covariate_names:
            factor_names = dataset.columns_with_role(ColumnRole.FACTOR)
            covariate_names = dataset.columns_with_role(ColumnRole.COVARIATE)
        interaction_names = tuple(tuple(str(value) for value in term) for term in interactions)

    if response_name not in dataset.frame.columns:
        raise ValueError(f"Unknown factorial response column: {response_name!r}.")
    if dataset.kind_of(response_name) is not ColumnKind.NUMERIC:
        raise ValueError("Factorial ANOVA/ANCOVA requires a numeric response.")
    if dataset.role_of(response_name) in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED}:
        raise ValueError("The factorial response cannot be an identifier/excluded column.")

    if len(set(factor_names)) != len(factor_names):
        raise ValueError("factors cannot contain duplicates.")
    if len(set(covariate_names)) != len(covariate_names):
        raise ValueError("covariates cannot contain duplicates.")
    overlap = set(factor_names) & set(covariate_names)
    if overlap:
        raise ValueError(f"Columns cannot be both factors and covariates: {sorted(overlap)}.")
    if response_name in set(factor_names) | set(covariate_names):
        raise ValueError("The response cannot also be a predictor.")

    unknown = [name for name in (*factor_names, *covariate_names) if name not in dataset.frame.columns]
    if unknown:
        raise ValueError(f"Unknown factorial predictor columns: {unknown}.")
    for factor in factor_names:
        role = dataset.role_of(factor)
        kind = dataset.kind_of(factor)
        if role is not ColumnRole.FACTOR and kind not in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}:
            raise ValueError(
                f"Factor {factor!r} must be categorical/boolean or explicitly assigned the factor role."
            )
    for covariate in covariate_names:
        if dataset.kind_of(covariate) is not ColumnKind.NUMERIC:
            raise ValueError(f"Covariate {covariate!r} must be numeric.")
        if dataset.role_of(covariate) is ColumnRole.FACTOR:
            raise ValueError(f"Factor-role column {covariate!r} cannot be used as a numeric covariate.")

    predictor_order = (*factor_names, *covariate_names)
    predictor_set = set(predictor_order)
    normalized_interactions: list[tuple[str, ...]] = []
    for raw in interaction_names:
        term = tuple(raw)
        if len(term) < 2:
            raise ValueError("Each factorial interaction requires at least two predictors.")
        if len(term) > max_interaction_order:
            raise ValueError(f"Interaction {term} exceeds max_interaction_order={max_interaction_order}.")
        if len(set(term)) != len(term):
            raise ValueError(f"Interaction cannot repeat a predictor: {term}.")
        missing = [name for name in term if name not in predictor_set]
        if missing:
            raise ValueError(
                f"Interaction predictors must also be declared as main effects; missing={missing}."
            )
        canonical = tuple(sorted(term, key=predictor_order.index))
        if canonical not in normalized_interactions:
            normalized_interactions.append(canonical)

    # For interactions above order two, retain strong hierarchy by including every
    # lower-order interaction among their constituent predictors.
    hierarchical: list[tuple[str, ...]] = []
    for term in normalized_interactions:
        for order in range(2, len(term) + 1):
            for combo in combinations(term, order):
                canonical = tuple(sorted(combo, key=predictor_order.index))
                if canonical not in hierarchical:
                    hierarchical.append(canonical)

    factor_set = set(factor_names)
    terms: list[FactorialTerm] = [
        FactorialTerm(
            columns=(name,),
            kinds=("factor" if name in factor_set else "covariate",),
        )
        for name in predictor_order
    ]
    terms.extend(
        FactorialTerm(
            columns=term,
            kinds=tuple("factor" if name in factor_set else "covariate" for name in term),
        )
        for term in hierarchical
    )

    if not terms:
        raise ValueError("Factorial analysis requires at least one factor or covariate.")
    rhs = " + ".join(term.label for term in terms)
    resolved_formula = f"{response_name} ~ {rhs}"
    return FactorialDesign(
        response=response_name,
        factors=tuple(factor_names),
        covariates=tuple(covariate_names),
        interactions=tuple(hierarchical),
        terms=tuple(terms),
        requested_formula=requested_formula,
        resolved_formula=resolved_formula,
        ss_type=resolved_ss,
    )


__all__ = [
    "FactorialDesign",
    "FactorialTerm",
    "build_factorial_design",
]
