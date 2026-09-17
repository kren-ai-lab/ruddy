"""Stable tabular result schemas."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import polars as pl

from ruddy.core.enums import ResultStatus
from ruddy.core.exceptions import ResultContractError

RESULT_STATUS_COLUMN = "status"
RESULT_REASON_COLUMN = "reason"


def validate_result_table(
    frame: pl.DataFrame | pd.DataFrame,
    *,
    required_columns: Iterable[str] = (),
) -> pl.DataFrame | pd.DataFrame:
    """Validate a result table and return it (a defensive copy for pandas).

    Analysis-specific tables declare their own columns through ``required_columns``.
    """
    if isinstance(frame, pl.DataFrame):
        return _validate_polars_result_table(frame, required_columns=required_columns)
    # ponytail: temporary pandas adapter, removed in phase 5
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Result tables must be Polars or pandas DataFrames.")

    required = {RESULT_STATUS_COLUMN, RESULT_REASON_COLUMN, *required_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ResultContractError(f"Result table is missing required columns: {missing}.")

    valid_statuses = {status.value for status in ResultStatus}
    observed = set(frame[RESULT_STATUS_COLUMN].dropna().astype(str))
    invalid = sorted(observed - valid_statuses)
    if invalid:
        raise ResultContractError(f"Result table contains invalid statuses: {invalid}.")

    status = frame[RESULT_STATUS_COLUMN].astype(str)
    reason = frame[RESULT_REASON_COLUMN]
    bad_ok = (status == ResultStatus.OK.value) & reason.notna()
    bad_non_ok = status.isin([ResultStatus.DEGENERATE.value, ResultStatus.SKIPPED.value]) & reason.isna()
    if bad_ok.any() or bad_non_ok.any():
        raise ResultContractError(
            "Result table violates status/reason consistency: OK rows require no "
            "reason "
            "and degenerate/skipped rows require an explicit reason."
        )

    return frame.copy(deep=True)


def _validate_polars_result_table(frame: pl.DataFrame, *, required_columns: Iterable[str]) -> pl.DataFrame:
    required = {RESULT_STATUS_COLUMN, RESULT_REASON_COLUMN, *required_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ResultContractError(f"Result table is missing required columns: {missing}.")

    status = frame.get_column(RESULT_STATUS_COLUMN).cast(pl.String)
    reason = frame.get_column(RESULT_REASON_COLUMN)
    valid_statuses = {item.value for item in ResultStatus}
    invalid = sorted(set(status.drop_nulls().to_list()) - valid_statuses)
    if invalid:
        raise ResultContractError(f"Result table contains invalid statuses: {invalid}.")

    non_ok = [ResultStatus.DEGENERATE.value, ResultStatus.SKIPPED.value]
    bad_ok = (status == ResultStatus.OK.value) & reason.is_not_null()
    bad_non_ok = status.is_in(non_ok) & reason.is_null()
    if bad_ok.any() or bad_non_ok.any():
        raise ResultContractError(
            "Result table violates status/reason consistency: OK rows require no "
            "reason "
            "and degenerate/skipped rows require an explicit reason."
        )
    return frame
