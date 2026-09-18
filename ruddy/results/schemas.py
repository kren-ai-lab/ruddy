"""Stable tabular result schemas."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from ruddy.core.enums import ResultStatus
from ruddy.core.exceptions import ResultContractError

RESULT_STATUS_COLUMN = "status"
RESULT_REASON_COLUMN = "reason"


def validate_result_table(
    frame: pl.DataFrame,
    *,
    required_columns: Iterable[str] = (),
) -> pl.DataFrame:
    """Validate a result table and return it.

    Analysis-specific tables declare their own columns through ``required_columns``.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("Result tables must be Polars DataFrames.")

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
