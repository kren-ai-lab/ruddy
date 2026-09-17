"""Stable tabular result schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ruddy.core.enums import ResultStatus
from ruddy.core.exceptions import ResultContractError

if TYPE_CHECKING:
    from collections.abc import Iterable

RESULT_STATUS_COLUMN = "status"
RESULT_REASON_COLUMN = "reason"


def validate_result_table(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate a result table and return a defensive copy.

    Analysis-specific tables declare their own columns through ``required_columns``.
    """
    if not isinstance(frame, pd.DataFrame):
        msg = "Result tables must be pandas DataFrames."
        raise TypeError(msg)

    required = {RESULT_STATUS_COLUMN, RESULT_REASON_COLUMN, *required_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        msg = f"Result table is missing required columns: {missing}."
        raise ResultContractError(msg)

    valid_statuses = {status.value for status in ResultStatus}
    observed = set(frame[RESULT_STATUS_COLUMN].dropna().astype(str))
    invalid = sorted(observed - valid_statuses)
    if invalid:
        msg = f"Result table contains invalid statuses: {invalid}."
        raise ResultContractError(msg)

    status = frame[RESULT_STATUS_COLUMN].astype(str)
    reason = frame[RESULT_REASON_COLUMN]
    bad_ok = (status == ResultStatus.OK.value) & reason.notna()
    bad_non_ok = status.isin([ResultStatus.DEGENERATE.value, ResultStatus.SKIPPED.value]) & reason.isna()
    if bad_ok.any() or bad_non_ok.any():
        msg = (
            "Result table violates status/reason consistency: OK rows require no "
            "reason "
            "and degenerate/skipped rows require an explicit reason."
        )
        raise ResultContractError(msg)

    return frame.copy(deep=True)
