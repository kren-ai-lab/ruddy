"""Stable tabular result schemas."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ruddy.core.enums import ResultStatus
from ruddy.core.exceptions import ResultContractError

RESULT_STATUS_COLUMN = "status"
RESULT_REASON_COLUMN = "reason"


def validate_result_table(
    frame: pd.DataFrame,
    *,
    required_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate a result table and return a defensive copy.

    Analysis-specific schemas will extend this small global contract in later phases.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Result tables must be pandas DataFrames.")

    required = {RESULT_STATUS_COLUMN, RESULT_REASON_COLUMN, *required_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ResultContractError(
            f"Result table is missing required columns: {missing}."
        )

    valid_statuses = {status.value for status in ResultStatus}
    observed = set(frame[RESULT_STATUS_COLUMN].dropna().astype(str))
    invalid = sorted(observed - valid_statuses)
    if invalid:
        raise ResultContractError(f"Result table contains invalid statuses: {invalid}.")

    status = frame[RESULT_STATUS_COLUMN].astype(str)
    reason = frame[RESULT_REASON_COLUMN]
    bad_ok = (status == ResultStatus.OK.value) & reason.notna()
    bad_non_ok = status.isin(
        [ResultStatus.DEGENERATE.value, ResultStatus.SKIPPED.value]
    ) & reason.isna()
    if bad_ok.any() or bad_non_ok.any():
        raise ResultContractError(
            "Result table violates status/reason consistency: OK rows require no "
            "reason "
            "and degenerate/skipped rows require an explicit reason."
        )

    return frame.copy(deep=True)
