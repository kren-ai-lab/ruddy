from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from ruddy import Advisory, AnalysisProvenance, AnalysisResult, ResultStatus
from ruddy.core.exceptions import ResultContractError
from ruddy.results import validate_result_table


def test_global_result_status_contract() -> None:
    assert AnalysisResult.ok().status is ResultStatus.OK
    assert AnalysisResult.degenerate("constant").reason == "constant"
    assert AnalysisResult.skipped("not_applicable").status is ResultStatus.SKIPPED


def test_non_ok_result_requires_explicit_reason() -> None:
    with pytest.raises(ResultContractError):
        AnalysisResult(status=ResultStatus.DEGENERATE)


def test_ok_result_rejects_degeneracy_reason() -> None:
    with pytest.raises(ResultContractError):
        AnalysisResult(status=ResultStatus.OK, reason="constant")


def test_advisories_and_provenance_are_serializable() -> None:
    provenance = AnalysisProvenance(
        analysis="test",
        parameters={"alpha": 0.05},
        input_summary={"n": 10},
        random_state=7,
    )
    result = AnalysisResult.ok(
        advisories=(Advisory(code="small_n", message="Small sample."),),
        provenance=provenance,
    )

    payload = result.to_dict()
    assert payload["status"] == "ok"
    assert payload["advisories"][0]["code"] == "small_n"
    assert payload["provenance"]["parameters"] == {"alpha": 0.05}


def test_result_table_validates_global_columns_and_statuses() -> None:
    frame = pd.DataFrame({"status": ["ok", "degenerate"], "reason": [None, "constant"], "x": [1, 2]})
    validated = validate_result_table(frame, required_columns=["x"])
    assert isinstance(validated, pd.DataFrame)
    pd.testing.assert_frame_equal(validated, frame)

    with pytest.raises(ResultContractError):
        validate_result_table(pd.DataFrame({"status": ["weird"], "reason": [None]}))


def test_result_table_enforces_status_reason_consistency() -> None:
    with pytest.raises(ResultContractError, match="status/reason consistency"):
        validate_result_table(pd.DataFrame({"status": ["ok"], "reason": ["constant"]}))

    with pytest.raises(ResultContractError, match="status/reason consistency"):
        validate_result_table(pd.DataFrame({"status": ["skipped"], "reason": [None]}))


def test_polars_result_table_contract() -> None:
    frame = pl.DataFrame({"x": [1, 2], "status": ["ok", "skipped"], "reason": [None, "too few"]})
    assert validate_result_table(frame, required_columns=["x"]) is frame
    with pytest.raises(ResultContractError):
        validate_result_table(pl.DataFrame({"status": ["weird"], "reason": [None]}))
    with pytest.raises(ResultContractError):
        validate_result_table(pl.DataFrame({"status": ["ok"], "reason": ["constant"]}))
    with pytest.raises(ResultContractError):
        validate_result_table(pl.DataFrame({"status": ["skipped"], "reason": [None]}))
