"""Global result, status, reason, and advisory contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ruddy.core.enums import AdvisoryLevel, ResultStatus
from ruddy.core.exceptions import ResultContractError
from ruddy.results.provenance import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class Advisory:
    """Non-fatal scientific diagnostic attached to a result."""

    code: str
    message: str
    level: AdvisoryLevel = AdvisoryLevel.WARNING
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        level = self.level if isinstance(self.level, AdvisoryLevel) else AdvisoryLevel(self.level)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "level": self.level.value,
            "context": dict(self.context),
        }


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Base scientific result contract used across Ruddy."""

    status: ResultStatus
    reason: str | None = None
    advisories: tuple[Advisory, ...] = ()
    provenance: AnalysisProvenance | None = None

    def __post_init__(self) -> None:
        status = self.status if isinstance(self.status, ResultStatus) else ResultStatus(self.status)
        object.__setattr__(self, "status", status)
        if self.status is ResultStatus.OK and self.reason is not None:
            raise ResultContractError("An OK result cannot carry a degeneracy reason.")
        if self.status in {ResultStatus.DEGENERATE, ResultStatus.SKIPPED} and not self.reason:
            raise ResultContractError(f"A {self.status.value} result must provide an explicit reason.")

    @classmethod
    def ok(
        cls,
        *,
        advisories: tuple[Advisory, ...] = (),
        provenance: AnalysisProvenance | None = None,
    ) -> AnalysisResult:
        return cls(
            status=ResultStatus.OK,
            advisories=advisories,
            provenance=provenance,
        )

    @classmethod
    def degenerate(
        cls,
        reason: str,
        *,
        advisories: tuple[Advisory, ...] = (),
        provenance: AnalysisProvenance | None = None,
    ) -> AnalysisResult:
        return cls(
            status=ResultStatus.DEGENERATE,
            reason=reason,
            advisories=advisories,
            provenance=provenance,
        )

    @classmethod
    def skipped(
        cls,
        reason: str,
        *,
        advisories: tuple[Advisory, ...] = (),
        provenance: AnalysisProvenance | None = None,
    ) -> AnalysisResult:
        return cls(
            status=ResultStatus.SKIPPED,
            reason=reason,
            advisories=advisories,
            provenance=provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "advisories": [advisory.to_dict() for advisory in self.advisories],
            "provenance": (None if self.provenance is None else self.provenance.to_dict()),
        }
