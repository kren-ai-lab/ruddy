"""Shared contracts, enumerations, and exceptions."""

from ruddy.core.enums import (
    AdvisoryLevel,
    AlignmentMode,
    ColumnKind,
    ColumnRole,
    ResultStatus,
)
from ruddy.core.exceptions import (
    AlignmentError,
    ContractError,
    DataValidationError,
    DuplicateObservationIDError,
    FeatureMatrixValidationError,
    KindConflictError,
    MissingObservationIDError,
    ResultContractError,
    RoleConflictError,
    RuddyError,
    UnknownColumnError,
)

__all__ = [
    "AdvisoryLevel",
    "AlignmentError",
    "AlignmentMode",
    "ColumnKind",
    "ColumnRole",
    "ContractError",
    "DataValidationError",
    "DuplicateObservationIDError",
    "FeatureMatrixValidationError",
    "KindConflictError",
    "MissingObservationIDError",
    "ResultContractError",
    "ResultStatus",
    "RoleConflictError",
    "RuddyError",
    "UnknownColumnError",
]
