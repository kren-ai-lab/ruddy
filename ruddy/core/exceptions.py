"""Ruddy exception hierarchy."""

from __future__ import annotations


class RuddyError(Exception):
    """Base exception for Ruddy."""


class ContractError(RuddyError):
    """Base exception for invalid public data/result contracts."""


class DataValidationError(ContractError):
    """Raised when input data violate a Ruddy data contract."""


class UnknownColumnError(DataValidationError):
    """Raised when a configured column is absent from the dataset."""


class DuplicateObservationIDError(DataValidationError):
    """Raised when observation identifiers are not unique."""


class MissingObservationIDError(DataValidationError):
    """Raised when observation identifiers contain missing values."""


class RoleConflictError(DataValidationError):
    """Raised when column-role declarations conflict."""


class KindConflictError(DataValidationError):
    """Raised when a data-kind override is incompatible with observed data."""


class AlignmentError(DataValidationError):
    """Raised when external annotations cannot be aligned under the chosen policy."""


class FeatureMatrixValidationError(DataValidationError):
    """Raised when a feature matrix violates shape or numeric-data requirements."""


class OptionalDependencyError(RuddyError):
    """Raised when an optional analysis dependency is unavailable."""


class ResultContractError(ContractError):
    """Raised when an analysis result violates the global result contract."""


class RuddyIOError(RuddyError):
    """Raised when an input/output path cannot be read or written."""
