"""Shared type aliases used by Ruddy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import spmatrix

from ruddy.core.enums import ColumnKind, ColumnRole

ObservationID: TypeAlias = Any
RoleOverrides: TypeAlias = Mapping[str, ColumnRole | str]
KindOverrides: TypeAlias = Mapping[str, ColumnKind | str]
ObservationIDs: TypeAlias = Sequence[ObservationID] | pd.Index | pd.Series | np.ndarray | pl.Series
FeatureInput: TypeAlias = np.ndarray | pd.DataFrame | pl.DataFrame | spmatrix
TableInput: TypeAlias = pl.DataFrame | pd.DataFrame
