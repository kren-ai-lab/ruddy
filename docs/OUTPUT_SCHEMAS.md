# Output schema reference

This document inventories the stable tabular column schemas explicitly declared by the current Ruddy implementation. It is intended for downstream visualization/export/application work. Some result objects also contain dictionaries, matrices or tables whose columns are constructed dynamically; those are documented in `RESULTS_AND_PROVENANCE.md` and `METHODS.md`.

## Tabular conventions

All tabular outputs across Ruddy follow these core invariants:

- **Polars DataFrames with declared schemas**: Every result table is returned as a `polars.DataFrame` (`pl.DataFrame`). Empty result tables and tables with null entries retain their fully typed column schemas.
- **Missing values as `null`, never `NaN`**: Missing values in Polars tables are represented as `null`. Float `NaN` is treated as missing on input but normalized to `null` in results.
- **Observation ID dtype preservation**: Columns containing observation identifiers (such as `observation_id` in exclusions tables, outlier flags, and projection scores) dynamically preserve the dataset's or feature matrix's original observation ID dtype (e.g., `pl.Int64`, `pl.String`), rather than coercing them to a single hardcoded type.
- **Labeled and square matrix layout**: Square matrices (covariance, correlation, pairwise counts, variation matrix, Aitchison distances) and coordinate tables place the row label in the first column (`feature` or `observation_id`) preserving its original dtype, followed by value columns named `str(label)`. Export to CSV writes the table directly without writing an index column.

## `ruddy/bivariate/associations.py`

### `ASSOCIATION_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `test` |
| 2 | `column_x` |
| 3 | `column_y` |
| 4 | `column_x_role` |
| 5 | `column_y_role` |
| 6 | `n_x_levels` |
| 7 | `n_y_levels` |
| 8 | `n_total` |
| 9 | `n_used` |
| 10 | `n_missing` |
| 11 | `statistic` |
| 12 | `df` |
| 13 | `p_value` |
| 14 | `q_value` |
| 15 | `family_id` |
| 16 | `family_size` |
| 17 | `correction` |
| 18 | `effect_size_name` |
| 19 | `effect_size` |
| 20 | `min_expected_count` |
| 21 | `n_expected_lt5` |
| 22 | `fraction_expected_lt5` |
| 23 | `advisory_codes` |
| 24 | `contingency_json` |
| 25 | `status` |
| 26 | `reason` |

## `ruddy/bivariate/comparisons.py`

### `COMPARISON_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `group_column` |
| 2 | `group_role` |
| 3 | `scope` |
| 4 | `test` |
| 5 | `feature` |
| 6 | `feature_role` |
| 7 | `n_groups` |
| 8 | `group_a` |
| 9 | `group_b` |
| 10 | `group_a_n` |
| 11 | `group_b_n` |
| 12 | `min_group_n_used` |
| 13 | `max_group_n_used` |
| 14 | `n_total` |
| 15 | `n_used` |
| 16 | `n_missing_or_nonfinite` |
| 17 | `statistic` |
| 18 | `df1` |
| 19 | `df2` |
| 20 | `p_value` |
| 21 | `q_value` |
| 22 | `family_id` |
| 23 | `family_size` |
| 24 | `correction` |
| 25 | `effect_size_name` |
| 26 | `effect_size` |
| 27 | `status` |
| 28 | `reason` |

## `ruddy/bivariate/contingency.py`

### `SUMMARY_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column_x` |
| 2 | `column_y` |
| 3 | `n_total` |
| 4 | `n_used` |
| 5 | `n_x_levels` |
| 6 | `n_y_levels` |
| 7 | `chi_square` |
| 8 | `df` |
| 9 | `p_value` |
| 10 | `q_value` |
| 11 | `family_id` |
| 12 | `family_size` |
| 13 | `correction` |
| 14 | `cramers_v` |
| 15 | `min_expected_count` |
| 16 | `n_expected_lt5` |
| 17 | `fraction_expected_lt5` |
| 18 | `status` |
| 19 | `reason` |

### `CELL_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column_x` |
| 2 | `column_y` |
| 3 | `level_x` |
| 4 | `level_y` |
| 5 | `observed` |
| 6 | `expected` |
| 7 | `pearson_residual` |
| 8 | `standardized_residual` |
| 9 | `chi_square_contribution` |
| 10 | `chi_square_contribution_fraction` |
| 11 | `row_fraction` |
| 12 | `column_fraction` |
| 13 | `status` |
| 14 | `reason` |

## `ruddy/bivariate/correlations.py`

### `CORRELATION_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `method` |
| 2 | `column_x` |
| 3 | `column_y` |
| 4 | `n_total` |
| 5 | `n_complete` |
| 6 | `n_missing_pair` |
| 7 | `coefficient` |
| 8 | `statistic` |
| 9 | `p_value` |
| 10 | `q_value` |
| 11 | `family_id` |
| 12 | `family_size` |
| 13 | `correction` |
| 14 | `status` |
| 15 | `reason` |

## `ruddy/bivariate/dependence.py`

### `PARTIAL_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `method` |
| 2 | `column_x` |
| 3 | `column_y` |
| 4 | `covariates` |
| 5 | `n_total` |
| 6 | `n_complete` |
| 7 | `n_covariates` |
| 8 | `coefficient` |
| 9 | `statistic` |
| 10 | `df` |
| 11 | `p_value` |
| 12 | `q_value` |
| 13 | `family_id` |
| 14 | `family_size` |
| 15 | `correction` |
| 16 | `status` |
| 17 | `reason` |

### `DEPENDENCE_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `method` |
| 2 | `column_x` |
| 3 | `column_y` |
| 4 | `column_x_kind` |
| 5 | `column_y_kind` |
| 6 | `n_total` |
| 7 | `n_complete` |
| 8 | `statistic` |
| 9 | `p_value` |
| 10 | `q_value` |
| 11 | `family_id` |
| 12 | `family_size` |
| 13 | `correction` |
| 14 | `n_permutations` |
| 15 | `status` |
| 16 | `reason` |

## `ruddy/bivariate/groups.py`

### `GROUP_COVERAGE_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `group_column` |
| 2 | `group_role` |
| 3 | `data_kind` |
| 4 | `n_dataset` |
| 5 | `n_group_present` |
| 6 | `n_group_missing` |
| 7 | `coverage_fraction` |
| 8 | `n_levels` |
| 9 | `levels_json` |
| 10 | `status` |
| 11 | `reason` |

### `RESPONSE_CATALOG_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `response` |
| 2 | `declared_role` |
| 3 | `data_kind` |
| 4 | `n_dataset` |
| 5 | `n_present` |
| 6 | `n_missing` |
| 7 | `n_finite` |
| 8 | `n_non_finite` |
| 9 | `status` |
| 10 | `reason` |

### `NUMERIC_GROUP_SUMMARY_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `group_column` |
| 2 | `group_level` |
| 3 | `response` |
| 4 | `response_role` |
| 5 | `n_dataset` |
| 6 | `n_group_observations` |
| 7 | `n_response_present` |
| 8 | `n_response_finite` |
| 9 | `n_response_missing` |
| 10 | `n_response_non_finite` |
| 11 | `mean` |
| 12 | `std` |
| 13 | `median` |
| 14 | `q25` |
| 15 | `q75` |
| 16 | `iqr` |
| 17 | `min` |
| 18 | `max` |
| 19 | `status` |
| 20 | `reason` |

### `CATEGORICAL_GROUP_SUMMARY_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `group_column` |
| 2 | `group_level` |
| 3 | `response` |
| 4 | `response_role` |
| 5 | `response_level` |
| 6 | `n_dataset` |
| 7 | `n_group_observations` |
| 8 | `n_response_present` |
| 9 | `n_response_missing` |
| 10 | `count` |
| 11 | `fraction` |
| 12 | `status` |
| 13 | `reason` |

### `ANNOTATION_COVERAGE_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `source_name` |
| 2 | `coverage` |
| 3 | `base_count` |
| 4 | `annotation_count` |
| 5 | `covered_count` |
| 6 | `coverage_fraction` |
| 7 | `n_missing_ids` |
| 8 | `n_unmatched_ids` |
| 9 | `columns_json` |

## `ruddy/bivariate/posthoc.py`

### `PAIRWISE_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `response` |
| 2 | `factor` |
| 3 | `method` |
| 4 | `group_a` |
| 5 | `group_b` |
| 6 | `n_a` |
| 7 | `n_b` |
| 8 | `mean_a` |
| 9 | `mean_b` |
| 10 | `mean_difference` |
| 11 | `std_error` |
| 12 | `df` |
| 13 | `statistic` |
| 14 | `p_value` |
| 15 | `confidence_level` |
| 16 | `ci_lower` |
| 17 | `ci_upper` |
| 18 | `correction` |
| 19 | `family_size` |
| 20 | `status` |
| 21 | `reason` |

## `ruddy/factorial/diagnostics.py`

### `DIAGNOSTIC_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `diagnostic` |
| 2 | `statistic` |
| 3 | `p_value` |
| 4 | `alpha` |
| 5 | `flagged` |
| 6 | `details` |
| 7 | `status` |
| 8 | `reason` |

### `OBSERVATION_DIAGNOSTIC_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `source_row_index` |
| 2 | `observation_id` |
| 3 | `fitted_value` |
| 4 | `residual` |
| 5 | `studentized_residual` |
| 6 | `leverage` |
| 7 | `cooks_distance` |
| 8 | `large_residual_threshold` |
| 9 | `high_leverage_threshold` |
| 10 | `cooks_distance_threshold` |
| 11 | `is_large_residual` |
| 12 | `is_high_leverage` |
| 13 | `is_influential` |
| 14 | `status` |
| 15 | `reason` |

## `ruddy/factorial/marginal_means.py`

### `MEAN_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `response` |
| 2 | `term` |
| 3 | `levels_json` |
| 4 | `estimate` |
| 5 | `std_error` |
| 6 | `df` |
| 7 | `confidence_level` |
| 8 | `ci_lower` |
| 9 | `ci_upper` |
| 10 | `weighting` |
| 11 | `status` |
| 12 | `reason` |

### `CONTRAST_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `response` |
| 2 | `term` |
| 3 | `levels_a_json` |
| 4 | `levels_b_json` |
| 5 | `estimate_difference` |
| 6 | `std_error` |
| 7 | `df` |
| 8 | `t_value` |
| 9 | `p_value` |
| 10 | `q_value` |
| 11 | `correction` |
| 12 | `family_size` |
| 13 | `confidence_level` |
| 14 | `ci_lower` |
| 15 | `ci_upper` |
| 16 | `status` |
| 17 | `reason` |

### `EXCLUSION_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `source_row_index` |
| 2 | `observation_id` |
| 3 | `stage` |
| 4 | `reason` |

## `ruddy/factorial/mixed_effects.py`

### `FIXED_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `parameter` |
| 2 | `estimate` |
| 3 | `std_error` |
| 4 | `z_value` |
| 5 | `p_value` |
| 6 | `ci_lower` |
| 7 | `ci_upper` |
| 8 | `status` |
| 9 | `reason` |

### `VARIANCE_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `component` |
| 2 | `row` |
| 3 | `column` |
| 4 | `estimate` |
| 5 | `status` |
| 6 | `reason` |

### `RANDOM_EFFECT_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `group` |
| 2 | `effect` |
| 3 | `estimate` |
| 4 | `status` |
| 5 | `reason` |

### `EXCLUSION_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `source_row_index` |
| 2 | `observation_id` |
| 3 | `stage` |
| 4 | `reason` |

## `ruddy/factorial/models.py`

### `EFFECT_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `term` |
| 2 | `term_type` |
| 3 | `order` |
| 4 | `components_json` |
| 5 | `df` |
| 6 | `sum_sq` |
| 7 | `mean_sq` |
| 8 | `f_value` |
| 9 | `p_value` |
| 10 | `q_value` |
| 11 | `correction` |
| 12 | `family_id` |
| 13 | `family_size` |
| 14 | `eta_squared` |
| 15 | `partial_eta_squared` |
| 16 | `omega_squared` |
| 17 | `partial_omega_squared` |
| 18 | `status` |
| 19 | `reason` |

### `COEFFICIENT_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `parameter` |
| 2 | `estimate` |
| 3 | `std_error` |
| 4 | `t_value` |
| 5 | `p_value` |
| 6 | `ci_lower` |
| 7 | `ci_upper` |
| 8 | `status` |
| 9 | `reason` |

### `EXCLUSION_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `source_row_index` |
| 2 | `observation_id` |
| 3 | `stage` |
| 4 | `reason` |

### `DESIGN_TERM_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `term` |
| 2 | `term_type` |
| 3 | `order` |
| 4 | `components_json` |
| 5 | `component_kinds_json` |

## `ruddy/multivariate/permutation.py`

### `SUMMARY_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `analysis` |
| 2 | `factor` |
| 3 | `statistic` |
| 4 | `value` |
| 5 | `df_between` |
| 6 | `df_within` |
| 7 | `r_squared` |
| 8 | `p_value` |
| 9 | `n_permutations` |
| 10 | `status` |
| 11 | `reason` |

### `GROUP_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `factor` |
| 2 | `level` |
| 3 | `n` |
| 4 | `mean_distance_to_centroid` |

### `CENTROID_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `source_row_index` |
| 2 | `observation_id` |
| 3 | `factor` |
| 4 | `level` |
| 5 | `distance_to_centroid` |
| 6 | `status` |
| 7 | `reason` |

### `EXCLUSION_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `source_row_index` |
| 2 | `observation_id` |
| 3 | `stage` |
| 4 | `reason` |

## `ruddy/profiling/columns.py`

### `COLUMN_PROFILE_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column` |
| 2 | `dtype` |
| 3 | `role` |
| 4 | `data_kind` |
| 5 | `excluded` |
| 6 | `analysis_eligible` |
| 7 | `n_total` |
| 8 | `n_present` |
| 9 | `n_missing` |
| 10 | `missing_fraction` |
| 11 | `n_unique` |
| 12 | `unique_fraction` |
| 13 | `finite_count` |
| 14 | `non_finite_count` |
| 15 | `is_all_missing` |
| 16 | `is_constant` |

## `ruddy/profiling/missingness.py`

### `MISSINGNESS_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column` |
| 2 | `role` |
| 3 | `data_kind` |
| 4 | `n_total` |
| 5 | `n_present` |
| 6 | `fraction_present` |
| 7 | `n_missing` |
| 8 | `fraction_missing` |
| 9 | `finite_count` |
| 10 | `non_finite_count` |
| 11 | `is_all_missing` |

### `PAIRWISE_COMPLETENESS_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column_x` |
| 2 | `column_y` |
| 3 | `n_total` |
| 4 | `n_complete` |
| 5 | `n_incomplete` |
| 6 | `fraction_complete` |

### `MISSINGNESS_PATTERN_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `rank` |
| 2 | `missing_columns` |
| 3 | `n_missing_columns` |
| 4 | `count` |
| 5 | `fraction` |
| 6 | `n_patterns_total` |
| 7 | `patterns_truncated` |
| 8 | `unreported_count` |
| 9 | `unreported_fraction` |

## `ruddy/univariate/categorical.py`

### `CATEGORICAL_STATISTICS_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column` |
| 2 | `role` |
| 3 | `n_total` |
| 4 | `n_present` |
| 5 | `n_missing` |
| 6 | `n_levels` |
| 7 | `mode` |
| 8 | `mode_count` |
| 9 | `mode_fraction` |
| 10 | `entropy` |
| 11 | `normalized_entropy` |
| 12 | `entropy_base` |
| 13 | `n_levels_reported` |
| 14 | `frequencies_truncated` |
| 15 | `unreported_count` |
| 16 | `unreported_fraction` |
| 17 | `status` |
| 18 | `reason` |

### `CATEGORICAL_FREQUENCY_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column` |
| 2 | `role` |
| 3 | `level` |
| 4 | `count` |
| 5 | `fraction` |
| 6 | `rank` |

## `ruddy/univariate/diagnostics.py`

### `NORMALITY_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `method` |
| 2 | `column` |
| 3 | `role` |
| 4 | `n_total` |
| 5 | `n_finite` |
| 6 | `statistic` |
| 7 | `p_value` |
| 8 | `q_value` |
| 9 | `family_id` |
| 10 | `family_size` |
| 11 | `correction` |
| 12 | `status` |
| 13 | `reason` |

### `DISPERSION_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `method` |
| 2 | `response` |
| 3 | `group` |
| 4 | `n_total` |
| 5 | `n_used` |
| 6 | `n_groups` |
| 7 | `group_sizes_json` |
| 8 | `statistic` |
| 9 | `p_value` |
| 10 | `q_value` |
| 11 | `family_id` |
| 12 | `family_size` |
| 13 | `correction` |
| 14 | `status` |
| 15 | `reason` |

## `ruddy/univariate/distributions.py`

### `DATETIME_STATISTICS_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column` |
| 2 | `role` |
| 3 | `n_total` |
| 4 | `n_present` |
| 5 | `n_missing` |
| 6 | `min` |
| 7 | `max` |
| 8 | `range_seconds` |
| 9 | `status` |
| 10 | `reason` |

## `ruddy/univariate/numeric.py`

### `NUMERIC_STATISTICS_BASE_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column` |
| 2 | `role` |
| 3 | `n_total` |
| 4 | `n_finite` |
| 5 | `n_missing` |
| 6 | `n_non_finite` |
| 7 | `mean` |
| 8 | `std` |
| 9 | `variance` |
| 10 | `min` |
| 11 | `max` |
| 12 | `range` |
| 13 | `median` |
| 14 | `iqr` |
| 15 | `mad` |
| 16 | `skewness` |
| 17 | `kurtosis` |
| 18 | `zero_count` |
| 19 | `zero_fraction` |
| 20 | `status` |
| 21 | `reason` |

## `ruddy/univariate/outliers.py`

### `OUTLIER_SUMMARY_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column` |
| 2 | `role` |
| 3 | `method` |
| 4 | `n_total` |
| 5 | `n_finite` |
| 6 | `n_missing` |
| 7 | `n_non_finite` |
| 8 | `center` |
| 9 | `scale` |
| 10 | `lower_bound` |
| 11 | `upper_bound` |
| 12 | `criterion` |
| 13 | `threshold` |
| 14 | `n_flagged` |
| 15 | `flagged_fraction` |
| 16 | `status` |
| 17 | `reason` |

### `OUTLIER_FLAG_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column` |
| 2 | `role` |
| 3 | `method` |
| 4 | `observation_id` |
| 5 | `source_row_index` |
| 6 | `value` |
| 7 | `score` |
| 8 | `direction` |
| 9 | `threshold` |
| 10 | `status` |
| 11 | `reason` |

### `NUMERIC_QUALITY_COLUMNS`

| # | Column |
| ---: | --- |
| 1 | `column` |
| 2 | `role` |
| 3 | `n_total` |
| 4 | `n_present` |
| 5 | `n_finite` |
| 6 | `n_missing` |
| 7 | `n_non_finite` |
| 8 | `n_unique_finite` |
| 9 | `missing_fraction` |
| 10 | `non_finite_fraction` |
| 11 | `iqr` |
| 12 | `mad` |
| 13 | `has_missing` |
| 14 | `has_non_finite` |
| 15 | `zero_iqr_nonconstant` |
| 16 | `zero_mad_nonconstant` |
| 17 | `status` |
| 18 | `reason` |
