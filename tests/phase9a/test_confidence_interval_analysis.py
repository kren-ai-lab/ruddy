from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import TabularDataset, analyze_confidence_intervals


def _dataset(zero_cell: bool=False) -> TabularDataset:
    rng=np.random.default_rng(10); n=60
    group=np.repeat(['A','B'],30)
    x=rng.normal(loc=np.where(group=='A',0.0,1.0),scale=1.0)
    y=.6*x+rng.normal(scale=.7,size=n)
    if zero_cell:
        c=np.where(group=='A','X','Y')
    else:
        c=np.array(['X']*20+['Y']*10+['X']*8+['Y']*22)
    frame=pd.DataFrame({'id':range(n),'x':x,'y':y,'group':group,'c':c})
    return TabularDataset(frame,id_column='id',role_overrides={'group':'factor'})


def test_interval_analysis_contains_core_estimands() -> None:
    result=analyze_confidence_intervals(_dataset(),bootstrap_resamples=150,bootstrap_method='percentile',random_state=2)
    assert set(result.means.column) == {'x','y'}
    assert len(result.correlations) == 1
    assert not result.mean_differences.empty
    assert not result.effect_sizes.empty
    assert not result.odds_ratios.empty


def test_pearson_interval_contains_estimate() -> None:
    result=analyze_confidence_intervals(_dataset(),bootstrap_resamples=100,bootstrap_method='percentile')
    row=result.correlations.iloc[0]
    assert row.ci_method == 'fisher_z'
    assert row.confidence_low < row.estimate < row.confidence_high


def test_spearman_interval_bootstrap_is_deterministic() -> None:
    kwargs=dict(correlation_methods=('spearman',),bootstrap_resamples=120,bootstrap_method='percentile',random_state=8)
    a=analyze_confidence_intervals(_dataset(),**kwargs)
    b=analyze_confidence_intervals(_dataset(),**kwargs)
    pd.testing.assert_frame_equal(a.correlations,b.correlations)


def test_zero_cell_odds_ratio_is_explicitly_degenerate() -> None:
    result=analyze_confidence_intervals(_dataset(True),bootstrap_resamples=100,bootstrap_method='percentile')
    rows=result.odds_ratios[(result.odds_ratios.column_x=='group') | (result.odds_ratios.column_y=='group')]
    assert (rows.status=='degenerate').any()
    assert 'zero_cell_or_invalid_table' in set(rows.reason.dropna())
