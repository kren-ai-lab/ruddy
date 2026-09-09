import numpy as np
from ruddy import FeatureMatrix, analyze_anomalies


def _data():
    rng=np.random.default_rng(8); x=rng.normal(size=(80,3)); x[-1]=[9,9,9]
    return FeatureMatrix(x,observation_ids=[f'o{i}' for i in range(80)])


def test_anomaly_methods_return_scores():
    r=analyze_anomalies(_data(),lof_neighbors=15,random_state=3)
    assert set(r.scores['method'])=={'isolation_forest','lof'}
    assert len(r.scores)==160


def test_extreme_point_is_isolation_forest_flagged():
    r=analyze_anomalies(_data(),methods=('isolation_forest',),contamination=0.05,random_state=2)
    row=r.scores[r.scores.observation_id=='o79'].iloc[0]; assert bool(row.is_flagged)


def test_anomaly_is_deterministic():
    a=analyze_anomalies(_data(),methods=('isolation_forest',),random_state=12)
    b=analyze_anomalies(_data(),methods=('isolation_forest',),random_state=12)
    np.testing.assert_allclose(a.scores.anomaly_score,b.scores.anomaly_score)


def test_lof_invalid_neighbors_is_skipped_not_crash():
    f=_data(); r=analyze_anomalies(f,methods=('lof',),lof_neighbors=80)
    assert r.methods.loc[0,'status']=='skipped'


def test_nonfinite_row_is_excluded():
    f=_data(); x=f.to_array(); x[0,0]=np.nan
    r=analyze_anomalies(FeatureMatrix(x,observation_ids=f.observation_ids),methods=('isolation_forest',))
    assert len(r.exclusions)==1 and r.exclusions.loc[0,'observation_id']=='o0'
