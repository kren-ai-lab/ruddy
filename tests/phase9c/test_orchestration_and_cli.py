import numpy as np
import pandas as pd
import pytest

from ruddy import AnalysisConfig, FeatureMatrix, TabularDataset, analyze, analyze_anomalies, analyze_bayesian_eda, analyze_composition, analyze_representation_similarity
from ruddy.cli.main import main


def _bundle(tmp_path=None):
    rng=np.random.default_rng(22); n=36; ids=[f'o{i}' for i in range(n)]
    frame=pd.DataFrame({'id':ids,'y':rng.normal(size=n),'g':['A']*(n//2)+['B']*(n//2)})
    ds=TabularDataset(frame,id_column='id',role_overrides={'y':'response','g':'factor'})
    x=rng.normal(size=(n,3)); y=x@rng.normal(size=(3,4))+0.1*rng.normal(size=(n,4))
    return ds, FeatureMatrix(x,observation_ids=ids,feature_names=['x1','x2','x3']), FeatureMatrix(y,observation_ids=ids,feature_names=['y1','y2','y3','y4'])


def test_phase9c_config_validation():
    with pytest.raises(ValueError): AnalysisConfig(enabled_blocks=('representation',),representation_distance_similarity_method='bad')
    with pytest.raises(ValueError): AnalysisConfig(enabled_blocks=('anomaly',),anomaly_methods=('bad',))
    with pytest.raises(ValueError): AnalysisConfig(enabled_blocks=('bayesian',),bayesian_draws=20)


def test_unified_representation_matches_standalone():
    ds,x,y=_bundle(); cfg=AnalysisConfig(enabled_blocks=('representation',),representation_cca_components=2,representation_mantel_permutations=9,random_state=5)
    u=analyze(ds,config=cfg,features=x,comparison_features=y).representation
    s=analyze_representation_similarity(x,y,cca_components=2,mantel_permutations=9,random_state=5)
    pd.testing.assert_frame_equal(u.cka,s.cka); pd.testing.assert_frame_equal(u.mantel,s.mantel)


def test_unified_requires_second_representation():
    ds,x,_=_bundle(); cfg=AnalysisConfig(enabled_blocks=('representation',))
    with pytest.raises(ValueError,match='comparison_features'): analyze(ds,config=cfg,features=x)


def test_unified_anomaly_matches_standalone():
    ds,x,_=_bundle(); cfg=AnalysisConfig(enabled_blocks=('anomaly',),anomaly_methods=('isolation_forest',),random_state=2)
    u=analyze(ds,config=cfg,features=x).anomaly; s=analyze_anomalies(x,methods=('isolation_forest',),random_state=2)
    pd.testing.assert_frame_equal(u.scores,s.scores)


def test_unified_bayesian_matches_standalone():
    ds,_,_=_bundle(); cfg=AnalysisConfig(enabled_blocks=('bayesian',),responses=('y',),groups=('g',),bayesian_variables=('y',),bayesian_groups=('g',),bayesian_draws=500,random_state=3)
    u=analyze(ds,config=cfg).bayesian; s=analyze_bayesian_eda(ds,variables=('y',),groups=('g',),draws=500,random_state=3)
    pd.testing.assert_frame_equal(u.mean_differences,s.mean_differences)


def test_unified_compositional_matches_standalone():
    ds,_,_=_bundle(); rng=np.random.default_rng(1); c=FeatureMatrix(np.abs(rng.normal(size=(36,4)))+0.1,observation_ids=ds.observation_ids)
    cfg=AnalysisConfig(enabled_blocks=('compositional',),compositional_transform='ilr')
    u=analyze(ds,config=cfg,features=c).compositional; s=analyze_composition(c,transform='ilr')
    np.testing.assert_allclose(u.transformed.to_array(),s.transformed.to_array())


def test_representation_cli_writes_artifacts(tmp_path):
    ds,x,y=_bundle(); xa=tmp_path/'x.csv'; ya=tmp_path/'y.csv'; out=tmp_path/'out'
    pd.DataFrame({'id':x.observation_ids,**{n:x.to_array()[:,i] for i,n in enumerate(x.feature_names)}}).to_csv(xa,index=False)
    pd.DataFrame({'id':y.observation_ids,**{n:y.to_array()[:,i] for i,n in enumerate(y.feature_names)}}).to_csv(ya,index=False)
    code=main(['representation',str(xa),str(ya),'--x-id-column','id','--y-id-column','id','--cca-components','2','--mantel-permutations','9','--output-dir',str(out)])
    assert code==0 and (out/'cka.csv').exists() and (out/'cca_correlations.csv').exists()


def test_compositional_cli_writes_artifacts(tmp_path):
    rng=np.random.default_rng(3); p=tmp_path/'c.csv'; out=tmp_path/'c_out'
    frame=pd.DataFrame(np.abs(rng.normal(size=(20,3)))+0.2,columns=['a','b','c']); frame.insert(0,'id',[f'o{i}' for i in range(20)]); frame.to_csv(p,index=False)
    code=main(['compositional',str(p),'--id-column','id','--compositional-transform','clr','--output-dir',str(out)])
    assert code==0 and (out/'compositional_transformed.csv').exists()


def test_bayesian_cli_writes_artifacts(tmp_path):
    ds,_,_=_bundle(); p=tmp_path/'d.csv'; out=tmp_path/'b'; ds.to_frame().to_csv(p,index=False)
    code=main(['bayesian',str(p),'--id-column','id','--response','y','--factor','g','--bayesian-variable','y','--bayesian-group','g','--bayesian-draws','500','--output-dir',str(out)])
    assert code==0 and (out/'bayesian_means.csv').exists()


def test_anomaly_cli_writes_artifacts(tmp_path):
    _,x,_=_bundle(); p=tmp_path/'x.csv'; out=tmp_path/'a'
    pd.DataFrame({'id':x.observation_ids,**{n:x.to_array()[:,i] for i,n in enumerate(x.feature_names)}}).to_csv(p,index=False)
    code=main(['anomaly',str(p),'--id-column','id','--anomaly-method','isolation_forest','--output-dir',str(out)])
    assert code==0 and (out/'anomaly_scores.csv').exists()
