import numpy as np
from tidal_enkf.filter import EnsembleKalmanFilter, FilterConfig
from tidal_enkf.model import ModelConfig, ShallowWater1D


def test_enkf_output_and_reproducibility():
    cfg=ModelConfig(cells=35)
    model=ShallowWater1D(cfg)
    t=np.arange(60)*cfg.dt_s
    boundary=1.5*np.sin(2*np.pi*t/(12*3600))
    truth=model.run(boundary)
    locations=np.array([25_000., 50_000., 75_000.])
    obs=model.sample_heights(truth,locations)
    fcfg=FilterConfig(ensemble_size=25,seed=123)
    a=EnsembleKalmanFilter(model,fcfg).assimilate(boundary,obs,locations,[0,1])
    b=EnsembleKalmanFilter(model,fcfg).assimilate(boundary,obs,locations,[0,1])
    assert a["analysis_states"].shape==(60,model.nstate+1)
    np.testing.assert_allclose(a["analysis_states"],b["analysis_states"])
    np.testing.assert_allclose(a["analysis_states"][:,-2],0,atol=1e-12)
    assert np.all(np.isfinite(a["analysis_states"]))
    assert a["assimilated_columns"]==[0,1]


def test_ar1_stationary_variance_relation():
    model=ShallowWater1D(ModelConfig(cells=20))
    conf=FilterConfig(ensemble_size=10,boundary_sd_m=0.2)
    enkf=EnsembleKalmanFilter(model,conf)
    np.testing.assert_allclose(enkf.innovation_sd_m**2/(1-enkf.alpha**2),0.2**2)


def test_held_out_station_values_do_not_affect_result():
    model=ShallowWater1D(ModelConfig(cells=35))
    t=np.arange(24)*model.config.dt_s
    base=np.sin(t/3600.)
    positions=np.array([25_000.,50_000.,75_000.])
    observed=np.zeros((24,3))
    mutated=observed.copy()
    mutated[:,2]=1000.0  # Never passed to the assimilation update.
    cfg=FilterConfig(ensemble_size=12,seed=44)
    a=EnsembleKalmanFilter(model,cfg).assimilate(base,observed,positions,[0,1])
    b=EnsembleKalmanFilter(model,cfg).assimilate(base,mutated,positions,[0,1])
    np.testing.assert_array_equal(a["analysis_states"],b["analysis_states"])
