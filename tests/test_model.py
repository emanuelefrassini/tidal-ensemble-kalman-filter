import numpy as np
from tidal_enkf.model import ModelConfig, ShallowWater1D


def test_at_rest_stays_at_rest():
    m = ShallowWater1D(ModelConfig(cells=30))
    x = m.run(np.zeros(40))
    assert x.shape == (40,60)
    np.testing.assert_allclose(x,0,atol=1e-12)


def test_forced_boundary_and_eastern_wall_are_respected():
    m = ShallowWater1D(ModelConfig(cells=30))
    t=np.arange(48)*600
    boundary=2.0*np.sin(2*np.pi*t/(12*3600))
    x=m.run(boundary)
    np.testing.assert_allclose(x[:,0],boundary,atol=1e-12)
    np.testing.assert_allclose(x[:,-1],0,atol=1e-12)
    assert np.all(np.isfinite(x))


def test_linear_wave_speed_from_moving_pulse():
    # A localized right-going wave propagates approximately at sqrt(g D).
    cfg=ModelConfig(cells=250,dt_s=30.,friction_per_s=0.)
    model=ShallowWater1D(cfg)
    h=np.exp(-0.5*((model.height_positions_m-30_000)/4000)**2)
    u=np.sqrt(cfg.gravity/cfg.depth_m)*np.exp(-0.5*((model.velocity_positions_m-30_000)/4000)**2)
    x=model.initial_state(0.0)
    x[model.h_idx] = h
    x[model.u_idx]=u
    x[0]=0.
    before=np.average(model.height_positions_m,weights=np.maximum(x[model.h_idx],0))
    for _ in range(18):
        x=model.step(x,0.)
    after=np.average(model.height_positions_m,weights=np.maximum(x[model.h_idx],0))
    speed=(after-before)/(18*cfg.dt_s)
    assert abs(speed-model.analytical_wave_speed_m_s)<0.8


def test_sampling_interpolation():
    model=ShallowWater1D(ModelConfig(cells=20))
    states=np.zeros(model.nstate)
    states[model.h_idx]=model.height_positions_m/1000
    out=model.sample_heights(states,np.array([5000.,25000.]))
    np.testing.assert_allclose(out,[5,25],atol=1e-12)
