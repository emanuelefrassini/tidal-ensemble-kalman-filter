"""Generate deterministic numerical experiments and publishable figures.

Usage:
    python -m experiments.run_all
    python -m experiments.run_all --ensemble-size 100 --seed 2026
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tidal_enkf.data import (STATIONS, STATION_X_M, load_observations,
                             load_western_tide)
from tidal_enkf.filter import EnsembleKalmanFilter, FilterConfig
from tidal_enkf.metrics import metrics
from tidal_enkf.model import ModelConfig, ShallowWater1D

ROOT = Path(__file__).resolve().parents[1]


def plot_stations(times, observed, baseline, enkf, filename, *, title, holdout=None):
    fig, axes = plt.subplots(2,2,figsize=(13,7.2),sharex=True)
    tt = np.arange(len(times))*10/60  # ten-minute samples, in hours
    for i,(ax,station) in enumerate(zip(axes.flat,STATIONS)):
        ax.plot(tt,observed[:,i],label="Observation", color="#1F2937",linewidth=1.7)
        ax.plot(tt,baseline[:,i],label="Baseline",color="#DC7B30",linewidth=1.45,alpha=.9)
        if enkf is not None:
            ax.plot(tt,enkf[:,i],label="EnKF analysis",color="#2876C5",linewidth=1.2)
        label=f"{station.capitalize()} · {STATION_X_M[i]/1000:g} km"
        if holdout==i:
            label+=" · HELD OUT"
        ax.set_title(label,fontsize=11)
        ax.set_ylabel("Water level (m)")
        ax.grid(alpha=.2)
    for ax in axes[-1,:]:
        ax.set_xlabel("Hours since 5 Dec 2013 00:00 UTC")
    axes[0,0].legend(loc="upper left",ncol=2,fontsize=9)
    fig.suptitle(title,fontsize=14,y=.995)
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(filename,dpi=170,bbox_inches="tight")
    plt.close(fig)


def plot_rmse_comparison(baseline_error, updated_error, filename):
    x=np.arange(4)
    fig, ax=plt.subplots(figsize=(9.2,4.4))
    ax.bar(x-.18,baseline_error,width=.35,color="#DC7B30",label="Baseline")
    ax.bar(x+.18,updated_error,width=.35,color="#2876C5",label="EnKF analysis")
    ax.set_xticks(x,list(s.capitalize() for s in STATIONS))
    ax.set_ylabel("RMSE (m), hours 6–48")
    ax.set_title("Storm-water-level assimilation · Bath withheld from updates")
    ax.legend()
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(filename,dpi=170,bbox_inches="tight");plt.close(fig)


def make_twin(model, base, rng, boundary_sd_m=0.2, obs_sd_m=0.12):
    alpha=np.exp(-model.config.dt_s/(6*3600))
    innovations=boundary_sd_m*np.sqrt(1-alpha**2)
    eta=np.zeros(len(base))
    eta[0]=rng.normal(0,boundary_sd_m)
    for i in range(1,len(eta)):
        eta[i]=alpha*eta[i-1]+innovations*rng.standard_normal()
    truth_states=model.run(base+eta)
    truth_y=model.sample_heights(truth_states,STATION_X_M)
    measured=truth_y+rng.normal(0,obs_sd_m,size=truth_y.shape)
    return truth_states,truth_y,measured,eta


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--ensemble-size",type=int,default=100)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--figures",type=Path,default=ROOT/"figures")
    args=parser.parse_args()
    args.figures.mkdir(exist_ok=True,parents=True)

    times, storm_obs=load_observations(ROOT/"data","waterlevel")
    tt, tide_obs=load_observations(ROOT/"data","tide")
    tb, tide_boundary=load_western_tide(ROOT/"data")
    assert np.array_equal(times,tt) and np.array_equal(tt,tb)

    model=ShallowWater1D(ModelConfig())
    baseline_states=model.run(tide_boundary)
    baseline_y=model.sample_heights(baseline_states,STATION_X_M)
    plot_stations(times,tide_obs,baseline_y,None,args.figures/"01_baseline_tide.png",
                  title="Shallow-water model vs astronomical tides (no data assimilation)")
    # Keep a full six hours of model spinup out of diagnostic metrics.
    burnin=36
    base_tide_metrics=metrics(baseline_y,tide_obs,burnin)

    # Real measurements: first 3 stations in analysis, Bath strictly withheld.
    fcfg=FilterConfig(ensemble_size=args.ensemble_size,seed=args.seed)
    filter_=EnsembleKalmanFilter(model,fcfg)
    out=filter_.assimilate(tide_boundary,storm_obs,STATION_X_M,
                           assimilated_columns=(0,1,2))
    analysis_y=model.sample_heights(out["analysis_states"][:,:-1],STATION_X_M)
    prior_y=model.sample_heights(out["forecast_states"][:,:-1],STATION_X_M)
    plot_stations(times,storm_obs,baseline_y,analysis_y,
                  args.figures/"02_storm_assimilation_holdout.png",
                  title="2013 storm data assimilation: three stations used, Bath withheld",
                  holdout=3)
    storm_base=metrics(baseline_y,storm_obs,burnin)
    storm_enkf=metrics(analysis_y,storm_obs,burnin)
    storm_prior=metrics(prior_y,storm_obs,burnin)
    plot_rmse_comparison(storm_base["rmse_m"],storm_enkf["rmse_m"],
                         args.figures/"03_rmse_comparison.png")

    # TRUE FREE FORECASTS: after the cutoff no observations are ingested.
    # Forecasts use the then-current augmented analysis mean and propagate the
    # expected AR(1) boundary error forward without observing future values.
    forecast_scores=[]
    fig, axes=plt.subplots(1,2,figsize=(13,3.8),sharey=True)
    steps=48 # 8-hour forecast horizon, 10-minute time steps
    for ax,cutoff_hour in zip(axes,[20,24]):
        cutoff=int(cutoff_hour*3600/model.config.dt_s)
        x=out["analysis_states"][cutoff,:-1].copy()
        eta_mean=float(out["analysis_states"][cutoff,-1])
        forecast=[model.sample_heights(x, STATION_X_M)[3]]
        for next_idx in range(cutoff+1, cutoff+steps+1):
            eta_mean*=out["alpha"]
            x=model.step(x,tide_boundary[next_idx]+eta_mean)
            forecast.append(model.sample_heights(x,STATION_X_M)[3])
        observed=storm_obs[cutoff:cutoff+steps+1,3]
        err=np.asarray(forecast)-observed
        forecast_scores.append({"cutoff_hour":cutoff_hour,"horizon_hours":8,
                                "bath_forecast_rmse_m":float(np.sqrt(np.mean(err**2)))})
        hr=np.arange(steps+1)/6
        ax.plot(hr,observed,color="#1F2937",label="Bath observed",linewidth=1.8)
        ax.plot(hr,forecast,color="#2876C5",label="Forecast from cutoff",linewidth=1.5)
        ax.set_title(f"Stop assimilating at +{cutoff_hour} h")
        ax.set_xlabel("Hours after cutoff")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Water level (m)")
    axes[0].legend()
    fig.suptitle("Eight-hour free forecasts at withheld Bath (no observations after cutoff)")
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(args.figures/"06_free_forecast.png",dpi=170,bbox_inches="tight")
    plt.close(fig)

    # Independent identical-twin experiment with synthetic stochastic boundary.
    rng=np.random.default_rng(args.seed+103)
    _,twin_truth,twin_obs,eta=make_twin(model,tide_boundary,rng)
    twin_out=EnsembleKalmanFilter(model,fcfg).assimilate(tide_boundary,twin_obs,
                       STATION_X_M,assimilated_columns=(0,1,2))
    twin_analysis_y=model.sample_heights(twin_out["analysis_states"][:,:-1],STATION_X_M)
    plot_stations(times,twin_truth,baseline_y,twin_analysis_y,
                  args.figures/"04_identical_twin.png",
                  title="Identical twin: synthetic truth vs baseline and EnKF (Bath held out)",
                  holdout=3)
    twin_base=metrics(baseline_y,twin_truth,burnin)
    twin_enkf=metrics(twin_analysis_y,twin_truth,burnin)

    # Repeated-seed convergence study against the known twin truth, not noisy obs.
    sizes=[20,50,100,200]
    seeds=[1234,1235,1236]
    convergence=[]
    for N in sizes:
        scores=[]
        for seed in seeds:
            res=EnsembleKalmanFilter(model,FilterConfig(ensemble_size=N,seed=seed)).assimilate(
                tide_boundary,twin_obs,STATION_X_M,(0,1,2))
            pred=model.sample_heights(res["analysis_states"][:,:-1],STATION_X_M)
            scores.append(float(metrics(pred,twin_truth,burnin)["rmse_m"][3]))
        convergence.append({"ensemble_size":N,"holdout_rmse_m_mean":float(np.mean(scores)),
                            "holdout_rmse_m_std":float(np.std(scores,ddof=1)),
                            "n_repeats":len(scores)})
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.errorbar([q["ensemble_size"] for q in convergence],
                [q["holdout_rmse_m_mean"] for q in convergence],
                yerr=[q["holdout_rmse_m_std"] for q in convergence],
                marker="o", capsize=4,color="#2876C5")
    ax.set_title("Identical twin · held-out Bath RMSE across three filter seeds")
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Holdout RMSE (m)")
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(args.figures/"05_convergence.png",dpi=170);plt.close(fig)

    report={
        "validation": "exploratory numerical experiments; see README for scope and limitations",
        "configuration": {"seed":args.seed,"ensemble_size":args.ensemble_size,
                          "hours_excluded_from_metrics":burnin*model.config.dt_s/3600,
                          "model_length_m":model.config.length_m,
                          "model_depth_m":model.config.depth_m,
                          "wave_speed_m_s":float(model.analytical_wave_speed_m_s),
                          "boundary_AR1_alpha":float(out["alpha"]),
                          "boundary_AR1_innovation_sd_m":float(out["innovation_sd_m"]),
                          "assimilation_stations":list(STATIONS[:3]),
                          "held_out_station":"bath"},
        "astronomical_tide_baseline":{"rmse_m":dict(zip(STATIONS,base_tide_metrics["rmse_m"].tolist())),
                                      "bias_m":dict(zip(STATIONS,base_tide_metrics["bias_m"].tolist()))},
        "storm_baseline_rmse_m":dict(zip(STATIONS,storm_base["rmse_m"].tolist())),
        "storm_enkf_analysis_rmse_m":dict(zip(STATIONS,storm_enkf["rmse_m"].tolist())),
        "storm_enkf_prior_rmse_m":dict(zip(STATIONS,storm_prior["rmse_m"].tolist())),
        "identical_twin_baseline_rmse_m":dict(zip(STATIONS,twin_base["rmse_m"].tolist())),
        "identical_twin_enkf_rmse_m":dict(zip(STATIONS,twin_enkf["rmse_m"].tolist())),
        "identical_twin_convergence":convergence,
        "eight_hour_free_forecasts":forecast_scores,
        "warning":"Assimilated-station analysis errors are in-sample; Bath is withheld. No parameter tuning used Bath."
    }
    path=ROOT/"results.json"
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Saved",path)
    print("Analytical wave speed:",report["configuration"]["wave_speed_m_s"])
    print("Storm baseline RMSE:",report["storm_baseline_rmse_m"])
    print("Storm EnKF analysis RMSE:",report["storm_enkf_analysis_rmse_m"])
    print("Storm EnKF forecast RMSE:",report["storm_enkf_prior_rmse_m"])
    print("Twin baseline RMSE:",report["identical_twin_baseline_rmse_m"])
    print("Twin EnKF RMSE:",report["identical_twin_enkf_rmse_m"])
    print("Convergence:",convergence)
    print("Free forecasts:",forecast_scores)

if __name__=="__main__":
    main()
