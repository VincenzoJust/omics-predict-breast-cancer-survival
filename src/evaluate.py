"""Metrics (sksurv), paired bootstrap, calibration, figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.nonparametric import kaplan_meier_estimator

from .data import SEED
from .models import TIMES

REF = "cox|C"
REF2 = "deepsurv|C"  # flexible clinical-only model: the stronger clinical comparator
I3 = int(np.argmin(abs(TIMES - 3)))
# BERU palettes (Gold Tangerine + Lavender Void)
COLORS = {"cox": "#28283d", "coxnet": "#fe5416", "rsf": "#b287fd", "deepsurv": "#f9d84b", "mofa": "#b2f142"}
NAMES = {"cox": "Clinical Cox", "coxnet": "CoxNet", "rsf": "RSF", "deepsurv": "DeepSurv", "mofa": "MOFA+ -> Cox"}
STAGE_COLORS = ["#b287fd", "#f9d84b", "#fe5416", "#28283d"]


def metrics(y, S, y_cens):
    """y_cens: survival data used for the IPCW censoring estimate (the whole evaluation cohort)."""
    r5 = 1 - S[:, -1]
    return {"C_uno_5y": concordance_index_ipcw(y_cens, y, r5, tau=5)[0],
            "C_harrell": concordance_index_censored(y["event"], y["time"], r5)[0],
            "AUC_3y": cumulative_dynamic_auc(y_cens, y, 1 - S[:, I3], 3)[0][0],
            "AUC_5y": cumulative_dynamic_auc(y_cens, y, r5, 5)[0][0],
            "IBS_1_5y": integrated_brier_score(y_cens, y, S, TIMES)}


def bootstrap(y, preds, n=1000):
    """Metrics per bootstrap resample; same resamples for all models, so differences are paired.
    Returns a DataFrame indexed by (b, metric) with one column per run."""
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(n):
        i = rng.integers(0, len(y), len(y))
        rows += [(b, k, met, v) for k, S in preds.items() for met, v in metrics(y[i], S[i], y).items()]
    return pd.DataFrame(rows, columns=["b", "run", "metric", "value"]).pivot_table("value", ["b", "metric"], "run")


def boot_deltas(boot, point, ref):
    """Paired difference vs ref: point estimate from `point` (run x metric), 95% CI from bootstrap percentiles."""
    d = boot.sub(boot[ref], axis=0).drop(columns=ref).stack().rename("delta").reset_index()
    out = (d.groupby(["run", "metric"], sort=False).delta.quantile([0.025, 0.975]).unstack()
           .set_axis(["lo", "hi"], axis=1).reset_index())
    out["est"] = [point.loc[r, mt] - point.loc[ref, mt] for r, mt in zip(out.run, out.metric)]
    return out


def _km_at(y, t0=5):
    t, s = kaplan_meier_estimator(y["event"], y["time"])
    return s[t <= t0][-1] if (t <= t0).any() else 1.0


def calibration(y, S, bins=10):
    """5-y predicted risk vs KM observed per risk decile; slope = Cox coef of cloglog(pred 5-y risk);
    O/E = KM 5-y risk / mean predicted 5-y risk."""
    p = 1 - S[:, -1]
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    dec = pd.DataFrame([(k, p[q == k].mean(), 1 - _km_at(y[q == k])) for k in np.unique(q)],
                       columns=["decile", "pred_5y_risk", "obs_5y_risk"])
    cll = np.log(-np.log(np.clip(S[:, -1], 1e-6, 1 - 1e-6)))[:, None]
    slope = CoxPHSurvivalAnalysis().fit(cll, y).coef_[0]
    return dec, {"calibration_slope": slope, "O/E_5y": (1 - _km_at(y)) / p.mean()}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_km(cohorts, path):
    fig, axs = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 4), sharey=True)
    for ax, (name, d) in zip(axs, cohorts.items()):
        for s, c in zip((1, 2, 3, 4), STAGE_COLORS):
            g = d[d.stage == s]
            t, p = kaplan_meier_estimator(g.event.astype(bool), g.time)
            ax.step(t, p, where="post", color=c, label=f"Stage {'I' * s if s < 4 else 'IV'} (n={len(g)})")
        ax.set(title=name, xlabel="Years", xlim=(0, 15))
        ax.legend(fontsize=8)
    axs[0].set_ylabel("Overall survival")
    _save(fig, path)


def plot_deltas(tab, path):
    """tab: rows = run, columns (metric, source) with est/lo/hi. Forest plot, internal vs external."""
    metrics_ = ["C_uno_5y", "IBS_1_5y"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 0.4 * tab.run.nunique() + 1.5), sharey=True)
    runs = list(dict.fromkeys(tab.run))
    for ax, met in zip(axs, metrics_):
        for off, (src, mk) in zip((-0.15, 0.15), (("internal", "o"), ("external", "s"))):
            d = tab[(tab.metric == met) & (tab.source == src)].set_index("run").reindex(runs)
            yy = np.arange(len(runs)) + off
            ok = d.est.notna().values
            cols = [COLORS[r.split("|")[0]] for r in runs]
            ax.errorbar(d.est[ok], yy[ok], xerr=[(d.est - d.lo)[ok], (d.hi - d.est)[ok]], fmt="none", ecolor="grey", lw=1)
            ax.scatter(d.est[ok], yy[ok], c=[c for c, o in zip(cols, ok) if o], marker=mk, edgecolors="k", zorder=3,
                       label=f"{src} ({'15-fold mean, 2.5-97.5% of folds' if src == 'internal' else 'METABRIC, bootstrap 95% CI'})")
        ax.axvline(0, color="k", lw=0.8)
        ax.set(title=f"Δ {met} vs clinical Cox" + (" (higher = better)" if met.startswith("C") else " (lower = better)"))
    axs[0].set_yticks(range(len(runs)), [f"{NAMES[r.split('|')[0]]} [{r.split('|')[1]}]" for r in runs])
    axs[0].invert_yaxis()
    axs[0].legend(fontsize=7, loc="lower left")
    _save(fig, path)


def plot_td_auc(y, preds, path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for k, S in preds.items():
        auc = [cumulative_dynamic_auc(y, y, 1 - S[:, j], t)[0][0] for j, t in enumerate(TIMES)]
        m, fs = k.split("|")
        ax.plot(TIMES, auc, color=COLORS[m], ls="-" if fs != "C" else "--", label=f"{NAMES[m]} [{fs}]")
    ax.set(xlabel="Years", ylabel="Cumulative/dynamic AUC(t)", title="METABRIC (external)")
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_calibration(cal, path):
    runs = list(dict.fromkeys(cal.run))
    fig, axs = plt.subplots(1, len(runs), figsize=(2.6 * len(runs), 2.9), sharex=True, sharey=True)
    lim = max(cal.pred_5y_risk.max(), cal.obs_5y_risk.max()) * 1.05
    for ax, r in zip(axs, runs):
        d = cal[cal.run == r]
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.plot(d.pred_5y_risk, d.obs_5y_risk, "o-", color=COLORS[r.split("|")[0]], mec="k", ms=4)
        ax.set(title=f"{NAMES[r.split('|')[0]]}\n[{r.split('|')[1]}]", xlabel="Predicted 5-y risk")
    axs[0].set(ylabel="Observed 5-y risk (KM)", xlim=(0, lim), ylim=(0, lim))
    for ax in axs:
        ax.title.set_fontsize(8)
    _save(fig, path)
