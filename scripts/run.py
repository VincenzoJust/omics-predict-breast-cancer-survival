"""Single entry point: python scripts/run.py [--quick]
download -> cohorts -> internal CV (TCGA) -> external validation (METABRIC) -> bootstrap, calibration, sanity, figures."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis

from src import data, evaluate as ev, models

QUICK = "--quick" in sys.argv  # 1 repeat, 50 bootstraps: for development only
REPEATS, N_BOOT = (1, 50) if QUICK else (3, 1000)
N_JOBS = 6  # 16 GB RAM: MOFA workers are memory-hungry; BLAS threads per worker change numerics slightly
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def blocks(fs, R, M):
    return {"C": [], "C+R": [R], "C+R+M": [R, M]}[fs]


def fit_predict(m, Xc_a, B_a, y_a, Xc_b, B_b):
    pred, info = models.MODELS[m](Xc_a, B_a, y_a)
    return pred(Xc_b, B_b), info


def one_fold(Xc, R, M, y, runs, tag, r, k, a, b):
    rows, hp = [], []
    for m, fs in runs:
        t0 = time.time()
        B = blocks(fs, R, M)
        S, info = fit_predict(m, Xc[a], [x[a] for x in B], y[a], Xc[b], [x[b] for x in B])
        key = {"setting": tag, "repeat": r, "fold": k, "run": f"{m}|{fs}"}
        if not (np.isfinite(S).all() and S.max() <= 1 + 1e-9):  # fail loudly on degenerate survival curves
            raise ValueError(f"invalid survival prediction: {key} {info}")
        rows.append({**key, **ev.metrics(y[b], S, y), "seconds": round(time.time() - t0, 1)})
        hp.append({**key, **info})
    return rows, hp


def cv(Xc, R, M, y, runs, repeats, tag):
    """Repeated stratified 5-fold CV; folds run in parallel (independent, so results do not depend on order)."""
    jobs = [(r, k, a, b) for r in range(repeats)
            for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(Xc, y["event"]))]
    out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one_fold)(Xc, R, M, y, runs, tag, *j) for j in jobs)
    return pd.DataFrame([x for o in out for x in o[0]]), pd.DataFrame([x for o in out for x in o[1]])


def fold_deltas(df, ref_run=ev.REF):
    """Per-fold paired difference vs a reference run, long format."""
    met = [c for c in df.columns if c[:2] in ("C_", "AU", "IB")]
    long = df.melt(["setting", "repeat", "fold", "run"], met, "metric")
    ref = long[long.run == ref_run].drop(columns="run").rename(columns={"value": "ref"})
    d = long[long.run != ref_run].merge(ref, on=["setting", "repeat", "fold", "metric"])
    return d.assign(delta=d.value - d.ref)


def summary_rows(name, d):
    return {"cohort": name, "n": len(d), "deaths": int(d.event.sum()),
            "deaths_within_5y": int(((d.time <= 5) & (d.event == 1)).sum()),
            "median_followup_censored_y": round(d.time[d.event == 0].median(), 2),
            "age_median": d.age.median(), **{f"stage_{s}": int((d.stage == i).sum())
                                              for i, s in enumerate(("I", "II", "III", "IV"), 1)}}


def main():
    md5 = data.download()
    pd.Series(md5, name="md5").rename_axis("file").to_csv(RES / "data_files.csv")
    co = data.load()
    tc, tr, tm = co["tcga"]
    mc, mr = co["metabric"]
    has_m = tm.notna().all(axis=1).values
    pd.DataFrame([summary_rows("TCGA-BRCA, RNA (external training set)", tc),
                  summary_rows("TCGA-BRCA, RNA + methylation (internal CV cohort)", tc[has_m]),
                  summary_rows("METABRIC (external test set)", mc)]).to_csv(RES / "data_summary.csv", index=False)
    pd.DataFrame([{"cohort": n, "term": t, "HR": h} for n, d in (("TCGA-BRCA", tc), ("METABRIC", mc))
                  for t, h in zip(["age per 10 y", "stage II vs I", "stage III vs I", "stage IV vs I"],
                                  np.exp(CoxPHSurvivalAnalysis().fit(data.clin_X(d), data.y_of(d)).coef_))]
                 ).to_csv(RES / "clinical_cox_hr.csv", index=False)
    ev.plot_km({"TCGA-BRCA": tc, "METABRIC": mc}, FIG / "km_by_stage.png")

    # ---- internal: repeated 5-fold CV on the TCGA patients with RNA + methylation, all feature sets
    Xc, y = data.clin_X(tc[has_m]), data.y_of(tc[has_m])
    R, M = tr.values[has_m], tm.values[has_m]
    cvm, hp = cv(Xc, R, M, y, models.RUNS, REPEATS, "internal")
    cvm.to_csv(RES / "cv_metrics.csv", index=False)
    met = ["C_uno_5y", "C_harrell", "AUC_3y", "AUC_5y", "IBS_1_5y"]
    cv_sum = (cvm.groupby("run", sort=False)[met].agg(["mean", "std"]).round(4))
    cv_sum.columns = [f"{a}_{b}" for a, b in cv_sum.columns]
    cv_sum.to_csv(RES / "cv_summary.csv")

    # ---- sanity: same CV (1 repeat) with omic rows permuted across patients (outcome kept)
    perm = np.random.default_rng(data.SEED).permutation(len(y))
    sruns = [r for r in models.RUNS if r[1] in ("C", "C+R")]
    san, _ = cv(Xc, R[perm], M[perm], y, sruns, 1, "permuted_omics")
    fold_deltas(san).groupby(["run", "metric"]).delta.agg(["mean", "std"]).round(4).to_csv(RES / "sanity.csv")

    # ---- external: fit on all TCGA with RNA, predict METABRIC (C and C+R only)
    Xt, yt, Rt = data.clin_X(tc), data.y_of(tc), tr.values
    Xm, ym, Rm = data.clin_X(mc), data.y_of(mc), mr.values
    ext_runs = [r for r in models.RUNS if r[1] != "C+R+M"]
    preds, ext, cal = {}, [], []
    for m, fs in ext_runs:
        k = f"{m}|{fs}"
        preds[k], info = fit_predict(m, Xt, blocks(fs, Rt, None), yt, Xm, blocks(fs, Rm, None))
        dec, cs = ev.calibration(ym, preds[k])
        cal.append(dec.assign(run=k))
        ext.append({"run": k, **ev.metrics(ym, preds[k], ym), **cs})
        hp = pd.concat([hp, pd.DataFrame([{"setting": "external", "run": k, **info}])])
        print(f"[external] {k}: C_uno {ext[-1]['C_uno_5y']:.3f}", flush=True)
    ext = pd.DataFrame(ext)
    ext.round(4).to_csv(RES / "external_metrics.csv", index=False)
    cal = pd.concat(cal)
    cal.round(4).to_csv(RES / "calibration.csv", index=False)
    hp.to_csv(RES / "hparams.csv", index=False)

    # ---- paired deltas vs each clinical reference: internal (fold distribution) + external (bootstrap)
    boot = ev.bootstrap(ym, preds, N_BOOT)
    point = ext.set_index("run")[met]
    tab = []
    for ref in (ev.REF, ev.REF2):
        ext_d = ev.boot_deltas(boot, point, ref)
        int_d = (fold_deltas(cvm, ref).groupby(["run", "metric"], sort=False).delta
                 .agg(est="mean", lo=lambda x: x.quantile(0.025), hi=lambda x: x.quantile(0.975)).reset_index())
        tab += [int_d.assign(source="internal", ref=ref), ext_d.assign(source="external", ref=ref)]
    tab = pd.concat(tab)
    tab[["ref", "source", "run", "metric", "est", "lo", "hi"]].round(4).to_csv(RES / "paired_deltas.csv", index=False)

    ev.plot_deltas(tab[tab.ref == ev.REF], FIG / "deltas_forest.png")
    ev.plot_td_auc(ym, preds, FIG / "td_auc_metabric.png")
    ev.plot_calibration(cal, FIG / "calibration_metabric.png")
    print("done")


if __name__ == "__main__":
    main()
