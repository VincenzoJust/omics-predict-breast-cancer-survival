"""Follow-up, external validation only: does the main result hold for disease-specific survival (DSS) and against a
richer clinical baseline (+ quadratic age, ER, PR, HER2, node status)? python scripts/followup.py [--quick]
All four endpoint x baseline cells use the same complete-case patients, so they are comparable with each other
(not exactly with the main analysis, which keeps patients missing receptor / node status)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from joblib import Parallel, delayed

from src import data, evaluate as ev, models

N_BOOT = 50 if "--quick" in sys.argv else 1000
RUNS = [("cox", "C"), ("deepsurv", "C"), ("coxnet", "C+R"), ("mofa", "C+R")]
RES = ROOT / "results"


def y_of(d, endpoint):
    return data.y_of(d if endpoint == "OS" else d.assign(time=d.dss_time, event=d.dss_event))


def cell(endpoint, baseline, tc, tr, mc, mr):
    X = data.clin_X if baseline == "age+stage" else data.clin_X_rich
    yt, ym = y_of(tc, endpoint), y_of(mc, endpoint)
    preds, rows = {}, []
    for m, fs in RUNS:
        B = (lambda R: [R]) if fs == "C+R" else (lambda R: [])
        try:
            pred, info = models.MODELS[m](X(tc), B(tr), yt)
        except ArithmeticError:  # CoxNet on DSS: unpenalised stage IV dummy (11/14 deaths) diverges in sksurv
            rows.append({"run": f"{m}|{fs}", "failed": True})
            continue
        S = preds[f"{m}|{fs}"] = pred(X(mc), B(mr))
        rows.append({"run": f"{m}|{fs}", **ev.metrics(ym, S, ym), **ev.calibration(ym, S)[1], **info})
    point = pd.DataFrame(rows).set_index("run")
    boot = ev.bootstrap(ym, preds, N_BOOT)
    tag = {"endpoint": endpoint, "baseline": baseline}
    deltas = pd.concat([ev.boot_deltas(boot, point, ref).assign(ref=ref) for ref in (ev.REF, ev.REF2)])
    return point.reset_index().assign(**tag), deltas.assign(**tag)


def main():
    data.download()
    co = data.load()
    (tc, tr, _), (mc, mr) = co["tcga"], co["metabric"]
    cc = lambda d, e: d.join(e).dropna().query("dss_time > 0")  # complete case, same patients in every cell
    tc, mc = cc(tc, data.tcga_extra()), cc(mc, data.metabric_extra())
    tr, mr = tr.loc[tc.index].values, mr.loc[mc.index].values
    pd.DataFrame([{"cohort": n, "n": len(d), "OS_deaths": int(d.event.sum()), "DSS_deaths": int(d.dss_event.sum()),
                   **{k: round(d[k].mean(), 3) for k in ("er", "pr", "her2", "node")}}
                  for n, d in (("TCGA-BRCA", tc), ("METABRIC", mc))]).to_csv(RES / "followup_cohorts.csv", index=False)
    out = Parallel(n_jobs=4)(delayed(cell)(e, b, tc, tr, mc, mr)
                             for e in ("OS", "DSS") for b in ("age+stage", "rich"))
    pd.concat([o[0] for o in out]).round(4).to_csv(RES / "followup_metrics.csv", index=False)
    d = pd.concat([o[1] for o in out])
    d[["endpoint", "baseline", "ref", "run", "metric", "est", "lo", "hi"]].round(4).to_csv(
        RES / "followup_deltas.csv", index=False)
    print(d[d.metric == "C_uno_5y"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
