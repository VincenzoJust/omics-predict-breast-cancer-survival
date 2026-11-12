"""Five model families. Each fit(Xc, blocks, y) -> (predict(Xc, blocks) -> S [n x len(TIMES)], hparams dict).
Xc = clinical matrix, blocks = list of omic matrices ([] for C, [RNA] for C+R, [RNA, meth] for C+R+M)."""
import contextlib
import copy
import io

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import concordance_index_censored
from torch import nn

from .data import SEED

TIMES = np.linspace(1, 5, 17)  # years; IBS over 1-5 y, AUC at 3 and 5 y
N_PC, N_FACTORS, L1_RATIO = 32, 15, 0.5


def _surv(fns):
    return np.array([f(TIMES) for f in fns])


def cox(Xc, blocks, y):
    m = CoxPHSurvivalAnalysis().fit(Xc, y)
    return lambda Xc, b: _surv(m.predict_survival_function(Xc)), {}


def coxnet(Xc, blocks, y):
    """Elastic net Cox, clinical terms unpenalised; alpha by 3-fold inner CV (Harrell C) over one shared path."""
    X = np.hstack([Xc, *blocks])
    pf = np.r_[np.zeros(Xc.shape[1]), np.ones(X.shape[1] - Xc.shape[1])]
    # alpha_min_ratio=0.1: smaller alphas give near-unpenalised fits (hundreds of features, ~75 events) whose
    # Breslow baseline degenerates (S > 1 / inf); one inner CV picked such an alpha.
    est = lambda **kw: CoxnetSurvivalAnalysis(l1_ratio=L1_RATIO, penalty_factor=pf, n_alphas=30, alpha_min_ratio=0.1, **kw)
    alphas = est().fit(X, y).alphas_
    score = np.zeros(len(alphas))
    for a, b in KFold(3, shuffle=True, random_state=SEED).split(X):
        m = est(alphas=alphas).fit(X[a], y[a])
        score += [concordance_index_censored(y[b]["event"], y[b]["time"], m.predict(X[b], alpha=al))[0] for al in alphas]
    best = alphas[score.argmax()]
    m = est(alphas=alphas, fit_baseline_model=True).fit(X, y)
    n_nz = int((m.coef_[Xc.shape[1]:, score.argmax()] != 0).sum())
    return (lambda Xc, b: _surv(m.predict_survival_function(np.hstack([Xc, *b]), alpha=best)),
            {"alpha": best, "n_omic_nonzero": n_nz})


def _pcs(Xc, blocks):
    """Clinical + N_PC principal components per omic block (PCA fit on training data), standardised."""
    pca = [PCA(N_PC, random_state=SEED).fit(b) for b in blocks]
    f = lambda Xc, b: np.hstack([Xc, *[p.transform(x) for p, x in zip(pca, b)]])
    sc = StandardScaler().fit(f(Xc, blocks))
    return lambda Xc, b: sc.transform(f(Xc, b))


def rsf(Xc, blocks, y):
    f = _pcs(Xc, blocks)
    X = f(Xc, blocks)
    fits = [RandomSurvivalForest(n_estimators=300, min_samples_leaf=leaf, max_features="sqrt", oob_score=True,
                                 n_jobs=-1, random_state=SEED).fit(X, y) for leaf in (10, 30)]
    m = max(fits, key=lambda m: m.oob_score_)
    return lambda Xc, b: _surv(m.predict_survival_function(f(Xc, b))), {"min_samples_leaf": m.min_samples_leaf}


def _nll(lp, t, e):
    """Negative Cox partial log-likelihood (Breslow ties), mean over events."""
    o = torch.argsort(t, descending=True)
    lp, e = lp[o], e[o]
    return -((lp - torch.logcumsumexp(lp, 0)) * e).sum() / e.sum().clamp(min=1)


def deepsurv(Xc, blocks, y):
    """DeepSurv-style MLP (one hidden layer) on clinical + PCs; early stopping on a 20% inner split."""
    f = _pcs(Xc, blocks)
    X = torch.tensor(f(Xc, blocks), dtype=torch.float32)
    t, e = (torch.tensor(y[k].astype(np.float32)) for k in ("time", "event"))
    tr, va = map(torch.as_tensor, train_test_split(np.arange(len(y)), test_size=0.2, stratify=y["event"],
                                                   random_state=SEED))
    torch.manual_seed(SEED)
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    best, wait, state, ep = np.inf, 0, None, 0
    for ep in range(1000):
        net.train(); opt.zero_grad()
        _nll(net(X[tr]).squeeze(1), t[tr], e[tr]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = _nll(net(X[va]).squeeze(1), t[va], e[va]).item()
        if v < best:
            best, wait, state = v, 0, copy.deepcopy(net.state_dict())
        elif (wait := wait + 1) > 50:
            break
    net.load_state_dict(state); net.eval()

    def lp(Z):
        with torch.no_grad():
            return net(torch.as_tensor(Z, dtype=torch.float32)).squeeze(1).numpy().astype(float)

    bre = BreslowEstimator().fit(lp(X), y["event"], y["time"])
    return lambda Xc, b: _surv(bre.get_survival_function(lp(f(Xc, b)))), {"best_epoch": ep - wait}


def mofa(Xc, blocks, y):
    """MOFA+ on the omic views (training data only); factors for any patient = least-squares projection on the
    stacked weights, Z = Y pinv(W)^T, used for train and test alike; then Cox(clinical + standardised factors)."""
    from mofapy2.run.entry_point import entry_point
    with contextlib.redirect_stdout(io.StringIO()):  # mofapy2 prints banners regardless of quiet=True
        ent = entry_point()
        ent.set_data_options(scale_views=False)
        ent.set_data_matrix([[b] for b in blocks], likelihoods=["gaussian"] * len(blocks))
        ent.set_model_options(factors=N_FACTORS)
        ent.set_train_options(iter=1000, convergence_mode="fast", seed=SEED, verbose=False, quiet=True)
        ent.build(); ent.run()
    P = np.linalg.pinv(np.vstack(ent.model.nodes["W"].getExpectation())).T
    z = lambda b: np.hstack(b) @ P
    sc = StandardScaler().fit(z(blocks))
    f = lambda Xc, b: np.hstack([Xc, sc.transform(z(b))])
    m = CoxPHSurvivalAnalysis(alpha=0.1).fit(f(Xc, blocks), y)
    return lambda Xc, b: _surv(m.predict_survival_function(f(Xc, b))), {"n_factors": P.shape[1]}


MODELS = {"cox": cox, "coxnet": coxnet, "rsf": rsf, "deepsurv": deepsurv, "mofa": mofa}
# (model, feature set). Non-linear models also run on C alone to separate "omics effect" from "model effect".
RUNS = [("cox", "C"), ("rsf", "C"), ("deepsurv", "C"),
        ("coxnet", "C+R"), ("rsf", "C+R"), ("deepsurv", "C+R"), ("mofa", "C+R"),
        ("coxnet", "C+R+M"), ("rsf", "C+R+M"), ("deepsurv", "C+R+M"), ("mofa", "C+R+M")]
