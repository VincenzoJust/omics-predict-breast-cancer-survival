"""Download, cohorts, harmonisation. Every omic transform here is outcome-free and done once per cohort."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 0
N_GENES, N_CPGS = 2000, 5000
XENA = "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/"
CBIO = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/"
URLS = {
    "HiSeqV2.gz": XENA + "HiSeqV2.gz",
    "HumanMethylation450.gz": XENA + "HumanMethylation450.gz",
    "BRCA_clinicalMatrix": XENA + "BRCA_clinicalMatrix",  # receptor / node status (follow-up only)
    "Survival_SupplementalTable_S1_20171025_xena_sp":  # TCGA-CDR (Liu et al. 2018)
        "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp",
    **{f: CBIO + f for f in ("data_clinical_patient.txt", "data_clinical_sample.txt", "data_mrna_illumina_microarray.txt")},
}
STAGE = {"I": 1, "II": 2, "III": 3, "IV": 4}


def download():
    """Fetch missing files; return {file: md5} of what is on disk."""
    DATA.mkdir(exist_ok=True)
    for f, u in URLS.items():
        if not (DATA / f).exists():
            with requests.get(u, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(DATA / f, "wb") as fh:
                    for c in r.iter_content(1 << 20):
                        fh.write(c)
    md5 = {}
    for f in URLS:
        with open(DATA / f, "rb") as fh:
            md5[f] = hashlib.file_digest(fh, "md5").hexdigest()
    return md5


def tcga_clin():
    s = pd.read_csv(DATA / "Survival_SupplementalTable_S1_20171025_xena_sp", sep="\t")
    s = s[(s["cancer type abbreviation"] == "BRCA") & s["sample"].str.endswith("-01") & (s.gender == "FEMALE")]
    st = s.ajcc_pathologic_tumor_stage.str.extract(r"^Stage (IV|I{1,3})")[0].map(STAGE)  # substages collapsed
    d = pd.DataFrame({"age": s.age_at_initial_pathologic_diagnosis.values, "stage": st.values,
                      "time": s["OS.time"].values / 365.25, "event": s.OS.values}, index=s["sample"].values)
    return d.dropna().query("time > 0")


def metabric_clin():
    p = pd.read_csv(DATA / "data_clinical_patient.txt", sep="\t", comment="#", index_col="PATIENT_ID")
    s = pd.read_csv(DATA / "data_clinical_sample.txt", sep="\t", comment="#", index_col="PATIENT_ID")
    d = pd.DataFrame({"age": p.AGE_AT_DIAGNOSIS, "stage": s.TUMOR_STAGE.reindex(p.index),
                      "time": p.OS_MONTHS / 12, "event": p.OS_STATUS.str[0].astype(float)})
    return d[d.stage.between(1, 4)].dropna().query("time > 0")  # stage 0 (DCIS) and missing stage dropped


PN = {"Positive": 1, "Negative": 0}  # Equivocal / Indeterminate -> missing


def tcga_extra():
    """Follow-up covariates: DSS (TCGA-CDR), ER/PR (IHC), HER2 (nature2012 final, else FISH, else IHC), node+."""
    s = pd.read_csv(DATA / "Survival_SupplementalTable_S1_20171025_xena_sp", sep="\t", index_col="sample")
    c = pd.read_csv(DATA / "BRCA_clinicalMatrix", sep="\t", index_col=0)
    n = c.pathologic_N.str.extract(r"^N([0-3])")[0].astype(float)  # NX -> missing
    her2 = (c.HER2_Final_Status_nature2012.map(PN).fillna(c.lab_procedure_her2_neu_in_situ_hybrid_outcome_type.map(PN))
            .fillna(c.lab_proc_her2_neu_immunohistochemistry_receptor_status.map(PN)))
    return pd.DataFrame({"dss_time": s["DSS.time"] / 365.25, "dss_event": s.DSS,
                         "er": c.breast_carcinoma_estrogen_receptor_status.map(PN),
                         "pr": c.breast_carcinoma_progesterone_receptor_status.map(PN),
                         "her2": her2, "node": (n > 0).astype(float).where(n.notna())})


def metabric_extra():
    """Follow-up covariates: DSS (death from other causes censored), ER/PR/HER2 status, any positive node."""
    p = pd.read_csv(DATA / "data_clinical_patient.txt", sep="\t", comment="#", index_col="PATIENT_ID")
    s = pd.read_csv(DATA / "data_clinical_sample.txt", sep="\t", comment="#", index_col="PATIENT_ID")
    ln = p.LYMPH_NODES_EXAMINED_POSITIVE
    return pd.DataFrame({"dss_time": p.OS_MONTHS / 12,
                         "dss_event": (p.VITAL_STATUS == "Died of Disease").astype(float).where(p.VITAL_STATUS.notna()),
                         "er": s.ER_STATUS.map(PN), "pr": s.PR_STATUS.map(PN), "her2": s.HER2_STATUS.map(PN),
                         "node": (ln > 0).astype(float).where(ln.notna())})


def zscore(x):
    return (x - x.mean()) / x.std()


def rna():
    """Shared genes, top N_GENES by TCGA variance (among genes with above-median METABRIC variance, to avoid
    microarray noise floor), z-scored within each cohort separately (RNA-seq vs microarray)."""
    cache = DATA / "rna.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    t = pd.read_csv(DATA / "HiSeqV2.gz", sep="\t", index_col=0).T
    t = t[t.index.str.endswith("-01")]
    m = pd.read_csv(DATA / "data_mrna_illumina_microarray.txt", sep="\t").drop(columns="Entrez_Gene_Id")
    m = m.dropna(subset=["Hugo_Symbol"]).groupby("Hugo_Symbol").mean()
    m = m[m.isna().mean(axis=1) < 0.2].T
    m = m.fillna(m.median())
    mv = m.var()
    genes = t.columns.intersection(mv[mv > mv.median()].index)
    top = t[genes].var().nlargest(N_GENES).index
    out = zscore(t[top]), zscore(m[top])
    pd.to_pickle(out, cache)
    return out


def meth():
    """TCGA 450k beta values: CpGs with <20% missing, top N_CPGS by variance (streamed), median-imputed, z-scored."""
    cache = DATA / "meth.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    best = None
    for c in pd.read_csv(DATA / "HumanMethylation450.gz", sep="\t", index_col=0, chunksize=20000):
        c = c.loc[:, c.columns.str.endswith("-01")].astype("float32")
        best = pd.concat([best, c[c.isna().mean(axis=1) < 0.2]])
        best = best.loc[best.var(axis=1).nlargest(N_CPGS).index]
    b = best.T
    out = zscore(b.fillna(b.median()))
    out.to_pickle(cache)
    return out


def clin_X(d):
    """Fixed (unfitted) clinical design: age per decade centred at 60, stage II/III/IV dummies vs I."""
    return np.column_stack([(d.age - 60) / 10, d.stage == 2, d.stage == 3, d.stage == 4]).astype(float)


def clin_X_rich(d):
    """clin_X + quadratic age, ER+, PR+, HER2+, any positive node (follow-up baseline)."""
    return np.column_stack([clin_X(d), ((d.age - 60) / 10) ** 2, d.er, d.pr, d.her2, d.node]).astype(float)


def y_of(d):
    return Surv.from_arrays(d.event.astype(bool).values, d.time.values)


def load():
    """Returns cohorts: tcga = (clin, rna, meth or NaN rows), metabric = (clin, rna). Patients need RNA."""
    tc, mc = tcga_clin(), metabric_clin()
    tr, mr = rna()
    tc, mc = tc.loc[tc.index.intersection(tr.index)], mc.loc[mc.index.intersection(mr.index)]
    return {"tcga": (tc, tr.loc[tc.index], meth().reindex(tc.index)), "metabric": (mc, mr.loc[mc.index])}
