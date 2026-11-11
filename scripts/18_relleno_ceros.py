"""Efecto de las 23 EPOC < 3 s (rellenadas con ceros): AUC sin ellas y probabilidad/umbral frente al resto de EPOC."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metricas import umbral_por_sensibilidad  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/validacion_cruzada/control_duracion"
meta = pd.read_csv(ROOT / "data/processed/copd_cough/metadata.csv")[["id", "grupo", "dur_s"]]
u = pd.read_csv(ROOT / "reports/validacion_cruzada/cnn_8s/predicciones_oof.csv").merge(meta, on="id")
u["corta"] = (u.dur_s < 3) & (u.label == 1)
mods = {"CNN audio completo": "p_CNN", "CNN 8 s": "p_CNN_8s"}
reps = sorted(u.repeticion.unique()); rng = np.random.default_rng(0)
lin = []


def log(s=""):
    print(s); lin.append(s)


def auc_ic(s):
    gi = {r: {g: d.index.to_numpy() for g, d in s[s.repeticion == r].groupby("grupo")} for r in reps}
    gs = s.grupo.unique(); boot = {m: [] for m in mods}
    for _ in range(1000):
        sel = rng.choice(gs, len(gs), replace=True); v = {m: [] for m in mods}
        for r in reps:
            d = s.loc[np.concatenate([gi[r][g] for g in sel if g in gi[r]])]
            if d.label.nunique() < 2:
                continue
            for m, c in mods.items():
                v[m].append(roc_auc_score(d.label, d[c]))
        for m in mods:
            boot[m].append(np.mean(v[m]))
    return {m: (np.mean([roc_auc_score(s[s.repeticion == r].label, s[s.repeticion == r][c]) for r in reps]),
                *np.percentile(boot[m], [2.5, 97.5])) for m, c in mods.items()}


log("== (a) AUC de la CNN con IC95 por participante ==")
for nom, s in (("Todas (853)", u), (f"Sin las {int(u[u.repeticion == 0].corta.sum())} EPOC < 3 s", u[~u.corta])):
    s = s.reset_index(drop=True); n = s[s.repeticion == 0]
    log(f"  {nom}  [EPOC {int((n.label == 1).sum())}, sanos {int((n.label == 0).sum())}]")
    for m, (pt, lo, hi) in auc_ic(s).items():
        log(f"    {m:20s} {pt:.3f}  [{lo:.3f}; {hi:.3f}]")

log("\n== (b) Probabilidad media y % sobre el umbral honesto, solo EPOC ==")
for m, c in mods.items():
    sup = np.zeros(len(u), bool)
    for r in reps:
        g = u[u.repeticion == r]
        for k in g.fold.unique():
            te, resto = g.fold == k, g.fold != k
            thr = umbral_por_sensibilidad(g.label[resto].to_numpy(), g[c][resto].to_numpy(), 0.85)
            sup[g.index[te]] = g[c][te].to_numpy() >= thr
    u[f"sup_{c}"] = sup
    e = u[u.label == 1]
    log(f"  {m}")
    for nom, d in (("EPOC < 3 s", e[e.corta]), ("Otras EPOC", e[~e.corta])):
        log(f"    {nom:12s} n={d.id.nunique():3d}  prob. media {d[c].mean():.3f}  sobre el umbral {d[f'sup_{c}'].mean():.1%}")
(OUT / "relleno_ceros.txt").write_text("\n".join(lin), encoding="utf-8")
