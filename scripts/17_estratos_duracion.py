"""AUC por estratos de duración (sin reentrenar), sobre las grabaciones elegibles de cargar_meta."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import cargar_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/validacion_cruzada/control_duracion"; OUT.mkdir(parents=True, exist_ok=True)
meta = cargar_meta(ROOT)
u = pd.read_csv(ROOT / "reports/validacion_cruzada/predicciones_oof_unido.csv").merge(
    meta[["id", "grupo", "dur_s"]], on="id")
mods = {"CNN": "p_CNN", "Fusión CNN+OPERA-CE": "p_Fusion_CNN_CE", "OPERA-CE": "p_OPERA_CE", "Metadatos": "p_Metadatos"}
estratos = {"(a) todas": lambda d: d >= 0, "(b) 5–10 s": lambda d: (d >= 5) & (d <= 10),
            "(c) < 5 s": lambda d: d < 5, "(d) > 10 s": lambda d: d > 10}
reps = sorted(u.repeticion.unique()); rng = np.random.default_rng(0)
lin = []


def log(s=""):
    print(s); lin.append(s)


log("== Conteo por estrato (grabaciones únicas) ==")
res = []
for nom, f in estratos.items():
    s = u[f(u.dur_s) & (u.repeticion == 0)]
    ne, ns = int((s.label == 1).sum()), int((s.label == 0).sum())
    ok = min(ne, ns) >= 20
    log(f"  {nom:10s} EPOC {ne:4d} | sanos {ns:4d} | {'AUC reportado' if ok else 'MENOS DE 20 de una clase: NO se reporta AUC'}")
    res.append((nom, f, ok))

log("\n== AUC (media de repeticiones) con IC95 por participante ==")
for nom, f, ok in res:
    if not ok:
        continue
    s = u[f(u.dur_s)].reset_index(drop=True)
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
    log(f"  {nom}")
    for m, c in mods.items():
        pt = np.mean([roc_auc_score(s[s.repeticion == r].label, s[s.repeticion == r][c]) for r in reps])
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        log(f"    {m:22s} {pt:.3f}  [{lo:.3f}; {hi:.3f}]")

(OUT / "estratos_duracion.txt").write_text("\n".join(lin), encoding="utf-8")
