"""Une las OOF de 06 y 08 por (repeticion, fold, id), corre 12 y añade diferencias CNN - fusión/OPERA-CE."""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports/validacion_cruzada"
a = pd.read_csv(R / "predicciones_oof.csv")
b = pd.read_csv(R / "desplegables_predicciones_oof.csv").drop(columns="label")
u = a.merge(b, on=["repeticion", "fold", "id"], validate="1:1")
assert len(u) == len(a) == len(b)
u.to_csv(R / "predicciones_oof_unido.csv", index=False)
mods = ["p_CNN", "p_Fusion_CNN_CE", "p_OPERA_CE", "p_Fusion"]
sal = subprocess.run([sys.executable, str(ROOT / "scripts/12_comparaciones_vs_metadatos.py"),
                      "--oof", str(R / "predicciones_oof_unido.csv"), "--ref", "p_Metadatos",
                      "--modelos", *mods, "--out", str(R / "comparaciones_vs_metadatos.txt")],
                     capture_output=True, text=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
print(sal.stdout.split("\nGuardado")[0] if sal.returncode == 0 else sal.stderr[-1500:])

# diferencias pareadas con IC95 por participante
u = u.merge(pd.read_csv(ROOT / "data/processed/copd_cough/metadata.csv")[["id", "grupo"]], on="id")
reps = sorted(u.repeticion.unique())
sub = {r: u[u.repeticion == r].reset_index(drop=True) for r in reps}
gi = {r: {g: d.index.to_numpy() for g, d in sub[r].groupby("grupo")} for r in reps}
grupos = u.grupo.unique(); rng = np.random.default_rng(0)
pares = [("p_CNN", "p_Fusion_CNN_CE"), ("p_CNN", "p_OPERA_CE")]
dif = {p: [] for p in pares}
for _ in range(1000):
    sel = rng.choice(grupos, len(grupos), replace=True); v = {p: [] for p in pares}
    for r in reps:
        d = sub[r].iloc[np.concatenate([gi[r][g] for g in sel if g in gi[r]])]
        if d.label.nunique() < 2: continue
        for x, y in pares:
            v[(x, y)].append(roc_auc_score(d.label, d[x]) - roc_auc_score(d.label, d[y]))
    for p in pares: dif[p].append(np.mean(v[p]))
print("== 5. Diferencia pareada de AUC, IC95 por participante ==")
for x, y in pares:
    pt = np.mean([roc_auc_score(sub[r].label, sub[r][x]) - roc_auc_score(sub[r].label, sub[r][y]) for r in reps])
    lo, hi = np.percentile(dif[(x, y)], [2.5, 97.5])
    print(f"  {x[2:]} - {y[2:]}: {pt:+.3f}  [{lo:+.3f}; {hi:+.3f}]  {'EXCLUYE 0' if lo > 0 or hi < 0 else 'incluye 0'}")
