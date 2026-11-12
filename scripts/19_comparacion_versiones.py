"""Compara v2 (853, con relleno de ceros) y v3 (830, sin relleno) -> reports/comparacion_v2_853_vs_v3_830.md"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metricas import umbral_por_sensibilidad  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
meta = pd.read_csv(ROOT / "data/processed/copd_cough/metadata.csv")[["id", "grupo"]]
VERS = {"v2 (853 grabaciones, con relleno de ceros)": ROOT / "reports/validacion_cruzada_v2_853",
        "v3 (830 grabaciones, sin relleno)": ROOT / "reports/validacion_cruzada"}
MODELOS = {"CNN": "p_CNN", "Fusión CNN+OPERA-CE": "p_Fusion_CNN_CE", "OPERA-CE": "p_OPERA_CE", "Metadatos": "p_Metadatos"}
AUCF = {"CNN": ("metricas_por_fold.csv", "CNN"), "Fusión CNN+OPERA-CE": ("desplegables_metricas_por_fold.csv", "Fusion_CNN_CE"),
        "OPERA-CE": ("desplegables_metricas_por_fold.csv", "OPERA_CE"), "Metadatos": ("metricas_por_fold.csv", "Metadatos")}
B = 1000


def boot(df, cols, pares):
    """AUC (media de repeticiones) y diferencias pareadas, con IC95 bootstrap por participante."""
    reps = sorted(df.repeticion.unique()); rng = np.random.default_rng(0)
    sub = {r: df[df.repeticion == r].reset_index(drop=True) for r in reps}
    gi = {r: {g: d.index.to_numpy() for g, d in sub[r].groupby("grupo")} for r in reps}
    gs = df.grupo.unique(); bs = {c: [] for c in cols}; bd = {p: [] for p in pares}
    for _ in range(B):
        sel = rng.choice(gs, len(gs), replace=True); v = {c: [] for c in cols}
        for r in reps:
            d = sub[r].iloc[np.concatenate([gi[r][g] for g in sel if g in gi[r]])]
            if d.label.nunique() < 2:
                continue
            for c in cols:
                v[c].append(roc_auc_score(d.label, d[c]))
        for c in cols:
            bs[c].append(np.mean(v[c]))
        for a, b in pares:
            bd[(a, b)].append(bs[a][-1] - bs[b][-1])
    pt = {c: np.mean([roc_auc_score(sub[r].label, sub[r][c]) for r in reps]) for c in cols}
    ic = lambda x: tuple(np.percentile(x, [2.5, 97.5]))
    return pt, {c: ic(bs[c]) for c in cols}, {p: (pt[p[0]] - pt[p[1]], *ic(bd[p])) for p in pares}


def honesto(df, col, sens=0.85):
    tp = fn = tn = fp = 0
    for r in sorted(df.repeticion.unique()):
        g = df[df.repeticion == r]; y, p = g.label.to_numpy(), g[col].to_numpy()
        for k in g.fold.unique():
            te, resto = (g.fold == k).to_numpy(), (g.fold != k).to_numpy()
            pred = p[te] >= umbral_por_sensibilidad(y[resto], p[resto], sens)
            tp += (pred & (y[te] == 1)).sum(); fn += (~pred & (y[te] == 1)).sum()
            tn += (~pred & (y[te] == 0)).sum(); fp += (pred & (y[te] == 0)).sum()
    return tp / (tp + fn), tn / (tn + fp)


f = lambda a: f"{a[0]:+.3f} [{a[1]:+.3f}; {a[2]:+.3f}]" + (" *" if a[1] > 0 or a[2] < 0 else "")
md = ["# Comparación v2 (853) vs v3 (830)", "",
      "- **v2:** 853 grabaciones (221 EPOC, 632 sanos); 23 EPOC de < 3 s se rellenan con ceros.",
      "- **v3:** 830 grabaciones (198 EPOC, 632 sanos); se excluyen las grabaciones de < 3 s.",
      "- Validación cruzada 5×5 agrupada por participante, mismas semillas. IC95 por bootstrap de participantes (B=1000).",
      "- `*` = el IC95 de la diferencia excluye 0. Umbral honesto: fijado con las OOF de los otros folds de la misma repetición, sensibilidad objetivo 85 %.",
      "- Piso de ruido = RF con los 100 cuadros de menor energía (128 rasgos, sin duración); sus comparaciones usan solo las grabaciones con ≥ 100 cuadros.", ""]
for nombre, d in VERS.items():
    u = pd.read_csv(d / "predicciones_oof_unido.csv").merge(meta, on="id")
    pf = pd.read_csv(d / "control_duracion/predicciones_oof_piso_fijo.csv")[["repeticion", "fold", "id", "p_PisoFijo_RF"]]
    up = u.merge(pf, on=["repeticion", "fold", "id"])
    cols = list(MODELOS.values())
    pt, ic, dm = boot(u, cols, [(c, "p_Metadatos") for c in cols if c != "p_Metadatos"])
    cp = cols + ["p_PisoFijo_RF"]
    ptp, icp, dp = boot(up, cp, [(c, "p_PisoFijo_RF") for c in cols])
    md += [f"## {nombre}", "",
           f"Grabaciones: {u.id.nunique()} ({int(u[u.repeticion == 0].label.sum())} EPOC) | con piso de ruido: {up.id.nunique()}", "",
           "| Modelo | AUC por fold (media ± DE) | AUC OOF [IC95] | Δ vs metadatos [IC95] | Δ vs piso de ruido [IC95] | Umbral honesto: sens / esp |",
           "|---|---|---|---|---|---|"]
    for m, c in MODELOS.items():
        arch, mod = AUCF[m]; a = pd.read_csv(d / arch); a = a[a.modelo == mod].auc
        sh, eh = honesto(u, c)
        dmx = "—" if c == "p_Metadatos" else f((c, "p_Metadatos") and dm[(c, "p_Metadatos")])
        md.append(f"| {m} | {a.mean():.3f} ± {a.std(ddof=1):.3f} | {pt[c]:.3f} [{ic[c][0]:.3f}; {ic[c][1]:.3f}] | {dmx} | {f(dp[(c, 'p_PisoFijo_RF')])} | {sh:.3f} / {eh:.3f} |")
    md += ["", f"Piso de ruido (RF, 100 cuadros): AUC OOF {ptp['p_PisoFijo_RF']:.3f} [{icp['p_PisoFijo_RF'][0]:.3f}; {icp['p_PisoFijo_RF'][1]:.3f}]", ""]
    print(nombre, "listo", flush=True)
(ROOT / "reports/comparacion_v2_853_vs_v3_830.md").write_text("\n".join(md), encoding="utf-8")
