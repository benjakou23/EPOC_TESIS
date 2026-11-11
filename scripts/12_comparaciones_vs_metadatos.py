"""Comparaciones contra el atajo de metadatos, IC por participante y umbral honesto.

Entrada : CSV de predicciones fuera de partición con columnas
          repeticion, fold, id, label, p_<modelo>... (el de 06 o el de 08).
Salida  : reports/validacion_cruzada/comparaciones_vs_metadatos.txt (+ se imprime)

Calcula, para cada modelo p_*:
  1. AUC fuera de partición (media de repeticiones) con IC95 por bootstrap de participantes.
  2. Diferencia de AUC pareada vs la referencia, con IC95 por participante, y DeLong
     por repetición (cuántas de 5 con p<0.05).
  3. Umbral de tamizaje HONESTO: para cada fold, umbral fijado (sens objetivo) con las
     predicciones fuera de partición de los OTROS folds de la misma repetición; se
     compara con el umbral único fijado sobre todo (re-sustitución, optimista).
  4. VPP / VPN / % derivados para prevalencias 6, 12 y 26 %.

Uso (desde la raíz del repo):
  python scripts/12_comparaciones_vs_metadatos.py
  python scripts/12_comparaciones_vs_metadatos.py --oof reports/desplegables/predicciones_oof.csv \
         --ref p_Metadatos --modelos p_Fusion_CE p_CNN
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metricas import delong, umbral_por_sensibilidad  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--oof", default=str(ROOT / "reports/validacion_cruzada/predicciones_oof.csv"))
ap.add_argument("--meta", default=str(ROOT / "data/processed/copd_cough/metadata.csv"))
ap.add_argument("--ref", default="p_Metadatos")
ap.add_argument("--modelos", nargs="*", default=None)
ap.add_argument("--sens", type=float, default=0.85)
ap.add_argument("--B", type=int, default=1000)
ap.add_argument("--out", default=str(ROOT / "reports/validacion_cruzada/comparaciones_vs_metadatos.txt"))
args = ap.parse_args()

oof = pd.read_csv(args.oof)
oof = oof.merge(pd.read_csv(args.meta)[["id", "grupo"]], on="id", how="left")
assert oof.grupo.notna().all(), "hay ids sin grupo en metadata.csv"
modelos = args.modelos or [c for c in oof.columns if c.startswith("p_")]
if args.ref not in modelos:
    modelos = [args.ref] + modelos
reps = sorted(oof.repeticion.unique())
sub = {r: oof[oof.repeticion == r].reset_index(drop=True) for r in reps}
gidx = {r: {g: d.index.to_numpy() for g, d in sub[r].groupby("grupo")} for r in reps}
grupos = oof.grupo.unique()
lineas = []


def log(s=""):
    print(s); lineas.append(s)


def auc_media(col):
    return float(np.mean([roc_auc_score(sub[r].label, sub[r][col]) for r in reps]))


# ---------------------------------------------- bootstrap por participante (pareado)
rng = np.random.default_rng(0)
boot = {m: [] for m in modelos}
for _ in range(args.B):
    sel = rng.choice(grupos, len(grupos), replace=True)
    vals = {m: [] for m in modelos}
    for r in reps:
        ii = np.concatenate([gidx[r][g] for g in sel if g in gidx[r]])
        d = sub[r].iloc[ii]
        if d.label.nunique() < 2:
            continue
        for m in modelos:
            vals[m].append(roc_auc_score(d.label, d[m]))
    for m in modelos:
        boot[m].append(np.mean(vals[m]))
boot = {m: np.array(v) for m, v in boot.items()}

log(f"Predicciones: {args.oof}")
log(f"{len(grupos)} grupos (carpetas participante/sesión), {len(reps)} repeticiones, B={args.B}\n")
log("== 1. AUC fuera de partición (media de repeticiones), IC95 bootstrap por participante ==")
for m in modelos:
    lo, hi = np.percentile(boot[m], [2.5, 97.5])
    log(f"  {m:18s} {auc_media(m):.3f}  [{lo:.3f}; {hi:.3f}]")

log(f"\n== 2. Diferencia de AUC vs {args.ref} (pareada, bootstrap por participante) ==")
for m in modelos:
    if m == args.ref:
        continue
    d = boot[m] - boot[args.ref]
    lo, hi = np.percentile(d, [2.5, 97.5])
    pt = auc_media(m) - auc_media(args.ref)
    n_sig, ps = 0, []
    for r in reps:
        g = sub[r]
        _, _, p = delong(g.label.to_numpy(), g[m].to_numpy(), g[args.ref].to_numpy())
        ps.append(p); n_sig += p < 0.05
    veredicto = "IC EXCLUYE 0" if (lo > 0 or hi < 0) else "IC incluye 0"
    log(f"  {m:18s} {pt:+.3f}  [{lo:+.3f}; {hi:+.3f}]  {veredicto} | DeLong p<0.05 en {n_sig}/{len(reps)} rep.")
log("  (DeLong asume casos independientes; hay varias grabaciones por participante: el IC por")
log("   participante es la medida principal, DeLong es complementaria.)")

# ---------------------------------------------- umbral honesto vs re-sustitución
log(f"\n== 3. Umbral de tamizaje (sensibilidad objetivo {args.sens:.0%}) ==")
log("  honesto = umbral fijado con las OOF de los otros folds de la misma repetición")
log("  resust. = umbral único fijado con TODAS las OOF (sens. optimista por construcción)")
prev = {}
for m in modelos:
    if m == args.ref:
        continue
    tp = fn = tn = fp = 0
    for r in reps:
        g = sub[r]
        y, p = g.label.to_numpy(), g[m].to_numpy()
        for k in g.fold.unique():
            te, resto = (g.fold == k).to_numpy(), (g.fold != k).to_numpy()
            thr = umbral_por_sensibilidad(y[resto], p[resto], args.sens)
            pred = p[te] >= thr
            tp += int((pred & (y[te] == 1)).sum()); fn += int((~pred & (y[te] == 1)).sum())
            tn += int((~pred & (y[te] == 0)).sum()); fp += int((pred & (y[te] == 0)).sum())
    sens_h, esp_h = tp / (tp + fn), tn / (tn + fp)
    yy = np.concatenate([sub[r].label for r in reps]); pp = np.concatenate([sub[r][m] for r in reps])
    thr_u = umbral_por_sensibilidad(yy, pp, args.sens)
    pred = pp >= thr_u
    sens_u = (pred & (yy == 1)).sum() / (yy == 1).sum(); esp_u = (~pred & (yy == 0)).sum() / (yy == 0).sum()
    log(f"  {m:18s} honesto: sens {sens_h:.3f} esp {esp_h:.3f} | resust. (umbral {thr_u:.4f}): sens {sens_u:.3f} esp {esp_u:.3f}")
    prev[m] = (sens_h, esp_h, sens_u, esp_u)

log("\n== 4. Valores predictivos según prevalencia (VPP / VPN / % derivados a espirometría) ==")
for m, (sh, eh, su, eu) in prev.items():
    for etiqueta, se, es in (("honesto", sh, eh), ("resust.", su, eu)):
        fila = []
        for p in (0.06, 0.12, 0.26):
            vpp = se * p / (se * p + (1 - es) * (1 - p))
            vpn = es * (1 - p) / (es * (1 - p) + (1 - se) * p)
            der = se * p + (1 - es) * (1 - p)
            fila.append(f"{p:.0%}: VPP {vpp:.1%} VPN {vpn:.1%} deriva {der:.0%}")
        log(f"  {m:18s} {etiqueta}: " + " | ".join(fila))

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(lineas), encoding="utf-8")
print(f"\nGuardado en {args.out}")