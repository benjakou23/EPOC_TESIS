"""Línea base: CNN liviana sobre log-Mel para tamizaje de EPOC por tos.

Criterio de tamizaje: el umbral se elige en VALIDACIÓN para alcanzar
sensibilidad >= SENS_OBJ y luego se aplica tal cual al TEST oficial.
Se entrenan varias semillas para reportar media ± desviación.

Uso:  python scripts/03_entrenar_baseline.py [--epocas 60] [--semillas 0 1 2]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from audio_utils import CNNTos, cargar_meta, entrenar_cnn, predecir_registros
from metricas import ic_bootstrap, metricas, umbral_por_sensibilidad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "baseline_cnn"
MODELS = ROOT / "models" / "baseline_cnn"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
SENS_OBJ = 0.90

ap = argparse.ArgumentParser()
ap.add_argument("--epocas", type=int, default=60)
ap.add_argument("--paciencia", type=int, default=15)
ap.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--lr", type=float, default=2e-3)
args = ap.parse_args()


def entrenar(seed, meta):
    return entrenar_cnn(ROOT, meta[meta.split == "train"], meta[meta.split == "val"],
                        seed, MODELS / f"cnn_seed{seed}.pt", args.epocas,
                        args.paciencia, args.lr, tag=f"s{seed}")


meta = cargar_meta(ROOT)
va, te = meta[meta.split == "val"], meta[meta.split == "test"]
n_params = sum(p.numel() for p in CNNTos().parameters())
print(f"Parámetros del modelo: {n_params:,}")

resultados, probs_test = [], []
for seed in args.semillas:
    print(f"\n=== Semilla {seed} ===")
    model, hist = entrenar(seed, meta)
    hist.to_csv(OUT / f"historial_seed{seed}.csv", index=False)
    p_val = predecir_registros(model, ROOT, va)
    p_te = predecir_registros(model, ROOT, te)
    thr = umbral_por_sensibilidad(va.label.to_numpy(), p_val, SENS_OBJ)
    m = metricas(te.label.to_numpy(), p_te, thr)
    m["auc_val"] = roc_auc_score(va.label, p_val); m["semilla"] = seed
    resultados.append(m); probs_test.append(p_te)
    print(f"  TEST -> AUC {m['auc']:.3f} | Sens {m['sensibilidad']:.3f} | "
          f"Esp {m['especificidad']:.3f} | VPN {m['vpn']:.3f} (umbral {thr:.3f})")

res = pd.DataFrame(resultados)
res.to_csv(OUT / "metricas_por_semilla.csv", index=False)

# Ensamble de semillas (promedio de probabilidades) con umbral de validación
y_te = te.label.to_numpy()
p_ens = np.mean(probs_test, axis=0)
modelos = []
for s in args.semillas:
    mm = CNNTos(); mm.load_state_dict(torch.load(MODELS / f"cnn_seed{s}.pt")); modelos.append(mm)
p_val_ens = np.mean([predecir_registros(mm, ROOT, va) for mm in modelos], axis=0)
thr_ens = umbral_por_sensibilidad(va.label.to_numpy(), p_val_ens, SENS_OBJ)
m_ens = metricas(y_te, p_ens, thr_ens)
m_ens["ic95"] = ic_bootstrap(y_te, p_ens, thr_ens)
pd.DataFrame({"id": te.id, "label": y_te, "prob": p_ens}).to_csv(OUT / "predicciones_test.csv", index=False)

resumen = {
    "parametros": n_params, "sens_objetivo": SENS_OBJ, "semillas": args.semillas,
    "por_semilla_media": res[["auc", "sensibilidad", "especificidad", "vpp", "vpn", "bal_acc"]].mean().round(4).to_dict(),
    "por_semilla_std": res[["auc", "sensibilidad", "especificidad", "vpp", "vpn", "bal_acc"]].std().round(4).to_dict(),
    "ensamble": m_ens,
}
(OUT / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

fpr, tpr, _ = roc_curve(y_te, p_ens)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"Ensamble CNN (AUC={m_ens['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.scatter(1 - m_ens["especificidad"], m_ens["sensibilidad"], c="red", zorder=3,
            label=f"Punto de tamizaje (umbral val, sens obj {SENS_OBJ:.0%})")
plt.xlabel("1 - Especificidad"); plt.ylabel("Sensibilidad")
plt.title("Curva ROC - test oficial COPD-Cough"); plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(OUT / "roc_test.png", dpi=150)

print("\n=== Resumen ===")
print(res[["semilla", "auc_val", "auc", "sensibilidad", "especificidad", "vpn"]].round(3).to_string(index=False))
print("\nEnsamble:", {k: round(v, 3) if isinstance(v, float) else v for k, v in m_ens.items() if k != "ic95"})
print("IC95 bootstrap:", {k: tuple(round(x, 3) for x in v) for k, v in m_ens["ic95"].items()})
