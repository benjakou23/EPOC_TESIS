"""Clasificador sobre embeddings OPERA (sonda lineal), mismo protocolo que la CNN.

- Regresión logística con pesos por clase; C elegido por CV estratificada
  5-fold en train (AUC).
- Umbral fijado en VALIDACIÓN para sensibilidad >= SENS_OBJ, aplicado al TEST.
- Compara contra la línea base CNN (reports/baseline_cnn/predicciones_test.csv).

Uso: python scripts/05_clasificador_opera.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_utils import cargar_meta
from metricas import ic_bootstrap, metricas, umbral_por_sensibilidad

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
OUT = ROOT / "reports" / "opera"
OUT.mkdir(parents=True, exist_ok=True)
SENS_OBJ = 0.90

meta = cargar_meta(ROOT).set_index("id")


def cargar(nombre):
    d = np.load(FEAT / f"opera_{nombre}.npz", allow_pickle=True)
    return pd.DataFrame(d["X"], index=d["ids"])


feats = {n: cargar(n) for n in ("CT", "CE") if (FEAT / f"opera_{n}.npz").exists()}
if len(feats) == 2:
    feats["CT+CE"] = pd.concat([feats["CT"], feats["CE"]], axis=1)

filas, preds = [], {}
for nombre, F in feats.items():
    F = F.loc[meta.index]
    X = {s: F[meta.split == s].to_numpy() for s in ("train", "val", "test")}
    y = {s: meta.label[meta.split == s].to_numpy() for s in ("train", "val", "test")}

    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=5000))
    gs = GridSearchCV(pipe, {"logisticregression__C": np.logspace(-4, 1, 11)},
                      scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=0))
    gs.fit(X["train"], y["train"])

    p_val = gs.predict_proba(X["val"])[:, 1]
    p_te = gs.predict_proba(X["test"])[:, 1]
    thr = umbral_por_sensibilidad(y["val"], p_val, SENS_OBJ)
    m = metricas(y["test"], p_te, thr)
    m["ic95"] = ic_bootstrap(y["test"], p_te, thr)
    m.update(modelo=f"OPERA-{nombre} + LR", C=gs.best_params_["logisticregression__C"],
             auc_cv_train=gs.best_score_)
    filas.append(m); preds[nombre] = p_te
    print(f"OPERA-{nombre}: CV-AUC {gs.best_score_:.3f} | TEST AUC {m['auc']:.3f} "
          f"Sens {m['sensibilidad']:.3f} Esp {m['especificidad']:.3f} VPN {m['vpn']:.3f}")

base = ROOT / "reports" / "baseline_cnn" / "resumen.json"
if base.exists():
    b = json.loads(base.read_text())["ensamble"]
    b["modelo"] = "CNN liviana (ensamble 3 semillas)"
    filas.append(b)

res = pd.DataFrame(filas)
cols = ["modelo", "auc", "sensibilidad", "especificidad", "vpp", "vpn", "bal_acc"]
res[cols].to_csv(OUT / "comparacion_test.csv", index=False)
(OUT / "resultados.json").write_text(json.dumps(filas, indent=2, ensure_ascii=False, default=float))
te_ids = meta.index[meta.split == "test"]
pd.DataFrame({"id": te_ids, "label": meta.label[te_ids].to_numpy(),
              **{f"prob_{k}": v for k, v in preds.items()}}).to_csv(OUT / "predicciones_test.csv", index=False)

print("\n=== Comparación en TEST (umbral fijado en val, sens obj 90 %) ===")
print(res[cols].round(3).to_string(index=False))
