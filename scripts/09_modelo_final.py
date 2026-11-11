"""Entrena el modelo final seleccionado para el aplicativo: fusión CNN + OPERA-CE.

Se aplica a las 853 grabaciones la misma receta evaluada en 06/08:
- CNN ligera: entrenamiento con el 85 % de los participantes y parada temprana
  con el 15 % restante (división agrupada por participante, como en cada fold).
- OPERA-CE + regresión logística: entrenada con las 853 grabaciones; C elegido
  por validación cruzada interna agrupada por participante.
- Fusión: promedio de ambas probabilidades.
- Umbral de tamizaje: el que da sensibilidad >= 85 % sobre las predicciones
  fuera de partición de la fusión en la validación cruzada 5x5 (08).

También guarda las probabilidades del modelo final para las 853 grabaciones,
que servirán de referencia para comprobar la consistencia PC <-> TFLite.

Salida: models/final/
Uso:    python scripts/09_modelo_final.py
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_utils import HOP_S, N_MELS, SEG_S, SR, cargar_meta, entrenar_cnn, predecir_registros
from metricas import metricas, umbral_por_sensibilidad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "final"
CV = ROOT / "reports" / "validacion_cruzada"
OUT.mkdir(parents=True, exist_ok=True)
SEMILLA, SENS_OBJ = 2026, 0.85

meta = cargar_meta(ROOT)
y = meta.label.to_numpy()
grupos = meta.grupo.to_numpy()

# ------------------------------------------------------------------ CNN
a, b = next(StratifiedGroupKFold(7, shuffle=True, random_state=SEMILLA).split(meta, y, grupos))
assert not set(grupos[a]) & set(grupos[b])
tr, va = meta.iloc[a], meta.iloc[b]
print(f"CNN: entrenamiento {len(tr)} grabaciones / {len(set(grupos[a]))} participantes; "
      f"parada temprana {len(va)} / {len(set(grupos[b]))}")
cnn, hist = entrenar_cnn(ROOT, tr, va, seed=SEMILLA, ckpt=OUT / "cnn_final.pt", tag="final")
hist.to_csv(OUT / "cnn_final_historial.csv", index=False)
mejor_ep = int(hist.loc[hist.auc_val.idxmax(), "epoca"])

# ------------------------------------------------------------------ OPERA-CE + regresión logística
d = np.load(ROOT / "data" / "features" / "opera_CE.npz", allow_pickle=True)
X_ce = pd.DataFrame(d["X"], index=d["ids"]).loc[meta.id].to_numpy()
lr = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
                  {"logisticregression__C": np.logspace(-4, 1, 11)}, scoring="roc_auc",
                  cv=StratifiedGroupKFold(5, shuffle=True, random_state=0))
lr.fit(X_ce, y, groups=grupos)
esc, reg = lr.best_estimator_.named_steps["standardscaler"], lr.best_estimator_.named_steps["logisticregression"]
(OUT / "opera_ce_logreg.json").write_text(json.dumps({
    "C": float(reg.C), "media": esc.mean_.tolist(), "escala": esc.scale_.tolist(),
    "coef": reg.coef_[0].tolist(), "intercepto": float(reg.intercept_[0]),
    "auc_cv_interna": float(lr.best_score_)}))
print(f"OPERA-CE LogReg: C = {reg.C:.4g}, AUC CV interna agrupada = {lr.best_score_:.3f}")

# ------------------------------------------------------------------ umbral desde la validación cruzada
oof = pd.read_csv(CV / "desplegables_predicciones_oof.csv")
umbral = float(umbral_por_sensibilidad(oof.label.to_numpy(), oof.p_Fusion_CNN_CE.to_numpy(), SENS_OBJ))
m_oof = metricas(oof.label.to_numpy(), oof.p_Fusion_CNN_CE.to_numpy(), umbral)
print(f"Umbral de tamizaje = {umbral:.4f} -> en CV: sens {m_oof['sensibilidad']:.3f}, "
      f"esp {m_oof['especificidad']:.3f}, VPN {m_oof['vpn']:.3f}")

# ------------------------------------------------------------------ predicciones de referencia
p_cnn = predecir_registros(cnn, ROOT, meta)
p_ce = lr.predict_proba(X_ce)[:, 1]
p_fus = (p_cnn + p_ce) / 2
ref = pd.DataFrame({"id": meta.id, "label": y, "grupo": grupos, "en_entrenamiento_cnn": meta.index.isin(a),
                    "p_cnn": p_cnn, "p_opera_ce": p_ce, "p_fusion": p_fus, "riesgo_alto": p_fus >= umbral})
ref.to_csv(OUT / "predicciones_referencia.csv", index=False)
auc_cnn_va = roc_auc_score(va.label, p_cnn[b])

res = pd.read_csv(CV / "desplegables_metricas_por_fold.csv")
r = res[res.modelo == "Fusion_CNN_CE"]
config = {
    "modelo": "Fusión CNN ligera + OPERA-CE (regresión logística), promedio de probabilidades",
    "fecha": date.today().isoformat(),
    "audio": {"frecuencia_hz": SR, "canales": 1, "formato": "float32 en [-1, 1]"},
    "cnn": {"ventana_s": SEG_S, "salto_s": HOP_S, "n_mels": N_MELS, "n_fft": 512, "win_length": 400,
            "hop_length": 160, "f_min": 50, "f_max": 8000, "agregacion": "media de probabilidades por ventana",
            "mejor_epoca": mejor_ep, "auc_parada_temprana": round(float(auc_cnn_va), 4)},
    "opera_ce": {"duracion_minima_s": 8, "relleno": "repetición del audio hasta 8 s", "n_mels": 64,
                 "f_max": 8000, "dim_embedding": 1280},
    "umbral_tamizaje": umbral,
    "sensibilidad_objetivo": SENS_OBJ,
    "desempeno_cv_5x5_agrupada": {c: f"{r[c].mean():.3f} ± {r[c].std():.3f}" for c in
                                  ("auc", "sensibilidad", "especificidad", "vpp", "vpn", "f1", "exactitud")},
}
(OUT / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(config, indent=2, ensure_ascii=False))
print(f"\nRiesgo alto en las 853 grabaciones con el modelo final: {ref.riesgo_alto.mean():.1%} "
      "(optimista: el modelo vio casi todos estos audios)")
