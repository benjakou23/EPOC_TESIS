"""Evalúa variantes que caben en el teléfono, con las mismas particiones que 06.

OPERA-CT (32 M parámetros) no puede desplegarse en el móvil; OPERA-CE (5 M, ~5 MB
en int8) sí. Aquí se comparan, con la validación cruzada 5×5 agrupada por
participante de 06_validacion_cruzada.py:
  - OPERA-CE + regresión logística
  - Fusión CNN + OPERA-CE (promedio de probabilidades)
La CNN no se reentrena: se reutilizan sus predicciones guardadas por 06 (las
particiones se regeneran con las mismas semillas y se verifica que coincidan).

Salida: reports/validacion_cruzada/desplegables_*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_utils import CNNTos, cargar_meta, predecir_registros
from metricas import delong, metricas, umbral_por_sensibilidad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "validacion_cruzada"
SENS_OBJ, REPS, FOLDS = 0.85, 5, 5

meta = cargar_meta(ROOT)
y = meta.label.to_numpy()
grupos = meta.grupo.to_numpy()
d = np.load(ROOT / "data" / "features" / "opera_CE.npz", allow_pickle=True)
X_ce = pd.DataFrame(d["X"], index=d["ids"]).loc[meta.id].to_numpy()
oof06 = pd.read_csv(OUT / "predicciones_oof.csv").set_index(["repeticion", "id"])
idx = {i: k for k, i in enumerate(meta.id)}

filas, oof = [], []
for rep in range(REPS):
    skf = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=100 + rep)
    for k, (i_tr, i_te) in enumerate(skf.split(meta, y, grupos)):
        a, b = next(StratifiedGroupKFold(7, shuffle=True, random_state=rep * 10 + k)
                    .split(i_tr, y[i_tr], grupos[i_tr]))
        i_in, i_val = i_tr[a], i_tr[b]
        te_ids = meta.id.to_numpy()[i_te]
        guardadas = oof06.loc[rep].loc[te_ids]
        assert (guardadas.fold == k).all(), "las particiones no coinciden con 06"

        lr = GridSearchCV(make_pipeline(StandardScaler(),
                                        LogisticRegression(class_weight="balanced", max_iter=5000)),
                          {"logisticregression__C": np.logspace(-4, 1, 11)}, scoring="roc_auc",
                          cv=StratifiedGroupKFold(5, shuffle=True, random_state=0))
        lr.fit(X_ce[i_in], y[i_in], groups=grupos[i_in])
        p_ce = {"val": lr.predict_proba(X_ce[i_val])[:, 1], "te": lr.predict_proba(X_ce[i_te])[:, 1]}
        # CNN de 06: test desde lo guardado; validación interna recalculada con su checkpoint
        cnn = CNNTos()
        cnn.load_state_dict(torch.load(ROOT / "models" / "cv" / f"cnn_r{rep}_f{k}.pt", weights_only=True))
        cnn.eval()
        p_cnn = {"val": predecir_registros(cnn, ROOT, meta.iloc[i_val]), "te": guardadas.p_CNN.to_numpy()}
        p = {"OPERA_CE": p_ce, "Fusion_CNN_CE": {s: (p_cnn[s] + p_ce[s]) / 2 for s in ("val", "te")}}

        for nombre, pr in p.items():
            m = metricas(y[i_te], pr["te"], umbral_por_sensibilidad(y[i_val], pr["val"], SENS_OBJ))
            m.update(modelo=nombre, repeticion=rep, fold=k)
            filas.append(m)
        oof.append(pd.DataFrame({"repeticion": rep, "fold": k, "id": te_ids, "label": y[i_te],
                                 "p_OPERA_CE": p_ce["te"], "p_Fusion_CNN_CE": p["Fusion_CNN_CE"]["te"]}))
        print(f"r{rep} f{k}  CE {filas[-2]['auc']:.3f}  CNN+CE {filas[-1]['auc']:.3f}", flush=True)

res = pd.DataFrame(filas)
oof = pd.concat(oof, ignore_index=True)
res.to_csv(OUT / "desplegables_metricas_por_fold.csv", index=False)
oof.to_csv(OUT / "desplegables_predicciones_oof.csv", index=False)
cols = ["auc", "sensibilidad", "especificidad", "vpp", "vpn", "f1", "exactitud", "bal_acc"]
print(res.groupby("modelo")[cols].agg(["mean", "std"]).round(3).T.to_string())

# DeLong por repetición frente a la fusión completa (CNN + OPERA CT+CE)
comp = oof.merge(oof06.reset_index()[["repeticion", "id", "p_Fusion", "p_CNN"]], on=["repeticion", "id"])
pr = []
for rep, g in comp.groupby("repeticion"):
    for a_, b_ in [("p_Fusion_CNN_CE", "p_Fusion"), ("p_Fusion_CNN_CE", "p_CNN"), ("p_OPERA_CE", "p_CNN")]:
        aa, bb, pv = delong(g.label.to_numpy(), g[a_].to_numpy(), g[b_].to_numpy())
        pr.append({"repeticion": rep, "comparacion": f"{a_[2:]} vs {b_[2:]}", "auc_a": aa, "auc_b": bb, "p": pv})
pr = pd.DataFrame(pr)
pr.to_csv(OUT / "desplegables_delong.csv", index=False)
print(pr.groupby("comparacion")[["auc_a", "auc_b"]].mean().round(3).join(
    pr.assign(sig=pr.p < 0.05).groupby("comparacion").sig.sum()).to_string())
