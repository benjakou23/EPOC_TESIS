"""Validación cruzada estratificada y AGRUPADA por participante: CNN vs OPERA vs fusión.

- Se usan las 853 grabaciones. La partición oficial train/test de Zenodo v2
  NO se usa porque mezcla participantes (ver 02b_grupos_desde_v1.py).
- StratifiedGroupKFold 5 folds x 5 repeticiones = 25 particiones; un mismo
  participante (carpeta de Zenodo v1) nunca está en entrenamiento y prueba.
- En cada partición, la parte de entrenamiento se divide por grupos en
  entrenamiento interno (~85 %) y validación interna (~15 %), usada para early
  stopping de la CNN y para fijar el umbral de tamizaje (sens >= --sens).
- Referencia: Random Forest solo con metadatos de la grabación (duración,
  RMS, pico) para detectar atajos de grabación.
- OPERA: regresión logística sobre embeddings CT+CE, C elegido por CV interna.
- Fusión: promedio de probabilidades CNN y OPERA.
- Por repetición se juntan las predicciones fuera de fold y se comparan
  las AUC con la prueba de DeLong.

Salida: reports/validacion_cruzada/
Uso:    python scripts/06_validacion_cruzada.py [--repeticiones 3] [--folds 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_utils import cargar_meta, entrenar_cnn, predecir_registros
from metricas import delong, metricas, umbral_por_sensibilidad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "validacion_cruzada"
CKPT = ROOT / "models" / "cv"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)
MODELOS = ("Metadatos", "CNN", "OPERA", "Fusion")

ap = argparse.ArgumentParser()
ap.add_argument("--repeticiones", type=int, default=5)
ap.add_argument("--sens", type=float, default=0.85)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epocas", type=int, default=60)
ap.add_argument("--paciencia", type=int, default=15)
args = ap.parse_args()
SENS_OBJ = args.sens

meta = cargar_meta(ROOT)
y = meta.label.to_numpy()
grupos = meta.grupo.to_numpy()
inv = pd.read_csv(ROOT / "reports" / "copd_cough_inventario.csv")
inv["id"] = inv.split + "_" + inv.clase + "_" + inv.archivo.str.replace(".wav", "", regex=False)
X_meta = inv.set_index("id").loc[meta.id, ["duracion_s", "rms", "pico"]].to_numpy()


def cargar_feat(n):
    d = np.load(ROOT / "data" / "features" / f"opera_{n}.npz", allow_pickle=True)
    return pd.DataFrame(d["X"], index=d["ids"]).loc[meta.id].to_numpy()


X_opera = np.hstack([cargar_feat("CT"), cargar_feat("CE")])

# reanudar si el script se interrumpió
f_folds = OUT / "metricas_por_fold.csv"
f_oof = OUT / "predicciones_oof.csv"
filas = pd.read_csv(f_folds).to_dict("records") if f_folds.exists() else []
oof = pd.read_csv(f_oof) if f_oof.exists() else pd.DataFrame()
hechos = {(r["repeticion"], r["fold"]) for r in filas}

for rep in range(args.repeticiones):
    skf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=100 + rep)
    for k, (i_tr, i_te) in enumerate(skf.split(meta, y, grupos)):
        if (rep, k) in hechos:
            continue
        print(f"\n=== Repetición {rep} | fold {k} ===", flush=True)
        a, b = next(StratifiedGroupKFold(7, shuffle=True, random_state=rep * 10 + k)
                    .split(i_tr, y[i_tr], grupos[i_tr]))
        i_in, i_val = i_tr[a], i_tr[b]
        assert not set(grupos[i_in]) & set(grupos[i_te]) and not set(grupos[i_val]) & set(grupos[i_te])
        tr, va, te = meta.iloc[i_in], meta.iloc[i_val], meta.iloc[i_te]

        cnn, _ = entrenar_cnn(ROOT, tr, va, seed=rep * 10 + k,
                              ckpt=CKPT / f"cnn_r{rep}_f{k}.pt", epocas=args.epocas,
                              paciencia=args.paciencia, tag=f"r{rep}f{k}")
        p = {"CNN": {"val": predecir_registros(cnn, ROOT, va),
                     "te": predecir_registros(cnn, ROOT, te)}}

        lr = GridSearchCV(make_pipeline(StandardScaler(),
                                        LogisticRegression(class_weight="balanced", max_iter=5000)),
                          {"logisticregression__C": np.logspace(-4, 1, 11)}, scoring="roc_auc",
                          cv=StratifiedGroupKFold(5, shuffle=True, random_state=0))
        lr.fit(X_opera[i_in], y[i_in], groups=grupos[i_in])
        p["OPERA"] = {"val": lr.predict_proba(X_opera[i_val])[:, 1],
                      "te": lr.predict_proba(X_opera[i_te])[:, 1]}
        rf = RandomForestClassifier(300, class_weight="balanced", min_samples_leaf=3,
                                    random_state=0).fit(X_meta[i_in], y[i_in])
        p["Metadatos"] = {"val": rf.predict_proba(X_meta[i_val])[:, 1],
                          "te": rf.predict_proba(X_meta[i_te])[:, 1]}
        p["Fusion"] = {s: (p["CNN"][s] + p["OPERA"][s]) / 2 for s in ("val", "te")}

        for nombre in MODELOS:
            thr = umbral_por_sensibilidad(y[i_val], p[nombre]["val"], SENS_OBJ)
            m = metricas(y[i_te], p[nombre]["te"], thr)
            m.update(modelo=nombre, repeticion=rep, fold=k)
            filas.append(m)
            print(f"  {nombre:6s} AUC {m['auc']:.3f} Sens {m['sensibilidad']:.3f} "
                  f"Esp {m['especificidad']:.3f} VPN {m['vpn']:.3f}", flush=True)
        oof = pd.concat([oof, pd.DataFrame({
            "repeticion": rep, "fold": k, "id": te.id.to_numpy(), "label": y[i_te],
            **{f"p_{n}": p[n]["te"] for n in MODELOS}})], ignore_index=True)
        pd.DataFrame(filas).to_csv(f_folds, index=False)
        oof.to_csv(f_oof, index=False)

# ------------------------------------------------------------------ resumen
res = pd.DataFrame(filas)
cols = ["auc", "sensibilidad", "especificidad", "vpp", "vpn", "f1", "exactitud", "bal_acc"]
media = res.groupby("modelo")[cols].mean()
std = res.groupby("modelo")[cols].std()
tabla = media.round(3).astype(str) + " ± " + std.round(3).astype(str)
tabla = tabla.loc[list(MODELOS)]
tabla.to_csv(OUT / "resumen_media_std.csv")

pruebas = []
for rep, g in oof.groupby("repeticion"):
    for a, b in [("Fusion", "CNN"), ("Fusion", "OPERA"), ("CNN", "OPERA"), ("CNN", "Metadatos")]:
        auc_a, auc_b, pv = delong(g.label.to_numpy(), g[f"p_{a}"].to_numpy(), g[f"p_{b}"].to_numpy())
        pruebas.append({"repeticion": rep, "comparacion": f"{a} vs {b}",
                        "auc_a": auc_a, "auc_b": auc_b, "p_delong": pv})
pruebas = pd.DataFrame(pruebas)
pruebas.to_csv(OUT / "delong_por_repeticion.csv", index=False)

# victorias por fold (AUC) como evidencia complementaria
piv = res.pivot_table(index=["repeticion", "fold"], columns="modelo", values="auc")
victorias = {"Fusion > CNN": int((piv.Fusion > piv.CNN).sum()),
             "CNN > Metadatos": int((piv.CNN > piv.Metadatos).sum()),
             "CNN > OPERA": int((piv.CNN > piv.OPERA).sum()),
             "Fusion > OPERA": int((piv.Fusion > piv.OPERA).sum()), "total_folds": len(piv)}
(OUT / "resumen.json").write_text(json.dumps(
    {"media": media.round(4).to_dict("index"), "std": std.round(4).to_dict("index"),
     "victorias_auc_por_fold": victorias}, indent=2, ensure_ascii=False))

print("\n=== Media ± DE sobre", len(piv), "folds ===")
print(tabla.to_string())
print("\n=== DeLong (predicciones fuera de fold, por repetición) ===")
print(pruebas.round(4).to_string(index=False))
print("\nVictorias en AUC por fold:", victorias)
