"""Preprocesamiento del dataset COPD-Cough (CC).

- Convierte a mono y remuestrea a 16 kHz (formato que usará la app).
- Recorta silencio inicial/final y normaliza el pico a 0.95.
- Guarda cada audio como .npy int16 en data/processed/copd_cough/wav16k/.
- Detecta duplicados exactos (posible fuga train/test).
- Divide el train oficial en train/val estratificado; el test oficial
  NO se toca y queda reservado para la evaluación final.

Salida: data/processed/copd_cough/metadata.csv
"""
import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / "reports" / "copd_cough_inventario.csv"
OUT = ROOT / "data" / "processed" / "copd_cough"
WAV_OUT = OUT / "wav16k"
WAV_OUT.mkdir(parents=True, exist_ok=True)

SR = 16000
TOP_DB = 40      # umbral para recortar silencio en los extremos
VAL_FRAC = 0.15
SEED = 42

inv = pd.read_csv(INV)
rows = []
for r in tqdm(inv.itertuples(), total=len(inv)):
    y, _ = librosa.load(ROOT / r.ruta, sr=SR, mono=True)
    y_trim, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(y_trim) >= SR * 0.5:  # no recortar si deja casi nada
        y = y_trim
    y = 0.95 * y / (np.abs(y).max() + 1e-9)
    pcm = (y * 32767).astype(np.int16)
    nombre = f"{r.split}_{r.clase}_{Path(r.archivo).stem}.npy"
    np.save(WAV_OUT / nombre, pcm)
    rows.append({"id": Path(nombre).stem, "archivo_orig": r.archivo,
                 "split_orig": r.split, "clase": r.clase,
                 "label": int(r.clase == "COPD"),
                 "npy": f"wav16k/{nombre}", "dur_s": len(pcm) / SR,
                 "saturado": r.pico >= 0.999,
                 "hash": hashlib.md5(pcm.tobytes()).hexdigest()})

df = pd.DataFrame(rows)

dup = df[df.duplicated("hash", keep=False)].sort_values("hash")
print(f"Duplicados exactos: {len(dup)} archivos")
if len(dup):
    print(dup[["id", "split_orig", "clase"]].to_string(index=False))
    # se elimina la copia que no está en test para evitar fuga
    df = df.sort_values("split_orig", ascending=False)  # test antes que train
    df = df.drop_duplicates("hash", keep="first")

tr = df[df.split_orig == "train"]
tr_idx, val_idx = train_test_split(tr.index, test_size=VAL_FRAC,
                                   stratify=tr.label, random_state=SEED)
df["split"] = "test"
df.loc[tr_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"
df = df.sort_values(["split", "id"])
df.to_csv(OUT / "metadata.csv", index=False)

print("\n=== Particiones finales ===")
print(pd.crosstab(df.split, df.clase, margins=True))
print("\nDuración tras recorte (s):")
print(df.groupby("clase").dur_s.describe().round(2))
