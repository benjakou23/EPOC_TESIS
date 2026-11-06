"""Análisis exploratorio del dataset COPD-Cough (CC).

Genera reports/copd_cough_inventario.csv con una fila por audio y
muestra un resumen por split y clase.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "copd_cough" / "extracted" / "Cough-COPD(CC)"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)

rows = []
for wav in tqdm(sorted(DATA.rglob("*.wav"))):
    split, clase = wav.parts[-3], wav.parts[-2]
    row = {"archivo": wav.name, "split": split, "clase": clase,
           "ruta": str(wav.relative_to(ROOT)), "bytes": wav.stat().st_size}
    try:
        info = sf.info(wav)
        y, sr = sf.read(wav, always_2d=True)
        mono = y.mean(axis=1)
        row.update(sr=info.samplerate, canales=info.channels,
                   subtype=info.subtype, duracion_s=info.duration,
                   rms=float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0,
                   pico=float(np.abs(mono).max()) if len(mono) else 0.0,
                   error="")
    except Exception as e:  # audio corrupto
        row["error"] = str(e)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT / "copd_cough_inventario.csv", index=False)

print("\n=== Conteo por split y clase ===")
print(pd.crosstab(df["split"], df["clase"], margins=True))
print("\n=== Formato ===")
print(df.groupby(["sr", "canales", "subtype"]).size())
print("\n=== Duración (s) por clase ===")
print(df.groupby("clase")["duracion_s"].describe().round(2))
print("\n=== Problemas ===")
print("corruptos:", (df["error"] != "").sum())
print("muy cortos (<0.3 s):", (df["duracion_s"] < 0.3).sum())
print("silenciosos (rms<1e-3):", (df["rms"] < 1e-3).sum())
print("saturados (pico>=0.999):", (df["pico"] >= 0.999).sum())
print("\nEjemplos de nombres:")
print(df.groupby("clase")["archivo"].apply(lambda s: list(s[:4])))
