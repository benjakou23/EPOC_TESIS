"""Recupera el participante/sesión de cada audio a partir de Zenodo v1.

La v2 de Zenodo (14881818) trae los audios renombrados (COPD_0.wav, ...) y sin
las carpetas de participante; la v1 (10209837) sí las tiene. Se emparejan los
archivos por el hash de las muestras de audio decodificadas (no por nombre) y
se agrega la columna `grupo` a data/processed/copd_cough/metadata.csv.

También comprueba si la partición oficial train/test de la v2 mezcla
participantes (fuga entre entrenamiento y prueba).
"""
import hashlib
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "raw" / "copd_cough_v1" / "extracted" / "COPD-Cough (CC)"
INV = ROOT / "reports" / "copd_cough_inventario.csv"
META = ROOT / "data" / "processed" / "copd_cough" / "metadata.csv"
OUT = ROOT / "reports" / "mapeo_v1_v2.csv"


def huella(p):
    y, _ = sf.read(p, dtype="int16")
    return hashlib.md5(y.tobytes()).hexdigest()


v1 = pd.DataFrame([{"ruta_v1": str(p.relative_to(V1)), "grupo": p.parent.name,
                    "clase_v1": p.parent.parent.name, "hash": huella(p)}
                   for p in tqdm(sorted(V1.rglob("*.wav")), desc="v1")])
inv = pd.read_csv(INV)
inv["hash"] = [huella(ROOT / r) for r in tqdm(inv.ruta, desc="v2")]

print("Hashes duplicados en v1:", v1.hash.duplicated().sum(), "| en v2:", inv.hash.duplicated().sum())
m = inv.merge(v1, on="hash", how="left")
sin_par = m.grupo.isna().sum()
print(f"Emparejados: {len(m) - sin_par}/{len(m)}  (sin pareja: {sin_par})")
assert sin_par == 0, "Hay audios de v2 sin equivalente en v1"
incoh = ((m.clase == "COPD") != (m.clase_v1 == "COPD")).sum()
print("Etiquetas incoherentes entre versiones:", incoh)

m["id"] = m.split + "_" + m.clase + "_" + m.archivo.str.replace(".wav", "", regex=False)
m[["id", "archivo", "split", "clase", "ruta_v1", "grupo"]].to_csv(OUT, index=False)

meta = pd.read_csv(META).drop(columns="grupo", errors="ignore")
meta = meta.merge(m[["id", "grupo"]], on="id", how="left")
assert meta.grupo.notna().all()
meta.to_csv(META, index=False)

print("\nGrupos por clase:", m.groupby("clase").grupo.nunique().to_dict())
print("Grabaciones por grupo: mediana", m.groupby("grupo").size().median(),
      "| máx", m.groupby("grupo").size().max())
g_tr = set(m[m.split == "train"].grupo); g_te = set(m[m.split == "test"].grupo)
comp = g_tr & g_te
n_te_fuga = m[(m.split == "test") & m.grupo.isin(comp)].shape[0]
print(f"\nPartición oficial v2: grupos compartidos train/test = {len(comp)} "
      f"({n_te_fuga}/{(m.split == 'test').sum()} audios de test tienen su grupo también en train)")
