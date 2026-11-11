"""CNN con solo los primeros 8 s de cada audio (como hará la app). No modifica audio_utils."""
import os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import CNNTos, SR, cargar_wav, ventanas  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/validacion_cruzada/cnn_8s"; OUT.mkdir(parents=True, exist_ok=True)
MAX = 8 * SR
oof = pd.read_csv(ROOT / "reports/validacion_cruzada/predicciones_oof.csv")
meta = pd.read_csv(ROOT / "data/processed/copd_cough/metadata.csv").set_index("id")


@torch.no_grad()
def predecir_8s(model, npys, batch=64):
    """Igual que audio_utils.predecir_registros, pero recortando el audio a 8 s."""
    model.eval(); probs = []
    for p in npys:
        v = torch.from_numpy(ventanas(cargar_wav(ROOT, p)[:MAX]))
        pr = torch.cat([torch.sigmoid(model(v[i:i + batch])) for i in range(0, len(v), batch)])
        probs.append(pr.mean().item())
    return np.array(probs)


filas = []
for (r, k), g in oof.groupby(["repeticion", "fold"]):
    m = CNNTos(); m.load_state_dict(torch.load(ROOT / f"models/cv/cnn_r{r}_f{k}.pt"))
    g = g.copy(); g["p_CNN_8s"] = predecir_8s(m, meta.loc[g.id, "npy"])
    filas.append(g); print(f"r{r} f{k} ok", flush=True)
res = pd.concat(filas)[["repeticion", "fold", "id", "label", "p_Metadatos", "p_CNN", "p_CNN_8s"]]
res.to_csv(OUT / "predicciones_oof.csv", index=False)
s = subprocess.run([sys.executable, str(ROOT / "scripts/12_comparaciones_vs_metadatos.py"),
                    "--oof", str(OUT / "predicciones_oof.csv"), "--ref", "p_Metadatos",
                    "--modelos", "p_CNN", "p_CNN_8s", "--out", str(OUT / "comparaciones_vs_metadatos.txt")],
                   capture_output=True, text=True, encoding="utf-8",
                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
print(s.stdout.split("\nGuardado")[0] if s.returncode == 0 else s.stderr[-1500:])
