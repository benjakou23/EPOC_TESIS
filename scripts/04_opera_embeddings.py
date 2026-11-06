"""Extrae embeddings de los modelos fundacionales OPERA (Zhang et al., NeurIPS 2024).

- OPERA-CT (HTS-AT, 768-d) y OPERA-CE (EfficientNet, 1280-d), congelados.
- Entrada: los audios ya preprocesados (16 kHz, sin silencios) de 02_preprocesar.py.
- Mismo pre-proceso que OPERA: relleno por repetición hasta 8 s y log-Mel
  64 bandas (50-8000 Hz) normalizado 0-1; el audio completo entra de una vez.
- Pesos cargados con weights_only=True (solo tensores, sin ejecutar código).

Salida: data/features/opera_{CT,CE}.npz  (ids, X)
Uso:    python scripts/04_opera_embeddings.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OPERA = ROOT / "third_party" / "OPERA"
sys.path.insert(0, str(OPERA))
from src.model.models_cola import Cola  # noqa: E402
from src.util import pre_process_audio_mel_t, split_pad_sample  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from audio_utils import cargar_meta, cargar_wav  # noqa: E402

OUT = ROOT / "data" / "features"
OUT.mkdir(parents=True, exist_ok=True)
MODELOS = {"CT": ("htsat", 768), "CE": ("efficientnet", 1280)}
SR, INPUT_SEC = 16000, 8

meta = cargar_meta(ROOT)
torch.set_num_threads(8)

for nombre, (encoder, dim) in MODELOS.items():
    destino = OUT / f"opera_{nombre}.npz"
    if destino.exists():
        print(f"{destino.name} ya existe, se omite")
        continue
    model = Cola(encoder=encoder)
    ck = torch.load(OPERA / "cks" / "model" / f"encoder-opera{nombre}.ckpt",
                    map_location="cpu", weights_only=True)
    r = model.load_state_dict(ck["state_dict"], strict=False)
    faltan = [k for k in r.missing_keys if k.startswith("encoder")]
    assert not faltan, f"Pesos del encoder faltantes: {faltan[:5]}"
    model.eval()

    X = []
    for p in tqdm(meta.npy, desc=f"OPERA-{nombre}"):
        y = cargar_wav(ROOT, p)
        if len(y) < SR * INPUT_SEC:
            y = split_pad_sample([y, 0, 0], INPUT_SEC, SR)[0][0]
        S = pre_process_audio_mel_t(y.squeeze(), f_max=8000)
        with torch.no_grad():
            f = model.extract_feature(torch.tensor(S[None], dtype=torch.float), dim)
        X.append(f.numpy()[0])
    X = np.stack(X).astype(np.float32)
    np.savez(destino, ids=meta.id.to_numpy(), X=X)
    print(f"Guardado {destino} -> {X.shape}")
