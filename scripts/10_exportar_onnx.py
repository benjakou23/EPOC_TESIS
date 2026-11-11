"""Prepara el modelo final para el teléfono y lo exporta a ONNX.

Se generan dos modelos que reciben directamente la forma de onda (16 kHz, mono,
float32 en [-1, 1]); todo el procesamiento de señal va dentro:

  cnn_tos.onnx       (1, 48000)  -> probabilidad de EPOC de una ventana de 3 s
  opera_ce_tos.onnx  (1, 128000) -> probabilidad de EPOC (OPERA-CE + regresión logística)

El STFT se reescribe como convolución 1D (núcleos de Fourier con ventana Hann)
para que el modelo no dependa de operaciones de FFT (se ejecuta con ONNX Runtime).
Antes de exportar se verifica que las versiones reescritas den lo mismo que las
originales: torchaudio (CNN), librosa (espectrograma de OPERA) y los
embeddings OPERA-CE guardados en data/features/opera_CE.npz.

El aplicativo hace el resto (ver models/final/export/especificacion_app.json):
recorte de silencios, normalización de pico, ventanas para la CNN, relleno por
repetición a 8 s para OPERA-CE, promedio de ambas probabilidades y umbral.

Salida: models/final/export/
Uso:    python scripts/10_exportar_onnx.py
"""
import json
import math
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from audio_utils import SEG, SR, CNNTos, LogMel, cargar_meta, cargar_wav, ventanas

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "models" / "final"
OUT = FINAL / "export"
OUT.mkdir(parents=True, exist_ok=True)
CE_N = 8 * SR                 # OPERA-CE trabaja con 8 s
torch.manual_seed(0)


def nucleos_fourier(n_fft, ventana):
    """Matriz (n_fft, 2F) con la parte real e imaginaria de la DFT, ya multiplicada por la ventana."""
    k = torch.arange(n_fft, dtype=torch.float64)
    f = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
    ang = 2 * math.pi * k[:, None] * f[None, :] / n_fft
    w = ventana.double()[:, None]
    return torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w], dim=1).float()


def indices_cuadros(n, n_fft, hop, modo):
    """Índices (T, n_fft) de cada cuadro del STFT con center=True, con el relleno incorporado.
    reflect: refleja en los bordes (torch.stft); constant: apunta a la posición n, que vale cero."""
    pad = n_fft // 2
    p = np.arange(1 + n // hop)[:, None] * hop + np.arange(n_fft)[None, :] - pad
    if modo == "reflect":
        p = np.abs(p)
        p = np.where(p > n - 1, 2 * (n - 1) - p, p)
    else:
        p = np.where((p < 0) | (p > n - 1), n, p)
    return torch.from_numpy(p.astype(np.int64))


def potencia_stft(x, indices, nucleos, con_cero):
    """|STFT|^2 con Gather + MatMul (sin FFT ni conv1d). x: (B, N) -> (B, T, F)."""
    if con_cero:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    z = torch.matmul(x[:, indices], nucleos)
    nf = nucleos.shape[1] // 2
    return z[..., :nf] ** 2 + z[..., nf:] ** 2


class LogMelExportable(nn.Module):
    """Equivalente a audio_utils.LogMel (torchaudio) sin torch.stft."""

    def __init__(self, original: LogMel):
        super().__init__()
        n_fft, win = 512, 400
        izq = (n_fft - win) // 2
        ventana = F.pad(torch.hann_window(win), (izq, n_fft - win - izq))
        self.register_buffer("nucleos", nucleos_fourier(n_fft, ventana))
        self.register_buffer("indices", indices_cuadros(SEG, n_fft, 160, "reflect"))
        self.register_buffer("fb", original.mel.mel_scale.fb.clone())     # (257, 64)

    def forward(self, w):
        p = potencia_stft(w, self.indices, self.nucleos, con_cero=False)  # (B, T, 257)
        x = torch.log(torch.matmul(p, self.fb) + 1e-6).transpose(1, 2)    # (B, 64, T)
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        return x.unsqueeze(1)


class CNNExportable(nn.Module):
    def __init__(self, cnn: CNNTos):
        super().__init__()
        self.feat, self.red, self.cabeza = LogMelExportable(cnn.feat), cnn.red, cnn.cabeza

    def forward(self, w):
        x = self.red(self.feat(w))
        x = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return torch.sigmoid(self.cabeza(x))


class MelOpera(nn.Module):
    """librosa.melspectrogram + power_to_db(top_db=80) + normalización 0-1 (src/util.py de OPERA)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("nucleos", nucleos_fourier(1024, torch.hann_window(1024)))
        self.register_buffer("indices", indices_cuadros(CE_N, 1024, 512, "constant"))
        fb = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=64, fmin=50, fmax=8000)
        self.register_buffer("fb", torch.from_numpy(fb.T.copy()).float())  # (513, 64)

    def forward(self, w):
        s = torch.matmul(potencia_stft(w, self.indices, self.nucleos, con_cero=True), self.fb)
        db = 10 * torch.log10(torch.clamp(s, min=1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80)
        mn, mx = db.amin(dim=(1, 2), keepdim=True), db.amax(dim=(1, 2), keepdim=True)
        return (db - mn) / (mx - mn)                                          # (B, T, 64)


class OperaCEExportable(nn.Module):
    def __init__(self, ckpt, logreg):
        super().__init__()
        self.mel = MelOpera()
        self.cnn1 = nn.Conv2d(1, 3, kernel_size=3)
        # sin image_size: el padding "same" estático queda calculado para 224×224, igual
        # que en OPERA (Encoder de src/model/models_cola.py); cambiarlo altera la red
        self.efficientnet = EfficientNet.from_name("efficientnet-b0", include_top=False)
        self.efficientnet.set_swish(memory_efficient=False)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        r = self.load_state_dict(enc, strict=False)
        assert not [k for k in r.missing_keys if not k.startswith(("mel.", "media", "escala", "coef", "b"))], r
        self.register_buffer("media", torch.tensor(logreg["media"]))
        self.register_buffer("escala", torch.tensor(logreg["escala"]))
        self.register_buffer("coef", torch.tensor(logreg["coef"]))
        self.register_buffer("b", torch.tensor(logreg["intercepto"]))

    def embedding(self, w):
        x = self.cnn1(self.mel(w).unsqueeze(1))
        return self.efficientnet(x).flatten(1)

    def forward(self, w):
        z = (self.embedding(w) - self.media) / self.escala
        return torch.sigmoid((z * self.coef).sum(dim=1, keepdim=True) + self.b)


# --------------------------------------------------------------- preprocesamiento de la app
random.seed(7456)
RELLENO_ATRAS = random.random() < 0.5        # misma rama que _duplicate_padding de OPERA


def relleno_opera(y, n=CE_N):
    """Relleno por repetición de OPERA (split_pad_sample 'repeat'); se recorta a 8 s si es más largo."""
    if len(y) >= n:
        return y[:n].copy()
    falta = n - len(y)
    aug = y
    while len(aug) < falta:
        aug = np.concatenate([aug, aug])
    out = np.zeros(n, dtype=np.float32)
    if RELLENO_ATRAS:
        out[falta:], out[:falta] = y, aug[len(aug) - falta:]
    else:
        out[:len(y)], out[len(y):] = y, aug[:falta]
    return out


def mel_librosa(y):
    """Copia literal de pre_process_audio_mel_t de OPERA (f_max=8000)."""
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, fmin=50, fmax=8000, n_fft=1024, hop_length=512)
    S = librosa.power_to_db(S, ref=np.max)
    return ((S - S.min()) / (S.max() - S.min())).T


if __name__ == "__main__":
    meta = cargar_meta(ROOT)
    ref = pd.read_csv(FINAL / "predicciones_referencia.csv").set_index("id")
    config = json.loads((FINAL / "config.json").read_text(encoding="utf-8"))
    logreg = json.loads((FINAL / "opera_ce_logreg.json").read_text())

    cnn = CNNTos(); cnn.load_state_dict(torch.load(FINAL / "cnn_final.pt", weights_only=True)); cnn.eval()
    cnn_x = CNNExportable(cnn).eval()
    ce_x = OperaCEExportable(ROOT / "third_party" / "OPERA" / "cks" / "model" / "encoder-operaCE.ckpt", logreg).eval()

    d = np.load(ROOT / "data" / "features" / "opera_CE.npz", allow_pickle=True)
    emb_guardado = pd.DataFrame(d["X"], index=d["ids"])
    rng = np.random.default_rng(0)
    muestra = meta.iloc[rng.choice(len(meta), 60, replace=False)]
    cortos = meta[[len(cargar_wav(ROOT, p)) < CE_N for p in meta.npy]]
    muestra_ce = cortos.iloc[rng.choice(len(cortos), 40, replace=False)]

    with torch.no_grad():
        # 1. CNN: versión exportable vs original, ventana por ventana
        dif_cnn = []
        for p in muestra.npy:
            v = torch.from_numpy(ventanas(cargar_wav(ROOT, p)))
            dif_cnn.append((torch.sigmoid(cnn(v)) - cnn_x(v)[:, 0]).abs().max().item())
        # 2. espectrograma OPERA: conv vs librosa; 3. embedding y probabilidad vs lo guardado
        dif_mel, dif_emb, dif_p = [], [], []
        for _, r in muestra_ce.iterrows():
            y = relleno_opera(cargar_wav(ROOT, r.npy))
            w = torch.from_numpy(y)[None]
            dif_mel.append(np.abs(ce_x.mel(w)[0].numpy() - mel_librosa(y)).max())
            e = ce_x.embedding(w)[0].numpy()
            g = emb_guardado.loc[r.id].to_numpy()
            dif_emb.append(np.abs(e - g).max() / (np.abs(g).max() + 1e-9))
            dif_p.append(abs(ce_x(w).item() - ref.loc[r.id, "p_opera_ce"]))

    verif = {"cnn_dif_max_prob": float(np.max(dif_cnn)),
             "opera_mel_dif_max": float(np.max(dif_mel)),
             "opera_emb_dif_rel_max": float(np.max(dif_emb)),
             "opera_dif_max_prob": float(np.max(dif_p))}
    print(json.dumps(verif, indent=2))
    assert verif["cnn_dif_max_prob"] < 1e-3 and verif["opera_dif_max_prob"] < 1e-2, "las versiones exportables no coinciden"

    # --------------------------------------------------------------- ONNX
    ejemplos = {"cnn_tos": (cnn_x, SEG), "opera_ce_tos": (ce_x, CE_N)}
    for nombre, (m, n) in ejemplos.items():
        destino = OUT / f"{nombre}.onnx"
        torch.onnx.export(m, (torch.zeros(1, n),), destino, input_names=["audio"], output_names=["prob"],
                          opset_version=17, dynamo=False)
        print(f"{destino.name}: {destino.stat().st_size / 1e6:.2f} MB")

    import onnxruntime as ort
    dif_onnx = {}
    for nombre, (m, n) in ejemplos.items():
        s = ort.InferenceSession(str(OUT / f"{nombre}.onnx"))
        x = (np.random.default_rng(1).standard_normal((1, n)) * 0.1).astype(np.float32)
        dif_onnx[nombre] = float(abs(s.run(None, {"audio": x})[0].item() - m(torch.from_numpy(x)).item()))
    print("diferencia ONNX vs PyTorch:", dif_onnx)

    # --------------------------------------------------------------- referencia del flujo de la app
    # mismas operaciones que hará el aplicativo (audio ya recortado y normalizado por 02),
    # limitado a 8 s; servirá para medir la consistencia PC <-> TFLite <-> teléfono
    filas = []
    with torch.no_grad():
        for _, r in meta.iterrows():
            y8 = cargar_wav(ROOT, r.npy)[:CE_N]
            p_c = cnn_x(torch.from_numpy(ventanas(y8))).mean().item()
            p_o = ce_x(torch.from_numpy(relleno_opera(y8))[None]).item()
            filas.append({"id": r.id, "label": r.label, "p_cnn": p_c, "p_opera_ce": p_o, "p_fusion": (p_c + p_o) / 2})
    app = pd.DataFrame(filas)
    app["riesgo_alto"] = app.p_fusion >= config["umbral_tamizaje"]
    app.to_csv(OUT / "referencia_pytorch.csv", index=False)
    dif_8s = (app.set_index("id").p_fusion - ref.p_fusion).abs()
    print(f"Efecto de limitar a 8 s (vs modelo con audio completo): dif. media {dif_8s.mean():.4f}, "
          f"máx {dif_8s.max():.4f}; misma clasificación {np.mean(app.riesgo_alto.to_numpy() == ref.loc[app.id].riesgo_alto.to_numpy()):.1%}")

    espec = {
        "entrada": {"frecuencia_hz": SR, "canales": 1, "duracion_max_s": 8,
                    "formato": "float32 en [-1, 1] (PCM16 / 32768)"},
        "preprocesamiento": [
            "Convertir a mono y 16 kHz.",
            "Recortar silencio inicial y final: cuadros de 2048 muestras con salto 512; conservar desde el "
            "primer hasta el último cuadro con RMS a menos de 40 dB del RMS máximo (librosa.effects.trim, "
            "top_db=40). Si quedan menos de 0.5 s, no recortar.",
            "Normalizar el pico: y = 0.95 * y / max(|y|).",
            "Si dura más de 8 s, usar los primeros 8 s.",
        ],
        "cnn_tos": {"archivo": "cnn_tos.onnx", "entrada": [1, SEG], "ventana": SEG, "salto": SEG // 2,
                    "ventanas": "si el audio es <= 3 s: rellenar con ceros centrado hasta 48000; si no, "
                                "ventanas desde 0 con salto 24000 y una última que termine en el final del audio",
                    "agregacion": "promedio de la probabilidad de todas las ventanas"},
        "opera_ce_tos": {"archivo": "opera_ce_tos.onnx", "entrada": [1, CE_N],
                         "relleno": ("repetir el audio y anteponer los últimos (128000 - n) valores de la "
                                     "repetición" if RELLENO_ATRAS else
                                     "repetir el audio y añadir al final los primeros (128000 - n) valores "
                                     "de la repetición")},
        "fusion": "p = (p_cnn + p_opera_ce) / 2",
        "umbral_tamizaje": config["umbral_tamizaje"],
        "resultado": "riesgo alto (derivar a espirometría) si p >= umbral; si no, riesgo bajo",
        "verificacion_pytorch": verif,
        "diferencia_onnx_pytorch": dif_onnx,
    }
    (OUT / "especificacion_app.json").write_text(json.dumps(espec, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Listo:", OUT)
