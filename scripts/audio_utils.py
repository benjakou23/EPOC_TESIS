"""Utilidades compartidas: dataset, aumentación, features y modelo.

La extracción log-Mel vive dentro del modelo (LogMel) para que la app
móvil reciba directamente la forma de onda a 16 kHz.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SR = 16000
SEG_S = 3.0            # ventana de análisis
HOP_S = 1.5            # salto entre ventanas en evaluación
SEG = int(SR * SEG_S)
N_MELS = 64


# ---------------------------------------------------------------- datos
def cargar_meta(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "data" / "processed" / "copd_cough" / "metadata.csv")


def cargar_wav(root: Path, npy: str) -> np.ndarray:
    pcm = np.load(root / "data" / "processed" / "copd_cough" / npy)
    return pcm.astype(np.float32) / 32767


def ajustar(y: np.ndarray, n: int, aleatorio: bool, rng=None) -> np.ndarray:
    """Recorta (aleatorio o centrado) o rellena con ceros hasta n muestras."""
    if len(y) >= n:
        ini = rng.integers(0, len(y) - n + 1) if aleatorio else (len(y) - n) // 2
        return y[ini:ini + n]
    pad = n - len(y)
    izq = rng.integers(0, pad + 1) if aleatorio else pad // 2
    return np.pad(y, (izq, pad - izq))


def ventanas(y: np.ndarray) -> np.ndarray:
    """Divide un audio completo en ventanas de SEG con salto HOP_S."""
    if len(y) <= SEG:
        return ajustar(y, SEG, False)[None]
    hop = int(SR * HOP_S)
    inis = list(range(0, len(y) - SEG + 1, hop))
    if inis[-1] + SEG < len(y):
        inis.append(len(y) - SEG)
    return np.stack([y[i:i + SEG] for i in inis])


class Aumentador:
    """Aumentación en forma de onda pensada para variabilidad de celulares."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        rng = self.rng
        # cambio de velocidad (altera tempo y tono, ±10 %)
        if rng.random() < 0.5:
            f = rng.uniform(0.9, 1.1)
            idx = np.arange(0, len(y) - 1, f)
            y = np.interp(idx, np.arange(len(y)), y).astype(np.float32)
        y = ajustar(y, SEG, True, rng)
        # ganancia aleatoria (-6 a +6 dB)
        y = y * 10 ** (rng.uniform(-6, 6) / 20)
        # ruido de fondo (SNR 10-40 dB)
        if rng.random() < 0.6:
            p = np.mean(y ** 2) + 1e-9
            snr = rng.uniform(10, 40)
            y = y + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(y))
        # filtro paso bajo simple (micrófonos de gama baja)
        if rng.random() < 0.3:
            k = rng.integers(2, 6)
            y = np.convolve(y, np.ones(k) / k, mode="same")
        return np.clip(y, -1, 1).astype(np.float32)


class DatasetTos(torch.utils.data.Dataset):
    def __init__(self, root, meta, aumentar=False, seed=0):
        self.y = meta.label.to_numpy()
        self.wavs = [cargar_wav(root, p) for p in meta.npy]
        self.aug = Aumentador(seed) if aumentar else None

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, i):
        w = self.wavs[i]
        w = self.aug(w) if self.aug else ajustar(w, SEG, False)
        return torch.from_numpy(w), torch.tensor(self.y[i], dtype=torch.float32)


# ---------------------------------------------------------------- modelo
class LogMel(nn.Module):
    """Forma de onda -> log-Mel normalizado por instancia (robusto a ganancia)."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=512, win_length=400, hop_length=160,
            f_min=50, f_max=8000, n_mels=N_MELS)

    def forward(self, w):
        x = torch.log(self.mel(w) + 1e-6)
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        return x.unsqueeze(1)  # (B, 1, mels, frames)


def spec_augment(x, n_masks=2, f_max=8, t_max=30):
    x = x.clone()
    B, _, Fq, T = x.shape
    for b in range(B):
        for _ in range(n_masks):
            f = np.random.randint(0, f_max + 1); f0 = np.random.randint(0, Fq - f + 1)
            t = np.random.randint(0, t_max + 1); t0 = np.random.randint(0, T - t + 1)
            x[b, :, f0:f0 + f, :] = 0
            x[b, :, :, t0:t0 + t] = 0
    return x


def bloque_ds(cin, cout, stride):
    """Convolución separable en profundidad (estilo MobileNet)."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))


class CNNTos(nn.Module):
    """CNN liviana para tamizaje en dispositivo (~100k parámetros)."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.feat = LogMel()
        self.red = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            bloque_ds(32, 64, 1), bloque_ds(64, 64, 2),
            bloque_ds(64, 128, 1), bloque_ds(128, 128, 2),
            bloque_ds(128, 192, 1), bloque_ds(192, 192, 2))
        self.cabeza = nn.Sequential(nn.Dropout(dropout), nn.Linear(192 * 2, 1))

    def forward(self, w, aug=False):
        x = self.feat(w)
        if aug:
            x = spec_augment(x)
        x = self.red(x)
        x = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.cabeza(x).squeeze(1)  # logit


@torch.no_grad()
def predecir_registros(model, root, meta, batch=64):
    """Probabilidad por registro = media de las ventanas del audio."""
    model.eval()
    probs = []
    for p in meta.npy:
        v = torch.from_numpy(ventanas(cargar_wav(root, p)))
        pr = torch.cat([torch.sigmoid(model(v[i:i + batch])) for i in range(0, len(v), batch)])
        probs.append(pr.mean().item())
    return np.array(probs)


# ---------------------------------------------------------------- entrenamiento
def entrenar_cnn(root, tr, va, seed, ckpt, epocas=60, paciencia=15, lr=2e-3, tag="",
                 min_epocas=None):
    """Entrena CNNTos con early stopping por AUC en `va`, solo a partir de `min_epocas`.

    Con OneCycleLR el learning rate sube hasta ~la época 18 y el AUC de validación es ruidoso
    durante ese calentamiento: ni se guarda checkpoint ni cuenta la paciencia antes de
    `min_epocas`. None -> epocas // 2 (30 con 60 épocas). Devuelve (modelo, historial).
    """
    import time

    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    if min_epocas is None:
        min_epocas = epocas // 2

    torch.manual_seed(seed); np.random.seed(seed)
    ds = DatasetTos(root, tr, aumentar=True, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    pos_weight = torch.tensor((tr.label == 0).sum() / (tr.label == 1).sum())

    model = CNNTos()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epocas * len(dl))
    mejor, sin_mejora, hist = -1, 0, []
    for ep in range(1, epocas + 1):
        model.train(); t0 = time.time(); perdida = 0
        for w, y in dl:
            loss = F.binary_cross_entropy_with_logits(model(w, aug=True), y, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            perdida += loss.item() * len(y)
        auc_val = roc_auc_score(va.label, predecir_registros(model, root, va))
        hist.append({"epoca": ep, "loss": perdida / len(ds), "auc_val": auc_val})
        marca = ""
        if ep >= min_epocas:                      # antes: calentamiento, no se evalúa
            if auc_val > mejor:
                mejor, sin_mejora = auc_val, 0
                torch.save(model.state_dict(), ckpt); marca = " *"
            else:
                sin_mejora += 1
        print(f"  [{tag}] ep {ep:3d} loss {perdida / len(ds):.4f} "
              f"AUC_val {auc_val:.3f} ({time.time() - t0:.0f}s){marca}", flush=True)
        if ep >= min_epocas and sin_mejora >= paciencia:
            break
    model.load_state_dict(torch.load(ckpt))
    return model, pd.DataFrame(hist)
