"""Genera las figuras (PNG, 300 dpi) y tablas (CSV) para el documento de tesis.

Figuras y tablas de datos se generan siempre; las de resultados requieren
reports/validacion_cruzada/ (06_validacion_cruzada.py).

Salida: reports/tesis/figuras/*.png y reports/tesis/tablas/*.csv
Uso:    python scripts/07_figuras_tablas.py
"""
import json
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

from audio_utils import SR, SEG, CNNTos, cargar_meta, cargar_wav

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "tesis" / "figuras"
TAB = ROOT / "reports" / "tesis" / "tablas"
CV = ROOT / "reports" / "validacion_cruzada"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
C_SANO, C_EPOC = "#4C9F70", "#C8553D"
COLORES = {"Metadatos": "#9AA0A6", "CNN": "#1F77B4", "OPERA": "#FF7F0E", "Fusion": "#2CA02C"}
NOMBRE = {"Metadatos": "Solo metadatos (RF)", "CNN": "CNN ligera (desde cero)",
          "OPERA": "OPERA CT+CE + LogReg (transferencia)", "Fusion": "Fusión CNN + OPERA"}

meta = cargar_meta(ROOT)
inv = pd.read_csv(ROOT / "reports" / "copd_cough_inventario.csv")
inv["id"] = inv.split + "_" + inv.clase + "_" + inv.archivo.str.replace(".wav", "", regex=False)
meta = meta.merge(inv[["id", "duracion_s", "rms", "pico"]], on="id")
meta["Clase"] = meta.label.map({0: "Sano", 1: "EPOC"})


def guardar(nombre, ancho_px=None):
    """Guarda a 300 dpi; con `ancho_px` reajusta los dpi para obtener ese ancho exacto."""
    fig = plt.gcf()
    fig.savefig(FIG / nombre)
    if ancho_px:
        from PIL import Image
        w = Image.open(FIG / nombre).size[0]
        fig.savefig(FIG / nombre, dpi=300 * ancho_px / w)
    plt.close(fig); print("  figura", nombre)


def tabla(df, nombre):
    df.to_csv(TAB / nombre, index=False, encoding="utf-8-sig"); print("  tabla ", nombre)


# =============================================================== DATOS
print("Datos")
res = meta.groupby("Clase").agg(Grabaciones=("id", "count"), Participantes=("grupo", "nunique"),
                                Minutos=("duracion_s", lambda d: round(d.sum() / 60, 1))).loc[["Sano", "EPOC"]]
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
for a, c in zip(ax, res.columns):
    a.bar(res.index, res[c], color=[C_SANO, C_EPOC])
    for i, v in enumerate(res[c]):
        a.text(i, v, f"{v:g}", ha="center", va="bottom")
    a.set_title(c)
guardar("fig_distribucion_clases.png")

fig, ax = plt.subplots(figsize=(6, 3.2))
bins = np.linspace(0, 20, 41)
for cl, col in (("Sano", C_SANO), ("EPOC", C_EPOC)):
    ax.hist(meta[meta.Clase == cl].duracion_s.clip(upper=20), bins=bins, alpha=.6,
            density=True, color=col, label=cl)
ax.set_xlabel("Duración de la grabación (s)"); ax.set_ylabel("Densidad"); ax.legend()
guardar("fig_duraciones.png")

fig, axs = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 2]},
                        constrained_layout=True)
for j, (cl, col) in enumerate((("EPOC", C_EPOC), ("Sano", C_SANO))):
    fila = meta[meta.Clase == cl].sample(1, random_state=3).iloc[0]
    y = cargar_wav(ROOT, fila.npy)[: SR * 6]
    t = np.arange(len(y)) / SR
    axs[0, j].plot(t, y, color=col, lw=.5); axs[0, j].set_title(f"{cl}: forma de onda")
    axs[0, j].set_xlim(0, t[-1]); axs[0, j].set_ylabel("Amplitud")
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, win_length=400,
                                                           hop_length=160, n_mels=64, fmin=50, fmax=8000), ref=np.max)
    librosa.display.specshow(S, sr=SR, hop_length=160, x_axis="time", y_axis="mel", fmin=50, fmax=8000, ax=axs[1, j])
    axs[1, j].set_title(f"{cl}: espectrograma log-Mel (64 bandas)")
    axs[1, j].set_xlabel("Tiempo (s)"); axs[0, j].set_xlabel("Tiempo (s)")
guardar("fig_espectrogramas.png")

calidad = meta.groupby("Clase").agg(
    n=("id", "count"), dur_media=("duracion_s", "mean"), dur_de=("duracion_s", "std"),
    dur_min=("duracion_s", "min"), dur_max=("duracion_s", "max"), rms_medio=("rms", "mean"),
    saturados=("pico", lambda p: int((p >= 0.999).sum()))).loc[["Sano", "EPOC"]].round(3).reset_index()
tabla(calidad, "tab_calidad_por_clase.csv")

# fuga de la partición oficial v2
g_tr = set(meta[meta.split_orig == "train"].grupo); te = meta[meta.split_orig == "test"]
fuga = pd.DataFrame([{"particion": "Oficial Zenodo v2 (train/test)",
                      "audios_test": len(te), "audios_test_con_participante_en_train": int(te.grupo.isin(g_tr).sum()),
                      "porcentaje": round(100 * te.grupo.isin(g_tr).mean(), 1)}])
tabla(fuga, "tab_fuga_particion_v2.csv")

# =============================================================== MODELO
print("Modelo")
cnn = CNNTos().eval()
n_par = sum(p.numel() for p in cnn.parameters())
x = torch.zeros(1, SEG)
with torch.no_grad():
    for _ in range(5):
        cnn(x)
    t0 = time.perf_counter()
    for _ in range(50):
        cnn(x)
lat = (time.perf_counter() - t0) / 50 * 1000
hip = pd.DataFrame([
    ("Entrada", "Forma de onda mono 16 kHz, ventanas de 3 s (salto 1,5 s)"),
    ("Representación", "Log-Mel 64 bandas, 50–8000 Hz, ventana 25 ms, salto 10 ms, normalización por instancia"),
    ("Arquitectura", "CNN con convoluciones separables en profundidad (estilo MobileNet), 7 bloques, pooling medio+máximo"),
    ("Parámetros", f"{n_par:,}".replace(",", " ")),
    ("Tamaño (float32 / int8 estimado)", f"{n_par * 4 / 1e6:.2f} MB / {n_par / 1e6:.2f} MB"),
    ("Aumento de datos", "Velocidad ±10 %, ganancia ±6 dB, ruido SNR 10–40 dB, paso bajo, SpecAugment"),
    ("Desbalance", "Pérdida BCE con pos_weight = sanos/EPOC"),
    ("Optimización", "AdamW (lr 2e-3, weight decay 1e-3), OneCycle, lote 32, máx. 60 épocas, paciencia 15 (AUC val)"),
    ("Predicción por grabación", "Promedio de probabilidades de las ventanas"),
    ("Latencia CPU laptop (3 s de audio)", f"{lat:.1f} ms (Intel Core 5 210H)"),
], columns=["Elemento", "Configuración"])
tabla(hip, "tab_configuracion_cnn.csv")

# =============================================================== RESULTADOS CV
if not (CV / "predicciones_oof.csv").exists():
    print("Sin resultados de validación cruzada todavía; se omiten figuras de resultados.")
    raise SystemExit
print("Resultados")
folds = pd.read_csv(CV / "metricas_por_fold.csv")
oof = pd.read_csv(CV / "predicciones_oof.csv").merge(meta[["id", "grupo"]], on="id")
MODELOS = [m for m in ("Metadatos", "CNN", "OPERA", "Fusion") if f"p_{m}" in oof]
n_rep = oof.repeticion.nunique()

# todas las predicciones fuera de fold apiladas (cada grabación aparece una vez por repetición);
# NO se promedian probabilidades entre repeticiones porque eso sería un ensamble de 5 modelos
prom = oof


def auc_por_rep(df, col):
    """AUC fuera de fold de cada repetición, promediada."""
    return np.mean([roc_auc_score(g.label, g[col]) for _, g in df.groupby("repeticion")])


def boot_grupos(df, col, n=1000, seed=0):
    """IC95 de la AUC media por repetición remuestreando participantes (bootstrap por conglomerados)."""
    rng = np.random.default_rng(seed)
    reps = [g.reset_index(drop=True) for _, g in df.groupby("repeticion")]
    gs = df.grupo.unique()
    idx = [r.groupby("grupo").indices for r in reps]
    v = []
    for _ in range(n):
        elegidos = rng.choice(gs, len(gs))
        aucs = []
        for r, ix in zip(reps, idx):
            sel = np.concatenate([ix[g] for g in elegidos])
            y = r.label.to_numpy()[sel]
            if y.min() != y.max():
                aucs.append(roc_auc_score(y, r[col].to_numpy()[sel]))
        v.append(np.mean(aucs))
    return np.percentile(v, [2.5, 97.5])


cols = ["auc", "sensibilidad", "especificidad", "vpp", "vpn", "f1", "exactitud", "bal_acc"]
filas = []
for m in MODELOS:
    f = folds[folds.modelo == m]
    lo, hi = boot_grupos(prom, f"p_{m}")
    fila = {"Modelo": NOMBRE[m]}
    for c in cols:
        fila[c] = f"{f[c].mean():.3f} ± {f[c].std():.3f}"
    fila["AUC OOF (media de repeticiones)"] = f"{auc_por_rep(prom, f'p_{m}'):.3f}"
    fila["IC95 bootstrap por participante"] = f"[{lo:.3f}; {hi:.3f}]"
    fila["H1 (AUC>=0,70 y LI>0,60)"] = "Se cumple" if (f.auc.mean() >= 0.70 and lo > 0.60) else "No se cumple"
    filas.append(fila)
comp = pd.DataFrame(filas).rename(columns={"auc": "AUC", "sensibilidad": "Sensibilidad",
                                           "especificidad": "Especificidad", "vpp": "VPP", "vpn": "VPN",
                                           "f1": "F1", "exactitud": "Exactitud", "bal_acc": "Exactitud balanceada"})
tabla(comp, "tab_comparacion_modelos.csv")
if (CV / "delong_por_repeticion.csv").exists():
    tabla(pd.read_csv(CV / "delong_por_repeticion.csv").round(4), "tab_delong.csv")

# ROC
fig, ax = plt.subplots(figsize=(5.2, 5))
for m in MODELOS:
    fpr, tpr, _ = roc_curve(prom.label, prom[f"p_{m}"])
    ax.plot(fpr, tpr, color=COLORES[m], lw=1.8,
            label=f"{NOMBRE[m]} (AUC {auc_por_rep(prom, f'p_{m}'):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=.8)
ax.set_xlabel("1 − Especificidad"); ax.set_ylabel("Sensibilidad"); ax.legend(loc="lower right", fontsize=8)
ax.set_title(f"Curvas ROC, validación cruzada agrupada 5×{n_rep}")
guardar("fig_roc_modelos.png")

# Boxplot AUC por fold
fig, ax = plt.subplots(figsize=(6.5, 3.5))
datos = [folds[folds.modelo == m].auc for m in MODELOS]
bp = ax.boxplot(datos, patch_artist=True, widths=.5)
for patch, m in zip(bp["boxes"], MODELOS):
    patch.set_facecolor(COLORES[m]); patch.set_alpha(.6)
ax.set_xticks(range(1, len(MODELOS) + 1), [NOMBRE[m].split(" (")[0] for m in MODELOS], fontsize=8)
ax.axhline(0.70, color="red", ls="--", lw=.8, label="Umbral H1 (0.70)")
ax.axhline(0.50, color="gray", ls=":", lw=.8, label="Azar (0.50)")
ax.set_ylim(0.45, 0.90)
ax.set_ylabel("AUC por partición"); ax.legend(fontsize=8)
guardar("fig_boxplot_auc.png", ancho_px=1800)

# Mejor modelo (excluye metadatos)
mejor = max([m for m in MODELOS if m != "Metadatos"], key=lambda m: folds[folds.modelo == m].auc.mean())
fm = folds[folds.modelo == mejor]
tp, fn, tn, fp = (int(fm[c].sum() / n_rep) for c in ("tp", "fn", "tn", "fp"))
fig, ax = plt.subplots(figsize=(3.8, 3.4))
mat = np.array([[tn, fp], [fn, tp]])
ax.imshow(mat, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=13,
                color="white" if mat[i, j] > mat.max() / 2 else "black")
ax.set_xticks([0, 1], ["Riesgo bajo", "Riesgo alto"]); ax.set_yticks([0, 1], ["Sano", "EPOC"])
ax.set_xlabel("Predicción del tamizaje"); ax.set_ylabel("Clase real")
ax.set_title(f"{NOMBRE[mejor].split(' (')[0]}\n(promedio por repetición)", fontsize=9)
ax.spines[:].set_visible(False)
guardar("fig_matriz_confusion.png")

# VPP / VPN vs prevalencia
se, es = fm.sensibilidad.mean(), fm.especificidad.mean()
prev = np.linspace(0.02, 0.40, 100)
vpp = se * prev / (se * prev + (1 - es) * (1 - prev)); vpn = es * (1 - prev) / (es * (1 - prev) + (1 - se) * prev)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(prev * 100, vpp * 100, label="VPP", color=C_EPOC); ax.plot(prev * 100, vpn * 100, label="VPN", color=C_SANO)
for p_, txt in ((6, "Perú ~6 %"), (26, "Dataset CC 26 %")):
    ax.axvline(p_, color="gray", ls=":", lw=.8); ax.text(p_ + .5, 50, txt, fontsize=8, rotation=90)
ax.set_xlabel("Prevalencia de EPOC (%)"); ax.set_ylabel("%")
ax.set_xlim(2, 40); ax.set_ylim(0, 100); ax.legend()
ax.set_title(f"Valores predictivos ({NOMBRE[mejor].split(' (')[0]}, sens. {se * 100:.1f} %, "
             f"esp. {es * 100:.1f} %)", fontsize=9)
guardar("fig_vpp_vpn_prevalencia.png", ancho_px=1600)
tab_prev = pd.DataFrame([{"Escenario": n, "Prevalencia": f"{p:.0%}",
                          "VPP": f"{se * p / (se * p + (1 - es) * (1 - p)):.1%}",
                          "VPN": f"{es * (1 - p) / (es * (1 - p) + (1 - se) * p):.1%}",
                          "Derivados a espirometría": f"{se * p + (1 - es) * (1 - p):.0%}"}
                         for n, p in (("Población general > 40 años (Perú)", .06),
                                      ("Fumadores / humo de leña (aprox.)", .12), ("Dataset CC", .26))])
tabla(tab_prev, "tab_tamizaje_prevalencias.csv")

# Calibración
fig, ax = plt.subplots(figsize=(4.6, 4.4))
for m in MODELOS:
    fr, mp = calibration_curve(prom.label, prom[f"p_{m}"], n_bins=8, strategy="quantile")
    ax.plot(mp, fr, "o-", color=COLORES[m], ms=3, label=NOMBRE[m].split(" (")[0])
ax.plot([0, 1], [0, 1], "k--", lw=.8)
ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Proporción real de EPOC"); ax.legend(fontsize=7)
guardar("fig_calibracion.png")

# Partición con fuga vs agrupada
base = ROOT / "reports" / "baseline_cnn" / "resumen.json"
if base.exists():
    auc_fuga = json.loads(base.read_text())["ensamble"]["auc"]
    auc_ok = folds[folds.modelo == "CNN"].auc.mean()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(["Partición oficial v2\n(participantes mezclados)", "CV agrupada\npor participante"],
           [auc_fuga, auc_ok], color=["#B0B0B0", COLORES["CNN"]])
    for i, v in enumerate([auc_fuga, auc_ok]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.4, 1); ax.set_ylabel("AUC de la CNN"); ax.set_title("Efecto de la fuga de participantes", fontsize=9)
    guardar("fig_efecto_fuga.png")

print("\nListo:", FIG, TAB)
