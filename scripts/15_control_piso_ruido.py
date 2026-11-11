"""Control de piso de ruido: clasificadores que solo ven el 20 % (y 10 %) de cuadros de menor energía.

Log-Mel idéntico al de audio_utils.LogMel (64 bandas, 25 ms, salto 10 ms, 50-8000 Hz), sobre la
grabación ORIGINAL sin recortar. Rasgos: media y percentil 90 del log-Mel de esos cuadros + n cuadros.
Particiones, RF y LR: los mismos que 06 (metadatos) y 08 (regresión logística con C por validación agrupada).
"""
import os, subprocess, sys
from pathlib import Path
import librosa, numpy as np, pandas as pd, torch, torchaudio
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import SR, cargar_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/validacion_cruzada/control_piso_ruido"; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "log.txt", "a", encoding="utf-8")


def log(s=""):
    LOG.write(s + "\n"); LOG.flush()


meta = cargar_meta(ROOT)
inv = pd.read_csv(ROOT / "reports/copd_cough_inventario.csv").set_index(["archivo", "split", "clase"])
oof06 = pd.read_csv(ROOT / "reports/validacion_cruzada/predicciones_oof.csv")
y, grupos = meta.label.to_numpy(), meta.grupo.to_numpy()

# ------------------------------------------------------------ rasgos
cache = OUT / "rasgos.npz"
if cache.exists():
    z = np.load(cache); F = {k: z[k] for k in z.files}
else:
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, win_length=400, hop_length=160,
                                               f_min=50, f_max=8000, n_mels=64)
    F = {"20": [], "10": []}
    for r in meta.itertuples():
        w, _ = librosa.load(ROOT / inv.loc[(r.archivo_orig, r.split_orig, r.clase), "ruta"], sr=SR, mono=True)
        m = mel(torch.from_numpy(w))                       # (64, cuadros), potencia
        lm = torch.log(m + 1e-6).numpy()
        e = m.sum(0).numpy()                               # energía por cuadro
        orden = np.argsort(e)
        for pct, k in (("20", 0.20), ("10", 0.10)):
            n = max(1, int(round(k * len(e)))); sel = lm[:, orden[:n]]
            F[pct].append(np.concatenate([sel.mean(1), np.percentile(sel, 90, axis=1), [n]]))
    F = {k: np.array(v) for k, v in F.items()}
    np.savez(cache, **F)
log(f"rasgos: {F['20'].shape}, {F['10'].shape}")
sets = {"PisoRuido": F["20"], "PisoRuido10": F["10"], "PisoRuido_sinN": F["20"][:, :-1]}

# ------------------------------------------------------------ CV (mismas 25 particiones de 06)
o6 = oof06.set_index(["repeticion", "id"])
filas = []
for rep in range(5):
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=100 + rep)
    for k, (i_tr, i_te) in enumerate(skf.split(meta, y, grupos)):
        a, b = next(StratifiedGroupKFold(7, shuffle=True, random_state=rep * 10 + k).split(i_tr, y[i_tr], grupos[i_tr]))
        i_in = i_tr[a]                                    # igual que 06/08: ajuste con el train interno
        assert (o6.loc[rep].loc[meta.id.to_numpy()[i_te]].fold == k).all(), "particiones distintas a 06"
        d = pd.DataFrame({"repeticion": rep, "fold": k, "id": meta.id.to_numpy()[i_te], "label": y[i_te]})
        for nom, X in sets.items():
            rf = RandomForestClassifier(300, class_weight="balanced", min_samples_leaf=3,
                                        random_state=0).fit(X[i_in], y[i_in])
            d[f"p_{nom}_RF"] = rf.predict_proba(X[i_te])[:, 1]
            lr = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
                              {"logisticregression__C": np.logspace(-4, 1, 11)}, scoring="roc_auc",
                              cv=StratifiedGroupKFold(5, shuffle=True, random_state=0))
            lr.fit(X[i_in], y[i_in], groups=grupos[i_in])
            d[f"p_{nom}_LR"] = lr.predict_proba(X[i_te])[:, 1]
        filas.append(d); log(f"r{rep} f{k} ok")
oof = pd.concat(filas).merge(oof06[["repeticion", "fold", "id", "p_Metadatos", "p_CNN"]],
                             on=["repeticion", "fold", "id"], validate="1:1")
oof.to_csv(OUT / "predicciones_oof.csv", index=False)

# ------------------------------------------------------------ comparaciones
mods = ["p_CNN", "p_PisoRuido_RF", "p_PisoRuido_LR", "p_PisoRuido10_RF", "p_PisoRuido10_LR"]
s = subprocess.run([sys.executable, str(ROOT / "scripts/12_comparaciones_vs_metadatos.py"), "--oof", str(OUT / "predicciones_oof.csv"),
                    "--ref", "p_Metadatos", "--modelos", *mods, "--out", str(OUT / "comparaciones_vs_metadatos.txt")],
                   capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
log(s.stdout if s.returncode == 0 else "ERROR 12:\n" + s.stderr[-1500:])

u = oof.merge(pd.read_csv(ROOT / "data/processed/copd_cough/metadata.csv")[["id", "grupo"]], on="id")
reps = sorted(u.repeticion.unique())
sub = {r: u[u.repeticion == r].reset_index(drop=True) for r in reps}
gi = {r: {g: dd.index.to_numpy() for g, dd in sub[r].groupby("grupo")} for r in reps}
gs = u.grupo.unique(); rng = np.random.default_rng(0)
pares = [("p_CNN", m) for m in mods[1:]]
dif = {p: [] for p in pares}
for _ in range(1000):
    sel = rng.choice(gs, len(gs), replace=True); v = {p: [] for p in pares}
    for r in reps:
        dd = sub[r].iloc[np.concatenate([gi[r][g] for g in sel if g in gi[r]])]
        if dd.label.nunique() < 2: continue
        auc = {m: roc_auc_score(dd.label, dd[m]) for m in mods}
        for p in pares: v[p].append(auc[p[0]] - auc[p[1]])
    for p in pares: dif[p].append(np.mean(v[p]))
log("\n== 5. Diferencia pareada de AUC, CNN - piso de ruido (IC95 por participante) ==")
for x, yv in pares:
    pt = np.mean([roc_auc_score(sub[r].label, sub[r][x]) - roc_auc_score(sub[r].label, sub[r][yv]) for r in reps])
    lo, hi = np.percentile(dif[(x, yv)], [2.5, 97.5])
    log(f"  CNN - {yv[2:]}: {pt:+.3f}  [{lo:+.3f}; {hi:+.3f}]  {'EXCLUYE 0' if lo > 0 or hi < 0 else 'incluye 0'}")
log("\n== 6. Ablación: sin el número de cuadros (proporcional a la duración) ==")
for c in ("p_PisoRuido_sinN_RF", "p_PisoRuido_sinN_LR"):
    log(f"  {c[2:]}: AUC {np.mean([roc_auc_score(sub[r].label, sub[r][c]) for r in reps]):.3f}")
