"""Piso de ruido con número FIJO de cuadros (100 de menor energía): sin duración ni número de cuadros."""
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
OUT = ROOT / "reports/validacion_cruzada/control_duracion"; OUT.mkdir(parents=True, exist_ok=True)
NF = 100
LOG = open(OUT / "log_piso_fijo.txt", "a", encoding="utf-8")


def log(s=""):
    LOG.write(s + "\n"); LOG.flush()


meta = cargar_meta(ROOT)
inv = pd.read_csv(ROOT / "reports/copd_cough_inventario.csv").set_index(["archivo", "split", "clase"])
oof06 = pd.read_csv(ROOT / "reports/validacion_cruzada/predicciones_oof.csv")
y, grupos = meta.label.to_numpy(), meta.grupo.to_numpy()

cache = OUT / "rasgos_fijo100.npz"
if cache.exists():
    z = np.load(cache); X, ncuad = z["X"], z["ncuad"]
else:
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=512, win_length=400, hop_length=160,
                                               f_min=50, f_max=8000, n_mels=64)
    X, ncuad = [], []
    for r in meta.itertuples():
        w, _ = librosa.load(ROOT / inv.loc[(r.archivo_orig, r.split_orig, r.clase), "ruta"], sr=SR, mono=True)
        m = mel(torch.from_numpy(w)); ncuad.append(m.shape[1])
        if m.shape[1] < NF:
            X.append(np.full(128, np.nan)); continue
        sel = torch.log(m + 1e-6).numpy()[:, np.argsort(m.sum(0).numpy())[:NF]]
        X.append(np.concatenate([sel.mean(1), np.percentile(sel, 90, axis=1)]))
    X, ncuad = np.array(X), np.array(ncuad)
    np.savez(cache, X=X, ncuad=ncuad)
ok = ~np.isnan(X).any(1)
log(f"Grabaciones con < {NF} cuadros (excluidas): {(~ok).sum()} de {len(ok)}")

o6 = oof06.set_index(["repeticion", "id"])
filas = []
for rep in range(5):
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=100 + rep)
    for k, (i_tr, i_te) in enumerate(skf.split(meta, y, grupos)):
        a, _ = next(StratifiedGroupKFold(7, shuffle=True, random_state=rep * 10 + k).split(i_tr, y[i_tr], grupos[i_tr]))
        i_in = i_tr[a]
        assert (o6.loc[rep].loc[meta.id.to_numpy()[i_te]].fold == k).all(), "particiones distintas a 06"
        i_in, i_te = i_in[ok[i_in]], i_te[ok[i_te]]
        rf = RandomForestClassifier(300, class_weight="balanced", min_samples_leaf=3, random_state=0).fit(X[i_in], y[i_in])
        lr = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
                          {"logisticregression__C": np.logspace(-4, 1, 11)}, scoring="roc_auc",
                          cv=StratifiedGroupKFold(5, shuffle=True, random_state=0))
        lr.fit(X[i_in], y[i_in], groups=grupos[i_in])
        filas.append(pd.DataFrame({"repeticion": rep, "fold": k, "id": meta.id.to_numpy()[i_te], "label": y[i_te],
                                   "p_PisoFijo_RF": rf.predict_proba(X[i_te])[:, 1],
                                   "p_PisoFijo_LR": lr.predict_proba(X[i_te])[:, 1]}))
        log(f"r{rep} f{k} ok")
oof = pd.concat(filas).merge(oof06[["repeticion", "fold", "id", "p_Metadatos", "p_CNN"]],
                             on=["repeticion", "fold", "id"], validate="1:1")
oof.to_csv(OUT / "predicciones_oof_piso_fijo.csv", index=False)
mods = ["p_CNN", "p_PisoFijo_RF", "p_PisoFijo_LR"]
s = subprocess.run([sys.executable, str(ROOT / "scripts/12_comparaciones_vs_metadatos.py"), "--oof", str(OUT / "predicciones_oof_piso_fijo.csv"),
                    "--ref", "p_Metadatos", "--modelos", *mods, "--out", str(OUT / "comparaciones_piso_fijo.txt")],
                   capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
log(s.stdout if s.returncode == 0 else "ERROR 12:\n" + s.stderr[-1500:])

u = oof.merge(meta[["id", "grupo"]], on="id")
reps = sorted(u.repeticion.unique())
sub = {r: u[u.repeticion == r].reset_index(drop=True) for r in reps}
gi = {r: {g: d.index.to_numpy() for g, d in sub[r].groupby("grupo")} for r in reps}
gs = u.grupo.unique(); rng = np.random.default_rng(0)
pares = [("p_CNN", "p_PisoFijo_RF"), ("p_CNN", "p_PisoFijo_LR")]
dif = {p: [] for p in pares}
for _ in range(1000):
    sel = rng.choice(gs, len(gs), replace=True); v = {p: [] for p in pares}
    for r in reps:
        d = sub[r].iloc[np.concatenate([gi[r][g] for g in sel if g in gi[r]])]
        if d.label.nunique() < 2: continue
        au = {m: roc_auc_score(d.label, d[m]) for m in mods}
        for p in pares: v[p].append(au[p[0]] - au[p[1]])
    for p in pares: dif[p].append(np.mean(v[p]))
log("\n== 5. Diferencia pareada de AUC, CNN - piso fijo (IC95 por participante) ==")
for x, yv in pares:
    pt = np.mean([roc_auc_score(sub[r].label, sub[r][x]) - roc_auc_score(sub[r].label, sub[r][yv]) for r in reps])
    lo, hi = np.percentile(dif[(x, yv)], [2.5, 97.5])
    log(f"  CNN - {yv[2:]}: {pt:+.3f}  [{lo:+.3f}; {hi:+.3f}]  {'EXCLUYE 0' if lo > 0 or hi < 0 else 'incluye 0'}")
