"""Métricas de tamizaje compartidas por todos los experimentos."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def umbral_por_sensibilidad(y, p, objetivo):
    """Mayor umbral que alcanza la sensibilidad objetivo (se fija en validación)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= objetivo)[0]
    return float(thr[ok[0]]) if len(ok) else float(thr[-1])


def metricas(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); esp = tn / max(tn + fp, 1)
    return {"auc": roc_auc_score(y, p), "sensibilidad": sens, "especificidad": esp,
            "vpp": tp / max(tp + fp, 1), "vpn": tn / max(tn + fn, 1),
            "bal_acc": (sens + esp) / 2, "exactitud": (tp + tn) / len(y),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "umbral": thr,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def ic_bootstrap(y, p, thr, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("auc", "sensibilidad", "especificidad", "vpn")}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        m = metricas(y[i], p[i], thr)
        for k in acc:
            acc[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items()}


def delong(y, p1, p2):
    """Prueba de DeLong (implementación rápida de Sun & Xu, 2014) para dos AUC
    correlacionadas sobre los mismos casos. Devuelve (auc1, auc2, p_valor)."""
    from scipy.stats import norm, rankdata

    y = np.asarray(y); pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    P = np.vstack([p1, p2])
    tx = np.array([rankdata(s[pos]) for s in P])
    ty = np.array([rankdata(s[neg]) for s in P])
    tz = np.array([rankdata(np.concatenate([s[pos], s[neg]])) for s in P])
    auc = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (auc[0] - auc[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(auc[0]), float(auc[1]), float(2 * norm.sf(abs(z)))
