"""Genera los 7 diagramas metodológicos de la tesis (tamizaje de EPOC por tos).

Salida: PNG a 300 dpi en la misma carpeta que este script.
Uso:    python generar_diagramas.py

Al guardar cada figura se verifica que ningún texto se salga de su caja ni se
superponga con otro, y se informa el tamaño de letra efectivo al insertar la
imagen con 15 cm de ancho en Word (mínimo exigido: 9 pt).
"""
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parent
ANCHO_WORD_CM = 15.0
FS = 11          # 11 pt a 7 in de ancho equivalen a ~9.3 pt a 15 cm
FS_MIN_WORD = 9.0

for _f in ("Arial", "Liberation Sans", "DejaVu Sans"):
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        FUENTE = _f
        break
    except ValueError:
        continue
plt.rcParams.update({"font.family": FUENTE, "font.size": FS, "mathtext.fontset": "custom",
                     "mathtext.rm": FUENTE, "mathtext.it": f"{FUENTE}:italic"})

C = {"sano": "#4C9F70", "epoc": "#C8553D", "modelo": "#1F77B4", "opera": "#FF7F0E",
     "fusion": "#2CA02C", "gris": "#9EA1A8"}
LW_CAJA, LW_FLECHA = 1.2, 1.3
_pares = []   # (texto, caja) para verificar que el texto quede dentro


def suave(color, a=0.15):
    """Equivalente opaco del color con alpha `a` sobre fondo blanco."""
    r, g, b = to_rgb(C.get(color, color))
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def lienzo(alto):
    fig = plt.figure(figsize=(7, alto), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7); ax.set_ylim(0, alto); ax.axis("off")
    _pares.clear()
    return fig, ax


def caja(ax, cx, cy, w, h, texto="", color="modelo", punteada=False, negrita=False,
         relleno=0.15, dx_texto=0.0, z=2):
    col = C["gris"] if punteada else C.get(color, color)
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                       fc=suave(col, relleno), ec=col, lw=LW_CAJA, ls="--" if punteada else "-", zorder=z)
    ax.add_patch(p)
    if texto:
        t = ax.text(cx + dx_texto, cy, texto, ha="center", va="center", fontsize=FS, zorder=z + 1,
                    fontweight="bold" if negrita else "normal", linespacing=1.15)
        _pares.append((t, p))
    return p


def contenedor(ax, x0, y0, x1, y1, titulo="", punteada=False, centrado=False):
    """Recuadro grande con título arriba (zonas, bandas, capas, nodos)."""
    p = caja(ax, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, punteada=punteada,
             color="gris", relleno=0.06, z=0)
    if titulo:
        x = (x0 + x1) / 2 if centrado else x0 + 0.1
        t = ax.text(x, y1 - 0.07, titulo, ha="center" if centrado else "left", va="top",
                    fontsize=FS, fontweight="bold", linespacing=1.1, zorder=1)
        _pares.append((t, p))
    return p


def flecha(ax, a, b, punteada=False, doble=False, lw=LW_FLECHA, ms=11):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>" if doble else "-|>", mutation_scale=ms,
                                 lw=lw, color="black", ls="--" if punteada else "-",
                                 shrinkA=0, shrinkB=0, zorder=5))


def codo(ax, puntos):
    """Polilínea con punta de flecha solo en el último tramo."""
    xs, ys = zip(*puntos[:-1])
    ax.plot(xs, ys, color="black", lw=LW_FLECHA, zorder=5, solid_capstyle="butt")
    flecha(ax, puntos[-2], puntos[-1])


def texto(ax, x, y, s, **kw):
    kw.setdefault("fontsize", FS); kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    return ax.text(x, y, s, linespacing=1.15, zorder=6, **kw)


def candado(ax, x, y, s=0.17):
    ax.add_patch(Rectangle((x - s / 2, y - s * 0.55), s, s * 0.7, fc="black", ec="black", zorder=6))
    ax.add_patch(Arc((x, y + s * 0.15), s * 0.62, s * 0.8, theta1=0, theta2=180, lw=1.3,
                     color="black", zorder=6))


def reajustar_ancho(fig, ruta, ancho_px):
    """Vuelve a exportar con los dpi necesarios para un ancho exacto en píxeles."""
    w = Image.open(ruta).size[0]
    fig.savefig(ruta, dpi=300 * ancho_px / w, bbox_inches="tight", pad_inches=0.1, facecolor="white")


def guardar(fig, ax, nombre, ancho_px=None):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    avisos, m = [], 2   # margen interior mínimo en píxeles de pantalla
    for t, p in _pares:
        bt, bp = t.get_window_extent(r), p.get_window_extent(r)
        if bt.x0 < bp.x0 + m or bt.x1 > bp.x1 - m or bt.y0 < bp.y0 + m or bt.y1 > bp.y1 - m:
            avisos.append(f"texto fuera de su caja: {t.get_text()!r}")
    textos = [t for t in ax.texts if t.get_text().strip()]
    for a, b in combinations(textos, 2):
        if a.get_window_extent(r).overlaps(b.get_window_extent(r)):
            avisos.append(f"textos superpuestos: {a.get_text()!r} / {b.get_text()!r}")
    ruta = OUT / nombre
    fig.savefig(ruta, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    ancho_in = Image.open(ruta).size[0] / 300
    if ancho_px:
        reajustar_ancho(fig, ruta, ancho_px)
    plt.close(fig)
    w, h = Image.open(ruta).size
    fmin = min(t.get_fontsize() for t in textos) * ANCHO_WORD_CM / (ancho_in * 2.54)
    if fmin < FS_MIN_WORD:
        avisos.append(f"letra mínima {fmin:.1f} pt a 15 cm (< 9 pt)")
    if w > 2400:
        avisos.append(f"ancho {w} px > 2400 px")
    print(f"{nombre:30s} {w} × {h} px | letra mínima a 15 cm: {fmin:.1f} pt | "
          + ("OK" if not avisos else "AVISOS:"))
    for a in avisos:
        print("   -", a.replace("\n", " "))


def miniatura_onda(ax, x0, y0, w, h):
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 1600)
    env = 0.95 * np.exp(-((t - 0.25) / 0.05) ** 2 * 0.5) * (t > 0.2) \
        + 0.55 * np.exp(-((t - 0.62) / 0.05) ** 2 * 0.5) * (t > 0.58) + 0.03
    ins = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
    ins.plot(t, env * rng.standard_normal(t.size), color="black", lw=0.4)
    ins.set_ylim(-2.8, 2.8); ins.axis("off")


def miniatura_espectrograma(ax, x0, y0, w, h):
    rng = np.random.default_rng(2)
    f = np.linspace(0, 1, 64)[:, None]; t = np.linspace(0, 1, 120)[None, :]
    burst = lambda c, a: a * np.exp(-((t - c) / 0.06) ** 2) * (t > c - 0.03) * np.exp(-2.2 * f)
    S = burst(0.25, 1.0) + burst(0.62, 0.6) + 0.08 * rng.random((64, 120))
    ins = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
    ins.imshow(np.log(S + 0.02), origin="lower", aspect="auto", cmap="magma")
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_linewidth(0.8)


# --------------------------------------------------------------- 1. log-Mel / MFCC
def fig_flujo_mel():
    fig, ax = lienzo(3.75)
    anchos, gap, x = [0.95, 0.95, 1.1, 1.0, 0.9, 1.2], 0.15, 0.07
    cx = []
    for w in anchos:
        cx.append(x + w / 2); x += w + gap
    Y, H = 1.95, 0.62
    etiquetas = ["Señal\nde tos", "Ventanas\nsolapadas", "Transformada\nde Fourier",
                 "Banco de\nfiltros Mel", "Logaritmo"]
    params = ["16 kHz, mono", "25 ms,\nsalto de 10 ms", "espectro de\nmagnitud", "64 bandas,\n50–8000 Hz", ""]
    for i, (e, p) in enumerate(zip(etiquetas, params)):
        caja(ax, cx[i], Y, anchos[i], H, e)
        if p:
            texto(ax, cx[i], Y - H / 2 - 0.06, p, va="top")
        if i < 4:
            flecha(ax, (cx[i] + anchos[i] / 2, Y), (cx[i + 1] - anchos[i + 1] / 2, Y))
    texto(ax, cx[3], 0.85, r"$m = 2595\cdot\log_{10}(1 + f/700)$")
    # rama superior: espectrograma log-Mel
    ys, yd = 2.75, 1.12
    caja(ax, cx[5], ys, anchos[5], H, "Espectrograma\nlog-Mel", color="modelo")
    texto(ax, cx[5], ys - H / 2 - 0.2, "entrada de la CNN")
    codo(ax, [(cx[4], Y + H / 2), (cx[4], ys), (cx[5] - anchos[5] / 2, ys)])
    # rama inferior: DCT -> MFCC
    caja(ax, cx[5], yd, anchos[5], 0.8, "Transformada\ndel coseno\ndiscreta", color="gris")
    codo(ax, [(cx[4], Y - H / 2), (cx[4], yd), (cx[5] - anchos[5] / 2, yd)])
    caja(ax, cx[5], 0.3, anchos[5], 0.42, "MFCC", color="gris")
    flecha(ax, (cx[5], yd - 0.4), (cx[5], 0.51))
    miniatura_onda(ax, cx[0] - 0.44, Y + H / 2 + 0.1, 0.88, 0.5)
    miniatura_espectrograma(ax, cx[5] - 0.5, ys + H / 2 + 0.1, 1.0, 0.55)
    guardar(fig, ax, "fig_flujo_mel.png")


# --------------------------------------------------------------- 2. CNN
def fig_cnn():
    fig, ax = lienzo(2.95)
    Y = 2.05
    miniatura_espectrograma(ax, 0.15, Y - 0.6, 0.72, 1.2)
    texto(ax, 0.51, Y - 0.7, "Espectrograma\nlog-Mel\n(64 × T)", va="top")
    bloques = [(1.1, 3, 1.1), (1.65, 4, 0.8), (2.25, 5, 0.55)]   # (x inicio, n mapas, alto)
    dx, dy, wm = 0.07, 0.05, 0.1
    previo = 0.87
    for bx, n, hm in bloques:
        flecha(ax, (previo + 0.02, Y), (bx - 0.02, Y))
        for i in range(n):
            y0 = Y - hm / 2 + (i - (n - 1) / 2) * dy
            ax.add_patch(Rectangle((bx + i * dx, y0), wm, hm, fc=suave("modelo"), ec=C["modelo"],
                                   lw=LW_CAJA, zorder=2 + i))
        previo = bx + (n - 1) * dx + wm
    # llave bajo los bloques
    xa, xb, yl = 1.05, 2.68, 1.4
    ax.plot([xa, xa, xb, xb], [yl + 0.08, yl, yl, yl + 0.08], color="black", lw=1)
    texto(ax, (xa + xb) / 2, yl - 0.08, "Convolución separable\n+ ReLU → Pooling", va="top")
    texto(ax, (xa + xb) / 2, yl - 0.58, "× 7 bloques", va="top", fontweight="bold")
    caja(ax, 3.6, Y, 1.3, 0.75, "Pooling global\nmedio + máximo")
    caja(ax, 4.78, Y, 0.75, 0.75, "Capa\ndensa")
    caja(ax, 6.175, Y, 1.45, 0.75, "Probabilidad\nde EPOC\n(valor entre 0 y 1)", color="epoc")
    flecha(ax, (previo + 0.02, Y), (2.95, Y))
    flecha(ax, (4.25, Y), (4.405, Y))
    flecha(ax, (5.155, Y), (5.45, Y))
    caja(ax, 4.9, 0.5, 4.0, 0.82, "Ventanas de 3 s (salto 1.5 s)\nPredicción por grabación = promedio de las\n"
         "probabilidades de sus ventanas", color="gris")
    guardar(fig, ax, "fig_cnn.png")


# --------------------------------------------------------------- 3. transferencia
def fig_transferencia():
    fig, ax = lienzo(3.5)
    filas = [2.4, 1.45, 0.5]
    caja(ax, 0.43, 1.45, 0.82, 0.85, "Grabación\nde tos\n(CC)", color="gris")
    contenedor(ax, 1.0, 0.02, 2.95, 3.45, "Preentrenamiento\n(otro dominio)", centrado=True)
    contenedor(ax, 3.05, 0.02, 4.95, 3.45, "Incrustaciones", centrado=True)
    contenedor(ax, 5.07, 0.02, 6.2, 3.45, "Tarea propia", centrado=True)
    pre = [("AudioSet → YAMNet\npesos congelados\n(exploratorio)", "gris"),
           ("AudioSet → VGGish\npesos congelados\n(exploratorio)", "gris"),
           ("Sonidos respiratorios\n→ OPERA\npesos congelados", "opera")]
    ax.plot([0.9, 0.9], [filas[2], filas[0]], color="black", lw=LW_FLECHA)
    for y, (t, col) in zip(filas, pre):
        caja(ax, 1.975, y, 1.8, 0.82, t, color=col, dx_texto=0.11)
        candado(ax, 1.2, y)
        flecha(ax, (0.9, y), (1.075, y))
    # barras (alto ~ raíz de la dimensión) y etiquetas sobre la flecha
    emb = [([("YAMNet", 1024, "gris")], "YAMNet (1024)"),
           ([("VGGish", 128, "gris")], "VGGish (128)"),
           ([("CT", 768, "opera"), ("CE", 1280, "opera")], "OPERA-CT (768) +\nOPERA-CE (1280)\nconcatenadas")]
    destinos = [1.75, 1.45, 1.15]
    for y, (barras, etiqueta), yd in zip(filas, emb, destinos):
        xs = [4.56] if len(barras) == 1 else [4.45, 4.67]
        for xb, (_, d, col) in zip(xs, barras):
            hb = 0.72 * np.sqrt(d / 1280)
            for k in range(8):
                ax.add_patch(Rectangle((xb - 0.08, y - hb / 2 + k * hb / 8), 0.16, hb / 8,
                                       fc=suave(col, 0.35), ec=C[col], lw=0.8, zorder=3))
        flecha(ax, (2.875, y), (xs[0] - 0.1, y))
        texto(ax, 3.65, y + 0.06, etiqueta, va="bottom")
        flecha(ax, (xs[-1] + 0.1, y), (5.2, yd))
    caja(ax, 5.665, 1.45, 0.93, 1.05, "Regresión\nlogística\n(solo se\nentrena este\nbloque)")
    caja(ax, 6.6, 1.45, 0.72, 0.6, "Riesgo\nde EPOC", color="epoc")
    flecha(ax, (6.13, 1.45), (6.24, 1.45), ms=9)
    guardar(fig, ax, "fig_transferencia.png")


# --------------------------------------------------------------- 4. TFLite
def fig_tflite():
    fig, ax = lienzo(3.65)
    xs, w = [0.97, 2.69, 4.41, 6.08], 1.48
    contenedor(ax, 0.05, 2.05, 6.95, 3.6, "Equipo de desarrollo (PC)")
    for x, t in zip(xs, ["Modelo entrenado\n(PyTorch)", "Conversión a\nTensorFlow Lite",
                         "Cuantización int8\n0.40 MB → 0.10 MB", "Archivo .tflite"]):
        caja(ax, x, 2.65, w, 0.75, t)
    contenedor(ax, 0.05, 0.05, 6.95, 1.6, "Teléfono Android")
    contenedor(ax, 0.12, 0.12, 6.9, 1.3, "")
    texto(ax, 0.3, 1.22, "Aplicativo móvil (el archivo .tflite se incrusta)", ha="left", va="top")
    for i, (x, t) in enumerate(zip(xs, ["Grabar tos", "Preprocesar y\nextraer log-Mel",
                                        "Inferencia\nTFLite", "Riesgo bajo /\nriesgo alto"])):
        caja(ax, x, 0.5, w, 0.6, t, color="epoc" if i == 3 else "modelo")
    for i in range(3):
        flecha(ax, (xs[i] + w / 2, 2.65), (xs[i + 1] - w / 2, 2.65))
        flecha(ax, (xs[i] + w / 2, 0.5), (xs[i + 1] - w / 2, 0.5))
    flecha(ax, (6.08, 2.275), (6.08, 1.3), lw=3, ms=20)
    texto(ax, 5.93, 1.83, "despliegue", ha="right")
    guardar(fig, ax, "fig_tflite.png")


# --------------------------------------------------------------- 5. validación cruzada
def fig_validacion_cruzada():
    fig, ax = lienzo(4.3)
    # leyenda
    for x, col, t in [(0.95, "epoc", "Prueba"), (2.0, "modelo", "Entrenamiento"),
                      (3.6, "opera", "Validación interna ~15 %, por participante")]:
        ax.add_patch(Rectangle((x, 4.03), 0.2, 0.16, fc=suave(col, 0.35), ec=C[col], lw=LW_CAJA))
        texto(ax, x + 0.28, 4.11, t, ha="left")
    texto(ax, 2.71, 3.72, "Grupos de participantes (una repetición)")
    x0, wb, gb, hb = 1.0, 0.62, 0.08, 0.34
    filas = [3.3, 2.85, 2.4, 1.95, 1.5]
    for k, y in enumerate(filas):
        texto(ax, 0.92, y, f"Partición {k + 1}", ha="right")
        jv = 4 if k != 4 else 3
        for j in range(5):
            xb = x0 + j * (wb + gb)
            if j == k:
                caja(ax, xb + wb / 2, y, wb, hb, "Prueba", color="epoc", relleno=0.3)
            else:
                caja(ax, xb + wb / 2, y, wb, hb, color="modelo", relleno=0.3)
            if j == jv:
                wv = 0.6 * wb
                ax.add_patch(Rectangle((xb + wb - wv - 0.03, y - hb / 2 + 0.04), wv, hb - 0.08,
                                       fc=suave("opera", 0.45), ec=C["opera"], lw=LW_CAJA, zorder=3))
    caja(ax, 5.8, 2.4, 2.3, 1.95,
         "• 289 participantes\n(78 EPOC, 211 sanos)\n\n• 5 particiones × 5 repeticiones\n= 25 particiones\n\n"
         "• Un mismo participante nunca\nestá en entrenamiento y prueba", color="gris")
    xv = x0 + 3 * (wb + gb) + wb - 0.3 * wb - 0.03
    caja(ax, 2.3, 0.45, 2.6, 0.62, "Parada temprana y umbral con\nsensibilidad objetivo 85 %")
    caja(ax, 5.6, 0.45, 2.3, 0.62, "Se aplica sin cambios\na la prueba", color="epoc")
    flecha(ax, (xv, filas[4] - hb / 2 + 0.04), (xv, 0.76))
    flecha(ax, (3.6, 0.45), (4.45, 0.45))
    guardar(fig, ax, "fig_validacion_cruzada.png")


# --------------------------------------------------------------- 6. arquitectura lógica
def fig_arquitectura_logica():
    fig, ax = lienzo(4.5)
    capas = [("Presentación", 3.37, 4.48, ["Inicio", "Grabación de tos",
              "Resultado: riesgo bajo /\nriesgo alto + recomendación\nde espirometría"], 0.66),
             ("Procesamiento", 1.95, 3.07, ["Detección de\nla tos y\ncontrol de\ncalidad",
              "Preprocesamiento\n(mono, 16 kHz,\nrecorte de\nsilencios,\nnormalización)",
              "Extracción\nlog-Mel", "Modelo\nTFLite", "Umbral de\ntamizaje"], 0.98),
             ("Datos", 1.0, 1.65, ["Modelo .tflite", "Registro local de resultados\n(con consentimiento)",
              "Configuración"], 0.5)]
    for nombre, y0, y1, cajas, hc in capas:
        p = contenedor(ax, 0.05, y0, 6.95, y1)
        t = texto(ax, 0.27, (y0 + y1) / 2, nombre, rotation=90, fontweight="bold")
        _pares.append((t, p))
        n = len(cajas)
        anchos = [1.2, 1.43, 1.13, 1.13, 1.13] if n == 5 else [(6.47 - 0.08 * (n - 1)) / n] * n
        x = 0.45
        for w, c in zip(anchos, cajas):
            caja(ax, x + w / 2, (y0 + y1) / 2, w, hc, c,
                 color="epoc" if c.startswith("Resultado") else "modelo")
            x += w + 0.08
    for xa in (2.4, 5.0):
        flecha(ax, (xa, 3.37), (xa, 3.07), doble=True, ms=9)
        flecha(ax, (xa, 1.95), (xa, 1.65), doble=True, ms=9)
    contenedor(ax, 0.05, 0.05, 6.95, 0.82, punteada=True)
    texto(ax, 0.15, 0.435, "Funciones secundarias /\ntrabajo futuro", ha="left")
    for i, c in enumerate(["Cuestionario\nde riesgo", "Multilingüe", "Guía por voz", "Ruta al centro\nde salud"]):
        caja(ax, 2.6 + i * 1.25, 0.435, 1.12, 0.55, c, punteada=True)
    guardar(fig, ax, "fig_arquitectura_logica.png")


# --------------------------------------------------------------- 7. arquitectura física
def fig_arquitectura_fisica():
    fig, ax = lienzo(3.4)
    contenedor(ax, 0.05, 0.6, 4.55, 3.3, "Teléfono Android")
    caja(ax, 0.9, 2.15, 1.0, 0.55, "Micrófono", color="gris")
    caja(ax, 2.85, 2.15, 1.9, 1.0, "Aplicativo móvil\n(interfaz + procesamiento\n+ modelo .tflite)")
    caja(ax, 2.85, 0.98, 1.9, 0.45, "Almacenamiento local", color="gris")
    flecha(ax, (1.4, 2.15), (1.9, 2.15))
    flecha(ax, (2.85, 1.65), (2.85, 1.205), doble=True)
    contenedor(ax, 4.9, 1.05, 6.95, 3.3, "Servicios externos\n(opcionales)", punteada=True)
    caja(ax, 5.925, 2.35, 1.85, 0.55, "Mapas / ruta a\ncentro de salud", punteada=True)
    caja(ax, 5.925, 1.53, 1.85, 0.75, "Sincronización de\ncasos (modo salud,\ncon consentimiento)", punteada=True)
    flecha(ax, (3.8, 2.15), (4.9, 2.15), punteada=True)
    texto(ax, 4.35, 2.22, "requiere\nconexión", va="bottom")
    # recorta el lienzo al borde inferior del teléfono (sin nota al pie)
    fig.set_size_inches(7, 2.8); ax.set_ylim(0.55, 3.35)
    guardar(fig, ax, "fig_arquitectura_fisica.png", ancho_px=2000)


# --------------------------------------------------------------- fases de la tos
def fig_fases_tos():
    """Tos sintética ilustrativa: inspiración, compresión y espiración explosiva."""
    import librosa
    import librosa.display
    sr, rng = 16000, np.random.default_rng(3)
    fases = [("Inspiración", 0.0, 0.40, "gris"), ("Compresión", 0.40, 0.55, "opera"),
             ("Espiración explosiva", 0.55, 1.0, "modelo")]
    t = np.arange(int(sr * 1.0)) / sr
    ruido = rng.standard_normal(t.size)
    # inspiración: soplo débil y largo (ruido filtrado paso bajo)
    from scipy.signal import butter, sosfilt
    soplo = sosfilt(butter(4, 1500, fs=sr, output="sos"), ruido)
    insp = soplo / np.abs(soplo).max() * 0.12 * np.sin(np.pi * np.clip(t / 0.4, 0, 1)) ** 2
    insp *= t < 0.4
    # compresión: glotis cerrada, casi silencio
    comp = 0.0008 * ruido * ((t >= 0.4) & (t < 0.55))
    # espiración: pico explosivo de banda ancha + componente sonora que decae
    te = np.clip(t - 0.55, 0, None)
    env = (t >= 0.55) * (1 - np.exp(-te / 0.004)) * np.exp(-te / 0.09)
    sonora = sum(np.sin(2 * np.pi * f0 * k * te) / k for k in range(1, 6) for f0 in [260])
    esp = env * (0.85 * ruido + 0.35 * sonora)
    y = insp + comp + esp
    y = 0.95 * y / np.abs(y).max()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.67, 3.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.2]}, constrained_layout=True)
    a1.plot(t, y, color="black", lw=0.5)
    a1.set_ylabel("Amplitud", fontsize=9); a1.set_ylim(-1.05, 1.3)
    for nombre, i0, i1, col in fases:
        a1.axvspan(i0, i1, color=suave(col, 0.25), zorder=0, lw=0)
        a1.text((i0 + i1) / 2, 1.12, nombre, ha="center", va="center", fontsize=9)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160,
                                       n_mels=64, fmin=50, fmax=8000)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, hop_length=160, x_axis="time",
                             y_axis="mel", fmin=50, fmax=8000, cmap="magma", ax=a2)
    for _, i0, i1, col in fases:
        a2.axvspan(i0, i1, color=C[col], alpha=0.18, lw=0)
    for x in (0.40, 0.55):
        for a in (a1, a2):
            a.axvline(x, color="white" if a is a2 else C["gris"], ls="--", lw=0.8)
    a2.set_ylabel("Frecuencia (Hz)", fontsize=9); a2.set_xlabel("Tiempo (s)", fontsize=9)
    a2.set_xlim(0, 1.0)
    for a in (a1, a2):
        a.tick_params(labelsize=8)
        for s in ("top", "right"):
            a1.spines[s].set_visible(False)
    ruta = OUT / "fig_fases_tos.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    reajustar_ancho(fig, ruta, 2000)
    plt.close(fig)
    w, h = Image.open(ruta).size
    print(f"{'fig_fases_tos.png':30s} {w} × {h} px")


if __name__ == "__main__":
    print("Fuente:", FUENTE)
    for f in (fig_flujo_mel, fig_cnn, fig_transferencia, fig_tflite, fig_validacion_cruzada,
              fig_arquitectura_logica, fig_arquitectura_fisica, fig_fases_tos):
        f()
