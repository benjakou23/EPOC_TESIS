"""Cuantiza los modelos ONNX para el teléfono y mide su consistencia con PyTorch.

La conversión de OPERA-CE a TensorFlow Lite con onnx2tf falló (interpreta mal la
disposición de ejes de la EfficientNet), por lo que el aplicativo usará
ONNX Runtime, que ejecuta directamente los .onnx verificados en 10_exportar_onnx.py.

Cuantización dinámica: pesos de las capas Conv y Gemm en int8; las MatMul del
espectrograma (matrices de Fourier y banco Mel) se dejan en float32.

Indicadores (853 grabaciones, flujo completo del aplicativo, frente a la
referencia de PyTorch models/final/export/referencia_pytorch.csv):
  - tamaño de los modelos (MB)
  - consistencia: diferencia absoluta media y máxima de la probabilidad final
    y % de grabaciones con la misma clasificación de riesgo
  - latencia de inferencia en la PC (ms)

Salida: models/final/export/{cnn_tos,opera_ce_tos}_int8.onnx, consistencia_onnx.{csv,json}
Uso:    python scripts/11_cuantizar_onnx.py
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import QuantType, quantize_dynamic

from audio_utils import cargar_meta, cargar_wav, ventanas

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "models" / "final" / "export"
_spec = importlib.util.spec_from_file_location("exportar", Path(__file__).with_name("10_exportar_onnx.py"))
exportar = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(exportar)
CE_N = exportar.CE_N
UMBRAL = json.loads((EXP / "especificacion_app.json").read_text(encoding="utf-8"))["umbral_tamizaje"]
MODELOS = ("cnn_tos", "opera_ce_tos")

for n in MODELOS:
    quantize_dynamic(EXP / f"{n}.onnx", EXP / f"{n}_int8.onnx", weight_type=QuantType.QInt8,
                     op_types_to_quantize=["Conv", "Gemm"])


def sesion(ruta):
    op = ort.SessionOptions(); op.intra_op_num_threads = 4
    s = ort.InferenceSession(str(ruta), op, providers=["CPUExecutionProvider"])
    return lambda x: float(s.run(None, {"audio": x[None].astype(np.float32)})[0].ravel()[0])


meta = cargar_meta(ROOT).set_index("id")
ref = pd.read_csv(EXP / "referencia_pytorch.csv")
resumen = {}
for tipo, sufijo in (("float32", ""), ("int8", "_int8")):
    cnn, ce = (sesion(EXP / f"{n}{sufijo}.onnx") for n in MODELOS)
    p, t_cnn, t_ce = [], [], []
    for id_ in ref.id:
        y = cargar_wav(ROOT, meta.loc[id_, "npy"])[:CE_N]
        t0 = time.perf_counter(); pc = np.mean([cnn(v) for v in ventanas(y)]); t_cnn.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); po = ce(exportar.relleno_opera(y)); t_ce.append(time.perf_counter() - t0)
        p.append((pc + po) / 2)
    p = np.array(p)
    dif = np.abs(p - ref.p_fusion.to_numpy())
    ref[f"p_fusion_onnx_{tipo}"] = p
    resumen[tipo] = {
        "tamano_mb": {n: round((EXP / f"{n}{sufijo}.onnx").stat().st_size / 1e6, 2) for n in MODELOS},
        "tamano_total_mb": round(sum((EXP / f"{n}{sufijo}.onnx").stat().st_size for n in MODELOS) / 1e6, 2),
        "dif_abs_media": round(float(dif.mean()), 6), "dif_abs_max": round(float(dif.max()), 6),
        "misma_clasificacion": round(float(np.mean((p >= UMBRAL) == ref.riesgo_alto.to_numpy())), 4),
        "latencia_pc_ms": {"cnn_por_grabacion": round(1000 * np.mean(t_cnn), 1),
                           "opera_ce": round(1000 * np.mean(t_ce), 1),
                           "total_media": round(1000 * (np.mean(t_cnn) + np.mean(t_ce)), 1),
                           "total_max": round(1000 * np.max(np.array(t_cnn) + np.array(t_ce)), 1)},
    }
    print(tipo, json.dumps(resumen[tipo], indent=2, ensure_ascii=False))
ref.to_csv(EXP / "consistencia_onnx.csv", index=False)
(EXP / "consistencia_onnx.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
