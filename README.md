# Sistema móvil inteligente basado en Deep Learning para la detección automática de enfermedades respiratorias mediante análisis de señales acústicas

Código y resultados de la tesis (Universidad Señor de Sipán, Pimentel, 2026).
Autores: Anderson Benjamín Girón Villegas y Lucrecia Victoria Montes Pozo.

**Caso de aplicación: la Enfermedad Pulmonar Obstructiva Crónica (EPOC).** El sistema analiza la tos grabada con un teléfono y es de **tamizaje**: identifica a las personas con riesgo que deben derivarse a espirometría. No es un diagnóstico.

## Datos

Conjunto **Cough-COPD (CC)**: 853 grabaciones de tos (221 EPOC, 632 sanos) de 289 participantes/sesiones. Licencia CC BY 4.0.

- Versión 1, con carpetas por participante: https://doi.org/10.5281/zenodo.10209837
- Versión 2, archivos renombrados y sin participantes: https://doi.org/10.5281/zenodo.14881818

Los audios **no** se incluyen en el repositorio. La partición oficial de la versión 2 mezcla participantes entre entrenamiento y prueba (162 de 172 audios de prueba), por lo que toda la evaluación usa **validación cruzada 5×5 agrupada por participante** y el participante de cada audio se recupera de la versión 1 (`02b_grupos_desde_v1.py`).

## Resultados

Validación cruzada estratificada 5 particiones × 5 repeticiones, agrupada por participante. IC95 por bootstrap remuestreando participantes. Umbral de cada partición fijado en su validación interna para una sensibilidad objetivo de 85 %. H1: AUC ≥ 0.70 y límite inferior del IC95 > 0.60.

| Modelo | AUC (media ± DE) | IC95 | Sensibilidad | Especificidad | VPN | ¿Cabe en el teléfono? | H1 |
|---|---|---|---|---|---|---|---|
| Fusión CNN + OPERA (CT + CE) | 0.741 ± 0.074 | 0.678–0.771 | 0.766 | 0.520 | 0.880 | No (OPERA-CT: 32 M parámetros) | Sí |
| **Fusión CNN + OPERA-CE (seleccionado)** | **0.721 ± 0.083** | **0.666–0.765** | 0.746 | 0.529 | 0.871 | **Sí** | **Sí** |
| OPERA CT + CE + regresión logística | 0.717 ± 0.081 | 0.652–0.751 | 0.764 | 0.480 | 0.875 | No | Sí |
| OPERA-CE + regresión logística | 0.688 ± 0.088 | 0.626–0.736 | 0.740 | 0.475 | 0.858 | Sí | No |
| CNN ligera desde cero | 0.658 ± 0.109 | 0.599–0.680 | 0.818 | 0.369 | 0.840 | Sí | No |
| Solo metadatos (referencia) | 0.649 ± 0.083 | 0.575–0.711 | 0.762 | 0.336 | 0.805 | — | No |

**Modelo final** (fusión CNN + OPERA-CE, entrenado con las 853 grabaciones): **umbral de tamizaje 0.3405**, fijado con las predicciones fuera de partición para sensibilidad ≥ 85 %. En la validación cruzada da sensibilidad 85.1 %, especificidad 39.4 % y VPN 88.3 %.

**Viabilidad móvil** (853 grabaciones, flujo completo del aplicativo, versión float32):

| Indicador | Resultado |
|---|---|
| Tamaño del modelo | 25.3 MB (CNN 2.8 MB + OPERA-CE 22.6 MB) |
| Consistencia con PyTorch | 100 % de grabaciones con la misma clasificación (dif. máx. 0.000003) |
| Latencia en laptop (Intel Core 5 210H) | 13.9 ms de media, 73 ms máximo |
| Efecto de limitar a 8 s de audio | 98.8 % de grabaciones con la misma clasificación |

La versión int8 (13.3 MB) cambia la clasificación en el 3.2 % de los casos y no es más rápida, por lo que se usa float32.

## De TensorFlow Lite a ONNX Runtime

La CNN se convirtió a TFLite sin problemas, pero la conversión de OPERA-CE (EfficientNet-B0) con `onnx2tf` falló repetidamente: la herramienta interpreta mal la disposición de los ejes de la red. El aplicativo usa **ONNX Runtime**, que también ejecuta el modelo dentro del teléfono y sin conexión, y tiene paquete para Flutter. Los `.onnx` se usan tal cual, ya verificados contra el modelo original.

## Estructura y orden de ejecución

```
scripts/
  01_analizar_copd_cough.py   Inventario y calidad de los audios
  02_preprocesar.py           Mono, 16 kHz, recorte de silencios, normalización de pico
  02b_grupos_desde_v1.py      Participante de cada audio (hash v1 ↔ v2) y fuga de la partición oficial
  03_entrenar_baseline.py     CNN con la partición oficial (solo para mostrar el efecto de la fuga)
  04_opera_embeddings.py      Embeddings OPERA-CT y OPERA-CE
  05_clasificador_opera.py    Sonda lineal sobre OPERA con la partición oficial (efecto de la fuga)
  06_validacion_cruzada.py    Validación cruzada 5×5 agrupada: metadatos, CNN, OPERA, fusión
  07_figuras_tablas.py        Figuras y tablas de la tesis
  08_modelos_desplegables.py  OPERA-CE y fusión CNN + OPERA-CE con las mismas particiones
  09_modelo_final.py          Modelo final y umbral de tamizaje
  10_exportar_onnx.py         Exportación a ONNX y verificación pieza por pieza
  11_cuantizar_onnx.py        Tamaño, latencia y consistencia (float32 e int8)
  audio_utils.py, metricas.py
data/features/                Embeddings OPERA (.npz)
data/processed/copd_cough/metadata.csv   Etiqueta y participante de cada grabación
models/cv/, models/baseline_cnn/          Pesos de las CNN
models/final/                 Modelo final, configuración, umbral y predicciones de referencia
models/final/export/          cnn_tos.onnx, especificacion_app.json, consistencia
reports/                      Resultados, figuras y tablas de la tesis
EPOC_CC_presentacion_FINAL.ipynb   Fase exploratoria en Google Colab (YAMNet, VGGish, rasgos clásicos)
```

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
pip install -r requirements.txt onnx onnxscript onnxruntime efficientnet_pytorch

python scripts/01_analizar_copd_cough.py
python scripts/02_preprocesar.py
python scripts/02b_grupos_desde_v1.py
python scripts/04_opera_embeddings.py
python scripts/06_validacion_cruzada.py
python scripts/07_figuras_tablas.py
python scripts/08_modelos_desplegables.py
python scripts/09_modelo_final.py
python scripts/10_exportar_onnx.py
python scripts/11_cuantizar_onnx.py
```

Los scripts 03 y 05 usan la partición oficial con fuga y solo se conservan para mostrar ese efecto; no deben reportarse como desempeño del modelo.

## OPERA

Los pesos de OPERA (`encoder-operaCT.ckpt`, `encoder-operaCE.ckpt`) **no se incluyen** y se descargan aparte con el repositorio oficial (https://github.com/evelyn0414/OPERA) en `third_party/OPERA`. Por la misma razón tampoco se incluye `opera_ce_tos.onnx`, que contiene esos pesos y se genera con `10_exportar_onnx.py`. **Revisar la licencia de OPERA antes de redistribuir sus pesos o cualquier modelo derivado.**

## Pendiente

- Aplicativo en Flutter (ONNX Runtime) según `models/final/export/especificacion_app.json`.
- Medición en el teléfono HONOR: latencia, tamaño instalado y consistencia con la PC.
- Corrección del early stopping (ver issue).
