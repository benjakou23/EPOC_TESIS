# Tamizaje de EPOC por análisis acústico de la tos con Deep Learning

Código y resultados de la tesis *Sistema móvil inteligente basado en Deep Learning para el tamizaje de la Enfermedad Pulmonar Obstructiva Crónica (EPOC) mediante análisis acústico de la tos* (Universidad Señor de Sipán, Pimentel, 2026).

Autores: Anderson Benjamín Girón Villegas y Lucrecia Victoria Montes Pozo.

> El sistema es de **tamizaje**: identifica a las personas con riesgo que deben derivarse a espirometría. No es un diagnóstico.

## Datos

Se usa el conjunto **Cough-COPD (CC)**: 853 grabaciones de tos (221 EPOC, 632 sanos) de 289 participantes/sesiones. Licencia CC BY 4.0.

- Versión 1 (con carpetas por participante): https://doi.org/10.5281/zenodo.10209837
- Versión 2 (archivos renombrados, sin participantes): https://doi.org/10.5281/zenodo.14881818

Los audios **no** se incluyen en este repositorio. La partición oficial de la versión 2 mezcla participantes entre entrenamiento y prueba (162 de 172 audios de prueba), por lo que toda la evaluación se hace con **validación cruzada 5×5 agrupada por participante**.

## Resultados (validación cruzada 5×5 agrupada por participante)

| Modelo | AUC (media ± DE) | AUC por repetición [IC95] | Sensibilidad | Especificidad | H1 |
|---|---|---|---|---|---|
| Solo metadatos (referencia) | 0.649 ± 0.083 | 0.647 [0.575; 0.711] | 0.762 | 0.336 | No |
| CNN ligera desde cero | 0.658 ± 0.109 | 0.639 [0.599; 0.680] | 0.818 | 0.369 | No |
| OPERA CT+CE + regresión logística | 0.717 ± 0.081 | 0.703 [0.652; 0.751] | 0.764 | 0.480 | Sí |
| **Fusión CNN + OPERA** | **0.741 ± 0.074** | **0.725 [0.678; 0.771]** | 0.766 | 0.520 | **Sí** |

H1: AUC ≥ 0.70 y límite inferior del IC95 (bootstrap por participante) > 0.60. Umbral de decisión fijado en validación interna para una sensibilidad objetivo de 85 %.

## Estructura

```
scripts/                     Pipeline (ejecutar en orden)
  01_analizar_copd_cough.py  Inventario y calidad de los audios
  02_preprocesar.py          Mono, 16 kHz, recorte de silencios, normalización
  02b_grupos_desde_v1.py     Recupera el participante de cada audio (hash v1 ↔ v2)
  03_entrenar_baseline.py    CNN con la partición oficial (muestra el efecto de la fuga)
  04_opera_embeddings.py     Incrustaciones OPERA-CT / OPERA-CE
  05_clasificador_opera.py   Sonda lineal sobre OPERA (partición oficial)
  06_validacion_cruzada.py   Validación cruzada 5×5 agrupada: metadatos, CNN, OPERA, fusión
  07_figuras_tablas.py       Figuras y tablas de la tesis
  audio_utils.py, metricas.py
data/features/               Incrustaciones OPERA (.npz)
data/processed/copd_cough/metadata.csv   Etiqueta y participante de cada grabación
models/                      Pesos de las CNN (validación cruzada y línea base)
reports/                     Resultados, figuras y tablas
  validacion_cruzada/        Métricas por partición, predicciones fuera de partición, DeLong
  tesis/figuras, tablas, diagramas, figuras_corregidas
EPOC_CC_presentacion_FINAL.ipynb   Fase exploratoria en Google Colab (YAMNet, VGGish, rasgos clásicos)
```

## Reproducir

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 1. Descargar Zenodo v1 y v2 en data/raw/ (ver rutas en cada script)
# 2. Clonar OPERA y sus pesos en third_party/OPERA (https://github.com/evelyn0414/OPERA)
python scripts/01_analizar_copd_cough.py
python scripts/02_preprocesar.py
python scripts/02b_grupos_desde_v1.py
python scripts/04_opera_embeddings.py
python scripts/06_validacion_cruzada.py
python scripts/07_figuras_tablas.py
```

Los resultados de `03` y `05` usan la partición oficial con fuga de participantes y solo se conservan para mostrar ese efecto; no deben reportarse como desempeño del modelo.

## Pendiente

- Destilar OPERA en la CNN ligera para el teléfono.
- Exportar a TensorFlow Lite y desarrollar el aplicativo Android (solo cuando el modelo final esté validado).
