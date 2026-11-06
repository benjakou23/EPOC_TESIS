Eres mi asistente de redacción de tesis. Edita este documento Word (formato IEEE, en español) aplicando TODAS las indicaciones siguientes. Reglas generales:
- No borres contenido que no se mencione aquí; modifica solo lo indicado.
- Formato IEEE igual al que ya tiene el documento: tablas numeradas con romanos (TABLA IV, TABLA V, …) con el título en cursiva ENCIMA de la tabla; figuras numeradas (Fig. 2, Fig. 3, …) con la leyenda DEBAJO; referencias numéricas entre corchetes, las nuevas desde [37].
- Donde diga «[INSERTAR IMAGEN: archivo.png]», deja un párrafo vacío centrado con ese texto resaltado en amarillo y la leyenda debajo; yo pego la imagen después (las imágenes están en reports/tesis/figuras/).
- Cada tabla y figura nueva debe citarse en el texto antes de aparecer («como se muestra en la TABLA V…»).
- Usa decimales con punto como en el resto del documento y «±» para la desviación estándar.
- Mantén el tiempo verbal del proyecto (futuro: «se empleará», «se evaluará») en Método; usa pasado solo en la sección de resultados preliminares o avance si existe.

==================================================================
1. TÍTULO Y ALCANCE
==================================================================
1.1 Cambia el título a: «Sistema móvil inteligente basado en Deep Learning para el tamizaje de la Enfermedad Pulmonar Obstructiva Crónica (EPOC) mediante análisis acústico de la tos» (ajusta mayúsculas al estilo de la portada).
1.2 En resumen, introducción, objetivos y conclusiones: el sistema es de TAMIZAJE (identifica personas con riesgo que deben ser derivadas a espirometría); NO es diagnóstico. Reemplaza «detección automática de enfermedades respiratorias» por «tamizaje de EPOC» cuando se refiera a este trabajo.
1.3 Funciones de la app: lo central es grabar la tos, procesarla y mostrar «riesgo bajo / riesgo alto, se recomienda espirometría». Lo multilingüe, la guía por voz y la navegación pásalos a «funciones secundarias / trabajo futuro» en una sola frase.
1.4 En la introducción, aclara en una frase que la mayoría de los antecedentes citados usan estetoscopio electrónico (sonidos pulmonares), mientras que este trabajo usa tos grabada con teléfono inteligente, que es más accesible para el tamizaje.

==================================================================
2. DATASET (Método – Materiales/Datos)
==================================================================
2.1 El dataset es Cough-COPD (CC), licencia CC BY 4.0. Cita la versión 1 de Zenodo (DOI 10.5281/zenodo.10209837) como fuente principal [37], porque conserva las carpetas por participante/sesión. Agrega este párrafo:
«Se empleará la versión 1 del conjunto de datos Cough-COPD (CC) [37], que contiene 853 grabaciones de tos (221 de pacientes con EPOC y 632 de personas sanas) organizadas en 289 carpetas de participante/sesión (78 EPOC y 211 sanos). La versión 2 publicada posteriormente renombra los archivos y elimina la información de participante; además, su partición oficial entrenamiento/prueba mezcla participantes: 162 de los 172 audios de prueba (94.2 %) pertenecen a participantes que también están en entrenamiento. Por ello no se utilizará esa partición y toda la evaluación se realizará agrupando por participante. Las 853 grabaciones de la versión 2 se emparejaron con la versión 1 mediante el hash de sus muestras de audio (853/853 coincidencias, sin etiquetas incoherentes).»
2.2 Verifica la afirmación de que las grabaciones se hicieron con teléfono inteligente: debe citarse al artículo original del dataset (Sun et al. [20]). Si el documento afirma algo que [20] no dice, suavízalo («grabaciones en condiciones no clínicas»).
2.3 Inserta la TABLA IV, «Características de las grabaciones por clase»:
| Clase | n | Duración media (s) | DE (s) | Mín (s) | Máx (s) | RMS medio | Saturadas |
| Sano | 632 | 7.24 | 1.75 | 3.82 | 29.92 | 0.056 | 3 |
| EPOC | 221 | 7.88 | 4.13 | 0.86 | 36.38 | 0.064 | 9 |
Nota bajo la tabla: «Todas las grabaciones originales: WAV PCM 16 bits, 48 kHz, estéreo; ninguna corrupta.»
2.4 Inserta Fig. 2: [INSERTAR IMAGEN: fig_distribucion_clases.png] Leyenda: «Fig. 2. Distribución de grabaciones y participantes por clase en el conjunto Cough-COPD (CC).»
2.5 Inserta Fig. 3: [INSERTAR IMAGEN: fig_duraciones.png] Leyenda: «Fig. 3. Distribución de la duración de las grabaciones por clase.» Texto que la cite: las diferencias de duración y de nivel entre clases pueden ser aprovechadas por un modelo como atajo, lo que justifica el control con metadatos (sección 4).
2.6 Inserta Fig. 4: [INSERTAR IMAGEN: fig_espectrogramas.png] Leyenda: «Fig. 4. Ejemplos de espectrogramas log-Mel de tos de una persona sana y de un paciente con EPOC.»
2.7 Inserta la TABLA V, «Fuga de participantes en la partición oficial de la versión 2»:
| Partición | Audios de prueba | Audios de prueba con participante en entrenamiento | % |
| Oficial Zenodo v2 | 172 | 162 | 94.2 |

==================================================================
3. PREPROCESAMIENTO Y MODELOS (Método)
==================================================================
3.1 Preprocesamiento: conversión a mono, remuestreo a 16 kHz, recorte de silencios inicial/final (umbral 40 dB, conservando al menos 0.5 s), normalización de pico a 0.95, verificación de duplicados por hash (0 duplicados). El aumento de datos se aplicará SOLO en entrenamiento (nunca en validación ni prueba). No se usarán audios sintéticos para evaluar.
3.2 Si el documento menciona «1000 segmentos» de la exploración inicial, muévelo a un anexo como prueba exploratoria y aclara que la evaluación final es por grabación.
3.3 Herramientas: actualiza a «Python 3.11, PyTorch 2 (CPU), scikit-learn y librosa» para el modelo principal; Google Colab con TensorFlow/YAMNet/VGGish queda como fase exploratoria de comparación. Equipo: laptop ASUS TUF Gaming F16 FX607VJ, Intel Core 5 210H, 16 GB RAM, Windows 11 Home. Cita PyTorch [42] y scikit-learn [41].
3.4 Se compararán cuatro modelos (agrega este párrafo):
a) Referencia de metadatos: Random Forest (300 árboles) entrenado solo con duración, RMS y pico de la grabación; NO escucha la tos. Sirve para medir cuánto puede «adivinarse» por las condiciones de grabación.
b) CNN ligera desde cero: convoluciones separables en profundidad estilo MobileNet [43], log-Mel dentro del modelo, SpecAugment [44].
c) OPERA (transferencia): incrustaciones congeladas de los modelos fundacionales respiratorios OPERA-CT (768 dimensiones) y OPERA-CE (1280 dimensiones) [38] concatenadas y clasificadas con regresión logística (C elegido por validación interna agrupada).
d) Fusión: promedio de las probabilidades de la CNN y de OPERA.
3.5 Inserta la TABLA VI, «Configuración de la CNN ligera»:
| Elemento | Configuración |
| Entrada | Forma de onda mono 16 kHz, ventanas de 3 s (salto 1.5 s) |
| Representación | Log-Mel 64 bandas, 50–8000 Hz, ventana 25 ms, salto 10 ms, normalización por instancia |
| Arquitectura | CNN con convoluciones separables en profundidad (estilo MobileNet), 7 bloques, pooling medio + máximo |
| Parámetros | 101 121 |
| Tamaño (float32 / int8 estimado) | 0.40 MB / 0.10 MB |
| Aumento de datos | Velocidad ±10 %, ganancia ±6 dB, ruido SNR 10–40 dB, paso bajo, SpecAugment |
| Desbalance | Pérdida BCE con pos_weight = sanos/EPOC |
| Optimización | AdamW (lr 2e-3, weight decay 1e-3), OneCycle, lote 32, máx. 60 épocas, paciencia 15 (AUC de validación) |
| Predicción por grabación | Promedio de las probabilidades de las ventanas |
| Latencia en CPU (3 s de audio) | 12.3 ms (Intel Core 5 210H) |

==================================================================
4. VALIDACIÓN Y MÉTRICAS (Método)
==================================================================
4.1 Redacta así el diseño de validación:
«Se empleará validación cruzada estratificada y agrupada por participante (StratifiedGroupKFold) de cinco particiones y cinco repeticiones (25 particiones), de modo que un mismo participante nunca aparezca simultáneamente en entrenamiento y prueba. Dentro de cada partición de entrenamiento se separará, también por participante, una validación interna (~15 %) para la parada temprana y para fijar el umbral de tamizaje con una sensibilidad objetivo del 85 %; ese umbral se aplicará sin cambios a la partición de prueba. Se reportarán AUC-ROC, sensibilidad, especificidad, VPP, VPN, F1, exactitud y exactitud balanceada como media ± DE sobre las 25 particiones; además, la AUC fuera de partición por repetición con intervalo de confianza del 95 % mediante bootstrap por conglomerados (remuestreo de participantes, 1000 réplicas). Las AUC de los modelos se compararán con la prueba de DeLong [39], [40] sobre las predicciones fuera de partición de cada repetición.»
4.2 Agrega los controles de atajos: (i) modelo solo con metadatos (ya descrito) y (ii) control de fondo sin tos realizado en la fase exploratoria (Colab): un clasificador entrenado con el ruido de fondo sin tos obtuvo AUC 0.554, cercano al azar.
4.3 Agrega «exactitud» y «F1» a la lista de métricas si no están.
4.4 Hipótesis H1: mantenla como está (AUC ≥ 0.70 con límite inferior del IC95 > 0.60, validación agrupada 5×5).

==================================================================
5. RESULTADOS (crea la sección «Resultados preliminares» o actualiza la existente)
==================================================================
5.1 Inserta la TABLA VII, «Desempeño de los modelos en validación cruzada agrupada por participante (5×5; media ± DE)»:
| Modelo | AUC | Sensibilidad | Especificidad | VPP | VPN | F1 | Exactitud | Exact. balanceada | AUC OOF | IC95 (bootstrap por participante) | H1 |
| Solo metadatos (RF) | 0.649 ± 0.083 | 0.762 ± 0.139 | 0.336 ± 0.150 | 0.289 ± 0.035 | 0.805 ± 0.072 | 0.416 ± 0.044 | 0.447 ± 0.087 | 0.549 ± 0.052 | 0.647 | [0.575; 0.711] | No |
| CNN ligera (desde cero) | 0.658 ± 0.109 | 0.818 ± 0.088 | 0.369 ± 0.158 | 0.319 ± 0.048 | 0.840 ± 0.074 | 0.456 ± 0.053 | 0.485 ± 0.109 | 0.593 ± 0.072 | 0.639 | [0.599; 0.680] | No |
| OPERA CT+CE + regresión logística | 0.717 ± 0.081 | 0.764 ± 0.206 | 0.480 ± 0.170 | 0.343 ± 0.040 | 0.875 ± 0.067 | 0.464 ± 0.062 | 0.553 ± 0.084 | 0.622 ± 0.052 | 0.703 | [0.652; 0.751] | Sí |
| Fusión CNN + OPERA | 0.741 ± 0.074 | 0.766 ± 0.188 | 0.520 ± 0.152 | 0.364 ± 0.044 | 0.880 ± 0.065 | 0.484 ± 0.057 | 0.583 ± 0.078 | 0.643 ± 0.054 | 0.725 | [0.678; 0.771] | Sí |
Nota bajo la tabla: «AUC OOF: AUC de las predicciones fuera de partición de cada repetición, promediada en las cinco repeticiones. Umbral fijado en validación interna para sensibilidad objetivo de 85 %.»
Si la tabla no cabe a una columna, ponla a doble columna o en orientación de página completa; si aún no cabe, divide en dos tablas (VII-a: AUC, sensibilidad, especificidad, VPP, VPN; VII-b: F1, exactitudes, AUC OOF, IC95, H1).
5.2 Inserta Fig. 5: [INSERTAR IMAGEN: fig_roc_modelos.png] Leyenda: «Fig. 5. Curvas ROC de los cuatro modelos con las predicciones fuera de partición de la validación cruzada agrupada 5×5.»
5.3 Inserta Fig. 6: [INSERTAR IMAGEN: fig_boxplot_auc.png] Leyenda: «Fig. 6. Distribución de la AUC en las 25 particiones; la línea roja marca el umbral de la hipótesis (0.70).»
5.4 Inserta la TABLA VIII, «Comparación de AUC mediante la prueba de DeLong (5 repeticiones)»:
| Comparación | AUC media A | AUC media B | Repeticiones con p < 0.05 | Interpretación |
| Fusión vs CNN | 0.725 | 0.639 | 5 de 5 | La fusión supera a la CNN de forma consistente |
| Fusión vs OPERA | 0.725 | 0.703 | 3 de 5 | Mejora pequeña y no siempre significativa |
| OPERA vs CNN | 0.703 | 0.639 | 3 de 5 (OPERA mayor en las 5) | OPERA supera a la CNN |
| CNN vs Metadatos | 0.639 | 0.647 | 2 de 5 (en sentidos opuestos) | Sin diferencia consistente: la CNN no supera al atajo de grabación |
5.5 Inserta Fig. 7: [INSERTAR IMAGEN: fig_matriz_confusion.png] Leyenda: «Fig. 7. Matriz de confusión del modelo de fusión (suma de las cinco particiones, promedio de las cinco repeticiones).» Texto: de 220 grabaciones con EPOC detecta 169 y pierde 51; de 631 sanas deriva 303 (falsos positivos).
5.6 Inserta Fig. 8: [INSERTAR IMAGEN: fig_vpp_vpn_prevalencia.png] Leyenda: «Fig. 8. Valores predictivos del modelo de fusión según la prevalencia de EPOC.»
5.7 Inserta la TABLA IX, «Valores predictivos esperados del tamizaje según el escenario (fusión; sensibilidad 76.6 %, especificidad 52.0 %)»:
| Escenario | Prevalencia | VPP | VPN | Derivados a espirometría |
| Población general > 40 años (Perú) | 6 % | 9.2 % | 97.2 % | 50 % |
| Fumadores / exposición a humo de leña (aprox.) | 12 % | 17.9 % | 94.2 % | 51 % |
| Dataset CC | 26 % | 35.9 % | 86.3 % | 55 % |
Texto: el VPN alto (≈97 % en población general) respalda el uso como tamizaje para descartar; el VPP bajo confirma que el resultado positivo NO es diagnóstico y requiere espirometría.
5.8 Inserta Fig. 9: [INSERTAR IMAGEN: fig_calibracion.png] Leyenda: «Fig. 9. Curvas de calibración de los modelos (probabilidad predicha vs proporción real de EPOC).»
5.9 Inserta Fig. 10: [INSERTAR IMAGEN: fig_efecto_fuga.png] Leyenda: «Fig. 10. AUC de la CNN con la partición oficial con fuga de participantes (0.878) frente a la validación agrupada por participante (0.658).» Texto: este resultado demuestra que la partición oficial sobreestima el desempeño, coherente con el sesgo descrito por Coppock et al. [16].
5.10 Inserta la TABLA X, «Resultados de la fase exploratoria en Google Colab (validación agrupada por participante)»:
| Experimento | AUC | Observación |
| Solo metadatos (RF) | 0.670 | Atajo de grabación |
| YAMNet, audio completo | 0.72 | Transferencia de audio general |
| RF con rasgos clásicos de tos limpia | 0.670 | Tras detección de tos y reducción de ruido |
| Control: solo fondo sin tos | 0.554 | Cercano al azar |
| Tamizaje (umbral para sens. ≈ 85 %) | — | Sensibilidad 86.6 %, especificidad 33.0 % |
Texto: los resultados de Colab y los locales coinciden: los modelos entrenados desde cero rinden al nivel del atajo de metadatos (~0.65–0.67), mientras que la transferencia desde modelos preentrenados (YAMNet 0.72; OPERA 0.70) y la fusión (0.73–0.74) son los únicos que superan 0.70.

==================================================================
6. CONTRASTE DE HIPÓTESIS Y DISCUSIÓN
==================================================================
6.1 Redacta el contraste de H1:
«La hipótesis H1 se cumple para el modelo de fusión CNN + OPERA (AUC 0.741 ± 0.074 por partición; AUC fuera de partición 0.725, IC95 0.678–0.771, cuyo límite inferior supera 0.60) y para OPERA solo (0.717 ± 0.081; IC95 0.652–0.751). No se cumple para la CNN entrenada desde cero (0.658 ± 0.109; IC95 0.599–0.680), cuyo desempeño no difiere de forma consistente del modelo que solo usa metadatos de la grabación.»
6.2 Discusión (agrega estos puntos):
- Con solo 289 participantes, una CNN desde cero tiende a aprender condiciones de grabación; el preentrenamiento en sonidos respiratorios (OPERA) aporta información de la tos que generaliza a participantes nuevos.
- La sensibilidad lograda en prueba (76.6 %) quedó por debajo del objetivo de 85 % fijado en validación, por la variabilidad entre participaciones (DE 0.19). Se propone calibrar el umbral con más datos y combinar el audio con un cuestionario de factores de riesgo (tabaquismo, humo de leña, edad, disnea) como tamizaje híbrido.
- La especificidad (52 %) implica que alrededor de la mitad de las personas sanas serían derivadas; es aceptable para tamizaje solo si el costo de la espirometría lo permite, lo que se discute con la TABLA IX.
- Limitaciones: un solo dataset, sin validación externa, sin datos de población peruana, etiquetas solo EPOC/sano (sin gravedad GOLD ni otras enfermedades).
- La exportación a TensorFlow Lite y la integración en la app se harán con el modelo final validado; OPERA requiere un codificador más grande que la CNN, por lo que se evaluará destilarlo a la CNN ligera para el móvil.

==================================================================
7. REFERENCIAS NUEVAS (agrégalas al final, en formato IEEE, continuando desde [37])
==================================================================
[37] Cough-COPD (CC) dataset, versión 1, Zenodo, 2023. DOI: 10.5281/zenodo.10209837. (Completa los autores y el año exactos según la página del registro en Zenodo.)
[38] Y. Zhang, T. Xia, J. Han, Y. Wu, G. Rizos, Y. Liu, M. Mosuily, J. Chauhan y C. Mascolo, “Towards open respiratory acoustic foundation models: Pretraining and benchmarking,” en Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2024.
[39] E. R. DeLong, D. M. DeLong y D. L. Clarke-Pearson, “Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach,” Biometrics, vol. 44, no. 3, pp. 837–845, 1988.
[40] X. Sun y W. Xu, “Fast implementation of DeLong’s algorithm for comparing the areas under correlated receiver operating characteristic curves,” IEEE Signal Processing Letters, vol. 21, no. 11, pp. 1389–1393, 2014.
[41] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
[42] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” en Advances in Neural Information Processing Systems, vol. 32, 2019.
[43] A. G. Howard et al., “MobileNets: Efficient convolutional neural networks for mobile vision applications,” arXiv:1704.04861, 2017.
[44] D. S. Park et al., “SpecAugment: A simple data augmentation method for automatic speech recognition,” en Proc. Interspeech, 2019, pp. 2613–2617.
Si alguna de estas obras ya está en la lista, no la dupliques: usa su número existente y renumera las siguientes para que sean consecutivas en orden de primera aparición.

==================================================================
8. ANEXOS Y REVISIÓN FINAL
==================================================================
8.1 Anexo «Ficha de registro de datos»: tabla con columnas ID de grabación, participante/sesión, clase, duración (s), frecuencia de muestreo, canales, saturación (sí/no), partición (repetición/fold).
8.2 Al terminar: revisa que las figuras (Fig. 2 a Fig. 10) y las tablas (TABLA IV a TABLA X) estén numeradas en orden de aparición y todas citadas en el texto; que no queden valores antiguos (AUC 0.88, 0.83 u 0.888 presentados como resultado válido; solo pueden aparecer en la Fig. 10 como ejemplo de fuga); y que no queden marcadores [Zhou], [Sun], [licencia] o similares sin resolver (Zhou = [22], Sun = [20], licencia = CC BY 4.0).
8.3 Al final, dame una lista breve de los cambios hechos y de lo que no pudiste aplicar.
