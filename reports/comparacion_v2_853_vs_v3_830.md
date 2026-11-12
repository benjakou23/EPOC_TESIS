# Comparación v2 (853) vs v3 (830)

- **v2:** 853 grabaciones (221 EPOC, 632 sanos); 23 EPOC de < 3 s se rellenan con ceros.
- **v3:** 830 grabaciones (198 EPOC, 632 sanos); se excluyen las grabaciones de < 3 s.
- Validación cruzada 5×5 agrupada por participante, mismas semillas. IC95 por bootstrap de participantes (B=1000).
- `*` = el IC95 de la diferencia excluye 0. Umbral honesto: fijado con las OOF de los otros folds de la misma repetición, sensibilidad objetivo 85 %.
- Piso de ruido = RF con los 100 cuadros de menor energía (128 rasgos, sin duración); sus comparaciones usan solo las grabaciones con ≥ 100 cuadros.

## v2 (853 grabaciones, con relleno de ceros)

Grabaciones: 853 (221 EPOC) | con piso de ruido: 852

| Modelo | AUC por fold (media ± DE) | AUC OOF [IC95] | Δ vs metadatos [IC95] | Δ vs piso de ruido [IC95] | Umbral honesto: sens / esp |
|---|---|---|---|---|---|
| CNN | 0.747 ± 0.050 | 0.731 [0.685; 0.776] | +0.084 [+0.010; +0.170] * | +0.076 [+0.012; +0.140] * | 0.837 / 0.426 |
| Fusión CNN+OPERA-CE | 0.753 ± 0.068 | 0.748 [0.696; 0.794] | +0.101 [+0.029; +0.183] * | +0.093 [+0.030; +0.154] * | 0.843 / 0.455 |
| OPERA-CE | 0.688 ± 0.088 | 0.684 [0.628; 0.736] | +0.037 [-0.036; +0.121] | +0.029 [-0.042; +0.099] | 0.849 / 0.337 |
| Metadatos | 0.649 ± 0.083 | 0.647 [0.575; 0.711] | — | -0.008 [-0.099; +0.082] | 0.842 / 0.227 |

Piso de ruido (RF, 100 cuadros): AUC OOF 0.653 [0.593; 0.715]

## v3 (830 grabaciones, sin relleno)

Grabaciones: 830 (198 EPOC) | con piso de ruido: 830

| Modelo | AUC por fold (media ± DE) | AUC OOF [IC95] | Δ vs metadatos [IC95] | Δ vs piso de ruido [IC95] | Umbral honesto: sens / esp |
|---|---|---|---|---|---|
| CNN | 0.708 ± 0.063 | 0.688 [0.633; 0.738] | +0.069 [-0.007; +0.153] | +0.078 [+0.003; +0.148] * | 0.825 / 0.371 |
| Fusión CNN+OPERA-CE | 0.733 ± 0.066 | 0.726 [0.666; 0.780] | +0.107 [+0.033; +0.190] * | +0.116 [+0.051; +0.176] * | 0.851 / 0.423 |
| OPERA-CE | 0.697 ± 0.070 | 0.694 [0.633; 0.749] | +0.075 [-0.005; +0.163] | +0.084 [+0.017; +0.146] * | 0.846 / 0.357 |
| Metadatos | 0.621 ± 0.073 | 0.619 [0.538; 0.690] | — | +0.009 [-0.091; +0.095] | 0.845 / 0.213 |

Piso de ruido (RF, 100 cuadros): AUC OOF 0.610 [0.549; 0.674]
