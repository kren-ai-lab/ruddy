Inventario y estimación de los reemplazos para una Ruddy nativa de Polars
======================================================================

Estimación de ingeniería del 17 de septiembre de 2026, sobre `f332721`.
Complementa el [plan de migración](POLARS_MIGRATION_PLAN.md) y la
[revisión de Fable](POLARS_MIGRATION_REVIEW_FABLE.md). Los rangos de este documento
son estimaciones propias posteriores a esa revisión; Fable no los ha estimado
ni validado. No se han implementado o cronometrado estos reemplazos.

El escenario presupuestado conserva todos los métodos, usa Polars para tablas,
NumPy/SciPy/sklearn para cálculo y Patsy sobre arrays para diseños. El paquete
final no instala pandas, statsmodels ni Formulaic. El prototipo previo de Patsy
sin pandas solo verificó un diseño y una grilla básicos.

La estimación es de **1.845–3.355 LOC de implementación de reemplazos
estadísticos**, más **400–720 LOC de soporte de datos**, y **3.140–5.600 LOC de
pruebas específicas para ambos**. El motor mixto representa aproximadamente la
mitad del esfuerzo de reemplazo estadístico. Adaptar los módulos existentes a
Polars es trabajo adicional, aunque gran parte reutilice sus algoritmos.

**Cómo interpretar LOC y esfuerzo**

- LOC significa líneas físicas de código con tokens significativos, excluyendo
  comentarios, líneas vacías y docstrings. Incluye tipos, validaciones y
  construcción de resultados. Con documentación y formato, los archivos serán
  más largos. No se cuentan los datos de fixtures JSON/NPZ como código.
- Las LOC presupuestadas representan implementación que Ruddy deberá mantener,
  escrita o adaptada de fuentes compatibles. No son el crecimiento neto del
  repositorio: reemplazan parte del código actual y conviven con código reutilizado.
- Un día equivale a una jornada efectiva de una persona con experiencia en
  Python científico y los métodos correspondientes. Las filas incluyen
  implementación, pruebas específicas y revisión numérica del bloque. No son
  estimaciones de tiempo de generación de un agente ni compromisos de calendario.
- Los costes compartidos se cuentan una vez: OLS alimenta ANOVA, VIF,
  Breusch–Pagan y medias marginales; el diseño se reutiliza entre modelos.
  El arnés de referencias, integración global y adaptación tabular se presupuestan
  aparte. No sumar subtotales y desgloses del mismo bloque.
- Los límites inferiores requieren reutilización efectiva y paridad sin grandes
  discrepancias. Los superiores son rangos de planificación, no límites máximos
  garantizados. Especialmente en mixtos, los prototipos pueden obligar a revisarlos.

**Base medida para dimensionar**

El recuento local con AST/tokenización encontró 15.562 LOC en `ruddy/`, 5.328 en
`tests/` y 2.010 en `examples/`. La superficie de migración contiene 45 archivos
de librería con imports pandas, 52 de pruebas y 15 de ejemplos. Estos conteos no
son una estimación de líneas que necesariamente habrá que editar.

En las fuentes instaladas de statsmodels, las funciones tienen aproximadamente:

| Implementación de referencia | LOC medidas | Interpretación |
| --- | ---: | --- |
| `normal_ad` | 34 | Más el cálculo del estadístico AD y utilidades. |
| `jarque_bera` | 10 | Usa SciPy; el cálculo ya tiene primitivas reutilizables. |
| `het_breuschpagan` | 12 | La regresión auxiliar queda delegada en OLS. |
| `variance_inflation_factor` | 28 | También depende de OLS. |
| `anova2_lm_single` + `anova3_lm_single` | 78 | Dependen de diseño, ajuste, covarianzas y test F. |
| Cuatro funciones centrales de MANOVA | 170 | Ajuste, estadísticos y pruebas; falta integración y validación. |
| Módulo completo `mixed_linear_model.py` | 1.507 | Incluye funciones fuera de nuestro alcance, pero depende de infraestructura de otros módulos. |

Por eso una función de referencia de doce líneas no equivale a un reemplazo
independiente de doce líneas. Tampoco necesitamos portar todo statsmodels.

**Inventario de reemplazos estadísticos, conservando Patsy**

Los nombres propuestos identifican responsabilidades privadas; no son una
propuesta de ampliar la API pública. Las referencias a consumidores son relativas
a `ruddy/`. La columna de pruebas presupone aserciones de paridad y casos
patológicos, además de las pruebas ya existentes que se adaptarán.

| Funciones/responsabilidades propuestas | Consumidor actual / dependencia que se sustituye | LOC implementación | LOC pruebas | Días |
| --- | --- | ---: | ---: | ---: |
| `build_model_design`, `apply_model_design`, metadatos de términos/niveles | `factorial/{models,marginal_means,mixed_effects}.py`, `multivariate/manova.py`; preparar diccionarios/arrays para Patsy, conservar Sum/Treatment y nombres | 100–180 | 140–260 | 1,5–3 |
| `fit_ols`, `LinearFit` | `ols(...).fit()`, `OLS(...).fit()`; rango, QR/SVD, beta, residuos, SSR, df y covarianza clásica | 130–220 | 180–300 | 2–4 |
| `heteroskedastic_covariance` | Covarianzas HC0, HC1, HC2, HC3 usadas en ANOVA | 80–140 | 100–180 | 1–2 |
| `coefficient_inference` | `params`, `bse`, `tvalues`, `pvalues`, `conf_int` del ajuste | 45–80 | 70–120 | 0,5–1 |
| `linear_hypothesis_f` | Test F de contrastes, rango/estimabilidad, df y p-value | 45–85 | 80–140 | 1–2 |
| `anova_type_ii`, `anova_type_iii`, contrastes por término | `factorial/models.py`; `anova_lm`, incluida variante robusta y SS reconstruida | 120–210 | 180–300 | 2–4 |
| `ols_influence` | `factorial/diagnostics.py`; leverage, residuos studentizados internos y Cook | 45–85 | 80–140 | 1–2 |
| `fit_multivariate_ols`, `manova_hypothesis`, `manova_statistics` | `multivariate/manova.py`; H/E, autovalores, Wilks/Pillai/Hotelling–Lawley/Roy, F y df | 220–380 | 250–450 | 3–6 |
| Adaptar `_safe_model`, `_grid_l_vectors` | `factorial/marginal_means.py`; consumir diseño/OLS propios y conservar el cálculo de medias/contrastes existente | 40–90 | 80–140 | 0,5–1,5 |
| `normal_ad_statistic`, `normal_ad_pvalue` | `univariate/diagnostics.py`; `normal_ad` y su aproximación de p-value | 50–90 | 80–150 | 0,5–1,5 |
| `jarque_bera_diagnostic` | `factorial/diagnostics.py`; envolver SciPy y recuperar asimetría/curtosis para `details` | 10–25 | 30–60 | 0,25–0,5 |
| `breusch_pagan_koenker` | `factorial/diagnostics.py`; regresión auxiliar con OLS compartido, LM/F y p-values | 30–65 | 50–100 | 0,5–1 |
| `variance_inflation_factors` | `multivariate/collinearity.py`; OLS auxiliares y convenciones del VIF de referencia | 30–65 | 60–110 | 0,5–1 |
| `sample_skewness`, `sample_excess_kurtosis` | `univariate/numeric.py`; sustitución de `Series.skew/kurt` y tratamiento de bordes casi constantes | 35–70 | 60–110 | 0,5–1 |
| `spearman_matrix` | `multivariate/covariance.py`; rankings con empates y correlación, forma y constantes | 25–50 | 50–100 | 0,5–1 |
| **Subtotal sin mixtos** | Incluye el adaptador de diseño, no un codificador Patsy propio | **1.005–1.835** | **1.490–2.660** | **15,25–31,5** |
| **Motor mixto restringido** | Desglosado debajo | **840–1.520** | **1.150–2.040** | **15–30** |
| **Total reemplazos estadísticos** | | **1.845–3.355** | **2.640–4.700** | **30,25–61,5** |

Para ANOVA se conservan la jerarquía y los contrastes actuales. La opción robusta
actual afecta los tests ANOVA; la tabla de coeficientes usa todavía inferencia OLS
clásica. El coste incluye preservar esa distinción. MANOVA debe conservar Treatment
y su fila de intercepto. Las medias marginales ya calculan estimaciones,
intervalos y diferencias en NumPy/SciPy; su fila cuenta únicamente código de
integración que se escribe o sustituye, no toda su implementación actual.

**Desglose del motor mixto: la parte de mayor riesgo**

El alcance de `analyze_mixed_effects` es un agrupador, intercepto aleatorio y cero
o más pendientes numéricas con covarianza entre efectos. Incluye ML y REML,
inferencia de efectos fijos, efectos condicionales por grupo, varianzas, ICC,
log-verosimilitud, AIC/BIC y diagnósticos. No incorpora GLMM, efectos cruzados,
DSL nueva, regularización o nuevas familias estadísticas.

| Funciones/responsabilidades propuestas | LOC implementación | LOC pruebas | Días |
| --- | ---: | ---: | ---: |
| `pack_covariance`, `unpack_covariance`, parametrización por Cholesky y escala | 70–130 | 100–180 | 1–2 |
| `prepare_group_blocks`, `solve_group_covariance`, `group_logdet`, GLS de beta | 120–220 | 150–260 | 2–4 |
| `profile_loglikelihood`, variantes ML/REML y recuperación de escala | 90–160 | 150–250 | 2–4 |
| `mixed_score`, `mixed_hessian`, transformación de derivadas entre parametrizaciones | 220–400 | 250–450 | 4–8 |
| `fit_mixed_model`, inicialización, mapeo de optimizadores, tolerancias y convergencia | 120–220 | 120–240 | 2–4 |
| `mixed_fixed_inference`, recuperación de covarianza/escala, AIC/BIC/ICC | 100–180 | 180–300 | 2–4 |
| `conditional_random_effects` (BLUP) | 60–110 | 100–180 | 1–2 |
| `mixed_fit_diagnostics`, singularidad, fallos y contrato del ajuste | 60–100 | 100–180 | 1–2 |
| **Subtotal: no sumar de nuevo al total anterior** | **840–1.520** | **1.150–2.040** | **15–30** |

NumPy/SciPy aportan factorizaciones y optimizadores. El código propio define el
modelo, su objetivo, derivadas, inferencia y estados. SciPy permite suministrar
objetivo, gradiente y Hessiano a sus
[optimizadores](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html);
eso evita escribir un L-BFGS, pero no evita implementar el estimador mixto.
La [documentación de MixedLM](https://www.statsmodels.org/stable/mixed_linear.html)
describe la estimación ML/REML y sus cálculos de verosimilitud/derivadas.

El Hessiano no es un detalle opcional: interviene en los errores estándar y debe
evaluarse en la parametrización correcta. La cuenta presupone derivadas
analíticas o una implementación de equivalencia demostrada. Reemplazarlo por la
aproximación inversa que devuelve un optimizador podría acortar código y alterar
la inferencia; no se contabiliza como solución válida.

La API actual expone `optimizer` como cadena y solo exige que no esté vacía en
`analysis/config.py`. La estimación incluye caracterizar los nombres actualmente
utilizables y mapearlos explícitamente a SciPy, sin fallback automático. No basta
probar únicamente `lbfgs` y declarar equivalente todo el parámetro público.

**Soporte custom de datos, separado de los motores**

| Funciones/responsabilidades | LOC implementación | LOC pruebas | Días |
| --- | ---: | ---: | ---: |
| `validate_observation_ids`, `align_id_positions`, selección/preservación de IDs | 140–240 | 180–320 | 1,5–3 |
| Máscaras missing/finite, extracción numérica, constructores tipados y validación común de resultados | 160–300 | 200–360 | 1,5–3 |
| `LabeledMatrix`, validación de ejes, conversión/exportación explícita | 100–180 | 120–220 | 1–2 |
| **Subtotal** | **400–720** | **500–900** | **4–8** |

Estas cifras cubren las utilidades compartidas. Las declaraciones de schema por
tabla y la adaptación de sus consumidores se incluyen en el trabajo tabular,
no se cuentan nuevamente como motores estadísticos. También habrá código que se
elimine al centralizar patrones ya repetidos.

**Lo que ya existe y se reutiliza**

No se presupuestan implementaciones nuevas de PCA, t-SNE, UMAP, CCA,
Isolation Forest, LOF, covarianza empírica/robusta de sklearn, distancias de SciPy,
correlaciones y tests ya calculados con SciPy, bootstrap, FDR, tamaños de efecto,
PERMANOVA/PERMDISP, CKA, Mantel, transformaciones composicionales o cálculo
bayesiano actualmente en NumPy. Sí hay adaptación de sus entradas/salidas e IDs.

Tukey–Kramer/Games–Howell y Welch ANOVA ya están implementados en Ruddy;
statsmodels aparece como referencia en sus tests. El trabajo allí es conservar
las referencias y migrar pruebas/tablas, no crear otro motor. El parser restringido
de fórmulas también existe en `factorial/design.py`.

Retener Patsy evita escribir codificación de categorías, contrastes, interacciones,
orden de columnas, slices por término y reconstrucción de diseños para nuevas
filas. Si después se decide eliminar también Patsy, presupuestaría **350–650 LOC
de implementación, 450–900 LOC de pruebas y 5–10 días adicionales**, además del
adaptador/metadatos ya contado. Es un escenario opcional; no es necesario para
cumplir cero pandas y no está sumado al total base.

**Esfuerzo de la migración completa**

| Trabajo sin doble conteo | Días efectivos |
| --- | ---: |
| Reemplazos estadísticos, con sus pruebas específicas | 30–62 |
| Soporte compartido de datos, con sus pruebas específicas | 4–8 |
| Adaptar consumidores a Polars: contratos, análisis existentes, schemas, CLI, tests existentes, ejemplos y documentación | 12–22 |
| Generador de referencias, comparador global, instalación limpia, CI y verificación integrada de entrega | 5–8 |
| **Total de planificación redondeado** | **51–100** |

Para una persona dedicada son aproximadamente **10–20 semanas efectivas**. La
estimación incluye programación, validación científica y revisión; no incluye
esperas de revisión externa, nuevas funciones o investigación abierta si el
motor mixto no alcanza equivalencia. Herramientas de asistencia pueden reducir
trabajo repetitivo; no hay medición local para asignarles un factor de aceleración.

La adaptación tabular toca código existente: presupuestar miles de líneas
editadas tiene sentido, pero no hay una estimación fiable del diff antes de
probar una ruta completa. No convertir los 15.562 LOC actuales en "código nuevo".
El arnés/generador de referencias podría añadir otras 200–400 LOC de herramientas,
fuera del código de producción y de los tests específicos presupuestados.

La aproximación de tamaño de la nueva implementación mantenida queda así:

| Concepto | LOC |
| --- | ---: |
| Motores y soporte de producción | **2.245–4.075** |
| Tests específicos para esos componentes | **3.140–5.600** |
| Herramientas compartidas de referencia | **200–400** |
| Adaptación de código, tests y ejemplos existentes | **No es código enteramente nuevo; no sumado a las cifras anteriores** |

Los mayores factores de incertidumbre son la convergencia e inferencia del motor
mixto, la equivalencia de ANOVA robusta, las discrepancias casi constantes y las
semánticas de categorías/IDs. La evidencia actual es más fuerte para diagnósticos
y reutilización NumPy/SciPy que para mixtos; estos rangos no son intervalos de
confianza estadísticos.

Propongo dedicar **3–5 días del presupuesto de mixtos**, no adicionales, a un
prototipo que contraste ML y REML, intercepto y pendiente aleatoria, efectos fijos,
varianzas, log-verosimilitud y al menos una comprobación de errores estándar.
Antes de extrapolar debe producir un informe de diferencias y confirmar qué
algoritmos/derivadas habrá que adaptar. Si aparecen discrepancias importantes,
reestimar esa fila antes de comprometer el calendario de la migración completa.

Mantener el código significa también mantener sus decisiones numéricas y tests
de regresión. Adaptar funciones de fuentes compatibles puede reducir el trabajo
inicial, pero exige conservar licencia/atribución y no elimina esa responsabilidad.
No se ha validado todavía una biblioteca alternativa sin pandas que elimine todo
el coste de MixedLM; no he descontado ese ahorro hipotético.
