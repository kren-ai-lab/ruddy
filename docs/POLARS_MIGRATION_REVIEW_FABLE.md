Revisión externa de Claude Fable
===============================

Fecha: 17 de septiembre de 2026. Solicitada explícitamente por el usuario y
realizada mediante Claude Code con el modelo `fable`; la respuesta identifica
`claude-fable-5-1`. Acceso de lectura al repositorio, sin permisos de edición.

A continuación se conserva íntegra la respuesta del revisor. Leyó una versión
intermedia del plan mientras se incorporaba el requisito de cero pandas; sus
referencias a fases o líneas del documento pueden corresponder a esa versión.
Sus propuestas de tolerancias son recomendaciones pendientes de validación.
Las observaciones aceptadas, matices y discrepancias están en el
[plan consolidado](POLARS_MIGRATION_PLAN.md).

---

He terminado la revisión. Toda la evidencia proviene de lectura estática del repo, del lockfile y del código fuente de statsmodels/patsy/pandas instalado en `.venv`. No ejecuté la suite ni prototipos, así que distingo abajo lo comprobado en código de lo que exige prototipo.

## 1. Diagnóstico independiente y hallazgos priorizados

**Conclusión principal.** El requisito "cero pandas directo ni transitivo" obliga a retirar tres paquetes del grafo de ejecución, no uno. Está comprobado en `uv.lock`:

| Paquete | Evidencia | Consecuencia |
| --- | --- | --- |
| pandas | dependencia directa en `pyproject.toml:45` | se retira |
| statsmodels 0.15.0 | exige pandas, patsy y formulaic en `uv.lock:1851-1863` | se retira completo |
| formulaic 1.2.2 | exige pandas en `uv.lock:438-450` | no es alternativa a patsy |
| patsy 1.0.3 | solo numpy y packaging en `uv.lock:1168-1175` | candidato, ver abajo |
| scikit-learn 1.9.1 | joblib, narwhals, numpy, scipy, threadpoolctl en `uv.lock:1643-1654` | se conserva |
| typer, rich, umap-learn | sin pandas en el lock | se conservan |

El plan actual ya reconoce esto en su cabecera, pero su secuencia de fases sigue escrita para el adaptador. Los hallazgos, ordenados por riesgo:

- **P0, modelos mixtos.** `ruddy/factorial/mixed_effects.py:227-230` delega en MixedLM. Lo que habría que portar está repartido en un módulo de unas 3000 líneas: verosimilitud perfilada ML/REML con solver de Woodbury en `mixed_linear_model.py:1588-1696` y `472-560`, GLS de efectos fijos en `1381-1459`, gradiente analítico, Hessiano analítico en `2017+` del que salen los errores estándar, cadena de ajuste con reparametrización raíz cuadrada en `2249-2486`, BLUP condicionales en `2729-2775`, y AIC/BIC que devuelven NaN bajo REML en `2995-3014`. La paridad numérica exacta de un estimador iterativo no está garantizada aunque se replique el algoritmo. Es el único reemplazo cuya viabilidad no puedo afirmar desde el código.
- **P0, pruebas actuales de mixtos son débiles.** `tests/factorial/test_mixed_effects.py` solo comprueba convergencia, una varianza mayor a un umbral y el ICC en rango; tres de sus casos usan `reml=False` y la ruta por defecto REML no tiene ninguna aserción numérica. Hoy no existe pin que detecte una regresión del motor.
- **P0, oráculos vivos en la suite.** Cinco archivos importan statsmodels como referencia: `tests/factorial/test_models.py:6-7`, `tests/multivariate/test_manova.py:6`, `tests/multivariate/test_collinearity.py:5`, `tests/bivariate/test_posthoc.py:3` y `tests/bivariate/test_comparisons.py:5`. Además 52 archivos de tests importan pandas. La suite nativa debe correr sin ellos, así que estos oráculos tienen que convertirse en referencias congeladas.
- **P1, ANOVA II/III y robusta.** `ruddy/factorial/models.py:423,480` usa `ols` y `anova_lm`. La matemática está verificada en fuente: Tipo II construye el complemento ortogonal con QR de L1·Cov·L2ᵀ en `anova.py:222-291`; Tipo III usa las filas del término en `294-346`; ambas recuperan `sum_sq` como F·df·SSR/df_resid; HC0 a HC3 están en `linear_model.py:2213-2259`; el test F usa pseudoinversa de R·Cov·Rᵀ en `base/model.py:2098-2107`. Es reimplementable con NumPy/SciPy y verificable a `rel=1e-10`, la tolerancia de los tests actuales.
- **P1, codificación y nombres de parámetros son salida pública.** `models.py:183-194` traduce solo el prefijo `C(F0, Sum)` y deja el sufijo patsy `[S.nivel]`, así que la columna `parameter` expone nombres como `factor_a[S.A0]:factor_b[S.B1]`. El contraste Sum omite el último nivel y los niveles se ordenan con `SortAnythingKey` salvo Categorical explícito, según `patsy/contrasts.py:395-425` y `patsy/categorical.py:182-204`. El motor nuevo debe reproducir orden de niveles, nivel omitido y nombres, o documentar el cambio con test.
- **P1, mixtos exponen nombres seguros sin traducir.** `mixed_effects.py:255` publica `parameter=str(name)` y `test_mixed_effects.py:106` consulta la fila `X0`. Es una fuga del nombre interno que hoy es contrato de facto. Decidir si se conserva o se corrige con pin.
- **P1, MANOVA.** `ruddy/multivariate/manova.py:246-249,296-297` usa codificación Treatment `C(F0)` y reporta la fila `Intercept`. El test de un factor es invariante a la codificación, pero el `Intercept` no lo es. La matemática está en `multivariate_ols.py:65-133` para el ajuste SVD, `136-259` para los cuatro estadísticos con sus aproximaciones F y `385` para los autovalores de solve(E+H, H). Reimplementable; hay que conservar Treatment para la fila Intercept.
- **P1, medias marginales.** `ruddy/factorial/marginal_means.py:97-99,149` depende de `dmatrices` con `return_type="dataframe"` y de `build_design_matrices` para la grilla. Requiere el mismo codificador que ANOVA, aplicado a filas nuevas con niveles ya fijados.
- **P2, diagnósticos de statsmodels.** Todos son fórmulas cortas verificadas en fuente: `normal_ad` con la corrección de tamaño y las cuatro aproximaciones de p en `_adnorm.py:104-119`; `jarque_bera` con asimetría y curtosis sesgadas de SciPy en `stattools.py:127-139`; `het_breuschpagan` variante Koenker, LM = n·R² y chi² con `nvars-1` en `diagnostic.py:1211-1222`; VIF con estandarización de columnas no constantes, R² auxiliar y recorte a `1-1e-15` en `outliers_influence.py:208-239`; leverage, residuo studentizado interno y Cook en `857-860`, `930-961` y `978-981`. Nota: VIF en 0.15 añade un warning por condición mayor a 1e4 que `collinearity.py:109-110` suprime.
- **P2, cálculos pandas propiamente dichos.** Solo dos: asimetría y curtosis en `ruddy/univariate/numeric.py:190-192`, y Spearman en `ruddy/multivariate/covariance.py:158`. pandas usa los estimadores insesgados G1/G2 y anula momentos menores a 1e-14, según `pandas/core/nanops.py:1264-1272` y `1357-1362`. SciPy con `bias=False` da la misma fórmula; el comportamiento casi constante debe fijarse con pruebas.
- **P2, identidad.** `ruddy/data/validation.py:19-31` valida con `pd.Index`; `dataset.py:58` cae al índice; `feature_matrix.py:42` usa RangeIndex; `annotations.py:168` y `analysis/engine.py:75` construyen frames sin columnas con solo índice; `multivariate/permutation.py:80` asigna el índice para alinear. `ObservationID` es `Any` en `core/types.py:14`.
- **P2, matrices etiquetadas y exportación.** `covariance.py:36-43` y `compositional/analysis.py:112,118` usan IDs como ejes; los escritores `cli/commands/multivariate.py:33-44` y `cli/commands/specialized.py:58-59` exportan con `index=True`.
- **P3, contrato de resultados.** `ruddy/results/schemas.py:16-50` exige pandas y decide estado/razón con `isna`/`notna`. En Polars, `reason` debe ser `null`, no cadena vacía ni NaN.
- **P3, patsy no está mantenido.** Su METADATA declara desde 2021 que no tiene desarrollo activo y recomienda formulaic. Su uso sin pandas está guardado por `have_pandas` en `patsy/util.py:41-45`, `highlevel.py:151` y `build.py:931,943`, y el sniffer de niveles trabaja con iterables planos en `categorical.py:205-229`. Es viable en principio, no está probado en este repo.
- **P3, CI y documentación.** `.github/workflows/tests.yml:32` instala con `--all-extras` y no prueba una instalación limpia. README, `docs/METHOD_INVENTORY.md:9-10`, `docs/METHODS.md:130`, `docs/DATA_CONTRACTS.md:22,62,80-86` y `docs/RESULTS_AND_PROVENANCE.md:456` nombran pandas o statsmodels como contrato.

Un dato favorable: la gramática de fórmulas ya es nativa. `ruddy/factorial/design.py:90-164` parsea nombres, `+`, `:` y `*` sin patsy y rechaza transformaciones. Lo único que patsy aporta hoy es la codificación de diseño.

## 2. Arquitectura que cumple cero pandas en ejecución

**Dos pistas desacopladas.** Los modelos ya convierten a arrays antes de llamar a statsmodels, así que retirar statsmodels no depende de Polars. Recomiendo separar el trabajo en una pista de motores numéricos sobre NumPy/SciPy y otra pista de contratos tabulares Polars. La primera se puede validar mientras pandas sigue instalado en desarrollo, lo que reduce el riesgo del cambio grande.

**Módulos privados propuestos:**

- `ruddy/_identity`: IDs como tupla inmutable más mapa ID a posición. Validación explícita de `None`, NaN y NaT como faltantes, duplicados por igualdad Python. Alineación devuelve posiciones, nunca reindexa por orden.
- `ruddy/_tabular`: utilidades Polars. Máscaras de faltante que cubren `null` y NaN en flotantes, máscaras de finito, constructores de tablas vacías desde esquemas declarados, inferencia de `ColumnKind` desde dtypes Polars incluyendo Categorical, Enum, Null y Object.
- `ruddy/_labeled`: `LabeledMatrix` con `values`, `row_ids`, `column_ids` y exportadores explícitos.
- `ruddy/_design`: codificador de diseño a partir de `FactorialDesign`. Niveles observados ordenados con la regla actual, contrastes Sum y Treatment, columnas por término, nombres de columna compatibles con los actuales, y aplicación a filas nuevas para la grilla de medias marginales.
- `ruddy/_linear`: OLS con verificación de rango antes de resolver, covarianza clásica y HC0 a HC3, test F para matrices L, ANOVA II con complemento ortogonal y ANOVA III, influencia, MANOVA con los cuatro estadísticos, y el motor de modelo mixto con un agrupador, intercepto aleatorio y pendientes numéricas.
- `ruddy/_diagnostics`: `normal_ad`, Jarque–Bera, Breusch–Pagan Koenker y VIF.

**Estrategia patsy.** Dos opciones, decisión por prototipo en la fase 1. Opción A: declarar patsy y usarla con diccionarios de arrays y `return_type="matrix"`, con niveles fijados vía `C(x, Sum, levels=...)`. Opción B: codificador propio en `_design`. Recomiendo B como destino, porque el codificador debe compartirse entre ANOVA, medias marginales y MANOVA, porque la gramática ya es propia, y porque patsy está en mantenimiento. A sirve como red de seguridad si B no alcanza paridad exacta de matrices en el plazo. En ambos casos el gate es idéntico: igualdad exacta de matriz de diseño, orden de columnas y nombres frente a la referencia congelada.

**Identidad, alternativas evaluadas:**

1. Tupla de objetos Python más diccionario de posiciones. Preserva tipos heterogéneos y la semántica de igualdad actual en los casos compatibles. Recomendada. Requiere pruebas de caracterización porque pandas y Python coinciden en tratar `1`, `1.0` y `True` como el mismo hash, pero pandas detecta NaN duplicados y Python no.
2. Serie Polars como almacén de IDs. Rápida para joins homogéneos, pero los IDs heterogéneos caen en dtype Object con soporte pobre. Útil solo como caché interna cuando el tipo es homogéneo.
3. Tipo de ID declarado, entero o cadena. Simplifica todo pero es una restricción pública que rompe casos actuales. Solo si el laboratorio lo acepta como cambio documentado.

**Matrices etiquetadas, alternativas:**

1. `LabeledMatrix` propio con exportador a tabla ancha, columna de etiqueta con el nombre que hoy da `index_label`, y etiquetas tipadas en Parquet. Recomendada.
2. Tabla Polars ancha con la etiqueta como primera columna. Pierde el tipo del eje de columnas cuando los IDs no son cadenas, exactamente el caso de Aitchison.
3. Tabla larga N². Rechazada, cambia el contrato y el coste de memoria.

**Entorno oráculo.** Grupo de dependencias `oracle` con pandas y statsmodels, usado solo por un script generador de referencias fuera de `tests/`. Las referencias se guardan con versiones, semilla y commit. La suite nativa compara contra archivos y nunca importa pandas, statsmodels ni patsy salvo que la opción A resulte elegida.

**Gates de dependencia verificables.** Un job de CI que instale el paquete en un entorno limpio sin grupos de desarrollo y falle si pandas, statsmodels o formulaic aparecen en la lista instalada. Un test que importe `ruddy`, ejecute un análisis de cada familia y compruebe que esos módulos no están en `sys.modules`. Una auditoría AST sobre `ruddy/` y `tests/` con lista de excepciones vacía.

## 3. Fases, gates científicos y cambios al documento

**Fase 0, congelar referencias.** Antes de tocar código: ejecutar la suite y registrar fallos previos; generar con el entorno oráculo referencias para cada método listado abajo, incluidas matrices de diseño, nombres de parámetros, estados degenerados y advisories; convertir los cinco tests con oráculo vivo a comparaciones contra archivo. Salida: referencias legibles, comparador con tolerancias por método.

**Fase 1, go/no-go de motores.** Prototipos sobre NumPy, sin Polars, comparados con las referencias:

| Método | Gate propuesto | Estado de la evidencia |
| --- | --- | --- |
| Matriz de diseño Sum y Treatment | igualdad exacta de valores, orden y nombres | comprobado en fuente, falta prototipo |
| OLS, coeficientes, SE, t, p, CI | rel 1e-10 | comprobado en fuente |
| ANOVA II y III, sum_sq, df, F, p | rel 1e-10 sobre balanceado y desbalanceado, más el test de invariancia al orden de categorías | comprobado en fuente |
| HC0 a HC3 | rel 1e-10 en F robusto para los cuatro tipos, hoy solo HC3 tiene test | comprobado en fuente |
| Medias marginales | vectores L exactos, estimaciones y contrastes rel 1e-10, términos conjuntos, covariables en la media | comprobado en fuente |
| MANOVA | valor, F, p y ambos df para los cuatro estadísticos, incluidas filas Intercept y covariable, rel 1e-10 | comprobado en fuente |
| normal_ad, Jarque–Bera, Breusch–Pagan, VIF, influencia | rel 1e-10 a 1e-12, más umbrales y estados skipped/degenerate sin cambio | comprobado en fuente |
| Asimetría, curtosis, Spearman | rel 1e-12 sobre el corpus de paridad, más casos casi constantes con decisión explícita sobre el anulado de 1e-14 | comprobado en fuente |
| Modelo mixto ML y REML | ver abajo | requiere prototipo |

Para mixtos el gate debe pactarse antes de medir. Propuesta inicial: log-verosimilitud en el óptimo con diferencia absoluta menor a 1e-6, efectos fijos, errores estándar, covarianza aleatoria, varianza residual y BLUP con rel 1e-6, flags de convergencia y singularidad iguales, AIC y BIC ausentes bajo REML como hoy. Incluir los tres fixtures actuales, la ruta REML por defecto y un caso con pendiente aleatoria. Si el prototipo no alcanza la tolerancia pactada, la migración se detiene y se informa. No se degrada a skipped ni se cambia de optimizador.

**Fase 2, sustituir statsmodels en producción.** Integrar los motores aprobados en `factorial/`, `multivariate/manova.py`, `multivariate/collinearity.py` y `univariate/diagnostics.py`, conservando parámetros públicos y textos de razones. Retirar statsmodels y patsy del `pyproject.toml` si la opción B ganó. Gate: suite completa verde, job de instalación limpia verde con pandas aún presente solo por dependencia directa.

**Fase 3 a 6, migración Polars.** Mantener las fases 3 a 7 del plan actual con dos correcciones: la fase 2 del documento deja de introducir "el adaptador de modelos" y las conversiones `from_pandas`/`to_pandas`, y la fase 6 deja de "conservar el motor". Los gates de esas fases ya están bien descritos en el plan.

**Fase 7, entrega.** Añadir a los comandos de cierre la instalación limpia y la auditoría AST, actualizar README, inventario, métodos y contratos, y regenerar los 13 ejemplos con Polars. Verificar antes que Polars publica ruedas para la matriz Python 3.11 a 3.14; no lo puedo confirmar desde el repo.

**Cambios concretos al documento:**

- Reescribir la fase 2, líneas 201 a 214, sin adaptador ni conversiones pandas, y añadir la pista de motores como fase previa a Polars.
- Reescribir la fase 6, líneas 258 a 269, como integración de motores propios ya validados, no como conservación del motor.
- Cambiar el criterio de terminación, líneas 324 a 327, de "conversiones acotadas" a "grafo de dependencias sin pandas, statsmodels ni formulaic, verificado en CI".
- Añadir la tabla de gates por método con tolerancias y el gate de mixtos como decisión explícita go/no-go.
- Añadir la conversión de los cinco tests con oráculo vivo y la estructura del grupo `oracle`.
- Añadir los contratos de facto a decidir: nombres `[S.nivel]` en coeficientes, nombres seguros `X0` en mixtos, fila Intercept con codificación Treatment en MANOVA, AIC/BIC ausentes bajo REML, y textos de advisories que hoy provienen de statsmodels.
- Corregir la afirmación de la tabla de reemplazos sobre Patsy: el lock confirma que no exige pandas, pero su estado de mantenimiento debe constar como riesgo.
- Declarar que Parquet y CSV se leen con Polars nativo, sin PyArrow.

**Recomendaciones que rechazaría:**

- Instalar statsmodels con `--no-deps` o copiar el paquete entero. No demuestra independencia y arrastra código no auditado.
- Usar formulaic. Exige pandas en el lock.
- Convertir modelos en extra opcional o marcarlos skipped sin la dependencia. Viola el requisito y el feature freeze.
- Sustituir `normal_ad` por los valores críticos de `scipy.stats.anderson`. Cambia la salida y elimina el p-value.
- Reemplazar el estimador mixto por otro más simple o cambiar de optimizador automáticamente. Cambia el método.
- Ampliar tolerancias de forma global. Solo tolerancias por método, pactadas y documentadas.
- Convertir IDs a texto, usar el índice de fila Polars como identidad o emitir matrices N×N como tablas largas.
- Mantener dos backends seleccionables o usar `polars.from_pandas` dentro del paquete o de la suite nativa.
- Añadir PyArrow como dependencia. Polars no lo necesita para CSV ni Parquet.

**Lo que queda pendiente de investigación o prototipo:** viabilidad y tolerancias del motor mixto; paridad exacta del codificador propio frente a patsy en niveles heterogéneos y categorías no observadas; comportamiento de asimetría y curtosis en columnas casi constantes; semántica de igualdad de IDs con NaN y booleanos; disponibilidad de ruedas Polars para Python 3.14; y si los textos de advisories que hoy copian warnings de statsmodels están pinneados en algún test, que no comprobé.
