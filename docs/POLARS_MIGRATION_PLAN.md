Plan de migración de pandas a Polars
===================================

Estado: ejecutado en la rama `feat/polars-migration` (18 de septiembre de 2026); pendiente revisión final y merge.
2026 sobre el commit `f332721`; ajustado el mismo día tras verificar qué
dependencias aceptan Polars. Deben conservarse los métodos científicos
existentes y las invariantes de `AGENTS.md`.

**Decisión de alcance (17 de septiembre de 2026).** La versión anterior de
este plan exigía cero pandas, directo o transitivo, y por tanto reemplazar
statsmodels y Patsy con motores propios. Esa parte concentraba 30–62 de los
51–100 días estimados en el
[inventario de reemplazos](POLARS_REPLACEMENT_ESTIMATES.md). Se descarta por
ahora. El alcance vigente es:

- Los contratos tabulares públicos, la preparación de datos y las tablas de
  resultados pasan a Polars.
- NumPy/SciPy siguen siendo el almacenamiento de matrices y el motor numérico.
- scikit-learn, SciPy, umap-learn y statsmodels se conservan y reciben Polars
  directamente. Patsy recibe pandas mediante `.to_pandas()` explícito.
- pandas deja de ser dependencia directa y de aparecer en `ruddy/`, pero sigue
  instalado como dependencia transitiva de statsmodels. pyarrow pasa a
  dependencia principal porque `.to_pandas()` lo requiere.
- Reimplementar motores queda como decisión posterior, si se justifica.

Los documentos [POLARS_MIGRATION_REVIEW_FABLE.md](POLARS_MIGRATION_REVIEW_FABLE.md)
y [POLARS_REPLACEMENT_ESTIMATES.md](POLARS_REPLACEMENT_ESTIMATES.md) se
conservan como registro del escenario descartado. Sus secciones sobre motores
(OLS, ANOVA, MANOVA, mixtos, VIF, Anderson–Darling) no forman parte de este plan.

**Fronteras con dependencias, verificadas**

Comprobación ejecutada en el entorno del proyecto con Polars 1.44.2, pyarrow
25.0.1 y las versiones fijadas en `uv.lock`. Está pinneada en
`tests/parity/test_polars_input_parity.py`.

| Dependencia | Versión | Entrada Polars | Mecanismo | Consecuencia |
| --- | --- | --- | --- | --- |
| scikit-learn | 1.9.1 | Sí, `DataFrame` y `Series` | narwhals | Sin cambios en llamadas. Devuelve NumPy. |
| SciPy | 1.18.1 | Sí, como array | `__array__` | Sin cambios en llamadas. Preferir `.to_numpy()` explícito donde hoy se pasa `.to_numpy()`/`.values`. |
| umap-learn | 0.5.12 | Sí | vía sklearn | Sin cambios. |
| statsmodels | 0.15.0 | Sí en `ols`, `mixedlm`, `MANOVA.from_formula`, `OLS`, `anova_lm`, VIF, `normal_ad`, `jarque_bera`, `het_breuschpagan` | `statsmodels.tools.data._to_pandas` llama `.to_pandas()` | Resultados idénticos a pasar pandas. Requiere pyarrow instalado. |
| Patsy | 1.0.3 | No: `PatsyError ... "C" not found` | — | `.to_pandas()` explícito antes de `dmatrices` en `factorial/marginal_means.py:97` y `multivariate/manova.py:249`. `build_design_matrices` recibe una grilla construida en pandas en `marginal_means.py:148`. |

Los objetos devueltos por statsmodels (`params`, tablas de `anova_lm`, `resid`,
`mv_test().results`) siguen siendo pandas. Se consumen internamente y se
convierten a los contratos Polars/NumPy de Ruddy en cada módulo; no se exponen.

**Evidencia del repositorio**

Inventario mediante AST de todos los archivos Python, contando imports de pandas
tanto en ejecución como en anotaciones de tipos:

| Superficie | Archivos Python | Archivos que importan pandas |
| --- | ---: | ---: |
| `ruddy/` | 97 | 45 |
| `tests/` | 78 | 52 |
| `examples/` | 15 | 15 |

Accesos sintácticos `pd.*` en `ruddy/` por paquete, como medida de esfuerzo
relativo: factorial 77, bivariate 63, multivariate 62, representation 54,
univariate 52, data 28, projections 21, profiling 16, analysis 12,
compositional 10, bayesian 7, core 6, statistics 5, anomaly 4, results 3,
cli 3. Hay 38 funciones que devuelven `pd.DataFrame`: bivariate 11,
univariate 5, profiling 4, multivariate 4, data 4, representation 3,
compositional 2, statistics 1, results 1, projections 1.

| Área | Evidencia y efecto sobre la migración |
| --- | --- |
| Contratos | `data/dataset.py` exige `pd.DataFrame`; `select()` y `to_frame()` devuelven pandas; los IDs proceden de una columna o del índice. |
| Matrices de entrada | `data/feature_matrix.py` ya almacena NumPy/SciPy. Pandas interviene en entradas tabulares, IDs y metadatos. |
| Identidad | `data/validation.py` valida con `pd.Index` y alinea con `reindex`; `data/annotations.py` usa índices y `join`. |
| Tipos | `data/roles.py` infiere clases estadísticas a partir de dtypes pandas; `core/types.py` expone sus tipos. |
| Resultados | `results/schemas.py` exige pandas. Los contratos de cada análisis y las tablas vacías se construyen de forma distribuida. |
| Descriptiva | `univariate/numeric.py` calcula asimetría y curtosis con pandas. Las otras estadísticas de ese módulo usan principalmente NumPy. |
| Multivariante | `multivariate/covariance.py` calcula Spearman con `DataFrame.corr`; sus matrices usan etiquetas en ambos ejes. |
| Modelos | `factorial/{models,marginal_means,mixed_effects}.py` y `multivariate/manova.py` construyen tablas para statsmodels/Patsy. Solo cambia la frontera de entrada y la conversión de salida. |
| Corrección múltiple | `statistics/multiple_testing.py` ya separa el cálculo NumPy de la selección/asignación tabular por familia. |
| Orquestación | `analysis/engine.py` representa los IDs de features mediante un DataFrame sin columnas y con índice. |
| Archivos y CLI | `core/io.py` centraliza CSV/TSV/Parquet, pero los escritores CLI también usan `.empty`, `.iloc` y exportaciones con `index=True`. |
| Ejemplos | Los ejemplos marimo, `_helpers.py` y el generador de datos usan pandas. |

Los paths de implementación son relativos a `ruddy/`.

**Contrato objetivo**

| Superficie | Destino |
| --- | --- |
| `TabularDataset` | Almacenamiento Polars expuesto como `frame`; entrada `pl.DataFrame` o `pd.DataFrame`, convertida en el constructor. Ver decisiones tomadas abajo. |
| `FeatureMatrix` | Mantener NumPy/SciPy; aceptar Polars numérico como entrada, con IDs separados y explícitos cuando corresponda. |
| IDs | Secuencia inmutable independiente de pandas, propuesta `tuple[ObservationID, ...]`, y operaciones internas de validación y selección. |
| Anotaciones/metadatos | Tablas Polars más identidad explícita; conservar cobertura y `AlignmentReport`. |
| Tablas de resultados | Polars con esquema declarado, incluso sin filas o con columnas enteramente nulas. |
| Matrices con etiquetas | Tabla Polars: la primera columna (`feature` u `observation_id`) conserva la etiqueta de fila con su tipo; las demás columnas son la matriz, con nombres `str(etiqueta)`. Decisión del 18 de septiembre de 2026: se descartó un contrato aparte con `values`/`row_ids`/`column_ids` por complejidad. |
| Modelos con fórmulas | Ruddy construye el marco seguro (`Y`, `F0..`, `X0..`, `G`) en Polars y lo convierte con `.to_pandas()` en un único punto por módulo justo antes de statsmodels/Patsy. Los resultados vuelven a NumPy/Polars ahí mismo. |
| I/O | Polars para tablas, NumPy/SciPy para `.npy`/`.npz`. |

Los IDs requieren una decisión explícita porque hoy `ObservationID = Any` y el
comportamiento real depende de pandas. Caracterizar enteros, cadenas, booleanos,
fechas e IDs heterogéneos antes de fijar los tipos admitidos. Preservar los tipos
y la igualdad actual en los casos compatibles; cualquier restricción debe ser
un cambio público documentado. No convertir automáticamente IDs a texto:
`1` y `"1"` pueden identificar observaciones diferentes.

Para un `TabularDataset` sin columna ID, mantener IDs generados `0..n-1`, fijos
desde su construcción, y registrar su origen. Añadir un argumento explícito
`observation_ids` permite preservar identidad al reconstruir tablas que antes
dependían de un índice. Rechazar la combinación ambigua de ese argumento e
`id_column`. En anotaciones externas exigir columna ID o IDs separados; la
ausencia de índice en Polars no debe convertirse en alineación por posición.

Las matrices con etiquetas (covarianza, correlaciones, conteos por pares,
distancias, variación composicional, Aitchison) se exportan como tabla Polars
con la etiqueta de fila en la primera columna. Polars exige nombres de columna
de tipo cadena, así que en el eje de columnas los IDs no textuales quedan como
`str(id)`; la primera columna conserva el tipo original. Es un cambio público
documentado. La CLI escribe la tabla sin `index=True` ni `index_label`.

Como el proyecto está en pre-release, el cambio de API es anunciado y
coordinado, sin período de compatibilidad ni doble backend. Los resultados
públicos tendrán un único formato nativo: Polars.

**Decisiones tomadas (17 de septiembre de 2026).**

- `TabularDataset` acepta `pl.DataFrame` y `pd.DataFrame`. La entrada pandas se
  convierte en el constructor con `pl.from_pandas` y se registra
  `input_backend` en procedencia. El índice pandas nunca es identidad
  implícita: si lo es, el caller lo pasa por `observation_ids` o lo vuelca a
  columna. Decisión revisable cuando la API se estabilice.
- `TabularDataset` se reduce a lo que un DataFrame no tiene: `frame` público
  en Polars, `observation_ids` validados, `schema` con roles y kinds, y las
  consultas `role_of`, `kind_of`, `columns_with_role`, `columns_with_kind`.
  Se eliminan los métodos que solo delegan (`n_observations`, `n_columns`,
  `columns`, `select`, `to_frame`, `__len__`) y las copias defensivas, que
  Polars hace innecesarias.

**Detalles verificados que deben conservarse**

Proceden de la revisión de Fable y se contrastaron con el código de Ruddy. Con
statsmodels conservado dejan de ser riesgos de reimplementación, pero siguen
siendo salidas públicas que la migración tabular no debe alterar.

| Hallazgo verificado | Acción en la migración |
| --- | --- |
| `factorial/models.py:183` traduce los nombres de factores, pero conserva sufijos Patsy `[S.nivel]` e interacciones. | Congelar nombres, orden de niveles, nivel omitido y orden de columnas. El orden de niveles depende hoy de `pd.Categorical`; al construir el marco en Polars hay que fijar el mismo orden antes de convertir. |
| `factorial/mixed_effects.py:255` y sus componentes de varianza exponen nombres seguros; `tests/factorial/test_mixed_effects.py:106` busca `X0`. | Conservarlos; corregir nombres, si se desea, como cambio separado. |
| `multivariate/manova.py:246` usa `C(F0)` (Treatment), y desde la línea 319 publica `Intercept`. | Sin cambios en fórmulas ni contrastes. |
| pandas anula términos muy pequeños en asimetría/curtosis mediante un umbral 1e-14. | `univariate/numeric.py` pierde `Series.skew/kurt`. Sustituir por SciPy con corrección de sesgo explícita y añadir fixtures casi constantes; esta es la única fórmula numérica que cambia de proveedor en este plan y necesita su test de regresión. |
| `multivariate/covariance.py` usa `DataFrame.corr(method="spearman")`. | Reemplazar por rankings con empates y correlación equivalentes; probar una/dos variables, columnas constantes y filas excluidas. |
| Los warnings capturados de statsmodels llegan a advisories. | Sin cambios: el motor se conserva. |

**Semánticas que deben fijarse antes de migrar consumidores**

1. **Faltantes y no finitos.** Ruddy cuenta `NaN` como faltante y `±inf` como
   presente no finito. Polars distingue `null` de `NaN`; `drop_nulls()` no cubre
   ambos. Implementar máscaras por dtype para faltantes (`null` o `NaN` en
   flotantes), presentes y finitos, sin transformar la tabla fuente. Mantener
   separadas las exclusiones locales de cada análisis. Véase la
   [documentación de faltantes de Polars](https://docs.pola.rs/user-guide/expressions/missing-data/).
   Al convertir con `.to_pandas()` para statsmodels/Patsy, `null` pasa a `NaN`;
   la selección de casos completos debe hacerse antes, en Polars, para que la
   muestra sea la misma que hoy.
2. **Orden e identidad.** Conservar el orden de los IDs base, las columnas, las
   familias de hipótesis y los grupos donde forme parte del comportamiento
   actual. No aplicar un orden global nuevo: algunas funciones ordenan niveles y
   otras preservan primera aparición. Para joins usar validación uno a uno y
   orden explícito, o un mapa ID→posición para identidades sin representación
   nativa; nunca inferir correspondencia por posición. Polars exige declarar el
   orden si se depende de él, según su
   [API de joins](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html).
3. **Dtypes y categorías.** Definir el mapeo de `String`, `Categorical`, `Enum`,
   booleanos, fechas, duraciones, decimales, tipos anidados y `Null` a
   `ColumnKind`. Una columna `Null` no aporta evidencia de ser numérica: exigir
   schema de entrada para ese caso. Caracterizar los `object` heterogéneos de
   pandas; no convertirlos a cadenas o nulos para hacerlos aceptables. Preservar
   las etiquetas normalizadas existentes y los niveles ordenados/no observados
   que influyen en diseños factoriales. Documentar el cambio de las cadenas
   `dtype` expuestas en perfiles.
4. **Inmutabilidad.** Conservar aislamiento de entradas, tablas devueltas y arrays.
   Verificar mutaciones del caller y de resultados de accesores, incluyendo
   buffers NumPy compartidos. Evaluar `clone()` según esos contratos; no suponer
   que toda conversión copia sus datos.
5. **Resultados vacíos.** Declarar tipos de columnas, orden y nulabilidad por
   tabla. Validar `status/reason` también en tablas vacías o enteramente nulas.
   Conservar los significados de valores ausentes; cualquier normalización de
   `NaN` a `null` en resultados debe ser explícita y probada.
6. **Tablas sin columnas.** Las anotaciones ausentes y la comprobación de
   alineación de features dependen hoy de un índice que conserva la cantidad de
   filas. Guardar IDs y cardinalidad fuera de la tabla para esos casos; no derivar
   el número de observaciones de una tabla vacía sin columnas.
7. **Archivos.** Caracterizar inferencia de CSV, tokens de NA, cadenas vacías,
   booleanos, fechas, enteros con null y etiquetas numéricas. Igualar las políticas
   actuales o documentar diferencias intencionales; evitar `ignore_errors` y
   casts permisivos. Parquet debe preservar tipos, categorías e identidad según
   el nuevo contrato, incluyendo archivos legados con índice pandas.

Polars no tiene índice de filas etiquetado; esta diferencia está descrita en su
[guía de migración desde pandas](https://docs.pola.rs/user-guide/migration/pandas/).
La primera versión migrada usará ejecución eager. Introducir `LazyFrame`,
streaming o nuevas optimizaciones después de alcanzar paridad mantendrá acotado
el cambio y hará explícitos el momento de validación y la materialización.

**Secuencia de implementación**

Un PR por fase, sobre una rama de migración. Cada fase migra sus tests, sus
escritores CLI y sus ejemplos marimo. La suite completa debe pasar al cierre de
cada fase; mientras un paquete siga en pandas, la fase que cambió los contratos
compartidos usa adaptadores privados temporales enumerados en el PR y retirados
antes de la fase 5.

1. **Referencia y dependencias.** Ejecutar la suite en el commit base y registrar
   fallos previos. Ampliar los fixtures congelados de
   `tests/parity/reference/` a modelos, matrices con etiquetas, alineación,
   exclusiones y serialización, generados con el código pandas actual. Añadir
   `polars` y `pyarrow` a `dependencies`. `pandas` sigue declarado como
   dependencia directa hasta que la fase 5 retire su último import.
   Comprobar en Python 3.11–3.14 sin actualizar NumPy/SciPy/sklearn.

   Salida: referencias legibles, comparador con igualdad exacta para
   identidades, estados, razones, orden y conteos, y tolerancias numéricas por
   método. Mantener `rtol=atol=1e-12` donde las pruebas actuales lo requieren.

2. **Contratos y soporte compartido.** Adaptar `core/types.py`, `data/`
   (`dataset`, `feature_matrix`, `validation`, `annotations`, `roles`),
   `results/schemas.py` y `core/io.py`. Introducir utilidades privadas para
   identidad, máscaras de faltantes, extracción numérica, tablas tipadas y
   matrices con etiquetas. Resolver la decisión pendiente sobre entrada pandas.

   Salida: pruebas de tipos, conversiones, inmutabilidad, alineación e I/O
   `CSV → TabularDataset → Polars → CSV`, con pruebas de I/O dedicadas en
   `tests/core/`.

3. **Descriptiva, perfilado e inferencia tabular.** Adaptar `profiling/`,
   `univariate/`, `statistics/`, `bivariate/`, `analysis/intervals.py` y
   `bayesian/`. Reutilizar los cálculos NumPy/SciPy; reemplazar selección,
   concatenación, agrupación y ensamblado de tablas. Sustituir asimetría/curtosis
   pandas con su test de regresión.

   Salida: mismas muestras por análisis, frecuencias y desempates; mismas
   familias FDR, p/q, semillas, flags y razones de degeneración. Las filas `ok`
   sin p finito siguen fuera de la familia inferencial.

4. **Representaciones y matrices.** Adaptar `projections/`,
   `multivariate/{covariance,collinearity,distances,permutation}.py`,
   `representation/`, `compositional/` y `anomaly/`. Conservar almacenamiento
   NumPy/SciPy y rechazos explícitos de densificación. Reemplazar Spearman pandas.

   Salida: paridad con IDs permutados y cobertura parcial; conservación de
   `source_row_index` y ejes de matrices; rutas sparse intactas.

5. **Modelos con fórmulas.** Adaptar `factorial/{models,marginal_means,
   mixed_effects,diagnostics}.py` y `multivariate/manova.py`. Construir el marco
   seguro en Polars, fijar el orden de niveles y convertir con `.to_pandas()` en
   un único punto por módulo; convertir salidas de statsmodels a NumPy/Polars
   ahí mismo. Retirar los adaptadores temporales de la fase 2 y `pandas` de
   `dependencies`.

   Salida: igualdad exacta de diseño, nombres de términos, SS II/III, HC0–HC3,
   medias marginales, MANOVA y mixtos frente a las referencias congeladas. Ningún
   `import pandas` queda en `ruddy/`.

6. **Orquestación, CLI, ejemplos y entrega.** Adaptar `analysis/engine.py`, los
   comandos restantes y los escritores con `index=True`. Migrar ejemplos marimo,
   helpers y generador. Actualizar README, DEVELOPMENT, DATA_CONTRACTS,
   PUBLIC_API, OUTPUT_SCHEMAS, CLI_REFERENCE, RESULTS_AND_PROVENANCE y
   TESTING_AND_REPRODUCIBILITY. Documentar los cambios incompatibles.

   Salida: auditoría AST sin imports pandas en `ruddy/`; wheel instalada en
   entorno limpio con todos los métodos ejecutables; ejemplos aprobados.

**Pruebas de aceptación y verificación**

| Riesgo | Casos mínimos | Evidencia requerida |
| --- | --- | --- |
| Identidad | IDs reordenados, duplicados, faltantes, enteros/cadenas, cobertura parcial, columnas con nombres conflictivos | Mismos pares de observaciones, orden base y reportes; rechazos explícitos. |
| Datos vacíos | Cero filas, cero variables, solo IDs, anotaciones ausentes, columnas totalmente nulas | Cardinalidad correcta, schema estable y estados explícitos. |
| Faltantes | `None`, `NaN`, `±inf`, enteros/booleanos nullable | Mismos conteos, máscaras, muestras y exclusiones. |
| Tipos | Strings numéricos, categorías ordenadas/no observadas, fechas con zona, heterogéneos | Sin coerción accidental; cambios de soporte documentados. |
| Estadística | Empates, constantes/casi constantes, n pequeño, rango deficiente, FDR con p ausentes | Paridad de valores y estados; semillas reproducibles. |
| Matrices | CSR, IDs numéricos, etiquetas que colisionan al convertirlas a texto, una variable | Sin densificación implícita ni pérdida de etiquetas o forma. |
| Modelos | SS II/III, HC3, medias marginales, mixtos, MANOVA | Mismo marco seguro, misma selección de filas y mismo orden de niveles que la referencia; igualdad exacta de nombres. |
| Frontera pandas | Cada `.to_pandas()` en `ruddy/` | Enumerado en `tests/parity/test_polars_input_parity.py` o equivalente; un import hook de prueba verifica que ningún otro módulo importa pandas. |
| Superficies | Función individual, `analyze()`, CLI, CSV/TSV/Parquet y ejemplos | Igualdad semántica y esquemas/exportaciones documentados. |
| Mutación | Cambiar entrada, accesores o arrays externos | Dataset e identidad originales intactos. |
| Dependencias | Wheel y extras en entorno limpio, Python 3.11–3.14 | pandas ausente de `ruddy/`; pyarrow presente; todas las rutas científicas funcionan. |

Reutilizar `tests/data`, `tests/results`, `tests/parity`, `tests/robustness`,
`tests/integration` y `tests/cli`, adaptando comparaciones a
`polars.testing.assert_frame_equal` y a NumPy según el contrato. Los tests que
usan statsmodels como oráculo (factorial, MANOVA, VIF, Tukey, Welch) siguen
válidos: statsmodels permanece instalado.

Comandos de cierre, después de migrar:

```bash
uv sync --all-extras --group examples
uv run task lint
uv run task pyrefly
uv run task test
bash examples/run_ci_examples.sh
```

Mantener la matriz CI Python 3.11–3.14 y la ejecución de ejemplos. Añadir un job
que construya e instale la wheel en un entorno limpio y ejecute la suite.

Registrar tiempo y memoria pico sobre perfilado ancho, agrupaciones, alineación
grande, matrices sparse y un modelo factorial, comparados con la referencia en
la misma máquina. No hay benchmarks que permitan prometer una aceleración; la
motivación de esta migración es el contrato de datos, no el rendimiento.

La migración estará terminada cuando los contratos públicos tabulares sean
Polars, la identidad sea independiente del índice pandas, las matrices preserven
ambos ejes, `ruddy/` no importe pandas fuera de las fronteras enumeradas, todos
los métodos mantengan paridad con las referencias congeladas, y tests, tipado,
CLI, documentación y ejemplos correspondan a la misma API.
