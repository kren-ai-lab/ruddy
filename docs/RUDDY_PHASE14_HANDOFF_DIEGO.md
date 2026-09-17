# Ruddy — Phase 14
## Especificación de productización, interfaz local y handoff a Diego

**Estado del documento:** especificación operacional de implementación posterior al freeze científico
**Idioma:** español
**Responsable del diseño científico y funcional:** equipo Ruddy
**Responsable principal de implementación/productización:** Diego
**Naturaleza de esta fase:** diseño y handoff. Este documento no implica que el equipo científico deba implementar los elementos descritos aquí.

---

# 1. Objetivo de este documento

Ruddy ya dispone de un núcleo científico funcional para análisis exploratorio de datos tabulares, espacios de características y comparación entre representaciones numéricas. El objetivo de esta fase es transformar ese núcleo en una **librería utilizable, mantenible y presentable como producto científico**, sin alterar innecesariamente la matemática ya congelada.

Este documento define con precisión qué debe implementar Diego después del handoff, cómo deben conectarse los distintos componentes y qué criterios deben cumplirse para considerar Ruddy una librería madura y lista para distribución local, uso científico reproducible y posterior utilización dentro del **Numerical Representation Framework**.

El trabajo posterior se concentra en seis frentes:

1. endurecimiento del paquete y del repositorio;
2. rediseño completo de la CLI;
3. creación de una capa de visualización reutilizable;
4. creación de una aplicación web local;
5. empaquetado, instalación, distribución y ejecución local;
6. fortalecimiento de tests, CI, documentación de usuario y experiencia de desarrollo.

El principio central es que **el núcleo científico ya existe y no debe reescribirse sólo por razones estéticas o de producto**. La productización debe envolverse alrededor de los contratos actuales y consumirlos.

---

# 2. Contexto del proyecto y rol de Ruddy

Ruddy forma parte de un framework más amplio de representación numérica orientado a aplicaciones moleculares y biomoleculares.

El marco general se compone de:

| Componente | Responsabilidad principal |
| --- | --- |
| **Sylphy** | Codificación de secuencias y extracción de embeddings |
| **Eris** | Representaciones estructurales |
| **Roxy** | Descriptores tabulares |
| **Ruddy** | EDA, análisis estadístico, comparación entre espacios y visualización |

Ruddy debe seguir siendo **agnóstico respecto al origen científico de la representación**. Para Ruddy, un output de Sylphy, Eris, Roxy o cualquier otra herramienta es un `FeatureMatrix` o un dataset tabular con observaciones identificables.

Esto es fundamental para la historia científica del futuro Application Note. Ruddy no debe contener lógica dependiente de proteínas, péptidos, SMILES, estructuras moleculares o nombres internos de las otras librerías. La integración ocurre por contratos de datos, no por acoplamiento entre proyectos.

---

# 3. Estado que recibe Diego

Diego recibe un proyecto cuyo **scientific core se considera funcionalmente congelado para el MVP**, con las siguientes familias ya implementadas:

- contratos `TabularDataset` y `FeatureMatrix`;
- roles estadísticos de columnas y alineación por observation IDs;
- profiling y missingness;
- análisis univariante numérico, categórico y datetime;
- correlations Pearson, Spearman y Kendall;
- partial correlations;
- distance correlation y mutual information;
- comparaciones entre grupos;
- effect sizes;
- contingency analysis;
- confidence intervals;
- bootstrap;
- grouped analyses;
- outlier diagnostics;
- PCA, t-SNE y UMAP;
- covariance, VIF, condition diagnostics y Mahalanobis;
- MANOVA;
- PERMANOVA y PERMDISP;
- factorial ANOVA/ANCOVA;
- Type II y Type III sums of squares;
- post-hoc Tukey–Kramer y Games–Howell;
- estimated marginal means y contrasts;
- mixed-effects models acotados;
- CCA;
- CKA;
- Procrustes;
- distance-space similarity y Mantel;
- compositional statistics;
- Bayesian EDA acotado;
- Isolation Forest y LOF;
- `ruddy.analyze()` como orquestador unificado;
- CLI funcional básica;
- notebooks fuertes de análisis y visualización;
- documentación científica y técnica.

La responsabilidad de Diego **no es volver a diseñar estos métodos**. Su responsabilidad es transformar el proyecto en una experiencia coherente y robusta de uso.

---

# 4. Reglas no negociables durante la productización

Estas reglas deben tratarse como invariantes de arquitectura.

## 4.1. No romper el scientific freeze

Las implementaciones científicas actuales sólo deben modificarse cuando exista:

- un bug reproducible;
- un test que demuestre el comportamiento incorrecto;
- una incompatibilidad real de backend;
- una inconsistencia entre API standalone y unified;
- un problema numérico demostrado.

No deben reescribirse módulos científicos simplemente para “ordenarlos mejor”.

Cuando sea necesario modificar ciencia, el procedimiento debe ser:

```text
bug reproducible
    ↓
test que falla
    ↓
fix mínimo
    ↓
suite científica completa
    ↓
documentación del cambio
```

## 4.2. Ruddy debe permanecer independiente

No deben existir referencias a otros proyectos en:

- configs;
- tests;
- módulos;
- fixtures;
- metadata;
- provenance;
- comentarios;
- nombres internos;
- comandos;
- interfaz;
- documentación interna de implementación.

Los ejemplos del Numerical Representation Framework pueden mencionar conceptualmente otros componentes en documentación externa o notebooks destinados al paper, pero el código de Ruddy no debe depender de ellos.

## 4.3. Visualización separada del core científico

`src/ruddy` puede incorporar una capa visual opcional durante la productización, pero el scientific core no debe depender de ella.

Debe conservarse esta dirección de dependencias:

```text
scientific core
      ↓
structured result objects
      ↓
visualization layer
      ↓
CLI / local app / figure export
```

Nunca:

```text
scientific method
      ↓
imports plotting framework
```

## 4.4. La aplicación es local

La interfaz web está destinada a ejecutarse **localmente en el computador del usuario**.

No se requiere diseñar:

- infraestructura cloud;
- multi-user server;
- autenticación remota;
- workers distribuidos;
- colas remotas;
- S3;
- bases de datos administradas;
- Kubernetes;
- despliegue SaaS.

El concepto objetivo es:

```bash
ruddy app
```

seguido de una interfaz disponible en `localhost`.

## 4.5. Ninguna capa debe recalcular estadística

La CLI y la aplicación local deben consumir los resultados del core.

Por ejemplo:

```text
FactorialResult.effects
        ↓
effect-size plot
```

La visualización no debe volver a ejecutar ANOVA para construir la figura.

## 4.6. No introducir comportamiento científico automático nuevo

La capa de producto no debe inventar reglas del tipo:

```text
normality failed → cambiar automáticamente a Kruskal
heteroscedasticity → cambiar automáticamente a Games–Howell
p >> n → ejecutar PCA automáticamente
zero composition → reemplazar ceros automáticamente
```

Puede **recomendar** o mostrar advertencias en UI, pero no modificar silenciosamente la configuración.

---

# 5. Prioridades del handoff

La implementación se debe organizar usando tres niveles.

| Prioridad | Interpretación |
| --- | --- |
| **P0** | Requerido para considerar Ruddy funcionalmente productizado |
| **P1** | Requerido para considerar Ruddy maduro y cómodo de usar |
| **P2** | Mejoras valiosas que pueden implementarse después del primer release estable |

Orden recomendado:

```text
P0 package/repository hardening
        ↓
P0 CLI 2.0
        ↓
P0 visualization layer
        ↓
P0 local application MVP
        ↓
P0 packaging/distribution
        ↓
P1 tests + CI + UX hardening
        ↓
P1 visual/export polish
        ↓
P2 enhancements
```

---

# 6. Workstream A — Limpieza inicial y baseline del handoff

**Prioridad:** P0

Antes de implementar producto, Diego debe congelar el estado que recibe.

## 6.1. Limpiar artifacts de desarrollo

Eliminar del repositorio cualquier artifact que no deba distribuirse:

- `__pycache__/`;
- `.pyc`;
- carpetas temporales;
- outputs locales de notebooks que no sean intencionales;
- archivos de apply-check;
- ZIP/patches históricos;
- archivos de debugging;
- caches de pytest/Jupyter.

## 6.2. Eliminar tests de documentación innecesarios

No mantener tests creados únicamente para comprobar que existe documentación Markdown o que sus links internos funcionan, salvo que Diego considere posteriormente que son útiles dentro de CI.

El scientific test suite debe priorizar comportamiento del software.

## 6.3. Registrar baseline real

Crear un tag o commit explícito de handoff.

Ejemplo conceptual:

```text
scientific-freeze-v0.1
```

Registrar:

- commit hash;
- Python soportado;
- versiones de dependencias principales;
- número real de tests que pasan;
- dependencias opcionales instaladas durante el gate;
- comandos necesarios para reproducir el gate.

No hardcodear en documentación permanente un número de tests que pueda cambiar con cada commit.

## 6.4. Gate inicial

Debe ejecutarse como mínimo:

```bash
python -m compileall -q src tests
pytest -q
```

El resultado de este gate se convierte en baseline de Diego.

### Criterio de aceptación

- working tree limpio;
- suite completa pasa;
- scientific freeze identificado por commit/tag;
- no existen artifacts temporales versionados.

---

# 7. Workstream B — Reorganización de dependencias y extras

**Prioridad:** P0

El paquete actual tiene un core científico relativamente compacto. La productización agregará dependencias que no deben instalarse obligatoriamente para todos los usuarios.

## 7.1. Separar grupos de dependencias

Diseñar extras conceptualmente similares a:

```toml
[project.optional-dependencies]
manifold = [...]
visual = [...]
app = [...]
dev = [...]
all = [...]
```

Propuesta conceptual:

### Core

Mantener como dependencias obligatorias únicamente lo necesario para la ciencia base.

### `visual`

Puede contener:

- matplotlib;
- plotly;
- kaleido u otro backend de exportación;
- dependencias estrictamente necesarias para figure generation.

### `app`

Debe contener:

- framework seleccionado para la aplicación local;
- `visual`;
- parser de configuración si es necesario;
- dependencias de UI.

### `dev`

Diego puede incorporar aquí:

- pytest;
- pytest-cov;
- ruff;
- mypy/pyright;
- build;
- pre-commit si se decide;
- herramientas de documentación.

El hecho de que Ruff no haya sido prioritario durante el desarrollo científico **no significa que esté prohibido durante productización**. Aquí ya es responsabilidad de Diego decidir y endurecer estilo/CI.

## 7.2. Instalaciones objetivo

Deben funcionar experiencias como:

```bash
pip install ruddy
pip install "ruddy[visual]"
pip install "ruddy[app]"
pip install "ruddy[all]"
```

Y para desarrollo:

```bash
pip install -e ".[dev,app]"
```

### Criterio de aceptación

Instalar `ruddy` sin extras **no debe instalar frameworks de visualización ni la aplicación local**.

---

# 8. Workstream C — CLI 2.0

**Prioridad:** P0

La CLI actual es funcional pero se construyó durante el desarrollo científico y su objetivo principal era exponer rápidamente capabilities. Diego debe transformarla en una interfaz de usuario coherente.

## 8.1. Stack recomendado

La primera opción a evaluar debe ser:

```text
Typer
+
Rich
```

Razones:

- command groups claros;
- typing natural;
- help automático;
- autocompletado;
- buen manejo de subcomandos;
- Rich permite paneles, tablas, progress, warnings y tracebacks legibles.

La elección puede cambiar si Diego demuestra una alternativa mejor, pero debe documentar la razón.

## 8.2. No duplicar lógica científica

La nueva CLI debe ser una capa delgada:

```text
CLI arguments/config
       ↓
AnalysisConfig / standalone API
       ↓
Ruddy core
       ↓
result writers / visualization
```

No debe existir estadística implementada dentro de `cli/`.

## 8.3. Reorganizar comandos

La CLI actual tiene muchos comandos de primer nivel. Se debe evaluar una estructura jerárquica más legible.

Propuesta:

```text
ruddy
├── inspect
│   ├── profile
│   └── schema
│
├── analyze
│   ├── run
│   ├── univariate
│   ├── bivariate
│   ├── groups
│   ├── factorial
│   ├── multivariate
│   ├── permanova
│   ├── representation
│   ├── compositional
│   ├── bayesian
│   └── anomaly
│
├── project
│   ├── pca
│   ├── tsne
│   └── umap
│
├── config
│   ├── init
│   ├── validate
│   └── show
│
├── app
│
└── version
```

No es obligatorio conservar esta estructura exacta. El requisito es eliminar la sensación de una lista plana de comandos crecientes.

## 8.4. Experiencia visual

Ejemplo esperado:

```text
Ruddy
Exploratory statistics for tabular and numerical spaces

Dataset
  observations      12,480
  columns                 38
  numeric                 25
  categorical              8
  responses                2
  factors                  3

Analysis
  ✓ profiling
  ✓ univariate
  ✓ dependence
  ✓ PCA
  ⚠ factorial       2 advisory messages

Results
  ./ruddy_results/
```

Se deben usar:

- `Rich.Table`;
- `Rich.Panel`;
- status spinners/progress cuando el análisis lo justifique;
- warnings bien visibles;
- mensajes de error entendibles;
- resumen al finalizar.

## 8.5. Progress reporting

Análisis rápidos no necesitan barra.

Análisis potencialmente pesados sí deben mostrar progreso o al menos etapa actual:

- bootstrap;
- permutations;
- PERMANOVA/PERMDISP;
- Mantel;
- t-SNE/UMAP;
- Bayesian draws;
- pairwise analyses grandes.

La implementación de progress **no debe cambiar la matemática ni el seed**.

## 8.6. Manejo de errores

Evitar tracebacks gigantes para errores de usuario previsibles.

Ejemplos:

```text
Error: column 'family' does not exist.
Available categorical/factor columns: group, source, batch
```

```text
Cannot run Mahalanobis
Reason: p >= n after complete-case filtering
Suggestion: reduce dimensionality explicitly before running this analysis
```

Los tracebacks completos pueden habilitarse mediante algo como:

```bash
ruddy ... --debug
```

## 8.7. Dry-run / plan

Agregar una opción valiosa:

```bash
ruddy analyze run ... --dry-run
```

que muestre:

- bloques que se ejecutarán;
- columnas usadas;
- feature matrices;
- parámetros relevantes;
- outputs previstos;

sin ejecutar análisis pesados.

## 8.8. Configuración desde archivo

La CLI 2.0 debe soportar:

```bash
ruddy analyze run data.csv --config analysis.yaml
```

Los argumentos de CLI deben poder sobreescribir config de archivo de forma explícita y documentada.

## 8.9. Salida estructurada

Mantener outputs tabulares/JSON existentes, pero estandarizar estructura de directorios.

Propuesta:

```text
results/
├── manifest.json
├── config.resolved.yaml
├── dataset/
├── profiling/
├── univariate/
├── dependence/
├── groups/
├── factorial/
├── projections/
├── multivariate/
├── representation/
├── anomaly/
├── figures/
└── logs/
```

`manifest.json` debe indicar qué artifacts existen.

## 8.10. Criterios de aceptación CLI

La CLI 2.0 se considera aceptable cuando:

- todos los bloques científicos principales son accesibles;
- la interfaz de ayuda es clara;
- no existe lógica científica duplicada;
- soporta config files;
- tiene error handling de usuario;
- produce un summary final;
- mantiene reproducibilidad;
- puede iniciar la aplicación local;
- los outputs siguen siendo machine-readable.

---

# 9. Workstream D — Sistema de configuración reproducible

**Prioridad:** P0

`AnalysisConfig` debe mantenerse como fuente conceptual de verdad.

La capa de producto debe agregar una representación portable en YAML o JSON.

## 9.1. Archivo de configuración

Ejemplo:

```yaml
version: 1

input:
  id_column: id

roles:
  responses:
    - activity
  factors:
    - family
    - source
  covariates:
    - length

analysis:
  profiling:
    enabled: true
  univariate:
    enabled: true
  dependence:
    enabled: true
    n_permutations: 999
  pca:
    enabled: true
    scaling: standard
    n_components: 10

random_state: 42
```

## 9.2. Requisitos

- schema versionado;
- validación antes de ejecutar;
- error messages claros;
- posibilidad de guardar config resuelta;
- round-trip YAML → config → YAML;
- defaults visibles;
- ningún parámetro científico debe existir sólo en UI y no en config.

## 9.3. Presets

P1 puede agregar presets explícitos, nunca mágicos.

Ejemplos conceptuales:

```text
quick-eda
representation-comparison
factorial-analysis
publication-summary
```

El preset debe expandirse a config visible antes de ejecutar.

---

# 10. Workstream E — Capa de visualización reutilizable

**Prioridad:** P0

Los notebooks demostraron que Ruddy tiene suficiente información estructurada para producir figuras fuertes. Diego debe transformar esos prototipos en una capa reutilizable.

## 10.1. Ubicación arquitectónica

Dos opciones aceptables:

### Opción A

```text
src/ruddy/visualization/
```

como subpackage opcional, siempre que el core no lo importe.

### Opción B

```text
src/ruddy_visual/
```

como package separado dentro del mismo repositorio.

Preferencia inicial: **Opción A**, porque simplifica la instalación y el descubrimiento por usuario, pero debe mantenerse la separación de dependencias.

## 10.2. Diseño por figure factories

No crear funciones gigantes por notebook.

Diseñar funciones reutilizables:

```python
plot_distribution(...)
plot_group_distribution(...)
plot_correlation_heatmap(...)
plot_dependence_matrix(...)
plot_effect_forest(...)
plot_factorial_interaction(...)
plot_pca_scores(...)
plot_pca_loadings(...)
plot_permdisp(...)
plot_cka_matrix(...)
plot_cca_scores(...)
plot_procrustes(...)
plot_anomaly_scores(...)
```

Estas funciones deben recibir **result objects o tablas producidas por Ruddy**, no datos crudos para recalcular estadísticas.

## 10.3. Backends

Se recomienda evaluar una arquitectura híbrida:

- **Plotly** para interacción dentro de la app local;
- **Matplotlib** para salida estática altamente controlable;
- exportación mediante Kaleido u otro mecanismo compatible cuando corresponda.

No es obligatorio que cada figura tenga dos implementaciones. Diego debe definir un backend primario y uno secundario donde tenga sentido.

## 10.4. Figure specification

Sería útil crear un contrato intermedio como:

```python
FigureSpec(
    kind="group_distribution",
    title="Activity by family",
    x="family",
    y="activity",
    group="source",
    options={...},
)
```

Esto permitiría que CLI y app compartan configuración visual.

No debe diseñarse una DSL gigantesca; sólo abstraer opciones comunes.

---

# 11. Workstream F — Catálogo visual mínimo

**Prioridad:** P0

El catálogo se deriva de los notebooks fuertes ya creados.

## 11.1. Profiling / univariate

Implementar como mínimo:

- missingness bars/matrix;
- histogram;
- KDE;
- ECDF;
- boxplot;
- violin;
- box + raw points;
- distribución estratificada por grupo;
- categorical frequency plot.

## 11.2. Bivariate / dependence

- scatter plot;
- scatter por grupo;
- marginal distributions;
- trend overlay visual;
- Pearson/Spearman/Kendall heatmaps;
- distance-correlation heatmap;
- mutual-information heatmap;
- contingency residual heatmap;
- raw vs partial association comparison.

## 11.3. Groups / inference

- grouped violin/box;
- mean/median + interval;
- effect-size forest;
- pairwise-difference forest;
- Tukey matrix/forest;
- Games–Howell matrix/forest.

## 11.4. Factorial / EMM

- main-effect plot;
- interaction plot;
- EMM plot;
- EMM contrast forest;
- coefficient forest;
- residual diagnostics;
- leverage/Cook plot.

## 11.5. Projections

- scree;
- cumulative explained variance;
- PCA score plot;
- loadings;
- biplot;
- t-SNE;
- UMAP;
- coloring by categorical group;
- coloring by continuous variable;
- anomaly overlay.

## 11.6. Multivariate / PERMANOVA

- covariance heatmap;
- correlation heatmap;
- VIF plot;
- condition-index plot;
- Mahalanobis ranking;
- PERMANOVA space view;
- PERMDISP distance-to-centroid plot.

## 11.7. Representation comparison

- CKA matrix;
- CCA correlation spectrum;
- CCA score scatter;
- CCA loadings;
- pairwise-distance scatter;
- Mantel summary;
- Procrustes connected-space plot.

## 11.8. Compositional

- composition profile;
- transformed-component heatmap;
- variation matrix;
- CLR/ILR view;
- Aitchison distance distribution;
- Aitchison within/between comparison.

## 11.9. Bayesian

- posterior density;
- posterior difference;
- credible-interval forest;
- ROPE plot;
- probability-of-direction summary.

## 11.10. Anomaly/outlier

- univariate flagged-distribution plot;
- anomaly score distribution;
- anomaly ranking;
- method agreement matrix;
- anomaly comparison scatter;
- projection with flags.

---

# 12. Workstream G — Local web application

**Prioridad:** P0

La aplicación local debe convertirse en la interfaz visual principal de Ruddy.

## 12.1. Objetivo UX

Un usuario debería poder:

```text
abrir Ruddy
   ↓
cargar datos
   ↓
inspeccionar schema
   ↓
asignar roles
   ↓
agregar feature matrices
   ↓
seleccionar análisis
   ↓
configurar parámetros
   ↓
ejecutar
   ↓
explorar resultados
   ↓
construir figuras
   ↓
exportar
```

## 12.2. Lanzamiento

Objetivo:

```bash
ruddy app
```

La CLI debe:

1. encontrar un puerto libre;
2. iniciar el backend/UI local;
3. abrir opcionalmente el navegador;
4. mostrar la URL local;
5. cerrar correctamente con Ctrl+C.

No se requiere servidor externo.

## 12.3. Selección de framework

Antes de construir la app completa, Diego debe realizar un spike pequeño comparando al menos:

- Streamlit;
- Panel;
- NiceGUI;
- Dash.

Criterios de evaluación:

| Criterio | Importancia |
| --- | --- |
| Uso completamente local | crítica |
| Integración con Plotly/Matplotlib | crítica |
| Manejo de estado | crítica |
| Tablas grandes | alta |
| Upload de archivos | alta |
| Multi-page app | alta |
| Componentes custom | media/alta |
| Facilidad de empaquetado | alta |
| Tiempo de desarrollo | alta |
| Control visual | alta |
| Mantenimiento | alta |

El spike debe concluir con una recomendación breve y razonada antes de implementar.

## 12.4. Estructura recomendada de la app

```text
app/
├── state/
├── pages/
├── components/
├── controllers/
├── visualization/
└── io/
```

La app no debe convertirse en un segundo motor científico.

## 12.5. Página 1 — Load / Inspect

Debe permitir:

- CSV;
- TSV;
- Parquet;
- matrices `.npy`;
- sparse `.npz` cuando corresponda;
- preview;
- dimensions;
- missingness;
- inferred kinds;
- observation IDs.

## 12.6. Página 2 — Data roles

UI para asignar:

- identifier;
- variable;
- response;
- factor;
- covariate;
- annotation;
- excluded.

Mostrar inferencia automática como sugerencia editable.

## 12.7. Página 3 — Feature spaces

Permitir cargar una o más matrices numéricas y mostrar:

- nombre del espacio;
- shape;
- IDs;
- alignment coverage;
- missing/unmatched IDs;
- dense/sparse;
- provenance opcional.

## 12.8. Página 4 — Analysis builder

Debe reflejar directamente `AnalysisConfig`.

El usuario activa bloques mediante toggles/checks y configura parámetros mediante widgets.

La app debe distinguir visualmente:

- descriptive;
- inferential;
- permutations;
- projection;
- specialized.

Análisis pesados deben advertir costo aproximado de manera cualitativa:

```text
fast
moderate
potentially expensive
```

No estimar tiempos falsamente precisos.

## 12.9. Página 5 — Run / status

Durante ejecución mostrar:

- bloque actual;
- bloques completados;
- advisories;
- errors;
- exclusions;
- seed/config.

La app corre todo localmente en el proceso del usuario.

## 12.10. Página 6 — Results explorer

Debe organizar resultados por familia.

Ejemplo:

```text
Overview
Distributions
Associations
Groups
Factorial
Projections
Multivariate
Representations
Compositional
Bayesian
Anomalies
```

Cada sección debe tener:

- summary;
- tables;
- diagnostics;
- plot suggestions;
- export controls.

## 12.11. Página 7 — Figure builder

El usuario debe poder seleccionar un result table y construir una figura sin escribir código.

Controles comunes:

- x/y;
- grouping;
- color variable;
- facet;
- labels;
- title;
- legend;
- dimensions;
- font size;
- palette;
- confidence intervals;
- annotations.

No todos los controles aplican a todas las figuras.

## 12.12. Comparación entre representaciones

Esta debe ser una experiencia destacada.

UI para seleccionar:

```text
Space A
Space B
```

Mostrar juntos:

- alignment;
- dimensions;
- PCA summaries;
- CKA;
- CCA;
- distance similarity;
- Mantel;
- Procrustes cuando corresponda;
- group-separation statistics.

Esto es central para el Numerical Representation Framework.

---

# 13. Workstream H — Figure export

**Prioridad:** P0/P1

El usuario científico debe poder llevar una figura desde Ruddy al manuscrito sin tener que reconstruirla manualmente.

## 13.1. Formatos

Mínimo:

```text
PNG
SVG
PDF
```

P1 opcional:

```text
HTML interactivo
```

## 13.2. Parámetros

Permitir:

- width;
- height;
- DPI para raster;
- transparent background;
- font sizes;
- title on/off;
- legend position;
- labels;
- palette;
- marker size;
- line width.

## 13.3. Publication presets

P1 puede agregar presets de dimensiones, por ejemplo:

```text
single-column
one-and-half-column
double-column
presentation
```

No hace falta usar nombres de journals salvo que tengamos requerimientos reales verificados.

## 13.4. Reproducibilidad

Cada figura exportada debe poder asociarse con:

- analysis config;
- figure config;
- source result artifact;
- Ruddy version.

Idealmente guardar un sidecar:

```text
figure_001.svg
figure_001.json
```

---

# 14. Workstream I — Persistencia local y estructura de proyecto

**Prioridad:** P1

No necesitamos base de datos remota.

## 14.1. Ruddy project directory

Diseñar una estructura local como:

```text
my_analysis/
├── ruddy.yaml
├── data/
├── features/
├── results/
├── figures/
└── metadata/
```

Alternativamente una carpeta oculta `.ruddy/` puede almacenar estado de aplicación, pero los artifacts científicos deberían seguir siendo visibles y portables.

## 14.2. Guardar/reabrir sesión

La app local debería poder:

- guardar configuración;
- recordar inputs mediante paths relativos;
- recargar resultados existentes;
- evitar recalcular si el usuario sólo quiere volver a visualizar;
- invalidar cache cuando cambie input/config.

## 14.3. Cache

P1 puede incorporar cache local basada en hashes de:

```text
input data
feature matrix
analysis config
Ruddy version
```

El cache debe ser transparente y borrable.

---

# 15. Workstream J — Packaging y “deployment” local

**Prioridad:** P0

En este proyecto, “deployment” significa **distribución e instalación local reproducible**, no despliegue en cloud.

## 15.1. Build reproducible

Validar:

```bash
python -m build
```

Debe producir wheel y sdist instalables.

## 15.2. Wheel smoke test

Crear environment limpio e instalar el wheel:

```bash
pip install dist/ruddy-*.whl
```

Probar:

```bash
python -c "import ruddy"
ruddy --version
ruddy --help
```

Y, con extra app:

```bash
ruddy app
```

## 15.3. PyPI

Preparar publicación en PyPI cuando corresponda.

La primera entrega puede probarse en TestPyPI si se desea.

## 15.4. Conda

No es requisito P0 si el release por pip funciona correctamente.

P1/P2 puede incluir recipe conda-forge posteriormente.

## 15.5. Local app packaging

La app debe instalarse mediante extras del mismo paquete.

Objetivo:

```bash
pip install "ruddy[app]"
ruddy app
```

Evitar exigir que el usuario clone el repositorio sólo para utilizar la interfaz.

## 15.6. Assets

Cualquier CSS, templates o assets de app deben empacarse correctamente dentro del wheel usando package data.

---

# 16. Workstream K — Test suite 2.0

**Prioridad:** P0/P1

La productización requiere extender tests hacia superficies que no eran prioritarias durante desarrollo científico.

## 16.1. Mantener scientific tests

La suite científica actual es baseline inmutable salvo cambios justificados.

## 16.2. CLI tests

Agregar tests para:

- root help;
- command groups;
- invalid arguments;
- config files;
- output directory;
- error messages;
- `--debug`;
- `--dry-run`;
- exit codes;
- deterministic outputs;
- representative end-to-end commands.

No hace falta snapshot-testear cada carácter/color de Rich.

## 16.3. Config tests

- YAML parsing;
- schema validation;
- round-trip;
- CLI overrides;
- invalid keys;
- version compatibility;
- resolved config export.

## 16.4. Visualization tests

No testear “la figura es bonita”.

Testear:

- función acepta result esperado;
- devuelve objeto de figura válido;
- no recalcula ciencia;
- handles empty/degenerate results;
- export PNG/SVG/PDF funciona;
- figure config round-trip cuando exista.

## 16.5. App tests

Dependiendo del framework:

- load dataset;
- assign roles;
- build config;
- run simple analysis;
- render results;
- export artifact;
- handle error state;
- reload saved project.

No intentar automatizar todos los clicks desde el primer día. Priorizar workflows críticos.

## 16.6. Packaging tests

En CI:

```text
build wheel
install wheel in clean env
import ruddy
run CLI smoke
```

## 16.7. Python matrix

Usar las versiones oficialmente soportadas por `pyproject.toml`.

Actualmente:

```text
3.11
3.12
3.13
```

Si alguna dependencia opcional no soporta una versión, reflejarlo explícitamente en extras/CI en lugar de romper el core.

## 16.8. Optional dependency matrix

Testear por lo menos:

```text
core only
core + manifold
core + visual
core + app
```

## 16.9. Coverage

Coverage puede usarse como diagnóstico, no como objetivo ciego.

Prioridades de cobertura:

1. contracts;
2. config;
3. CLI routing;
4. visualization dispatch;
5. app controllers;
6. serialization/export.

No reescribir tests científicos útiles sólo para subir porcentaje.

---

# 17. Workstream L — CI/CD y calidad del repositorio

**Prioridad:** P1

Ahora sí corresponde a Diego implementar las herramientas de calidad que deliberadamente no priorizamos durante el desarrollo científico.

## 17.1. CI mínima

Por push/PR:

```text
install
compile/import
scientific tests
product tests
build package
wheel smoke test
```

## 17.2. Lint/format

Diego puede estandarizar:

- Ruff;
- formatting;
- import ordering;
- dead-code checks razonables.

Debe evitar una mega-reforma que dificulte revisar qué cambió científicamente.

Idealmente hacer formatting masivo en un commit aislado.

## 17.3. Typing

Reforzar typing principalmente en:

- public APIs;
- configs;
- result contracts;
- CLI/app boundaries;
- visualization APIs.

No bloquear release por intentar tipar internals de dependencias que no exponen buenos stubs.

## 17.4. Pre-commit

Opcional P1.

Puede incluir formatting/linting rápido, nunca la suite científica completa.

## 17.5. Release automation

P1:

- tag-based build;
- wheel/sdist;
- release notes;
- publicación manual/aprobada a PyPI.

Evitar automatización excesiva antes del primer release.

---

# 18. Workstream M — Result serialization y manifest

**Prioridad:** P0

La app, CLI y sesiones locales necesitan una forma uniforme de saber qué existe.

## 18.1. Analysis manifest

Diseñar un `manifest.json` por run.

Debe contener al menos:

```json
{
  "ruddy_version": "...",
  "created_at": "...",
  "analysis_blocks": [],
  "artifacts": {},
  "figures": {},
  "config_file": "config.resolved.yaml",
  "status": "completed"
}
```

## 18.2. Mantener formatos simples

Preferir:

- CSV/Parquet para tablas;
- JSON para metadata;
- YAML para config;
- PNG/SVG/PDF/HTML para figuras.

No crear un formato binario propietario de Ruddy sin una necesidad real.

## 18.3. Schema versioning

Manifest/config deben tener `schema_version`.

Esto permitirá cambios futuros sin romper proyectos guardados.

---

# 19. Workstream N — UX de advisories, degeneracies y errores

**Prioridad:** P0

Ruddy tiene una ventaja fuerte: `ok / degenerate / skipped`, razones y advisories explícitos.

La productización debe hacerlos visibles.

## 19.1. CLI

Ejemplo:

```text
Completed with 4 advisories

2 degenerate analyses
  activity × batch      constant response within level
  PC3                   zero variance

2 skipped analyses
  Mahalanobis           singular covariance
  CCA                    requested components exceed rank
```

## 19.2. App

No esconder resultados fallidos.

Debe existir una vista de diagnostics donde el usuario pueda inspeccionar:

- block;
- method;
- status;
- reason;
- affected variables;
- advisory text;
- exclusions.

## 19.3. Explicaciones

La UI puede traducir un `reason` técnico a una explicación más legible, pero debe conservar el código original.

---

# 20. Workstream O — Rendimiento y datasets grandes

**Prioridad:** P1

No rediseñar prematuramente todo el core, pero sí medir.

## 20.1. Benchmark suite pequeña

Crear datasets sintéticos reproducibles:

```text
1k × 20
10k × 50
100k × 50
5k × 512
5k × 1024
```

No todos los análisis aplican a todos.

Medir:

- load;
- profiling;
- correlations;
- PCA;
- anomaly;
- pairwise dependence;
- permutations.

## 20.2. Memory awareness

La app debe advertir cuando una acción puede ser problemática, por ejemplo matrices de distancia O(n²).

No densificar sparse matrices sin confirmación/acción explícita.

## 20.3. Cancellation

Si el framework local lo permite razonablemente, P1 debería permitir cancelar análisis costosos desde la app.

No es requisito para el primer CLI 2.0.

---

# 21. Workstream P — Documentación de producto

**Prioridad:** P1

La documentación científica existente no debe reemplazarse. Debe complementarse con documentación orientada a usuario.

Agregar posteriormente:

- Installation;
- CLI quickstart;
- config recipes;
- local app quickstart;
- figure export;
- troubleshooting;
- common workflows;
- migration/version notes;
- FAQ.

Los documentos científicos actuales siguen siendo referencia de métodos y policies.

---

# 22. Workstream Q — Integración con notebooks existentes

**Prioridad:** P1

Los notebooks de Phase 11 son prototipos visuales y demos científicas.

Diego debe reutilizarlos de dos maneras:

## 22.1. Como tests de necesidades visuales

Cada plot útil debe mapearse a una function/factory reutilizable.

Ejemplo:

```text
Notebook 07 CKA heatmap
        ↓
plot_cka_matrix(result)
```

## 22.2. Como documentación

Después de migrar visualizaciones a helpers públicos, simplificar notebooks para que usen esas APIs y no tengan cientos de líneas de plotting manual.

Objetivo final:

```python
from ruddy.visualization import plot_cka_matrix

fig = plot_cka_matrix(result)
```

Esto transforma los notebooks en ejemplos reales del producto final.

---

# 23. Arquitectura objetivo propuesta

La arquitectura final debería aproximarse a:

```text
ruddy/
│
├── core scientific modules
│   ├── data
│   ├── profiling
│   ├── univariate
│   ├── bivariate
│   ├── multivariate
│   ├── factorial
│   ├── statistics
│   ├── representation
│   └── ...
│
├── analysis
│   ├── AnalysisConfig
│   └── unified orchestration
│
├── visualization        # optional dependency boundary
│   ├── distributions
│   ├── associations
│   ├── groups
│   ├── factorial
│   ├── projections
│   ├── multivariate
│   ├── representation
│   ├── compositional
│   ├── bayesian
│   ├── anomaly
│   ├── themes
│   └── export
│
├── io
│   ├── config
│   ├── results
│   ├── manifests
│   └── projects
│
├── cli
│   └── Typer/Rich interface
│
└── app                  # optional local web app
    ├── state
    ├── pages
    ├── components
    └── controllers
```

La dirección correcta de importación es:

```text
app ──────┐
          ↓
cli → analysis/core
          ↑
visual ───┘
```

El core nunca debe depender de `app` o `visualization`.

---

# 24. Roadmap de implementación recomendado para Diego

## Milestone D0 — Handoff baseline

**Objetivo:** recibir y congelar el estado actual.

Entregables:

- limpieza de repo;
- baseline tests;
- tag/commit freeze;
- environment reproducible.

**Exit:** suite científica completa pasa.

---

## Milestone D1 — Package hardening

**Objetivo:** preparar estructura de producto.

Implementar:

- dependency extras;
- package data;
- build wheel/sdist;
- clean install tests;
- config serialization skeleton;
- result manifest skeleton.

**Exit:** wheel core instalable en environment limpio.

---

## Milestone D2 — CLI 2.0

**Objetivo:** reemplazar la CLI de desarrollo por una CLI usable.

Implementar:

- Typer/Rich o alternativa justificada;
- command hierarchy;
- help;
- error handling;
- config file;
- dry-run;
- summaries;
- progress;
- manifest de run.

**Exit:** workflows científicos principales funcionan sólo mediante CLI nueva.

---

## Milestone D3 — Visualization API

**Objetivo:** convertir plots de notebooks en funciones reutilizables.

Implementar primero P0:

- univariate/group distributions;
- scatter/dependence;
- heatmaps;
- factorial/EMM;
- PCA/projections;
- PERMANOVA/PERMDISP;
- representation comparison;
- anomaly;
- export.

**Exit:** notebooks principales pueden usar `ruddy.visualization` en vez de plotting manual.

---

## Milestone D4 — Local app MVP

**Objetivo:** primer flujo visual completo.

Debe permitir:

```text
load CSV
→ inspect
→ assign roles
→ choose blocks
→ run
→ inspect tables
→ generate plots
→ export
```

No necesita inicialmente soportar el 100% de las capabilities.

Priorizar:

- profiling;
- univariate;
- groups;
- bivariate;
- PCA;
- representation comparison.

**Exit:** usuario sin Python puede completar un análisis básico y exportar una figura.

---

## Milestone D5 — Local app full scientific coverage

Agregar:

- factorial;
- post-hoc;
- EMMs;
- multivariate;
- PERMANOVA/PERMDISP;
- compositional;
- Bayesian;
- anomaly;
- mixed effects.

**Exit:** todas las familias importantes tienen UI funcional o una explicación clara de por qué se mantienen sólo en API/CLI.

---

## Milestone D6 — Test/CI hardening

Implementar:

- CLI tests;
- config tests;
- visual export tests;
- app workflow tests;
- wheel smoke;
- Python matrix;
- extras matrix;
- lint/typing razonable.

**Exit:** PR CI confiable.

---

## Milestone D7 — Release candidate

Implementar:

- docs usuario;
- packaging final;
- app assets;
- clean install;
- example project;
- versioning;
- release notes.

**Exit:** release candidate instalable por un tercero sin conocimiento interno del proyecto.

---

# 25. Matriz de aceptación final

Ruddy puede considerarse “full” para este ciclo cuando cumpla al menos:

| Área | Criterio |
| --- | --- |
| Core | Scientific suite permanece verde |
| API | Backward-compatible dentro de lo razonable para pre-1.0 |
| CLI | Interfaz reorganizada, usable y visualmente clara |
| Config | YAML/JSON reproducible y validable |
| Visual | API reutilizable derivada de resultados Ruddy |
| App | Local, lanzable con un comando |
| App workflow | Load → roles → analyze → visualize → export |
| Representations | Comparación A/B disponible visualmente |
| Export | PNG/SVG/PDF |
| Results | Manifest + config resuelta |
| Packaging | Wheel/sdist limpio |
| Installation | Core y extras instalables separadamente |
| Tests | Core + CLI + config + visual + app + packaging |
| CI | Ejecuta gates esenciales automáticamente |
| Docs | Scientific docs + user docs |
| Identity | Sin acoplamiento a otras librerías |
| Reproducibility | Seed/config/artifacts preservados |

---

# 26. Qué NO debe hacer Diego en esta etapa

Evitar scope creep.

No es objetivo:

- agregar nuevas familias estadísticas sólo porque existen;
- reescribir el scientific core;
- crear un servicio cloud;
- implementar autenticación;
- construir una plataforma multiusuario;
- acoplar Ruddy directamente con Sylphy, Eris o Roxy;
- interpretar automáticamente resultados científicos;
- añadir clustering porque “queda bonito”;
- inventar decisiones estadísticas automáticas;
- crear un sistema de reporting narrativo con LLM;
- construir un framework de dashboards genérico;
- optimizar prematuramente todos los métodos para millones de filas.

Estas extensiones pueden evaluarse posteriormente si existe una necesidad concreta.

---

# 27. Criterios de diseño de la aplicación local

La app debe sentirse como **una interfaz a Ruddy**, no como otro producto independiente.

Prioridades UX:

1. claridad científica;
2. reproducibilidad;
3. transparencia de configuración;
4. visualizaciones útiles;
5. exportación;
6. buena experiencia local;
7. estética.

La estética importa, pero nunca por encima de mostrar correctamente:

- qué se analizó;
- cuántas observaciones entraron;
- qué fue excluido;
- qué método se ejecutó;
- qué parámetros se usaron;
- qué warnings existen.

---

# 28. Diseño mínimo del Figure Builder

Una figura no debería aparecer como una caja negra.

Cada figura debe tener tres grupos de configuración.

## Data

- result source;
- variable(s);
- group;
- filter;
- facet.

## Appearance

- title;
- labels;
- palette;
- marker/line sizes;
- legend;
- grid;
- dimensions.

## Statistical overlays

Sólo overlays ya derivados de Ruddy:

- CI;
- effect size;
- fitted values;
- EMMs;
- anomaly flags;
- centroids;
- thresholds;
- advisories.

La visualización puede incorporar transformaciones puramente gráficas, pero no nuevos tests estadísticos.

---

# 29. Example project para distribución

Diego debería entregar un pequeño proyecto completo incluido en `examples/`:

```text
examples/project_demo/
├── data.csv
├── features_a.csv
├── features_b.csv
├── analysis.yaml
├── README.md
└── expected/
```

El README debe enseñar:

```bash
ruddy analyze run data.csv --config analysis.yaml
ruddy app
```

Y mostrar qué outputs deberían aparecer.

Este example project puede reutilizar los datasets deterministas de los notebooks.

---

# 30. Relación con el futuro Application Note

La productización debe apoyar directamente la historia del framework sin meter esa historia dentro del core.

El flujo demostrable deseado es:

```text
Sylphy       Eris       Roxy
  │           │           │
  └───────────┼───────────┘
              ↓
        numerical spaces
              ↓
            Ruddy
              ↓
  EDA / statistics / comparison
              ↓
        visual exploration
              ↓
       publication figures
```

Para el manuscrito, será especialmente importante que la app/notebooks puedan mostrar de forma coherente:

- PCA de distintos espacios;
- separabilidad de grupos;
- PERMANOVA/PERMDISP;
- CKA;
- CCA;
- distance-space similarity;
- Mantel;
- Procrustes;
- anomalías;
- distribuciones de descriptores/features.

Esto no requiere que Ruddy sepa quién generó cada representación. Los espacios pueden etiquetarse externamente.

---

# 31. Definition of Done del handoff de Diego

El trabajo puede considerarse terminado cuando un usuario nuevo pueda realizar el siguiente flujo sin editar el código fuente:

```text
1. Crear un environment limpio
2. Instalar Ruddy
3. Ejecutar `ruddy --help`
4. Cargar un dataset por CLI o app
5. Definir roles
6. Cargar uno o dos feature spaces
7. Crear/validar una configuración
8. Ejecutar análisis
9. Revisar advisories/exclusions
10. Explorar resultados
11. Generar varias figuras
12. Personalizarlas
13. Exportar SVG/PDF/PNG
14. Guardar el proyecto/config
15. Reabrir resultados localmente
16. Reproducir el análisis con el mismo seed/config
```

Y, paralelamente:

```text
pytest complete suite       PASS
package build               PASS
wheel clean install         PASS
CLI smoke                   PASS
local app smoke             PASS
figure export               PASS
no scientific regressions   PASS
```

---

# 32. Prioridad operacional resumida para Diego

Si Diego necesita una única secuencia de trabajo, utilizar ésta:

```text
D0  Freeze/clean baseline
D1  Packaging + dependency extras + manifests/config
D2  CLI 2.0
D3  Visualization API
D4  Local app MVP
D5  Expand local app to advanced analyses
D6  Tests/CI/typing/lint hardening
D7  Documentation polish + packaging + release candidate
```

No comenzar por la app antes de tener estabilizados **config + result serialization + visualization API**, porque eso probablemente generaría lógica duplicada dentro de la UI.

---

# 33. Resultado esperado

Al finalizar esta productización, Ruddy debería existir en tres superficies coherentes:

```text
Python API
    │
    ├──────── CLI 2.0
    │
    └──────── Local Web App
```

Las tres deben compartir:

```text
TabularDataset / FeatureMatrix
AnalysisConfig
scientific core
result contracts
provenance
visualization layer
serialization/export
```

La diferencia debe ser sólo **cómo el usuario interactúa con Ruddy**, no qué matemática recibe.

El objetivo final no es convertir Ruddy en una plataforma gigantesca. Es llevar el núcleo científico ya desarrollado a un nivel de software donde pueda instalarse fácilmente, ejecutarse de forma reproducible, explorarse visualmente y producir outputs y figuras de calidad científica sin exigir que cada usuario tenga que programar su propia capa de análisis.

---

# 34. Nota final para el handoff

Diego recibe un core deliberadamente amplio y científicamente testeado. La mayor parte del valor de su siguiente etapa estará en **integración, consistencia, experiencia de usuario y confiabilidad de distribución**, no en agregar más métodos estadísticos.

Cuando exista duda sobre una implementación, el orden de prioridad debe ser:

```text
preservar matemática existente
        ↓
preservar reproducibilidad
        ↓
preservar contratos de resultados
        ↓
mejorar experiencia de usuario
        ↓
mejorar estética
```

Esa jerarquía debe gobernar toda la productización de Ruddy.
