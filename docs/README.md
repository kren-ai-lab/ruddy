# Ruddy documentation

Start with the [quickstart](../README.md) to install Ruddy and run an analysis.
Use the guides below for input requirements, method choices and result interpretation.

| I want to… | Read |
| --- | --- |
| Load a table or feature matrix and align observations | [Data contracts](DATA_CONTRACTS.md) |
| Choose an analysis and understand its assumptions and limits | [Methods](METHODS.md) |
| Read, filter and export results; inspect exclusions and provenance | [Results](RESULTS_AND_PROVENANCE.md) |
| Run several analyses with one configuration | [Unified analysis](UNIFIED_ANALYSIS.md) |
| Run Ruddy from the terminal | [CLI reference](CLI_REFERENCE.md) |
| Explore complete workflows and plots | [Example notebooks](../examples/README.md) |
| Set up a checkout, change code and run checks | [Development guide](../DEVELOPMENT.md) |

## Python API help

Import public analysis functions and data contracts from `ruddy`. Use Python
help for the signatures and defaults of your installed version:

```python
import ruddy

help(ruddy.analyze_bivariate)
help(ruddy.AnalysisConfig)
```

The method guide links the main entry points to their scientific behavior.
Result tables expose their column names and types through `.schema`; see the
[results guide](RESULTS_AND_PROVENANCE.md#inspect-and-export-tables) for a working example.
