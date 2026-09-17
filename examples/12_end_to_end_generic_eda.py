import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 12 · End-to-end domain-agnostic EDA

    A compact but rich workflow showing how an arbitrary tabular dataset plus aligned feature spaces can be inspected, compared and visualized without any domain-specific logic inside Ruddy.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from _helpers import configure_plots, display, finish, group_violin_box_scatter, load_feature_demo, load_tabular_demo, score_plot, show
    configure_plots()

    from ruddy import AnalysisConfig, analyze
    frame,dataset=load_tabular_demo(); A=load_feature_demo('representation_a'); B=load_feature_demo('representation_b')
    config=AnalysisConfig(enabled_blocks=('profiling','univariate','bivariate','groups','outliers','pca','permanova','representation','anomaly'),responses=('activity',),groups=('group',),permanova_factor='group',permanova_permutations=19,representation_cca_components=3,representation_mantel_permutations=19,projection_n_components=3,random_state=42)
    res=analyze(dataset,config=config,features=A,comparison_features=B)
    print('Executed blocks:',[str(x) for x in res.executed_blocks]); display(pd.DataFrame([res.profiling.overview]))
    return (
        display,
        finish,
        frame,
        group_violin_box_scatter,
        np,
        pd,
        plt,
        res,
        score_plot,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From raw group distributions to structured inference
    """)
    return


@app.cell
def _(frame, group_violin_box_scatter, show):
    group_violin_box_scatter(frame,value='activity',group='group',title='Raw response distribution by factor'); show()
    return


@app.cell
def _(finish, np, plt, res, show):
    s = res.groups.numeric_summaries.query('response=="activity" and group_column=="group"')
    _fig, _ax = plt.subplots(figsize=(7, 4.4))
    _ax.errorbar(s['mean'], np.arange(len(s)), xerr=s['std'], fmt='o')
    _ax.set_yticks(np.arange(len(s)), s.group_level.astype(str))
    _ax.set_xlabel('Mean ± SD')
    _ax.set_title('Structured group summaries')
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature-space organization
    """)
    return


@app.cell
def _(frame, res, score_plot, show):
    score_plot(res.pca.scores,frame,x='PC1',y='PC2',group='group',title='Unified-analysis PCA result',ellipses=True); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Statistical outliers vs representation anomalies
    """)
    return


@app.cell
def _(finish, plt, res, show):
    _fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    res.outliers.summaries.query('status=="ok"').groupby('method').n_flagged.sum().plot(kind='bar', ax=axes[0])
    axes[0].set_title('Univariate flags')
    axes[0].set_ylabel('Count')
    res.anomaly.scores.groupby('method').is_flagged.sum().plot(kind='bar', ax=axes[1])
    axes[1].set_title('Multivariate anomaly flags')
    axes[1].set_ylabel('Count')
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Representation similarity
    """)
    return


@app.cell
def _(display, finish, pd, plt, res, show):
    metrics = pd.Series({'CKA': res.representation.cka.iloc[0].cka, 'Distance similarity': res.representation.distance_similarity.iloc[0].coefficient, 'Mantel': res.representation.mantel.iloc[0].correlation})
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    metrics.plot(kind='bar', ax=_ax)
    _ax.set_ylim(-0.1, 1.05)
    _ax.set_title('Comparison of aligned numerical spaces')
    finish(_fig)
    show()
    display(res.permanova.summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unified result map
    """)
    return


@app.cell
def _(finish, pd, plt, res, show):
    present = pd.Series({name: value is not None for name, value in res.components.items()})
    _fig, _ax = plt.subplots(figsize=(8, 4))
    present.astype(int).plot(kind='bar', ax=_ax)
    _ax.set_ylim(0, 1.15)
    _ax.set_ylabel('Available')
    _ax.set_title('Analysis blocks materialized in UnifiedAnalysisResult')
    _ax.tick_params(axis='x', rotation=35)
    finish(_fig)
    show()
    return


if __name__ == "__main__":
    app.run()
