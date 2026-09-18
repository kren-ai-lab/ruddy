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
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import configure_plots, display, finish, group_violin_box_scatter, load_feature_demo, load_tabular_demo, score_plot, show
    configure_plots()

    from ruddy import AnalysisConfig, analyze
    frame,dataset=load_tabular_demo(); A=load_feature_demo('representation_a'); B=load_feature_demo('representation_b')
    config=AnalysisConfig(enabled_blocks=('profiling','univariate','bivariate','groups','outliers','pca','permanova','representation','anomaly'),responses=('activity',),groups=('group',),permanova_factor='group',permanova_permutations=19,representation_cca_components=3,representation_mantel_permutations=19,projection_n_components=3,random_state=42)
    res=analyze(dataset,config=config,features=A,comparison_features=B)
    print('Executed blocks:',[str(x) for x in res.executed_blocks]); display(pl.DataFrame([res.profiling.overview]))
    return (
        display,
        finish,
        frame,
        group_violin_box_scatter,
        np,
        pl,
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
def _(finish, np, pl, plt, res, show):
    s = res.groups.numeric_summaries.filter((pl.col('response') == 'activity') & (pl.col('group_column') == 'group'))
    n = s.height
    _fig, _ax = plt.subplots(figsize=(7, 4.4))
    _ax.errorbar(s.get_column('mean').to_numpy(), np.arange(n), xerr=s.get_column('std').to_numpy(), fmt='o')
    _ax.set_yticks(np.arange(n), [str(x) for x in s.get_column('group_level').to_list()])
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
def _(finish, pl, plt, res, show):
    _fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    out_counts = (
        res.outliers.summaries.filter(pl.col('status') == 'ok')
        .group_by('method')
        .agg(n_flagged=pl.col('n_flagged').sum())
    )
    axes[0].bar(out_counts.get_column('method').to_list(), out_counts.get_column('n_flagged').to_numpy())
    axes[0].set_title('Univariate flags')
    axes[0].set_ylabel('Count')

    anom_counts = (
        res.anomaly.scores
        .group_by('method')
        .agg(n_flagged=pl.col('is_flagged').sum())
    )
    axes[1].bar(anom_counts.get_column('method').to_list(), anom_counts.get_column('n_flagged').to_numpy())
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
def _(display, finish, pl, plt, res, show):
    metrics = {
        'CKA': res.representation.cka.item(0, 'cka'),
        'Distance similarity': res.representation.distance_similarity.item(0, 'coefficient'),
        'Mantel': res.representation.mantel.item(0, 'correlation'),
    }
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.bar(list(metrics.keys()), list(metrics.values()))
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
def _(finish, plt, res, show):
    present = {name: int(value is not None) for name, value in res.components.items()}
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.bar(list(present.keys()), list(present.values()))
    _ax.set_ylim(0, 1.15)
    _ax.set_ylabel('Available')
    _ax.set_title('Analysis blocks materialized in UnifiedAnalysisResult')
    _ax.tick_params(axis='x', rotation=35)
    finish(_fig)
    show()
    return


if __name__ == "__main__":
    app.run()
