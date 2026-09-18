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
    # 09 · Statistical outliers and multivariate anomaly detection

    Contrast univariate statistical rules, multivariate distance diagnostics and algorithmic anomaly scores. The goal is not to label bad data, but to see where different notions of extremeness agree or disagree.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import anomaly_agreement, configure_plots, display, distance_scatter, finish, load_feature_demo, load_tabular_demo, scatter_by_group, show
    configure_plots()

    from ruddy import analyze_outliers, analyze_anomalies, analyze_multivariate, analyze_pca
    frame,dataset=load_tabular_demo(); features=load_feature_demo('representation_a')
    out=analyze_outliers(dataset,include_flags=True); an=analyze_anomalies(features,methods=('isolation_forest','lof'),scaling='standard',random_state=42); mv=analyze_multivariate(features,scaling='standard',random_state=42); pca=analyze_pca(features,n_components=2,scaling='standard')
    return (
        an,
        anomaly_agreement,
        distance_scatter,
        finish,
        frame,
        mv,
        np,
        out,
        pca,
        pl,
        plt,
        scatter_by_group,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Univariate flagged observations in context
    """)
    return


@app.cell
def _(finish, frame, np, out, pl, plt, show):
    flagged = set(out.flags.filter(pl.col('column') == 'activity').get_column('observation_id').to_list())
    f = (
        frame.select(['id', 'activity', 'group'])
        .with_columns(flagged=pl.col('id').is_in(list(flagged)))
    )
    _fig, _ax = plt.subplots(figsize=(7.6, 5))
    for _label in [False, True]:
        _sub = f.filter(pl.col('flagged') == _label)
        if _sub.height > 0:
            _ax.scatter(np.arange(_sub.height), _sub.get_column('activity').to_numpy(), label=f'flagged={_label}', alpha=0.65, s=24)
    _ax.set_ylabel('Activity')
    _ax.set_title('Activity observations flagged by univariate rules')
    _ax.legend()
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Anomaly-score distributions
    """)
    return


@app.cell
def _(an, finish, pl, plt, show):
    _fig, _ax = plt.subplots(figsize=(7.6, 4.6))
    for method in an.scores.get_column('method').unique(maintain_order=True):
        _sub = an.scores.filter(pl.col('method') == method)
        _ax.hist(_sub.get_column('anomaly_score').to_numpy(), bins=25, alpha=0.45, label=method)
    _ax.set_xlabel('Anomaly score (higher = more anomalous)')
    _ax.set_ylabel('Count')
    _ax.set_title('Isolation Forest vs LOF score distributions')
    _ax.legend()
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method agreement
    """)
    return


@app.cell
def _(an, anomaly_agreement, show):
    anomaly_agreement(an.scores,title='Top observations: anomaly-method flag agreement'); show()
    return


@app.cell
def _(an, frame, pl, scatter_by_group, show):
    wide = (
        an.scores.pivot(index='observation_id', on='method', values='anomaly_score')
        .drop_nulls()
        .join(frame.select(['id', 'group']), left_on='observation_id', right_on='id', how='left')
    )
    scatter_by_group(wide, x='isolation_forest', y='lof', group='group', title='Isolation Forest vs LOF anomaly scores', fit_lines=False)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Anomalies overlaid on PCA
    """)
    return


@app.cell
def _(an, finish, pca, pl, plt, show):
    flags = (
        an.scores.group_by('observation_id')
        .agg(anomaly=pl.col('is_flagged').any())
    )
    merged = pca.scores.join(flags, on='observation_id', how='left')
    _fig, _ax = plt.subplots(figsize=(7.4, 5.2))
    for _label in [False, True]:
        _sub = merged.filter(pl.col('anomaly') == _label)
        if _sub.height > 0:
            _ax.scatter(_sub.get_column('PC1').to_numpy(), _sub.get_column('PC2').to_numpy(), label=f'anomaly={_label}', alpha=0.7, s=32 if _label else 22)
    _ax.set_xlabel('PC1')
    _ax.set_ylabel('PC2')
    _ax.set_title('Consensus anomaly flags in PCA space')
    _ax.legend()
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classical vs robust Mahalanobis
    """)
    return


@app.cell
def _(distance_scatter, mv, pl, show):
    md = (
        mv.mahalanobis.distances.filter(pl.col('status') == 'ok')
        .pivot(index='observation_id', on='method', values='squared_distance')
        .drop_nulls()
    )
    methods = [c for c in md.columns if c != 'observation_id']
    distance_scatter(
        md.get_column(methods[0]).to_numpy(),
        md.get_column(methods[-1]).to_numpy(),
        title='Classical vs robust Mahalanobis distance',
        xlabel=str(methods[0]),
        ylabel=str(methods[-1]),
    )
    show()
    return


if __name__ == "__main__":
    app.run()
