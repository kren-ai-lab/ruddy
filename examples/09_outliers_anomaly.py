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
    import pandas as pd
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
def _(finish, frame, np, out, plt, show):
    f = frame[['id', 'activity', 'group']].copy()
    flagged = set(out.flags.query('column=="activity"').observation_id)
    f['flagged'] = f.id.isin(flagged)
    _fig, _ax = plt.subplots(figsize=(7.6, 5))
    for _label, _sub in f.groupby('flagged'):
        _ax.scatter(np.arange(len(_sub)), _sub.activity, label=f'flagged={_label}', alpha=0.65, s=24)
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
def _(an, finish, plt, show):
    _fig, _ax = plt.subplots(figsize=(7.6, 4.6))
    for method, _sub in an.scores.groupby('method'):
        _ax.hist(_sub.anomaly_score, bins=25, alpha=0.45, label=method)
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
def _(an, frame, scatter_by_group, show):
    wide=an.scores.pivot(index='observation_id',columns='method',values='anomaly_score').dropna(); scatter_by_group(wide.reset_index().assign(group=frame.set_index('id').loc[wide.index,'group'].values),x='isolation_forest',y='lof',group='group',title='Isolation Forest vs LOF anomaly scores',fit_lines=False); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Anomalies overlaid on PCA
    """)
    return


@app.cell
def _(an, finish, pca, plt, show):
    flags = an.scores.groupby('observation_id').is_flagged.any().rename('anomaly').reset_index()
    merged = pca.scores.merge(flags, on='observation_id')
    _fig, _ax = plt.subplots(figsize=(7.4, 5.2))
    for _label, _sub in merged.groupby('anomaly'):
        _ax.scatter(_sub.PC1, _sub.PC2, label=f'anomaly={_label}', alpha=0.7, s=32 if _label else 22)
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
def _(distance_scatter, mv, show):
    md=mv.mahalanobis.distances.query('status=="ok"').pivot(index='observation_id',columns='method',values='squared_distance').dropna(); ifcols=list(md.columns); distance_scatter(md[ifcols[0]].to_numpy(),md[ifcols[-1]].to_numpy(),title='Classical vs robust Mahalanobis distance',xlabel=str(ifcols[0]),ylabel=str(ifcols[-1])); show()
    return


if __name__ == "__main__":
    app.run()
