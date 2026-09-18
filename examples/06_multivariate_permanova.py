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
    # 06 · Multivariate diagnostics, PERMANOVA and dispersion

    Diagnose the geometry of a numerical space and then ask whether predefined groups differ in multivariate location and/or dispersion.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import bar_metric, configure_plots, display, group_violin_box_scatter, load_feature_demo, load_tabular_demo, matrix_heatmap, score_plot, show
    configure_plots()

    from ruddy import analyze_multivariate, analyze_permutation_group_structure, analyze_pca
    frame,dataset=load_tabular_demo(); features=load_feature_demo('representation_a')
    mv=analyze_multivariate(features,scaling='standard',random_state=42)
    perm=analyze_permutation_group_structure(features,dataset,factor='group',metric='euclidean',n_permutations=39,random_state=42)
    pca=analyze_pca(features,n_components=3,scaling='standard')
    return (
        bar_metric,
        display,
        frame,
        group_violin_box_scatter,
        matrix_heatmap,
        mv,
        pca,
        perm,
        pl,
        score_plot,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Covariance/correlation structure
    """)
    return


@app.cell
def _(matrix_heatmap, mv, show):
    matrix_heatmap(mv.covariance.pearson,title='Feature-space Pearson structure',annotate=False); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Collinearity diagnostics
    """)
    return


@app.cell
def _(display, mv, pl):
    _tbl = mv.collinearity.features if hasattr(mv.collinearity,'features') else mv.collinearity.table
    vif = _tbl.filter(pl.col('status') == 'ok')
    display(vif.head())
    return


@app.cell
def _(bar_metric, mv, pl, show):
    _tbl = mv.collinearity.features if hasattr(mv.collinearity,'features') else mv.collinearity.table
    if 'vif' in _tbl.columns:
        vv = _tbl.filter(pl.col('status') == 'ok').sort('vif', descending=True).head(12)
        bar_metric(vv, label='feature', value='vif', title='Variance inflation factors', ylabel='VIF')
        show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mahalanobis distance ranking
    """)
    return


@app.cell
def _(bar_metric, mv, pl, show):
    dist = (
        mv.mahalanobis.distances
        .filter((pl.col('method') == 'classical') & (pl.col('status') == 'ok'))
        .sort('squared_distance', descending=True)
        .head(25)
        .with_columns(label=pl.col('observation_id'))
        .sort('squared_distance')
    )
    bar_metric(dist, label='label', value='squared_distance', title='Largest classical Mahalanobis distances', ylabel='Squared distance')
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group location in PCA space
    """)
    return


@app.cell
def _(frame, pca, score_plot, show):
    score_plot(pca.scores,frame,x='PC1',y='PC2',group='group',title='Group separation in PCA space',ellipses=True); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PERMANOVA + PERMDISP together
    """)
    return


@app.cell
def _(display, frame, group_violin_box_scatter, perm, show):
    display(perm.summary)
    d = perm.distances_to_centroid.join(frame.select(['id', 'group']), left_on='observation_id', right_on='id', how='left')
    group_violin_box_scatter(d, value='distance_to_centroid', group='level', title='PERMDISP: distance to group centroid')
    show()
    return


@app.cell
def _(bar_metric, perm, show):
    g = perm.groups
    bar_metric(g, label='level', value='mean_distance_to_centroid', title='Mean multivariate dispersion by group', ylabel='Mean distance to centroid')
    show()
    return


if __name__ == "__main__":
    app.run()
