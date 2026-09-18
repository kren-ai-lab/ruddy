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
    # 13 · Visualization gallery for the future local app

    A compact gallery showing the plot families that structured Ruddy outputs can support. This notebook is intentionally presentation-oriented and acts as a visual requirements reference for later product design.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import anomaly_agreement, bar_metric, configure_plots, display, ecdf_by_group, finish, group_violin_box_scatter, load_feature_demo, load_tabular_demo, matrix_heatmap, scatter_by_group, score_plot, show
    configure_plots()

    from ruddy import analyze_bivariate, analyze_groups, analyze_factorial, analyze_pca, analyze_anomalies, analyze_representation_similarity
    frame,dataset=load_tabular_demo(); A=load_feature_demo('representation_a'); B=load_feature_demo('representation_b')
    biv=analyze_bivariate(dataset); grp=analyze_groups(dataset,responses=('activity',),groups=('group',)); fac=analyze_factorial(dataset,response='activity',factors=('group','source'),interactions=(('group','source'),)); pca=analyze_pca(A,n_components=3,scaling='standard'); an=analyze_anomalies(A,methods=('isolation_forest','lof'),random_state=42); rep=analyze_representation_similarity(A,B,cca_components=3,mantel_permutations=19,random_state=42)
    return (
        an,
        anomaly_agreement,
        bar_metric,
        biv,
        ecdf_by_group,
        fac,
        finish,
        frame,
        group_violin_box_scatter,
        grp,
        matrix_heatmap,
        np,
        pca,
        pl,
        plt,
        rep,
        scatter_by_group,
        score_plot,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distribution family
    """)
    return


@app.cell
def _(frame, group_violin_box_scatter, show):
    group_violin_box_scatter(frame,value='activity',group='group',title='Violin + box + observations'); show()
    return


@app.cell
def _(ecdf_by_group, frame, show):
    ecdf_by_group(frame,value='activity',group='group',title='Group-aware ECDF'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Relationship family
    """)
    return


@app.cell
def _(frame, scatter_by_group, show):
    scatter_by_group(frame,x='length',y='activity',group='group',title='Grouped scatter + trend + centroid'); show()
    return


@app.cell
def _(biv, matrix_heatmap, np, pl, show):
    num = ['activity', 'stability', 'length', 'charge']
    pear = biv.correlations.filter((pl.col('method') == 'pearson') & (pl.col('status') == 'ok'))
    idx = {name: i for i, name in enumerate(num)}
    mat = np.eye(len(num))
    for _r in pear.iter_rows(named=True):
        if _r['column_x'] in idx and _r['column_y'] in idx:
            i, j = idx[_r['column_x']], idx[_r['column_y']]
            mat[i, j] = mat[j, i] = _r['coefficient']
    matrix_heatmap(mat, title='Correlation heatmap', row_labels=num, col_labels=num)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group and factorial family
    """)
    return


@app.cell
def _(finish, grp, np, pl, plt, show):
    s = grp.numeric_summaries.filter(pl.col('response') == 'activity')
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.errorbar(s.get_column('mean').to_numpy(), np.arange(s.height), xerr=s.get_column('std').to_numpy(), fmt='o')
    _ax.set_yticks(np.arange(s.height), [str(x) for x in s.get_column('group_level').to_list()])
    _ax.set_title('Group estimate plot')
    finish(_fig)
    show()
    return


@app.cell
def _(bar_metric, fac, pl, show):
    e = fac.effects.filter(pl.col('status') == 'ok')
    bar_metric(e, label='term', value='partial_eta_squared', title='Factorial effect-size plot', ylabel='Partial η²')
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Projection and representation family
    """)
    return


@app.cell
def _(frame, pca, score_plot, show):
    score_plot(pca.scores,frame,x='PC1',y='PC2',group='group',title='Projection scatter with group ellipses',ellipses=True); show()
    return


@app.cell
def _(finish, plt, rep, show):
    labels = ['CKA', 'Distance similarity', 'Mantel']
    vals = [
        rep.cka.item(0, 'cka'),
        rep.distance_similarity.item(0, 'coefficient'),
        rep.mantel.item(0, 'correlation'),
    ]
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.bar(labels, vals)
    _ax.set_ylim(-0.1, 1.05)
    _ax.set_title('Representation similarity metrics')
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Anomaly family
    """)
    return


@app.cell
def _(an, anomaly_agreement, show):
    anomaly_agreement(an.scores,title='Anomaly agreement matrix'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visual contract
    These figures are downstream views of structured results. A future local application can reproduce the same plot families without adding visualization dependencies to the Ruddy core.
    """)
    return


if __name__ == "__main__":
    app.run()
