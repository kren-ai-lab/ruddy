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
    # 07 · Representation-space comparison

    Compare multiple numerical representations of the same observations using CKA, CCA, distance-space similarity, Mantel and Procrustes. This notebook is a direct blueprint for representation-comparison views in the future local application.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import bar_metric, configure_plots, display, distance_scatter, finish, load_feature_demo, load_tabular_demo, matrix_heatmap, procrustes_segments, scatter_by_group, show
    configure_plots()

    from ruddy import analyze_representation_similarity, analyze_pca
    from scipy.spatial.distance import pdist
    from scipy.spatial import procrustes
    frame,dataset=load_tabular_demo(); A=load_feature_demo('representation_a'); B=load_feature_demo('representation_b'); C=load_feature_demo('representation_c')
    AB=analyze_representation_similarity(A,B,cca_components=4,mantel_permutations=19,random_state=42)
    AC=analyze_representation_similarity(A,C,cca_components=4,mantel_permutations=19,random_state=42)
    BC=analyze_representation_similarity(B,C,cca_components=4,mantel_permutations=19,random_state=42)
    return (
        A,
        AB,
        AC,
        B,
        BC,
        C,
        bar_metric,
        display,
        distance_scatter,
        finish,
        frame,
        matrix_heatmap,
        np,
        pdist,
        pl,
        plt,
        procrustes,
        procrustes_segments,
        scatter_by_group,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pairwise representation similarity matrix (CKA)
    """)
    return


@app.cell
def _(AB, AC, BC, display, matrix_heatmap, np, pl, show):
    labels = ['Space A', 'Space B', 'Space C']
    mat = np.eye(3)
    vals = {
        (0, 1): AB.cka.item(0, 'cka'),
        (0, 2): AC.cka.item(0, 'cka'),
        (1, 2): BC.cka.item(0, 'cka'),
    }
    for (_i, _j), _v in vals.items():
        mat[_i, _j] = mat[_j, _i] = _v
    cka = pl.DataFrame({'representation': labels, **{lbl: mat[:, j] for j, lbl in enumerate(labels)}})
    matrix_heatmap(cka, title='Linear CKA across numerical representations')
    show()
    display(cka)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Canonical correlation spectrum
    """)
    return


@app.cell
def _(AB, bar_metric, pl, show):
    corr = AB.cca.correlations.filter(pl.col('status') == 'ok')
    bar_metric(corr, label='component', value='canonical_correlation', title='CCA spectrum: Space A ↔ Space B', ylabel='Canonical correlation')
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Canonical scores coloured by group
    """)
    return


@app.cell
def _(AB, frame, pl, scatter_by_group, show):
    xs = AB.cca.x_scores.rename({'CC1': 'X1', 'CC2': 'X2'})
    ys = AB.cca.y_scores.rename({'CC1': 'Y1', 'CC2': 'Y2'})
    joint = (
        xs.select(['observation_id', 'X1'])
        .join(ys.select(['observation_id', 'Y1']), on='observation_id')
        .join(frame.select(['id', 'group']), left_on='observation_id', right_on='id', how='left')
    )
    scatter_by_group(joint, x='X1', y='Y1', group='group', title='First canonical variates: X score vs Y score')
    show()
    return (joint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distance-space comparison
    """)
    return


@app.cell
def _(A, AB, B, display, distance_scatter, pdist, show):
    xa=A.to_array(); xb=B.to_array(); da=pdist(xa); db=pdist(xb); distance_scatter(da,db,title='Pairwise geometry: Space A vs Space B',xlabel='Distances in A',ylabel='Distances in B'); show(); display(AB.distance_similarity); display(AB.mantel)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Procrustes for same-dimensional related spaces
    """)
    return


@app.cell
def _(A, AC, C, display, procrustes, procrustes_segments, show):
    a=A.to_array(); c=C.to_array(); m1,m2,disp=procrustes(a,c); procrustes_segments(m1[:60,:2],m2[:60,:2],title=f'Procrustes A ↔ C | disparity={disp:.3f}'); show(); display(AC.procrustes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare all metrics across representation pairs
    """)
    return


@app.cell
def _(AB, AC, BC, display, finish, np, pl, plt, show):
    summary = pl.DataFrame([
        {'pair': 'A-B', 'CKA': AB.cka.item(0, 'cka'), 'distance_similarity': AB.distance_similarity.item(0, 'coefficient'), 'Mantel': AB.mantel.item(0, 'correlation')},
        {'pair': 'A-C', 'CKA': AC.cka.item(0, 'cka'), 'distance_similarity': AC.distance_similarity.item(0, 'coefficient'), 'Mantel': AC.mantel.item(0, 'correlation')},
        {'pair': 'B-C', 'CKA': BC.cka.item(0, 'cka'), 'distance_similarity': BC.distance_similarity.item(0, 'coefficient'), 'Mantel': BC.mantel.item(0, 'correlation')},
    ])
    _fig, ax = plt.subplots(figsize=(7.5, 4.5))
    metrics = ['CKA', 'distance_similarity', 'Mantel']
    x_pos = np.arange(summary.height)
    width = 0.25
    for _i, met in enumerate(metrics):
        ax.bar(x_pos + (_i - 1) * width, summary.get_column(met).to_numpy(), width, label=met)
    ax.set_xticks(x_pos, summary.get_column('pair').to_list())
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel('Similarity')
    ax.set_title('Representation-pair similarity profile')
    ax.legend()
    finish(_fig)
    show()
    display(summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive canonical-score explorer
    """)
    return


@app.cell
def _(joint):
    import plotly.express as px
    _fig = px.scatter(joint.to_pandas(), x='X1', y='Y1', color='group', hover_name='observation_id', title='Interactive CCA score alignment')
    _fig
    return


if __name__ == "__main__":
    app.run()
