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
    # 11 · Numerical representation framework demo

    A single workflow compares three aligned numerical spaces with the same observations. Ruddy remains agnostic to how each representation was produced and focuses on geometry, group structure, similarity and anomalies.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import configure_plots, display, finish, load_feature_demo, load_tabular_demo, matrix_heatmap, show
    configure_plots()

    from ruddy import analyze_pca, analyze_permutation_group_structure, analyze_representation_similarity, analyze_anomalies
    frame,dataset=load_tabular_demo(); spaces={'Embedding-like':load_feature_demo('representation_a'),'Descriptor-like':load_feature_demo('representation_b'),'Structure-like':load_feature_demo('representation_c')}
    pcs={name:analyze_pca(fm,n_components=3,scaling='standard') for name,fm in spaces.items()}
    perms={name:analyze_permutation_group_structure(fm,dataset,factor='group',n_permutations=19,random_state=42) for name,fm in spaces.items()}
    anoms={name:analyze_anomalies(fm,methods=('isolation_forest','lof'),scaling='standard',random_state=42) for name,fm in spaces.items()}
    AB=analyze_representation_similarity(spaces['Embedding-like'],spaces['Descriptor-like'],cca_components=3,mantel_permutations=49,random_state=42)
    AC=analyze_representation_similarity(spaces['Embedding-like'],spaces['Structure-like'],cca_components=3,mantel_permutations=49,random_state=42)
    BC=analyze_representation_similarity(spaces['Descriptor-like'],spaces['Structure-like'],cca_components=3,mantel_permutations=49,random_state=42)
    return (
        AB,
        AC,
        BC,
        anoms,
        display,
        finish,
        frame,
        matrix_heatmap,
        np,
        pcs,
        perms,
        pl,
        plt,
        show,
        spaces,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Same observations, three numerical spaces
    """)
    return


@app.cell
def _(finish, frame, pcs, pl, plt, show):
    _fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for _ax, (_name, _res) in zip(axes, pcs.items()):
        m = _res.scores.join(frame.select(['id', 'group']), left_on='observation_id', right_on='id', how='left')
        for level in m.get_column('group').unique(maintain_order=True):
            _sub = m.filter(pl.col('group') == level)
            _ax.scatter(_sub.get_column('PC1').to_numpy(), _sub.get_column('PC2').to_numpy(), label=str(level), alpha=0.65, s=22)
        _ax.set_title(_name)
        _ax.set_xlabel('PC1')
        _ax.set_ylabel('PC2')
    axes[-1].legend(title='group')
    finish(_fig, title='Parallel PCA views of aligned representations')
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group separation strength across spaces
    """)
    return


@app.cell
def _(display, finish, perms, pl, plt, show):
    _rows = []
    for _name, _res in perms.items():
        p = _res.summary.filter(pl.col('analysis') == 'permanova')
        d = _res.summary.filter(pl.col('analysis') == 'permdisp')
        _rows.append({'space': _name, 'PERMANOVA_R2': p.item(0, 'r_squared'), 'PERMANOVA_p': p.item(0, 'p_value'), 'PERMDISP_p': d.item(0, 'p_value')})
    sep = pl.DataFrame(_rows)
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    _ax.bar(sep.get_column('space').to_list(), sep.get_column('PERMANOVA_R2').to_numpy())
    _ax.set_ylabel('R²')
    _ax.set_title('Group-separation effect size by representation')
    _ax.tick_params(axis='x', rotation=15)
    finish(_fig)
    show()
    display(sep)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-representation CKA matrix
    """)
    return


@app.cell
def _(AB, AC, BC, matrix_heatmap, np, pl, show, spaces):
    labels = list(spaces)
    mat = np.eye(3)
    pairvals = {
        (0, 1): AB.cka.item(0, 'cka'),
        (0, 2): AC.cka.item(0, 'cka'),
        (1, 2): BC.cka.item(0, 'cka'),
    }
    for (_i, _j), v in pairvals.items():
        mat[_i, _j] = mat[_j, _i] = v
    cka = pl.DataFrame({'space': labels, **{lbl: mat[:, j] for j, lbl in enumerate(labels)}})
    matrix_heatmap(cka, title='CKA similarity across representation families')
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geometry-preservation profile
    """)
    return


@app.cell
def _(AB, AC, BC, finish, np, pl, plt, show):
    sim = pl.DataFrame([
        {'pair': 'Embedding–Descriptor', 'CKA': AB.cka.item(0, 'cka'), 'Distance similarity': AB.distance_similarity.item(0, 'coefficient'), 'Mantel': AB.mantel.item(0, 'correlation')},
        {'pair': 'Embedding–Structure', 'CKA': AC.cka.item(0, 'cka'), 'Distance similarity': AC.distance_similarity.item(0, 'coefficient'), 'Mantel': AC.mantel.item(0, 'correlation')},
        {'pair': 'Descriptor–Structure', 'CKA': BC.cka.item(0, 'cka'), 'Distance similarity': BC.distance_similarity.item(0, 'coefficient'), 'Mantel': BC.mantel.item(0, 'correlation')},
    ])
    _fig, _ax = plt.subplots(figsize=(8, 4.6))
    _metrics = ['CKA', 'Distance similarity', 'Mantel']
    _x_pos = np.arange(sim.height)
    _width = 0.25
    for _i, _met in enumerate(_metrics):
        _ax.bar(_x_pos + (_i - 1) * _width, sim.get_column(_met).to_numpy(), _width, label=_met)
    _ax.set_xticks(_x_pos, sim.get_column('pair').to_list())
    _ax.set_ylim(-0.1, 1.05)
    _ax.set_title('Cross-space geometry similarity')
    _ax.tick_params(axis='x', rotation=15)
    _ax.legend()
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Anomaly burden across representation spaces
    """)
    return


@app.cell
def _(anoms, finish, np, pl, plt, show):
    _rows = []
    for _name, _res in anoms.items():
        for method in _res.scores.get_column('method').unique(maintain_order=True):
            _sub = _res.scores.filter(pl.col('method') == method)
            _rows.append({'space': _name, 'method': method, 'n_flagged': int(_sub.get_column('is_flagged').sum())})
    flag = pl.DataFrame(_rows)
    pivot = flag.pivot(index='space', on='method', values='n_flagged')
    _fig, _ax = plt.subplots(figsize=(7.5, 4.4))
    _methods = [c for c in pivot.columns if c != 'space']
    _x_pos = np.arange(pivot.height)
    _width = 0.35
    for _i, _met in enumerate(_methods):
        _ax.bar(_x_pos + (_i - 0.5) * _width, pivot.get_column(_met).to_numpy(), _width, label=_met)
    _ax.set_xticks(_x_pos, pivot.get_column('space').to_list())
    _ax.set_ylabel('Flagged observations')
    _ax.set_title('Anomaly detection depends on representation geometry')
    _ax.tick_params(axis='x', rotation=15)
    _ax.legend()
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive side-by-side representation explorer
    """)
    return


@app.cell
def _(frame, pcs, pl):
    import plotly.express as px
    long = []
    for _name, _res in pcs.items():
        t = (
            _res.scores.select(['observation_id', 'PC1', 'PC2'])
            .join(frame.select(['id', 'group', 'activity']), left_on='observation_id', right_on='id', how='left')
            .with_columns(space=pl.lit(_name))
        )
        long.append(t)
    long_df = pl.concat(long).to_pandas()
    _fig = px.scatter(long_df, x='PC1', y='PC2', color='group', facet_col='space', hover_data=['observation_id', 'activity'], title='Interactive aligned-representation explorer')
    _fig
    return


if __name__ == "__main__":
    app.run()
