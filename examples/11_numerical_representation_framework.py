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
    import pandas as pd
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
        pd,
        perms,
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
def _(finish, frame, pcs, plt, show):
    _fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for _ax, (_name, _res) in zip(axes, pcs.items()):
        m = _res.scores.merge(frame[['id', 'group']], left_on='observation_id', right_on='id')
        for level, _sub in m.groupby('group'):
            _ax.scatter(_sub.PC1, _sub.PC2, label=level, alpha=0.65, s=22)
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
def _(display, finish, pd, perms, plt, show):
    _rows = []
    for _name, _res in perms.items():
        p = _res.summary.query('analysis=="permanova"').iloc[0]
        d = _res.summary.query('analysis=="permdisp"').iloc[0]
        _rows.append({'space': _name, 'PERMANOVA_R2': p.r_squared, 'PERMANOVA_p': p.p_value, 'PERMDISP_p': d.p_value})
    sep = pd.DataFrame(_rows).set_index('space')
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    sep[['PERMANOVA_R2']].plot(kind='bar', ax=_ax, legend=False)
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
def _(AB, AC, BC, matrix_heatmap, np, pd, show, spaces):
    labels=list(spaces); cka=pd.DataFrame(np.eye(3),index=labels,columns=labels); pairvals={(0,1):AB.cka.iloc[0].cka,(0,2):AC.cka.iloc[0].cka,(1,2):BC.cka.iloc[0].cka};
    for (i,j),v in pairvals.items(): cka.iat[i,j]=cka.iat[j,i]=v
    matrix_heatmap(cka,title='CKA similarity across representation families'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Geometry-preservation profile
    """)
    return


@app.cell
def _(AB, AC, BC, finish, pd, plt, show):
    sim = pd.DataFrame([{'pair': 'Embedding–Descriptor', 'CKA': AB.cka.iloc[0].cka, 'Distance similarity': AB.distance_similarity.iloc[0].coefficient, 'Mantel': AB.mantel.iloc[0].correlation}, {'pair': 'Embedding–Structure', 'CKA': AC.cka.iloc[0].cka, 'Distance similarity': AC.distance_similarity.iloc[0].coefficient, 'Mantel': AC.mantel.iloc[0].correlation}, {'pair': 'Descriptor–Structure', 'CKA': BC.cka.iloc[0].cka, 'Distance similarity': BC.distance_similarity.iloc[0].coefficient, 'Mantel': BC.mantel.iloc[0].correlation}]).set_index('pair')
    _fig, _ax = plt.subplots(figsize=(8, 4.6))
    sim.plot(kind='bar', ax=_ax)
    _ax.set_ylim(-0.1, 1.05)
    _ax.set_title('Cross-space geometry similarity')
    _ax.tick_params(axis='x', rotation=15)
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
def _(anoms, finish, pd, plt, show):
    _rows = []
    for _name, _res in anoms.items():
        for method, _sub in _res.scores.groupby('method'):
            _rows.append({'space': _name, 'method': method, 'n_flagged': int(_sub.is_flagged.sum())})
    flag = pd.DataFrame(_rows)
    pivot = flag.pivot(index='space', columns='method', values='n_flagged')
    _fig, _ax = plt.subplots(figsize=(7.5, 4.4))
    pivot.plot(kind='bar', ax=_ax)
    _ax.set_ylabel('Flagged observations')
    _ax.set_title('Anomaly detection depends on representation geometry')
    _ax.tick_params(axis='x', rotation=15)
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
def _(frame, pcs, pd):
    import plotly.express as px
    long = []
    for _name, _res in pcs.items():
        t = _res.scores[['observation_id', 'PC1', 'PC2']].merge(frame[['id', 'group', 'activity']], left_on='observation_id', right_on='id')
        t['space'] = _name
        long.append(t)
    long = pd.concat(long, ignore_index=True)
    _fig = px.scatter(long, x='PC1', y='PC2', color='group', facet_col='space', hover_data=['observation_id', 'activity'], title='Interactive aligned-representation explorer')
    _fig
    return


if __name__ == "__main__":
    app.run()
