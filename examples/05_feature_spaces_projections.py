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
    # 05 · Feature spaces, PCA and nonlinear projections

    Explore a high-dimensional numerical space from several angles: variance spectrum, group-aware PCA scores, loading structure, biplot and exploratory nonlinear projection.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import bar_metric, biplot, configure_plots, display, finish, load_feature_demo, load_tabular_demo, score_plot, show
    configure_plots()

    from ruddy import analyze_pca, analyze_tsne
    frame,dataset=load_tabular_demo(); features=load_feature_demo('representation_a')
    pca=analyze_pca(features,n_components=6,scaling='standard',random_state=42)
    tsne=analyze_tsne(features,n_components=2,scaling='standard',perplexity=25,random_state=42,max_iter=500)
    return (
        bar_metric,
        biplot,
        finish,
        frame,
        np,
        pca,
        pl,
        plt,
        score_plot,
        show,
        tsne,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scree and cumulative variance
    """)
    return


@app.cell
def _(finish, pca, plt, show):
    v = pca.variance
    _fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(v.get_column('component').to_list(), v.get_column('explained_variance_ratio').to_numpy(), label='Individual')
    ax.plot(v.get_column('component').to_list(), v.get_column('cumulative_explained_variance_ratio').to_numpy(), marker='o', label='Cumulative')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Explained variance ratio')
    ax.set_title('PCA variance spectrum')
    ax.legend()
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PCA scores by group and phenotype
    """)
    return


@app.cell
def _(frame, pca, score_plot, show):
    score_plot(pca.scores,frame,x='PC1',y='PC2',group='group',title='PCA score space by group',ellipses=True); show()
    return


@app.cell
def _(frame, pca, score_plot, show):
    score_plot(pca.scores,frame,x='PC1',y='PC3',group='phenotype',title='PC1–PC3 coloured by phenotype',ellipses=True); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loadings and biplot
    """)
    return


@app.cell
def _(bar_metric, np, pca, pl, show):
    load = (
        pca.loadings
        .with_columns(magnitude=(pl.col('PC1') ** 2 + pl.col('PC2') ** 2).sqrt())
        .sort('magnitude', descending=True)
        .head(10)
        .sort('magnitude')
    )
    bar_metric(load, label='feature', value='magnitude', title='Strongest PC1/PC2 loading vectors', ylabel='Loading magnitude')
    show()
    return


@app.cell
def _(biplot, frame, pca, show):
    biplot(pca.scores,pca.loadings,frame,group='group',top_n=8,title='PCA biplot: observations + strongest feature loadings'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory t-SNE
    Ruddy explicitly marks t-SNE coordinates as non-inferential.
    """)
    return


@app.cell
def _(frame, score_plot, show, tsne):
    coords = tsne.coordinates.rename({'component_1': 'TSNE1', 'component_2': 'TSNE2'})
    score_plot(coords, frame, x='TSNE1', y='TSNE2', group='group', title='Exploratory t-SNE by group')
    show()
    print('inferential_allowed =', tsne.inferential_allowed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive PCA explorer
    """)
    return


@app.cell
def _(frame, pca, pl):
    import plotly.express as px
    interactive = (
        pca.scores.join(frame.select(['id', 'group', 'source', 'phenotype', 'activity']), left_on='observation_id', right_on='id', how='left')
        .to_pandas()
    )
    _fig = px.scatter(interactive, x='PC1', y='PC2', color='group', symbol='source', hover_data=['observation_id', 'phenotype', 'activity'], title='Interactive PCA score explorer')
    _fig
    return


if __name__ == "__main__":
    app.run()
