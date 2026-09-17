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
    # 08 · Compositional data analysis and Aitchison geometry

    Show why compositional variables require log-ratio geometry: inspect raw compositions, zero handling, variation structure, ILR coordinates and Aitchison distances across groups.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from _helpers import configure_plots, display, finish, labeled_boxplot, load_compositional_demo, load_tabular_demo, matrix_heatmap, score_plot, show, stacked_composition
    configure_plots()

    from ruddy import analyze_composition, analyze_pca
    frame,dataset=load_tabular_demo(); comp_frame,comp=load_compositional_demo(); ilr=analyze_composition(comp,transform='ilr',replace_zeros=True); clr=analyze_composition(comp,transform='clr',replace_zeros=True)
    return (
        analyze_pca,
        clr,
        comp_frame,
        display,
        finish,
        frame,
        ilr,
        labeled_boxplot,
        matrix_heatmap,
        np,
        pd,
        plt,
        score_plot,
        show,
        stacked_composition,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw composition profiles
    """)
    return


@app.cell
def _(comp_frame, show, stacked_composition):
    stacked_composition(comp_frame,n=36,title='Raw compositional profiles (first 36 observations)'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zero replacement diagnostics
    """)
    return


@app.cell
def _(display, finish, ilr, np, plt, show):
    zr = ilr.zero_replacement
    _fig, _ax = plt.subplots(figsize=(7.2, 4.2))
    _ax.hist(zr.zero_count, bins=np.arange(zr.zero_count.max() + 2) - 0.5)
    _ax.set_xlabel('Zeros per composition')
    _ax.set_ylabel('Observations')
    _ax.set_title('Explicit zero-replacement demand')
    finish(_fig)
    show()
    display(zr.query('replaced').head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variation matrix
    """)
    return


@app.cell
def _(ilr, matrix_heatmap, show):
    matrix_heatmap(ilr.variation_matrix,title='Compositional variation matrix'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CLR feature structure
    """)
    return


@app.cell
def _(clr, matrix_heatmap, np, pd, show):
    arr=clr.transformed.to_array(); corr=pd.DataFrame(np.corrcoef(arr,rowvar=False),index=clr.transformed.feature_names,columns=clr.transformed.feature_names); matrix_heatmap(corr,title='Correlation structure in CLR coordinates'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ILR coordinates and PCA
    """)
    return


@app.cell
def _(analyze_pca, frame, ilr, score_plot, show):
    pca=analyze_pca(ilr.transformed,n_components=3,scaling='none'); score_plot(pca.scores,frame,x='PC1',y='PC2',group='group',title='PCA of ILR coordinates by group',ellipses=True); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aitchison distances by group membership
    """)
    return


@app.cell
def _(finish, frame, ilr, labeled_boxplot, plt, show):
    D = ilr.aitchison_distances
    ids = list(D.index)
    groups = frame.set_index('id').loc[ids, 'group']
    within = []
    between = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (within if groups.iloc[i] == groups.iloc[j] else between).append(float(D.iloc[i, j]))
    _fig, _ax = plt.subplots(figsize=(7.4, 4.6))
    labeled_boxplot(_ax, [within, between], ['Within group', 'Between groups'])
    _ax.set_ylabel('Aitchison distance')
    _ax.set_title('Compositional geometry: within vs between groups')
    finish(_fig)
    show()
    return


if __name__ == "__main__":
    app.run()
