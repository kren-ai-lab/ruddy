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
    # 02 · Bivariate relationships and nonlinear dependence

    Compare linear, rank-based, partial and general dependence. Raw grouped scatters are shown beside Ruddy correlation, distance-correlation and mutual-information results.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from _helpers import bar_metric, configure_plots, display, finish, load_tabular_demo, matrix_heatmap, scatter_by_group, show
    configure_plots()

    from ruddy import analyze_bivariate, analyze_dependence, analyze_contingency_diagnostics
    frame, dataset = load_tabular_demo()
    biv = analyze_bivariate(dataset)
    dep = analyze_dependence(dataset, partial_covariates=('length',), n_permutations=19, random_state=42)
    cont = analyze_contingency_diagnostics(dataset, pairs=(('group','phenotype'),))
    return (
        bar_metric,
        biv,
        cont,
        dep,
        display,
        finish,
        frame,
        matrix_heatmap,
        np,
        pd,
        plt,
        scatter_by_group,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Strong grouped scatter: linear association
    """)
    return


@app.cell
def _(frame, scatter_by_group, show):
    scatter_by_group(frame, x='length', y='activity', group='group', title='Activity vs length with group-specific trends'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Nonlinear relationship missed by linear correlation
    """)
    return


@app.cell
def _(frame, scatter_by_group, show):
    scatter_by_group(frame, x='charge', y='nonlinear_response', group='group', title='Quadratic dependence: charge vs nonlinear response', fit_lines=False); show()
    return


@app.cell
def _(biv, dep, display, finish, pd, plt, show):
    pairs = [('charge', 'nonlinear_response'), ('length', 'activity'), ('activity', 'stability')]
    rows = []
    for x, y in pairs:
        p = biv.correlations.query('column_x==@x and column_y==@y and method=="pearson"')
        d = dep.distance_correlations.query('column_x==@x and column_y==@y')
        m = dep.mutual_information.query('column_x==@x and column_y==@y')
        if len(p) and len(d) and len(m):
            rows.append({'pair': f'{x} ↔ {y}', 'Pearson': p.iloc[0].coefficient, 'Distance correlation': d.iloc[0].statistic, 'Mutual information': m.iloc[0].statistic})
    compare = pd.DataFrame(rows).set_index('pair')
    _fig, ax = plt.subplots(figsize=(8, 4.5))
    compare.plot(kind='bar', ax=ax)
    ax.set_ylabel('Dependence statistic')
    ax.set_title('Different dependence measures reveal different structure')
    ax.tick_params(axis='x', rotation=20)
    finish(_fig)
    show()
    display(compare)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Global dependence matrices
    """)
    return


@app.cell
def _(biv, matrix_heatmap, np, pd, show):
    numeric = ['activity', 'stability', 'length', 'charge', 'hydrophobicity', 'nonlinear_response']
    pear = biv.correlations.query('method=="pearson" and status=="ok"')
    pm = pd.DataFrame(np.eye(len(numeric)), index=numeric, columns=numeric)
    for _, _r in pear.iterrows():
        if _r.column_x in numeric and _r.column_y in numeric:
            pm.loc[_r.column_x, _r.column_y] = pm.loc[_r.column_y, _r.column_x] = _r.coefficient
    matrix_heatmap(pm, title='Pearson correlation structure')
    show()
    return (numeric,)


@app.cell
def _(dep, matrix_heatmap, np, numeric, pd, show):
    dc = dep.distance_correlations.query('status=="ok"')
    dm = pd.DataFrame(np.eye(len(numeric)), index=numeric, columns=numeric)
    for _, _r in dc.iterrows():
        if _r.column_x in numeric and _r.column_y in numeric:
            dm.loc[_r.column_x, _r.column_y] = dm.loc[_r.column_y, _r.column_x] = _r.statistic
    matrix_heatmap(dm, title='Distance-correlation structure')
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partial correlation: controlling for length
    """)
    return


@app.cell
def _(bar_metric, dep, show):
    pc=dep.partial_correlations.query('status=="ok"').copy(); pc['pair']=pc.column_x+' ↔ '+pc.column_y
    top=pc.reindex(pc.coefficient.abs().sort_values(ascending=False).index).head(12)
    bar_metric(top,label='pair',value='coefficient',title='Strongest partial correlations | controlling for length',ylabel='Partial correlation'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical association: which cells drive the result?
    """)
    return


@app.cell
def _(cont, display, matrix_heatmap, show):
    cells=cont.cells.query('column_x=="group" and column_y=="phenotype"')
    res=cells.pivot(index='level_x',columns='level_y',values='standardized_residual')
    matrix_heatmap(res,title='Standardized contingency residuals: group × phenotype'); show()
    display(cont.summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive grouped scatter
    """)
    return


@app.cell
def _(frame, np):
    import plotly.express as px
    _fig = px.scatter(frame.replace([np.inf, -np.inf], np.nan), x='charge', y='nonlinear_response', color='group', hover_data=['id', 'source', 'activity'], title='Interactive nonlinear dependence explorer')
    _fig
    return


if __name__ == "__main__":
    app.run()
