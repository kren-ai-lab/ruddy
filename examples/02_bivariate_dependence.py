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
    import polars as pl
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
        pl,
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
def _(biv, dep, display, finish, np, pl, plt, show):
    pairs = [('charge', 'nonlinear_response'), ('length', 'activity'), ('activity', 'stability')]
    rows = []
    for x, y in pairs:
        p = biv.correlations.filter((pl.col('column_x') == x) & (pl.col('column_y') == y) & (pl.col('method') == 'pearson'))
        d = dep.distance_correlations.filter((pl.col('column_x') == x) & (pl.col('column_y') == y))
        m = dep.mutual_information.filter((pl.col('column_x') == x) & (pl.col('column_y') == y))
        if p.height > 0 and d.height > 0 and m.height > 0:
            rows.append({'pair': f'{x} ↔ {y}', 'Pearson': p.item(0, 'coefficient'), 'Distance correlation': d.item(0, 'statistic'), 'Mutual information': m.item(0, 'statistic')})
    compare = pl.DataFrame(rows)
    _fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ['Pearson', 'Distance correlation', 'Mutual information']
    x_pos = np.arange(compare.height)
    width = 0.25
    for _i, met in enumerate(metrics):
        ax.bar(x_pos + (_i - 1) * width, compare.get_column(met).to_numpy(), width, label=met)
    ax.set_xticks(x_pos, compare.get_column('pair').to_list())
    ax.set_ylabel('Dependence statistic')
    ax.set_title('Different dependence measures reveal different structure')
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
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
def _(biv, matrix_heatmap, np, pl, show):
    numeric = ['activity', 'stability', 'length', 'charge', 'hydrophobicity', 'nonlinear_response']
    pear = biv.correlations.filter((pl.col('method') == 'pearson') & (pl.col('status') == 'ok'))
    _idx = {name: _i for _i, name in enumerate(numeric)}
    _mat = np.eye(len(numeric))
    for _r in pear.iter_rows(named=True):
        if _r['column_x'] in _idx and _r['column_y'] in _idx:
            _i, _j = _idx[_r['column_x']], _idx[_r['column_y']]
            _mat[_i, _j] = _mat[_j, _i] = _r['coefficient']
    matrix_heatmap(_mat, title='Pearson correlation structure', row_labels=numeric, col_labels=numeric)
    show()
    return (numeric,)


@app.cell
def _(dep, matrix_heatmap, np, numeric, pl, show):
    dc = dep.distance_correlations.filter(pl.col('status') == 'ok')
    _idx = {name: _i for _i, name in enumerate(numeric)}
    _mat = np.eye(len(numeric))
    for _r in dc.iter_rows(named=True):
        if _r['column_x'] in _idx and _r['column_y'] in _idx:
            _i, _j = _idx[_r['column_x']], _idx[_r['column_y']]
            _mat[_i, _j] = _mat[_j, _i] = _r['statistic']
    matrix_heatmap(_mat, title='Distance-correlation structure', row_labels=numeric, col_labels=numeric)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Partial correlation: controlling for length
    """)
    return


@app.cell
def _(bar_metric, dep, pl, show):
    pc = (
        dep.partial_correlations.filter(pl.col('status') == 'ok')
        .with_columns(pair=pl.concat_str([pl.col('column_x'), pl.lit(' ↔ '), pl.col('column_y')]))
    )
    top = (
        pc.with_columns(_abs=pl.col('coefficient').abs())
        .sort('_abs', descending=True)
        .head(12)
        .drop('_abs')
    )
    bar_metric(top, label='pair', value='coefficient', title='Strongest partial correlations | controlling for length', ylabel='Partial correlation')
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical association: which cells drive the result?
    """)
    return


@app.cell
def _(cont, display, matrix_heatmap, pl, show):
    cells = cont.cells.filter((pl.col('column_x') == 'group') & (pl.col('column_y') == 'phenotype'))
    res = cells.pivot(index='level_x', on='level_y', values='standardized_residual')
    matrix_heatmap(res, title='Standardized contingency residuals: group × phenotype')
    show()
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
    _fig = px.scatter(frame.to_pandas().replace([np.inf, -np.inf], np.nan), x='charge', y='nonlinear_response', color='group', hover_data=['id', 'source', 'activity'], title='Interactive nonlinear dependence explorer')
    _fig
    return


if __name__ == "__main__":
    app.run()
