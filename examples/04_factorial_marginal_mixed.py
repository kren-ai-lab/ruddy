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
    # 04 · Factorial effects, interactions, marginal means and mixed effects

    Demonstrate how Ruddy separates global factorial effects, adjusted means, pairwise contrasts, model diagnostics and hierarchical random effects.
    """)
    return


@app.cell
def _():
    import json
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import bar_metric, configure_plots, display, errorbar_table, finish, load_mixed_demo, load_tabular_demo, show
    configure_plots()

    from ruddy import analyze_factorial, analyze_marginal_means, analyze_mixed_effects
    frame,dataset=load_tabular_demo(); mf,mixed_ds=load_mixed_demo()
    fac=analyze_factorial(dataset,response='activity',factors=('group','source'),covariates=('length',),interactions=(('group','source'),),ss_type=3)
    emm=analyze_marginal_means(dataset,response='activity',factors=('group','source'),covariates=('length',),interactions=(('group','source'),),terms=('group',('group','source')))
    mix=analyze_mixed_effects(mixed_ds,group='batch',response='y',factors=('condition',),covariates=('x',))
    return (
        bar_metric,
        display,
        emm,
        errorbar_table,
        fac,
        finish,
        frame,
        json,
        mix,
        np,
        pl,
        plt,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Factorial effect sizes
    """)
    return


@app.cell
def _(bar_metric, display, fac, pl, show):
    effects = fac.effects.filter(pl.col('status') == 'ok')
    bar_metric(effects, label='term', value='partial_eta_squared', title='Factorial effects ranked by partial η²', ylabel='Partial η²')
    show()
    display(effects.select(['term', 'f_value', 'p_value', 'partial_eta_squared', 'omega_squared']))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw interaction view
    """)
    return


@app.cell
def _(finish, frame, np, pl, plt, show):
    means = (
        frame.filter(pl.col('activity').is_finite() & pl.col('activity').is_not_null())
        .group_by(['group', 'source'])
        .agg(pl.col('activity').mean())
        .sort(['source', 'group'])
    )
    _fig, _ax = plt.subplots(figsize=(7.5, 4.8))
    for _src in means.get_column('source').unique(maintain_order=True):
        _sub = means.filter(pl.col('source') == _src)
        _ax.plot(_sub.get_column('group').to_list(), _sub.get_column('activity').to_numpy(), marker='o', label=_src, linewidth=2)
    _ax.set_ylabel('Observed mean activity')
    _ax.set_title('Observed interaction: group × source')
    _ax.legend(title='source')
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimated marginal means and adjusted interaction
    """)
    return


@app.cell
def _(emm, finish, json, pl, plt, show):
    joint_raw = emm.means.filter((pl.col('term') == 'group:source') & (pl.col('status') == 'ok'))
    rows = []
    for r in joint_raw.iter_rows(named=True):
        lvl = json.loads(r['levels_json'])
        rows.append({
            'group': lvl['group'],
            'source': lvl['source'],
            'estimate': r['estimate'],
            'ci_lower': r['ci_lower'],
            'ci_upper': r['ci_upper'],
        })
    joint = pl.DataFrame(rows).sort(['source', 'group'])
    _fig, _ax = plt.subplots(figsize=(7.5, 4.8))
    for _src in joint.get_column('source').unique(maintain_order=True):
        _sub = joint.filter(pl.col('source') == _src)
        est = _sub.get_column('estimate').to_numpy()
        low = _sub.get_column('ci_lower').to_numpy()
        high = _sub.get_column('ci_upper').to_numpy()
        _ax.errorbar(_sub.get_column('group').to_list(), est, yerr=[est - low, high - est], marker='o', capsize=3, label=_src)
    _ax.set_ylabel('Estimated marginal mean')
    _ax.set_title('Adjusted interaction from Ruddy EMMs')
    _ax.legend(title='source')
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pairwise EMM contrasts
    """)
    return


@app.cell
def _(emm, errorbar_table, pl, show):
    c = (
        emm.contrasts.filter((pl.col('term') == 'group') & (pl.col('status') == 'ok'))
        .with_columns(contrast=pl.concat_str([pl.col('levels_a_json'), pl.lit(' vs '), pl.col('levels_b_json')]))
    )
    errorbar_table(c, label_col='contrast', estimate_col='estimate_difference', low_col='ci_lower', high_col='ci_upper', title='Adjusted group contrasts', xlabel='EMM difference')
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Residual and influence diagnostics
    """)
    return


@app.cell
def _(fac, finish, pl, plt, show):
    od = fac.observation_diagnostics.filter(pl.col('status') == 'ok')
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _ax.scatter(od.get_column('fitted_value').to_numpy(), od.get_column('studentized_residual').to_numpy(), s=20, alpha=0.65)
    _ax.axhline(0, linewidth=1)
    _ax.axhline(3, linestyle='--', linewidth=1)
    _ax.axhline(-3, linestyle='--', linewidth=1)
    _ax.set_xlabel('Fitted')
    _ax.set_ylabel('Studentized residual')
    _ax.set_title('Factorial residual diagnostics')
    finish(_fig)
    show()
    return (od,)


@app.cell
def _(finish, np, od, plt, show):
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.stem(np.arange(od.height), od.get_column('cooks_distance').to_numpy(), markerfmt='.', basefmt=' ')
    _ax.axhline(float(od.item(0, 'cooks_distance_threshold')), linestyle='--')
    _ax.set_xlabel('Observation')
    _ax.set_ylabel('Cook distance')
    _ax.set_title('Influence diagnostics')
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mixed-effects structure
    """)
    return


@app.cell
def _(bar_metric, display, mix, pl, show):
    re = mix.random_effects.filter(pl.col('status') == 'ok').sort('estimate')
    bar_metric(re, label='group', value='estimate', title='Random-intercept estimates by batch', ylabel='Conditional random effect')
    show()
    display(pl.DataFrame([mix.model_summary]))
    return


if __name__ == "__main__":
    app.run()
