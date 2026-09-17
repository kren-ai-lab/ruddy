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
    import pandas as pd
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
        pd,
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
def _(bar_metric, display, fac, show):
    effects=fac.effects.query('status=="ok"').copy(); bar_metric(effects,label='term',value='partial_eta_squared',title='Factorial effects ranked by partial η²',ylabel='Partial η²'); show(); display(effects[['term','f_value','p_value','partial_eta_squared','omega_squared']])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw interaction view
    """)
    return


@app.cell
def _(finish, frame, np, plt, show):
    means = frame.replace([np.inf, -np.inf], np.nan).groupby(['group', 'source'], as_index=False).activity.mean()
    _fig, _ax = plt.subplots(figsize=(7.5, 4.8))
    for _src, _sub in means.groupby('source'):
        _ax.plot(_sub.group, _sub.activity, marker='o', label=_src, linewidth=2)
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
def _(emm, finish, json, plt, show):
    joint = emm.means.query('term=="group:source" and status=="ok"').copy()
    joint['levels'] = joint.levels_json.map(json.loads)
    joint['group'] = joint.levels.map(lambda x: x['group'])
    joint['source'] = joint.levels.map(lambda x: x['source'])
    _fig, _ax = plt.subplots(figsize=(7.5, 4.8))
    for _src, _sub in joint.groupby('source'):
        _ax.errorbar(_sub.group, _sub.estimate, yerr=[_sub.estimate - _sub.ci_lower, _sub.ci_upper - _sub.estimate], marker='o', capsize=3, label=_src)
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
def _(emm, errorbar_table, show):
    c=emm.contrasts.query('term=="group" and status=="ok"').copy(); c['contrast']=c.levels_a_json+' vs '+c.levels_b_json
    errorbar_table(c,label_col='contrast',estimate_col='estimate_difference',low_col='ci_lower',high_col='ci_upper',title='Adjusted group contrasts',xlabel='EMM difference'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Residual and influence diagnostics
    """)
    return


@app.cell
def _(fac, finish, plt, show):
    od = fac.observation_diagnostics.query('status=="ok"')
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _ax.scatter(od.fitted_value, od.studentized_residual, s=20, alpha=0.65)
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
    _ax.stem(np.arange(len(od)), od.cooks_distance, markerfmt='.', basefmt=' ')
    _ax.axhline(float(od.cooks_distance_threshold.iloc[0]), linestyle='--')
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
def _(bar_metric, display, mix, pd, show):
    re=mix.random_effects.query('status=="ok"').sort_values('estimate'); bar_metric(re,label='group',value='estimate',title='Random-intercept estimates by batch',ylabel='Conditional random effect'); show(); display(pd.DataFrame([mix.model_summary]))
    return


if __name__ == "__main__":
    app.run()
