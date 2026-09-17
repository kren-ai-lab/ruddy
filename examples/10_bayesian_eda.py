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
    # 10 · Bayesian EDA and uncertainty summaries

    Use Ruddy Bayesian summaries to visualize posterior location, uncertainty, direction and practical-equivalence probabilities without introducing an external Bayesian modeling framework.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from _helpers import configure_plots, display, errorbar_table, finish, load_tabular_demo, show
    configure_plots()

    from ruddy import analyze_bayesian_eda
    frame,dataset=load_tabular_demo()
    # binary grouping for mean differences
    bay=analyze_bayesian_eda(dataset,variables=('activity','stability','length','charge'),groups=('flag',),draws=2500,rope=(-0.15,0.15),random_state=42)
    display(bay.means); display(bay.mean_differences)
    return bay, errorbar_table, finish, plt, show


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior location with credible intervals
    """)
    return


@app.cell
def _(bay, errorbar_table, show):
    m=bay.means.query('status=="ok"').copy(); errorbar_table(m,label_col='variable',estimate_col='posterior_mean',low_col='credible_low',high_col='credible_high',title='Posterior means with 95% credible intervals',xlabel='Posterior mean',zero_line=False); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Binary-group posterior differences
    """)
    return


@app.cell
def _(bay, errorbar_table, show):
    d=bay.mean_differences.query('status=="ok"').copy(); d['contrast']=d.variable+' | '+d.level_a.astype(str)+' − '+d.level_b.astype(str); errorbar_table(d,label_col='contrast',estimate_col='posterior_mean',low_col='credible_low',high_col='credible_high',title='Posterior mean differences',xlabel='Difference'); show()
    return (d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probability of direction and ROPE probability
    """)
    return


@app.cell
def _(d, finish, plt, show):
    summary = d.set_index('variable')[['probability_direction', 'rope_probability']]
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    summary.plot(kind='bar', ax=_ax)
    _ax.set_ylim(0, 1.05)
    _ax.set_ylabel('Posterior probability')
    _ax.set_title('Direction vs practical-equivalence evidence')
    _ax.tick_params(axis='x', rotation=20)
    finish(_fig)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior sign probabilities
    """)
    return


@app.cell
def _(d, finish, plt, show):
    sg = d.set_index('variable')[['probability_positive', 'probability_negative']]
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    sg.plot(kind='barh', ax=_ax)
    _ax.set_xlim(0, 1)
    _ax.set_xlabel('Posterior probability')
    _ax.set_title('Posterior sign probability by variable')
    finish(_fig)
    show()
    return


if __name__ == "__main__":
    app.run()
