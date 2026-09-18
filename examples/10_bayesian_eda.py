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
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import configure_plots, display, errorbar_table, finish, load_tabular_demo, show
    configure_plots()

    from ruddy import analyze_bayesian_eda
    frame,dataset=load_tabular_demo()
    # binary grouping for mean differences
    bay=analyze_bayesian_eda(dataset,variables=('activity','stability','length','charge'),groups=('flag',),draws=2500,rope=(-0.15,0.15),random_state=42)
    display(bay.means); display(bay.mean_differences)
    return bay, errorbar_table, finish, np, pl, plt, show


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Posterior location with credible intervals
    """)
    return


@app.cell
def _(bay, errorbar_table, pl, show):
    m = bay.means.filter(pl.col('status') == 'ok')
    errorbar_table(m, label_col='variable', estimate_col='posterior_mean', low_col='credible_low', high_col='credible_high', title='Posterior means with 95% credible intervals', xlabel='Posterior mean', zero_line=False)
    show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Binary-group posterior differences
    """)
    return


@app.cell
def _(bay, errorbar_table, pl, show):
    d = (
        bay.mean_differences.filter(pl.col('status') == 'ok')
        .with_columns(
            contrast=pl.concat_str([
                pl.col('variable'),
                pl.lit(' | '),
                pl.col('level_a').cast(pl.String),
                pl.lit(' − '),
                pl.col('level_b').cast(pl.String),
            ])
        )
    )
    errorbar_table(d, label_col='contrast', estimate_col='posterior_mean', low_col='credible_low', high_col='credible_high', title='Posterior mean differences', xlabel='Difference')
    show()
    return (d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probability of direction and ROPE probability
    """)
    return


@app.cell
def _(d, finish, np, plt, show):
    _variables = d.get_column('variable').to_list()
    x = np.arange(len(_variables))
    width = 0.35
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    _ax.bar(x - width / 2, d.get_column('probability_direction').to_numpy(), width, label='probability_direction')
    _ax.bar(x + width / 2, d.get_column('rope_probability').to_numpy(), width, label='rope_probability')
    _ax.set_xticks(x, _variables)
    _ax.set_ylim(0, 1.05)
    _ax.set_ylabel('Posterior probability')
    _ax.set_title('Direction vs practical-equivalence evidence')
    _ax.tick_params(axis='x', rotation=20)
    _ax.legend()
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
def _(d, finish, np, plt, show):
    _variables = d.get_column('variable').to_list()
    y = np.arange(len(_variables))
    height = 0.35
    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    _ax.barh(y - height / 2, d.get_column('probability_positive').to_numpy(), height, label='probability_positive')
    _ax.barh(y + height / 2, d.get_column('probability_negative').to_numpy(), height, label='probability_negative')
    _ax.set_yticks(y, _variables)
    _ax.set_xlim(0, 1)
    _ax.set_xlabel('Posterior probability')
    _ax.set_title('Posterior sign probability by variable')
    _ax.legend()
    finish(_fig)
    show()
    return


if __name__ == "__main__":
    app.run()
