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
    # 01 · Profiling, quality and group distributions

    A strong first-pass EDA: profile the dataset, inspect missingness and variable quality, then compare full distributions across groups rather than stopping at summary statistics.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from _helpers import configure_plots, density_by_group, display, ecdf_by_group, finish, group_violin_box_scatter, grouped_histograms, load_tabular_demo, show
    configure_plots()

    from ruddy import analyze_univariate, analyze_distribution_diagnostics
    frame, dataset = load_tabular_demo()
    univ = analyze_univariate(dataset)
    diag = analyze_distribution_diagnostics(dataset, responses=('activity','stability'), groups=('group','source'))
    display(pd.DataFrame([univ.profiling.overview]))
    display(univ.numeric_statistics[['column','mean','std','median','iqr','n_missing','n_non_finite','status']])
    return (
        density_by_group,
        diag,
        display,
        ecdf_by_group,
        finish,
        frame,
        group_violin_box_scatter,
        grouped_histograms,
        plt,
        show,
        univ,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data quality and missingness
    Ruddy reports missing and non-finite values separately. The visualization layer can turn those structured counts into a compact quality overview.
    """)
    return


@app.cell
def _(finish, plt, show, univ):
    q = univ.numeric_statistics.set_index('column')[['n_missing','n_non_finite']]
    fig, ax = plt.subplots(figsize=(8,4.5))
    q.plot(kind='bar', ax=ax)
    ax.set_ylabel('Count'); ax.set_title('Missing vs non-finite values by numeric variable'); ax.tick_params(axis='x', rotation=35)
    finish(fig); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Activity distributions by group
    The same response is viewed with complementary geometries: violin+box+raw observations, density, ECDF and faceted histograms.
    """)
    return


@app.cell
def _(frame, group_violin_box_scatter, show):
    group_violin_box_scatter(frame, value='activity', group='group', title='Activity distribution by group'); show()
    return


@app.cell
def _(density_by_group, frame, show):
    density_by_group(frame, value='activity', group='group', title='Activity density by group'); show()
    return


@app.cell
def _(ecdf_by_group, frame, show):
    ecdf_by_group(frame, value='activity', group='group', title='Activity ECDF by group'); show()
    return


@app.cell
def _(frame, grouped_histograms, show):
    grouped_histograms(frame, value='activity', group='group', title='Faceted activity histograms by group'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stability by experimental source
    A second response shows that group-aware visualization is not tied to one variable or one grouping factor.
    """)
    return


@app.cell
def _(frame, group_violin_box_scatter, show):
    group_violin_box_scatter(frame, value='stability', group='source', title='Stability by source'); show()
    return


@app.cell
def _(diag, display):
    disp = diag.dispersion[['response','group','method','statistic','p_value','q_value','status']]
    display(disp)
    return


if __name__ == "__main__":
    app.run()
