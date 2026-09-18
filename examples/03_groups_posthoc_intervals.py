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
    # 03 · Group comparisons, post-hoc tests and uncertainty

    Move from raw distributions to omnibus/group summaries, effect sizes, explicit post-hoc methods and confidence intervals. The plots emphasize magnitude and uncertainty, not only p-values.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from _helpers import configure_plots, density_by_group, display, errorbar_table, finish, group_violin_box_scatter, load_tabular_demo, matrix_heatmap, show
    configure_plots()

    from ruddy import analyze_groups, analyze_posthoc, analyze_confidence_intervals
    frame,dataset=load_tabular_demo()
    grp=analyze_groups(dataset,responses=('activity','stability'),groups=('group','source'))
    post=analyze_posthoc(dataset,response='activity',factor='group',methods=('tukey_hsd','games_howell'))
    ci=analyze_confidence_intervals(dataset,bootstrap_resamples=300,random_state=42)
    return (
        ci,
        density_by_group,
        display,
        errorbar_table,
        finish,
        frame,
        group_violin_box_scatter,
        grp,
        matrix_heatmap,
        np,
        pl,
        plt,
        post,
        show,
    )


@app.cell
def _(frame, group_violin_box_scatter, show):
    group_violin_box_scatter(frame,value='activity',group='group',title='Activity by group: raw observations + violin + box'); show()
    return


@app.cell
def _(density_by_group, frame, show):
    density_by_group(frame,value='activity',group='group',title='Activity density overlap between groups'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group means and uncertainty from Ruddy summaries
    """)
    return


@app.cell
def _(display, finish, grp, np, pl, plt, show):
    s = (
        grp.numeric_summaries.filter((pl.col('response') == 'activity') & (pl.col('group_column') == 'group'))
        .with_columns(label=pl.col('group_level').cast(pl.String))
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(s.get_column('mean').to_numpy(), np.arange(s.height), xerr=s.get_column('std').to_numpy(), fmt='o', capsize=4)
    ax.set_yticks(np.arange(s.height), s.get_column('label').to_list())
    ax.set_xlabel('Mean ± SD')
    ax.set_title('Group summaries returned by Ruddy')
    finish(fig)
    show()
    display(s.select(['group_level', 'mean', 'std', 'median', 'iqr']))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tukey and Games–Howell side by side
    """)
    return


@app.cell
def _(errorbar_table, pl, post, show):
    p = post.comparisons.with_columns(
        contrast=pl.concat_str([pl.col('group_a').cast(pl.String), pl.lit(' − '), pl.col('group_b').cast(pl.String)])
    )
    for method in ['tukey_hsd', 'games_howell']:
        q = p.filter(pl.col('method') == method)
        errorbar_table(q, label_col='contrast', estimate_col='mean_difference', low_col='ci_lower', high_col='ci_upper', title=f'{method}: pairwise activity differences', xlabel='Mean difference')
        show()
    return (p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect-size confidence intervals
    """)
    return


@app.cell
def _(ci, errorbar_table, pl, show):
    e = (
        ci.effect_sizes.filter((pl.col('response') == 'activity') & (pl.col('group') == 'group') & (pl.col('status') == 'ok'))
        .with_columns(contrast=pl.concat_str([pl.col('level_a').cast(pl.String), pl.lit(' − '), pl.col('level_b').cast(pl.String)]))
    )
    if e.height > 0:
        errorbar_table(e, label_col='contrast', estimate_col='estimate', low_col='confidence_low', high_col='confidence_high', title='Hedges g with bootstrap confidence intervals', xlabel='Standardized effect')
        show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pairwise mean-difference matrix
    """)
    return


@app.cell
def _(matrix_heatmap, np, p, pl, show):
    gh = p.filter(pl.col('method') == 'games_howell')
    levels = sorted(set(gh.get_column('group_a').to_list()).union(gh.get_column('group_b').to_list()))
    idx = {name: i for i, name in enumerate(levels)}
    mat = np.zeros((len(levels), len(levels)))
    for r in gh.iter_rows(named=True):
        i, j = idx[r['group_a']], idx[r['group_b']]
        mat[i, j] = r['mean_difference']
        mat[j, i] = -r['mean_difference']
    matrix_heatmap(mat, title='Games–Howell pairwise mean differences', row_labels=levels, col_labels=levels)
    show()
    return


if __name__ == "__main__":
    app.run()
