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
    import pandas as pd
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
        pd,
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
def _(display, finish, grp, np, plt, show):
    s=grp.numeric_summaries.query('response=="activity" and group_column=="group"').copy(); s['label']=s.group_level.astype(str)
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.errorbar(s['mean'],np.arange(len(s)),xerr=s['std'],fmt='o',capsize=4); ax.set_yticks(np.arange(len(s)),s['label']); ax.set_xlabel('Mean ± SD'); ax.set_title('Group summaries returned by Ruddy'); finish(fig); show(); display(s[['group_level','mean','std','median','iqr']])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tukey and Games–Howell side by side
    """)
    return


@app.cell
def _(errorbar_table, post, show):
    p=post.comparisons.copy(); p['contrast']=p.group_a.astype(str)+' − '+p.group_b.astype(str)
    for method in ['tukey_hsd','games_howell']:
        q=p.query('method==@method').copy(); errorbar_table(q,label_col='contrast',estimate_col='mean_difference',low_col='ci_lower',high_col='ci_upper',title=f'{method}: pairwise activity differences',xlabel='Mean difference'); show()
    return (p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect-size confidence intervals
    """)
    return


@app.cell
def _(ci, errorbar_table, show):
    e=ci.effect_sizes.query('response=="activity" and group=="group" and status=="ok"').copy(); e['contrast']=e.level_a.astype(str)+' − '+e.level_b.astype(str)
    if len(e): errorbar_table(e,label_col='contrast',estimate_col='estimate',low_col='confidence_low',high_col='confidence_high',title='Hedges g with bootstrap confidence intervals',xlabel='Standardized effect'); show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pairwise mean-difference matrix
    """)
    return


@app.cell
def _(matrix_heatmap, np, p, pd, show):
    gh=p.query('method=="games_howell"'); levels=sorted(set(gh.group_a).union(gh.group_b)); mat=pd.DataFrame(np.nan,index=levels,columns=levels)
    for _,r in gh.iterrows(): mat.loc[r.group_a,r.group_b]=r.mean_difference; mat.loc[r.group_b,r.group_a]=-r.mean_difference
    np.fill_diagonal(mat.values,0); matrix_heatmap(mat,title='Games–Howell pairwise mean differences'); show()
    return


if __name__ == "__main__":
    app.run()
