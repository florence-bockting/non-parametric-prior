"""
investigate slope of last loss values as preliminary convergence check
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import siegelslopes
import elicito as el
from typing import Optional

def calc_slope(
        res: np.array, start: int, end: int
) -> float:
    """
    compute the slope for the last x loss values

    Parameters
    ----------
    res
        simulated loss values for each epoch

    start
        start of range of loss values from which
        slope should be computed

    end
        end of range of loss values from which
        slope should be computed

    Returns
    -------
    :
        slope value
    """
    y = res[start:end]
    slope = siegelslopes(y)[0]
    return slope


def plot_conv_diagnostics(
    path_sim: str,
    scenario: str,
    start: int,
    end: int,
    last_vals: int,
    ymax_slope: float,
    max_seeds: int = 4,
    factor: int = 100,
    num_seeds: int=30,
    save_fig: Optional[str] = None,
) -> None:
    """
    plot slope of last loss values as preliminary convergence check

    Parameters
    ----------
    path_sim
        path to the simulation result

    scenario
        which scenario should be plotted
        (binomial, normal-independent, etc.)

    start
        start of range from which the slope should be computed

    end
        end of loss range from which the slope should be computed

    last_vals
        number of loss values that should be plotted on the x-axis

    ymax_slope
        upper limit of y-axis of slope-value plot (scatterplot)

    max_seeds
        number of seeds that are investigated as "highest slopes"

    factor
        factor by which slope value is multiplied for better visual
        discrimination

    num_seeds
        number of replications

    save_fig
        path to save figure. If none, no figure will be saved
    """
    all_files = os.listdir(path_sim)

    seed = []
    slopes = []
    slopes_orig = []
    single_losses = []

    for file in all_files:
        if file.startswith(f"{scenario}-deep_prior-elicits"):
            eliobj = el.utils.load(path_sim + "/" + file)
            res_loss = np.concatenate(eliobj.history[0]["loss"])
            slope_abs = abs(calc_slope(res_loss, start, end)) * factor
            slope = calc_slope(res_loss, start, end) * factor
            single_losses.append([res_loss])
            slopes.append(slope_abs)
            slopes_orig.append(slope)
            seed.append(eliobj.results[0]["seed"])

    res_dict = {"seed": seed, "slope": slopes, "slope_orig": slopes_orig}
    res_sorted = pd.DataFrame(res_dict).sort_values(by="slope")
    res_highest = res_sorted[-max_seeds:]
    res_lowest = res_sorted[0:0 + 1]
    res_single = pd.concat([res_lowest, res_highest])

    # extract seeds with lowest and highest loss slope
    single_loss = [single_losses[i] for i in res_single.seed.values]


    fig = plt.figure(layout="constrained", figsize=(6, 3.5))
    subfigs = fig.subfigures(2, 1, wspace=0.07)
    subfig0 = subfigs[0].subplots(1, 1)
    subfig1 = subfigs[1].subplots(1, 5, sharey=True, sharex=True)

    subfig0.axhline(0, linestyle="dashed", color="black", lw=1)
    subfig0.plot(range(num_seeds), res_sorted["slope"], "o", color="grey")
    subfig0.plot(
        range(num_seeds - max_seeds, num_seeds),
        res_highest["slope"],
        "o",
        color="#21284f",
    )
    subfig0.plot([0], res_lowest["slope"], "o", color="#2e5c2c")
    subfig0.set_xticks(range(num_seeds), res_sorted["seed"], fontsize="small")
    subfig0.spines[["right", "top"]].set_visible(False)
    subfig0.set_ylabel("|slope|*100", fontsize="small")
    subfig0.set_xlabel("seed", fontsize="small")
    subfig0.yaxis.set_tick_params(labelsize=7)
    subfig0.xaxis.set_tick_params(labelsize=7)
    subfig0.set_ylim(-0.001, ymax_slope)

    for i, c in enumerate(["#2e5c2c"] + ["#21284f"] * 4):
        subfig1[i].plot(single_loss[i][0][-last_vals:], color="grey")
        subfig1[i].plot(
            range(last_vals - 100, last_vals), single_loss[i][0][-100:], color=c
        )
        subfig1[i].plot(
            [last_vals - 100, last_vals],
            [
                np.mean(single_loss[i][0][-105:-95]),
                (res_single.iloc[i]["slope_orig"] / 100) * 100
                + np.mean(single_loss[i][0][-105:-95]),
            ],
            "-",
            color="red",
        )
        subfig1[i].set_title(f"seed: {res_single['seed'].iloc[i]}",
                             fontsize="medium")
        subfig1[i].set_xticks(
            np.linspace(0, last_vals, 3),
            np.linspace(end - last_vals, end, 3, dtype=int),
            fontsize="x-small",
        )
        subfig1[i].spines[["right", "top"]].set_visible(False)
        subfig1[i].set_xlabel("epochs", fontsize="small")
    subfig1[0].yaxis.set_tick_params(labelsize=7)
    subfig1[0].set_ylabel(r"$L(\lambda)$", fontsize="small")
    if save_fig is not None:
        plt.savefig(save_fig)
    plt.show()
