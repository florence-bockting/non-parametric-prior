"""
functions used for sensitivity analysis of binomial case study
"""
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from bayesflow.inference_networks import InvertibleNetwork
import elicito as el

from src.case_studies.case_study_1 import (
    BinomialModel,
    design_matrix
)

tfd = tfp.distributions

# prepare simulations
def run_prior_checks(
        seed: int, mu0: float, sigma0: float, mu1: float,
        sigma1: float, varying_target: str
) -> None:
    """
    set up of simulations for the binomial case study

    Parameters
    ----------
    seed
        seed value for this run

    mu0
        location hyperparameter of normal prior on
        intercept parameter (beta_0)

    sigma0
        scale hyperparameter of normal prior on
        intercept parameter (beta_0)

    mu1
        location hyperparameter of normal prior
        on slope parameter (beta_1)

    sigma1
        scale hyperparameter of normal prior
        on slope parameter (beta_1)

    varying_target
        which hyperparameter is varied in the
        current run?
    """

    ground_truth = {
        "b0": tfd.Normal(mu0, sigma0),
        "b1": tfd.Normal(mu1, sigma1)
    }

    parameters = [
        el.parameter(name="b0"),
        el.parameter(name="b1"),
    ]

    model = el.model(
        obj=BinomialModel,
        design_matrix=design_matrix(),
        total_count=30,
        temp=1.0
    )

    targets = [
        el.target(
            name=f"y_q{i}",
            query=el.queries.quantiles(
                (0.05, 0.25, 0.5, 0.75, 0.95)
            ),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0
        ) for i in [25, 75]
    ]

    expert = el.expert.simulator(
        ground_truth=ground_truth,
        num_samples=10_000
    )

    optimizer = el.optimizer(
        optimizer=tf.keras.optimizers.Adam,
        learning_rate=0.0001,
        clipnorm=1.0,
    )

    trainer = el.trainer(
        method="deep_prior",
        seed=seed,
        epochs=1,
        progress=0
    )

    normalizing_flow = el.networks.NF(
        inference_network=InvertibleNetwork,
        network_specs=dict(
            num_params=2,
            num_coupling_layers=3,
            coupling_design="affine",
            coupling_settings={
                "dropout": False,
                "dense_args": {
                    "units": 128,
                    "activation": "relu",
                    "kernel_regularizer": None,
                },
                "num_dense": 2,
            },
            permutation="fixed"
        ),
        base_distribution=el.networks.base_normal
    )

    eliobj = el.Elicit(
        model=model,
        parameters=parameters,
        targets=targets,
        expert=expert,
        optimizer=optimizer,
        trainer=trainer,
        network=normalizing_flow,
    )

    eliobj.fit()

    eliobj.save(f"sensitivity_binom_{mu0:.2f}_{sigma0:.2f}_{mu1:.2f}_{sigma1:.2f}_{varying_target}")

def run_sensitivity(
        seed: int, mu0_seq: list[float],
        mu1_seq: list[float], sigma0_seq: list[float], sigma1_seq: list[float]
) -> None:
    """
    run simulation-based workflow for various hyperparameter settings

    Parameters
    ----------
    seed
        seed value for this run

    mu0_seq
        range of location hyperparameters of normal prior for
        intercept parameter (beta_0)

    mu1_seq
        range of location hyperparameters of normal prior for
        slope parameter (beta_1)

    sigma0_seq
        range of location hyperparameters of normal prior for
        intercept parameter (beta_0)

    sigma1_seq
        range of scale hyperparameters of normal prior for
        slope parameter (beta_1)

    """
    # run simulations
    for mu0 in mu0_seq:
        run_prior_checks(
            seed=seed,
            mu0=mu0,
            sigma0=0.1,
            mu1=-0.1,
            sigma1=0.3,
            varying_target="mu0"
        )

    for sigma0 in sigma0_seq:
        run_prior_checks(
            seed=seed,
            mu0=0.1,
            sigma0=sigma0,
            mu1=-0.1,
            sigma1=0.3,
            varying_target="sigma0"
        )

    for mu1 in mu1_seq:
        run_prior_checks(
            seed=seed,
            mu0=0.1,
            sigma0=0.1,
            mu1=mu1,
            sigma1=0.3,
            varying_target="mu1"
        )

    for sigma1 in sigma1_seq:
        run_prior_checks(
            seed=seed,
            mu0=0.1,
            sigma0=0.1,
            mu1=-0.1,
            sigma1=sigma1,
            varying_target="sigma1"
        )

def prep_sensitivity_res(
        path: str = "./results/deep_prior/"
) -> pd.DataFrame:
    """
    preprocess data from sensitivity analysis

    Parameters
    ----------
    path
        path where sensitivity analysis results are saved

    Returns
    -------
    :
        DataFrame with all relevant results required for plotting
    """

    res_dict = {
        f"{n}": [] for n in ["vary", "mu0", "sigma0", "mu1", "sigma1","y_x0", "y_x1"]
    }

    all_files = os.listdir(path)
    for i, file in enumerate(all_files):
        if file.startswith("sensitivity_binom"):
            sensitivity_result = el.utils.load(path + file)
            true_elicited_statistics = sensitivity_result.results[0]["expert_elicited_statistics"]

            labels = file.split("_")
            res_dict["vary"].append(labels[-2])
            res_dict["mu0"].append(float(labels[2]))
            res_dict["sigma0"].append(float(labels[3]))
            res_dict["mu1"].append(float(labels[4]))
            res_dict["sigma1"].append(float(labels[5]))
            res_dict["y_x0"].append(true_elicited_statistics["quantiles_y_q25"][0,:].numpy())
            res_dict["y_x1"].append(true_elicited_statistics["quantiles_y_q75"][0,:].numpy())

    return pd.DataFrame(res_dict)


def plot_sensitivity(df, save_fig):
    cols_quantiles = ["#21284f", "#00537b", "#007d87", "#00ac79", "#83cf4a"]
    true_vals = {"mu0": 0.1, "sigma0": 0.1, "mu1": -0.1, "sigma1": 0.3}

    def re_dig(x):
        return [x[i].astype(str).replace("0.", ".") for i in range(len(x))]

    fig, axs = plt.subplots(4, 2, constrained_layout=True, figsize=(4, 4))
    for m, k in enumerate(["mu0", "sigma0", "mu1", "sigma1"]):
        df_hyper = df[df["vary"] == k]
        for j, elicit in enumerate(["y_x0", "y_x1"]):
            for i, col in list(enumerate(cols_quantiles)):
                axs[m, j].plot(
                    np.stack(df_hyper[k],0),
                    np.stack(df_hyper[elicit],1)[i],
                    "-o",
                    color=col,
                    ms=5,
                )
                axs[m, j].patch.set_alpha(0.0)

    for j in range(2):
        [
            axs[i, j].set_xlabel(lab, fontsize="small", labelpad=2)
            for i, lab in enumerate(
                [r"$\mu_0$", r"$\sigma_0$", r"$\mu_1$", r"$\sigma_1$"]
            )
        ]
        [
            axs[i, j].set_xticks(
                np.stack(df_hyper[k],0), re_dig(np.stack(df_hyper[k],0)),
                fontsize="x-small"
            ) for i, k in enumerate(["mu0","sigma0","mu1","sigma1"])
        ]
        [axs[i, j].tick_params(axis="y", labelsize="x-small") for
         i in range(4)]
    [
        axs[0, j].set_title(t, pad=10, fontsize="medium")
        for j, t in enumerate(
            [r"quantiles $y_i \mid x_0$", r"quantiles $y_i \mid x_1$"]
        )
    ]
    [
        axs[i, j].spines[["right", "top"]].set_visible(False)
        for i, j in itertools.product(range(4), range(2))
    ]
    [axs[i, 0].set_ylabel(" ", rotation=0, labelpad=10) for i in range(4)]
    for k, val in enumerate(true_vals):
        [axs[k, j].axvline(true_vals[val], color="darkred", lw=2) for
         j in range(2)]
    for i, lab, col in zip(
        [0, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.44],
        [
            "legend: ",
            r"$q_{05}$",
            r"$q_{25}$",
            r"$q_{50}$",
            r"$q_{75}$",
            r"$q_{95}$",
            "|",
            "ground truth",
        ],
        [
            "black",
            "#21284f",
            "#00537b",
            "#007d87",
            "#00ac79",
            "#83cf4a",
            "darkred",
            "black",
        ],
    ):
        fig.text(i, 1.02, lab, color=col, fontsize="small")

    for x, y, lab in zip([0.02] * 2, [0.73, 0.26],
                         [r"$\beta_0$", r"$\beta_1$"]):
        fig.text(
            x, y, lab, fontsize="medium", bbox=dict(facecolor="none",
                                                    edgecolor="grey")
        )
    if save_fig is not None:
        plt.savefig(save_fig)
    plt.show()
