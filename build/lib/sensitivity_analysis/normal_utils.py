# NOTE: If you want to run this file, you need to disable saving of the global
# dictionary
# you can do this by commenting out the respective line
# ('save_as_pkl(global_dict, path)' in the file run.py)
import os
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import elicito as el
from bayesflow.networks import InvertibleNetwork
from src.case_studies.case_study_2 import (
    NormalModel,
    design_matrix,
)
from typing import Optional

tfd = tfp.distributions

def run_prior_checks(
        seed: int, mu0: float, sigma0: float, mu1: float,
        sigma1: float, mu2: float, sigma2: float,
        alpha: float, beta: float, varying_target: str,
        scenario: str,
        cor01: Optional[float],
        cor02: Optional[float], cor12: Optional[float],
        skew1: Optional[float], skew2: Optional[float],
) -> None:
    if scenario == "independent":
        ground_truth = {
            "b0": tfd.Normal(mu0, sigma0),
            "b1": tfd.Normal(mu1, sigma1),
            "b2": tfd.Normal(mu2, sigma2),
            "sigma": tfd.Gamma(alpha, beta),
        }
    elif scenario == "correlated":
        S = [sigma0, sigma1, sigma2]
        M = [[1.0, cor01, cor02], [cor01, 1.0, cor12], [cor02, cor12, 1.0]]
        covariance_matrix = (tf.linalg.diag(S) @ M) @ tf.linalg.diag(S)

        ground_truth = {
            "theta": tfd.JointDistributionSequential(
                [
                    tfd.MultivariateNormalTriL(
                        loc=[mu0, mu1, mu2],
                        scale_tril=tf.linalg.cholesky(covariance_matrix)
                    ),
                    tfd.Gamma([alpha], [beta]),
                ]
            )
        }
    elif scenario == "skewed":
        ground_truth = {
            "b0": tfd.Normal(mu0, sigma0),
            "b1": tfd.TwoPieceNormal(mu1, sigma1, skew1),
            "b2": tfd.TwoPieceNormal(mu2, sigma2, skew2),
            "sigma": tfd.Gamma(alpha, beta),
        }
    else:
        raise ValueError("Couldn't determine scenario.")

    parameters = [
            el.parameter(name=f"b{i}") for i in range(3)
        ] + [
            el.parameter(name="sigma", lower=0)
        ]

    model = el.model(
        obj=NormalModel,
        design_matrix=design_matrix(n_group=30)
    )

    targets = [
        el.target(
            name=j,
            query=el.queries.quantiles((0.05, 0.25, 0.5, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0
        ) for j in ["y_gr1", "y_gr2", "y_gr3"]
    ] + [
        el.target(
            name="r2",
            query=el.queries.quantiles((0.05, 0.25, 0.5, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=10.0
        )
    ] + [
        el.target(
            name="cor",
            query=el.queries.correlation(),
            loss=el.losses.L2,
            weight=0.1
        )
    ]

    expert=el.expert.simulator(
        ground_truth=ground_truth,
        num_samples=10_000
    )

    optimizer = el.optimizer(
        optimizer=tf.keras.optimizers.Adam,
        learning_rate=0.00025,
        clipnorm=1.0,
    )

    trainer=el.trainer(
        method="deep_prior",
        seed=seed,
        epochs=1,
        progress=1
    )

    normalizing_flow=el.networks.NF(
        inference_network=InvertibleNetwork,
        network_specs=dict(
            num_params=4,
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

    path = (f"sensitivity_normal-{scenario}_{mu0:.2f}_{sigma0:.2f}_{mu1:.2f}_{sigma1:.2f}"
            + f"_{mu2:.2f}_{sigma2:.2f}_{alpha:.2f}_{beta:.2f}_{skew1:.2f}_{skew2:.2f}_"
            + f"{cor01:.2f}_{cor02:.2f}_{cor12:.2f}_{varying_target}")

    eliobj.save(path, overwrite=True)


def run_sensitivity(
    seed: int,
    scenario: str,
    mu0_seq: list[float],
    mu1_seq: list[float],
    mu2_seq: list[float],
    sigma0_seq: list[float],
    sigma1_seq: list[float],
    sigma2_seq: list[float],
    a_seq: list[float],
    b_seq: list[float],
    cor_seq: Optional[list[float]] = None,
    skewness_seq: Optional[list[float]] = None,
) -> None:
    # run simulations
    for mu0 in mu0_seq:
        run_prior_checks(
            scenario=scenario,
            seed = seed,
            varying_target="mu0",
            mu0=mu0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            alpha=5.0,
            beta=2.0,
            cor01=3.0,
            cor02=-3.0,
            cor12=-2.0,
            skew1=4.0,
            skew2=4.0,
        )

    for sigma0 in sigma0_seq:
        run_prior_checks(
            scenario=scenario,
            seed=seed,
            varying_target="sigma0",
            mu0=10.0,
            sigma0=sigma0,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            alpha=5.0,
            beta=2.0,
            cor01=3.0,
            cor02=-3.0,
            cor12=-2.0,
            skew1=4.0,
            skew2=4.0,
        )

    for mu1 in mu1_seq:
        run_prior_checks(
            scenario=scenario,
            seed=seed,
            varying_target="mu1",
            mu0=10.0,
            sigma0=2.5,
            mu1=mu1,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            alpha=5.0,
            beta=2.0,
            cor01=3.0,
            cor02=-3.0,
            cor12=-2.0,
            skew1=4.0,
            skew2=4.0,
        )

    for sigma1 in sigma1_seq:
        run_prior_checks(
            scenario=scenario,
            seed=seed,
            varying_target="sigma1",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=sigma1,
            mu2=2.5,
            sigma2=0.8,
            alpha=5.0,
            beta=2.0,
            cor01=3.0,
            cor02=-3.0,
            cor12=-2.0,
            skew1=4.0,
            skew2=4.0,
        )

    for mu2 in mu2_seq:
        run_prior_checks(
            scenario=scenario,
            seed=seed,
            varying_target="mu2",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=mu2,
            sigma2=0.8,
            alpha=5.0,
            beta=2.0,
            cor01=3.0,
            cor02=-3.0,
            cor12=-2.0,
            skew1=4.0,
            skew2=4.0,
        )

    for sigma2 in sigma2_seq:
        run_prior_checks(
            scenario=scenario,
            seed=1,
            varying_target="sigma2",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=sigma2,
            alpha=5.0,
            beta=2.0,
            cor01=3.0,
            cor02=-3.0,
            cor12=-2.0,
            skew1=4.0,
            skew2=4.0,
        )

    for a in a_seq:
        run_prior_checks(
            scenario=scenario,
            seed=seed,
            varying_target="alpha",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            alpha=a,
            beta=2.0,
            cor01=3.0,
            cor02=-3.0,
            cor12=-2.0,
            skew1=4.0,
            skew2=4.0,
        )

    for b in b_seq:
        run_prior_checks(
            scenario=scenario,
            seed=seed,
            varying_target="beta",
            mu0=10.0,
            sigma0=2.5,
            mu1=7.0,
            sigma1=1.3,
            mu2=2.5,
            sigma2=0.8,
            alpha=5.0,
            beta=b,
            cor01=3.0,
            cor02=-3.0,
            cor12=-2.0,
            skew1=4.0,
            skew2=4.0,
        )

    if cor_seq is not None:
        for cor in cor_seq:
            run_prior_checks(
                scenario=scenario,
                seed=seed,
                varying_target="cor01",
                mu0=10.0,
                sigma0=2.5,
                mu1=7.0,
                sigma1=1.3,
                mu2=2.5,
                sigma2=0.8,
                alpha=5.0,
                beta=2.0,
                cor01=cor,
                cor02=-0.3,
                cor12=-0.2,
                skew1=4.0,
                skew2=4.0,
            )
        for cor in cor_seq:
            run_prior_checks(
                scenario=scenario,
                seed=seed,
                varying_target="cor02",
                mu0=10.0,
                sigma0=2.5,
                mu1=7.0,
                sigma1=1.3,
                mu2=2.5,
                sigma2=0.8,
                alpha=5.0,
                beta=2.0,
                cor01=0.3,
                cor02=cor,
                cor12=-0.2,
                skew1=4.0,
                skew2=4.0,
            )
        for cor in cor_seq:
            run_prior_checks(
                scenario=scenario,
                seed=seed,
                varying_target="cor12",
                mu0=10.0,
                sigma0=2.5,
                mu1=7.0,
                sigma1=1.3,
                mu2=2.5,
                sigma2=0.8,
                alpha=5.0,
                beta=2.0,
                cor01=0.3,
                cor02=-0.3,
                cor12=cor,
                skew1=4.0,
                skew2=4.0,
            )
    if skewness_seq is not None:
        for skew in skewness_seq:
            run_prior_checks(
                scenario=scenario,
                seed=seed,
                varying_target="skew1",
                mu0=10.0,
                sigma0=2.5,
                mu1=7.0,
                sigma1=1.3,
                mu2=2.5,
                sigma2=0.8,
                alpha=5.0,
                beta=2.0,
                cor01=3.0,
                cor02=-3.0,
                cor12=-2.0,
                skew1=skew,
                skew2=4.0
            )
        for skew in skewness_seq:
            run_prior_checks(
                scenario=scenario,
                seed=seed,
                varying_target="skew2",
                mu0=10.0,
                sigma0=2.5,
                mu1=7.0,
                sigma1=1.3,
                mu2=2.5,
                sigma2=0.8,
                alpha=5.0,
                beta=2.0,
                cor01=3.0,
                cor02=-3.0,
                cor12=-2.0,
                skew1=4.0,
                skew2=skew,
            )


def prep_sensitivity_res(
    scenario: str, # independent, correlated, skewed
    path: str = "./results/sensitivity_analyses/"
):
    # save results in dictionary
    res_dict = {
        f"{n}": [] for n in [
            "vary", "mu0", "sigma0", "mu1", "sigma1","mu2", "sigma2",
            "alpha", "beta", "y_group1", "y_group2", "y_group3", "R2", "cor"
        ]
    }

    all_files = os.listdir(path)
    for i, file in enumerate(all_files):
        if file.startswith(f"sensitivity_normal-{scenario}"):
            sensitivity_result = el.utils.load(path + file)
            true_elicited_statistics = sensitivity_result.results[0]["expert_elicited_statistics"]

            labels = file.split("_")
            res_dict["vary"].append(labels[-2])
            res_dict["mu0"].append(float(labels[2]))
            res_dict["sigma0"].append(float(labels[3]))
            res_dict["mu1"].append(float(labels[4]))
            res_dict["sigma1"].append(float(labels[5]))
            res_dict["mu2"].append(float(labels[6]))
            res_dict["sigma2"].append(float(labels[7]))
            res_dict["alpha"].append(float(labels[8]))
            res_dict["beta"].append(float(labels[9]))
            res_dict["skew1"].append(float(labels[10]))
            res_dict["skew2"].append(float(labels[11]))
            res_dict["cor01"].append(float(labels[12]))
            res_dict["cor02"].append(float(labels[13]))
            res_dict["cor12"].append(float(labels[14]))
            res_dict["y_group1"].append(true_elicited_statistics["quantiles_y_gr1"][0, :].numpy())
            res_dict["y_group2"].append(true_elicited_statistics["quantiles_y_gr2"][0, :].numpy())
            res_dict["y_group3"].append(true_elicited_statistics["quantiles_y_gr3"][0, :].numpy())
            res_dict["R2"].append(true_elicited_statistics["quantiles_r2"][0, :].numpy())
            res_dict["cor"].append(true_elicited_statistics["cor_cor"][0, :].numpy())

    return pd.DataFrame(res_dict)


def plot_sensitivity(
    df, save_fig
):
    range_list2 = [np.sort(df[var].unique()) for var in [
        "mu0", "sigma0", "mu1", "sigma1", "mu2", "sigma2", "alpha", "beta"]]
    cols_quantiles = ["#21284f", "#00537b", "#007d87", "#00ac79", "#83cf4a"]
    true_vals = {
        "mu0": 10, "sigma0": 2.5, "mu1": 7, "sigma1": 1.3, "mu2": 2.5,
        "sigma2": 0.8, "a": 5, "b": 2,
    }

    def re_dig(x):
        return [x[i].astype(str).replace("0.", ".") for i in range(len(x))]

    fig, axs = plt.subplots(8, 4, constrained_layout=True, figsize=(7, 9))
    for m, k in enumerate(["mu0", "sigma0", "mu1", "sigma1", "mu2", "sigma2", "alpha",
                           "beta"]):

        df_hyper = df[df["vary"] == k]

        for j, elicit in enumerate(["y_group1", "y_group2", "y_group3", "R2"]):
            for i, col in list(enumerate(cols_quantiles)):
                sorted_idx = np.argsort(np.stack(df_hyper[k], 0))
                axs[m, j].plot(
                    np.sort(np.stack(df_hyper[k], 0)),
                    np.stack(df_hyper[elicit], 1)[i][sorted_idx],
                    "-o",
                    color=col,
                    ms=5,
                )
                axs[m, j].patch.set_alpha(0.0)

    for j in range(4):
        [
            axs[i, j].set_xlabel(lab, fontsize="small", labelpad=2)
            for i, lab in enumerate(
                [
                    r"$\mu_0$",
                    r"$\sigma_0$",
                    r"$\mu_1$",
                    r"$\sigma_1$",
                    r"$\mu_2$",
                    r"$\sigma_2$",
                    r"$a$",
                    r"$b$",
                ]
            )
        ]
        [
            axs[i, j].set_xticks(
                range_list2[i], np.array(range_list2[i]).astype(int),
                fontsize="x-small"
            )
            for i in range(8)
        ]
        [axs[i, j].tick_params(axis="y", labelsize="x-small") for
         i in range(8)]
    [
        axs[0, j].set_title(t, pad=10, fontsize="medium")
        for j, t in enumerate(
            [
                r"quantiles $y_i \mid gr_1$",
                r"quantiles $y_i \mid gr_2$",
                r"quantiles $y_i \mid gr_3$",
                r"quantiles $R^2$",
            ]
        )
    ]
    [
        axs[i, j].spines[["right", "top"]].set_visible(False)
        for i, j in itertools.product(range(8), range(4))
    ]
    [axs[i, -1].set_ylim(0, 1) for i in range(8)]
    [axs[i, 0].set_ylabel(" ", rotation=0, labelpad=10) for i in range(8)]
    for k, val in enumerate(true_vals):
        [axs[k, j].axvline(true_vals[val], color="darkred", lw=2) for
         j in range(4)]
    for i, lab, col in zip(
        [0, 0.08, 0.12, 0.16, 0.2, 0.24, 0.3, 0.32],
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
        fig.text(i, 1.02, lab, color=col)
    fig.suptitle("independent-normal model", x=0.5, y=1.07)
    # for y in [952, 639, 323]:
    #     fig.patches.extend(
    #         [
    #             plt.Rectangle(
    #                 (10, y),
    #                 1010,
    #                 3,
    #                 fill=True,
    #                 color="grey",
    #                 alpha=0.2,
    #                 zorder=-1,
    #                 transform=None,
    #                 figure=fig,
    #             )
    #         ]
    #     )
    for x, y, lab in zip(
        [0.005] * 4,
        [0.85, 0.61, 0.37, 0.13],
        [r"$\beta_0$", r"$\beta_1$", r"$\beta_2$", r"$\sigma$"],
    ):
        fig.text(
            x, y, lab, fontsize="large", bbox=dict(facecolor="none",
                                                   edgecolor="grey")
        )
    if save_fig is not None:
        plt.savefig(save_fig)
    plt.show()
