from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import elicito as el
import tensorflow as tf
import tensorflow_probability as tfp
from bayesflow.inference_networks import InvertibleNetwork
from copy import deepcopy
import patsy as pa
import sys

tfd = tfp.distributions


class NormalModel:
    """
    specify generative model for normal case study
    """
    def __call__(
            self, prior_samples: tf.Tensor, design_matrix: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        """
        simulate from a normal generative model

        Parameters
        ----------
        prior_samples
            prior samples

        design_matrix
            design matrix

        Returns
        -------
        :
            target quantities used for loss computation
        """
        epred = prior_samples[:, :, :-1] @ tf.transpose(design_matrix)
        sigma = tf.abs(prior_samples[:, :, -1][:, :, None])

        likelihood = tfd.Normal(loc=epred, scale=sigma)

        ypred = likelihood.sample()

        group1 = ypred[:, :, 0::3]
        group2 = ypred[:, :, 1::3]
        group3 = ypred[:, :, 2::3]

        # R2
        var_epred = tf.math.reduce_variance(epred, -1)
        # variance of difference between ypred and epred
        var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
        var_total = var_epred + var_diff
        # variance of linear predictor divided by total variance
        r2 = tf.divide(var_epred, var_total)


        prior_samples = tf.concat(
            [prior_samples[:, :, :-1],
             tf.abs(prior_samples[:, :, -1][:, :, None])],
            axis=-1,
        )

        return dict(
            r2=r2,
            y_gr1=group1, y_gr2=group2, y_gr3=group3,
            b0=prior_samples[:, :, 0], b1=prior_samples[:, :, 1], b2=prior_samples[:, :, 2],
            sigma=prior_samples[:, :, -1],
        )


def design_matrix(n_group: int) -> tf.Tensor:
    """
    define design matrix for normal model

    One three-level, dummy-coded predicto incl. intercept

    Parameters
    ----------
    n_group
        number of observations per group/level of predictor

    Returns
    -------
    :
        design matrix
    """
    # construct design matrix with a 3-level factor
    df = pa.dmatrix("a", pa.balanced(a=3, repeat=n_group),
                    return_type="dataframe")
    # save in correct format
    d_final = tf.cast(df, dtype=tf.float32)
    return d_final


def case_study_normal(scenario, seed, approach):
    if scenario == "independent":
        ground_truth = {
            "b0": tfd.Normal(10.0, 2.5),
            "b1": tfd.Normal(7.0, 1.3),
            "b2": tfd.Normal(2.5, 0.8),
            "sigma": tfd.Gamma(5.0, 2.0),
        }
    elif scenario == "skewed":
        ground_truth = {
            "b0": tfd.Normal(10.0, 2.5),
            "b1": tfd.TwoPieceNormal(7.0, 1.3, 4),
            "b2": tfd.TwoPieceNormal(2.5, 0.8, 4),
            "sigma": tfd.Gamma(5.0, 2.0),
        }
    elif scenario == "correlated":
        S = [2.5, 1.3, 0.8]
        M = [[1.0, 0.3, -0.3], [0.3, 1.0, -0.2], [-0.3, -0.2, 1.0]]
        covariance_matrix = (tf.linalg.diag(S) @ M) @ tf.linalg.diag(S)

        ground_truth = {
            "theta": tfd.JointDistributionSequential(
                [
                    tfd.MultivariateNormalTriL(
                        loc=[10, 7.0, 2.5],
                        scale_tril=tf.linalg.cholesky(covariance_matrix)
                    ),
                    tfd.Gamma([5.0], [2.0]),
                ]
            )
        }
    else:
        raise ValueError("Unknown scenario")


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
        epochs=800,
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

    if approach == "deep_prior-elicits":
        eliobj.fit()
        eliobj.save(f"normal-{scenario}-{approach}")


    elif approach == "deep_prior-params":
        eliobj_v2 = eliobj

        targets2 = [
            el.target(
                name=f"b{i}",
                query=el.queries.quantiles((0.05, 0.25, 0.5, 0.75, 0.95)),
                loss=el.losses.MMD2(kernel="energy"),
                weight=1.0
            ) for i in range(3)
        ] + [
            el.target(
                name="sigma",
                query=el.queries.quantiles((0.05, 0.25, 0.5, 0.75, 0.95)),
                loss=el.losses.MMD2(kernel="energy"),
                weight=1.0
            )
        ] + [
        el.target(
            name="cor",
            query=el.queries.correlation(),
            loss=el.losses.L2,
            weight=0.1
        )
        ]

        eliobj_v2.update(targets=targets2)
        eliobj_v2.fit()
        eliobj_v2.save(f"normal-{scenario}-{approach}")


    elif approach == "parametric_prior":
        eliobj_v3 = eliobj

        if scenario == "skewed":
            parameters3 = [
                              el.parameter(
                                  name=f"b0",
                                  family=tfd.Normal,
                                  hyperparams=dict(
                                      loc=el.hyper(f"mu0"),
                                      scale=el.hyper(f"sigma0", lower=0)
                                  ),
                              )
                          ] + [
                              el.parameter(
                                  name=f"b{i}",
                                  family=tfd.TwoPieceNormal,
                                  hyperparams=dict(
                                      loc=el.hyper(f"mu{i}"),
                                      scale=el.hyper(f"sigma{i}", lower=0),
                                      skewness=el.hyper(f"skew{i}", lower=0, shared=True),
                                  ),
                              ) for i in [1,2]
                         ] + [
                              el.parameter(
                                  name="sigma",
                                  family=tfd.Gamma,
                                  hyperparams=dict(
                                      concentration=el.hyper("alpha", lower=0),
                                      rate=el.hyper("beta", lower=0)
                                  )
                              )
            ]
        elif scenario == "correlated":
            raise NotImplementedError(
                "Optimization for covariance matrices is not yet supported in the elicito package"
            )
        else:
            parameters3 = [
                el.parameter(
                    name=f"b{i}",
                    family=tfd.Normal,
                    hyperparams=dict(
                        loc=el.hyper(f"mu{i}"),
                        scale=el.hyper(f"sigma{i}", lower=0)
                    ),
                ) for i in range(3)
            ] + [
                el.parameter(
                    name="sigma",
                    family=tfd.Gamma,
                    hyperparams=dict(
                        concentration=el.hyper("alpha", lower=0),
                        rate=el.hyper("beta", lower=0)
                    )
                )
            ]

        targets3 = [
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
        ]

        optimizer3 = el.optimizer(
            optimizer=tf.keras.optimizers.Adam,
            learning_rate=0.1,
            clipnorm=1.0,
        )

        trainer3 = el.trainer(
            method="parametric_prior",
            seed=seed,
            epochs=800,
            progress=1
        )

        network3 = None

        initializer3 = el.initializer(
            method="sobol",
            loss_quantile=0,
            iterations=32,
            distribution=el.initialization.uniform(radius=4.0, mean=0.0),
        )

        eliobj_v3.update(
            parameters=parameters3,
            targets=targets3,
            optimizer=optimizer3,
            trainer=trainer3,
            network=network3,
            initializer=initializer3,
        )

        eliobj_v3.fit()

        eliobj_v3.save(f"normal-{scenario}-{approach}")

    else:
        raise ValueError("Unknown approach")

if __name__ == "__main__":
    scenario = str(sys.argv[1])
    seed = int(sys.argv[2])
    approach = str(sys.argv[3])

    case_study_normal(scenario, seed, approach)