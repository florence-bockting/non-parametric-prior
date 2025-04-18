import elicito as el
import sys
import tensorflow as tf
import tensorflow_probability as tfp
from bayesflow.inference_networks import InvertibleNetwork
from copy import deepcopy
tfd = tfp.distributions
import numpy as np


class BinomialModel:
    """
    specify generative Binomial model
    """
    def __call__(self, prior_samples: tf.Tensor, design_matrix: tf.Tensor,
                 total_count: int, temp: float
                 ) -> dict[str, tf.Tensor]:
        """
        simulate from the Binomial model

        Parameters
        ----------
        prior_samples
            prior samples

        design_matrix
            design matrix

        total_count
            total number of observations in Binomial model

        temp
            temperature parameter for softmax-gumble trick

        Returns
        -------
        :
            prior predictions for the 25th and 75th quantile of the predictor x
        """
        # linear predictor
        epred = prior_samples @ tf.transpose(design_matrix)
        # link function
        probs = tf.sigmoid(epred)
        # likelihood
        likelihood = tfd.Binomial(total_count=total_count,
                                  probs=tf.expand_dims(probs, -1))
        # sample model predictions
        ypred = el.utils.softmax_gumbel_trick(
            likelihood=likelihood,
            upper_thres=total_count,
            temp=temp
        )

        return dict(
            y_q25 = ypred[:,:,0],
            y_q75 = ypred[:,:,1],
            b0 = prior_samples[:,:,0],
            b1 = prior_samples[:,:,1],
        )

def design_matrix() -> tf.Tensor:
    """
    build design matrix with one continuous predictor

    Returns
    -------
    :
        design matrix
    """
    x = np.arange(0, 50, 1)
    x_std = x/np.std(x)
    x_quantiles = np.percentile(x_std, [25, 75])
    return tf.stack([[1.] * len(x_quantiles), x_quantiles], axis=-1)

def case_study_binomial(seed, approach):
    ground_truth = {
        "b0": tfd.Normal(0.1, 0.1),
        "b1": tfd.Normal(-0.1, 0.3)
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
            query=el.queries.quantiles((0.05, 0.25, 0.5, 0.75, 0.95)),
            loss=el.losses.MMD2(kernel="energy"),
            weight=1.0
        ) for i in [25, 75]
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
        learning_rate=0.0001,
        clipnorm=1.0,
    )

    trainer=el.trainer(
        method="deep_prior",
        seed=seed,
        epochs=500,
        progress=1
    )

    normalizing_flow=el.networks.NF(
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
    if approach == "deep_prior-elicits":
        eliobj.fit()
        eliobj.save(f"binomial-{approach}")

        #el.plots.hyperparameter(eliobj)
        #el.plots.loss(eliobj)
        #el.plots.elicits(eliobj)
        #el.plots.prior_marginals(eliobj)

    elif approach == "deep_prior-params":
        # update: Optimize on the parameter space
        eliobj_v2 = deepcopy(eliobj)

        targets2 = [
            el.target(
                name=f"b{i}",
                query=el.queries.quantiles((0.05, 0.25, 0.5, 0.75, 0.95)),
                loss=el.losses.MMD2(kernel="energy"),
                weight=1.0
            ) for i in [0,1]
        ]

        eliobj_v2.update(targets=targets2)
        eliobj_v2.fit()
        eliobj_v2.save(f"binomial-{approach}")

        #el.plots.hyperparameter(eliobj_v2)
        #el.plots.loss(eliobj_v2)
        #el.plots.elicits(eliobj_v2)
        #el.plots.prior_joint(eliobj_v2)

    elif approach == "parametric_prior":
        # update: Compare with parametric prior method
        eliobj_v3 = deepcopy(eliobj)

        parameters3 = [
            el.parameter(
                name=f"b{i}",
                family=tfd.Normal,
                hyperparams=dict(
                    loc=el.hyper(f"mu{i}"),
                    scale=el.hyper(f"sigma{i}", lower=0)
                ),
            ) for i in range(2)
        ]

        targets3 = [
            el.target(
                name=f"y_q{i}",
                query=el.queries.quantiles((0.05, 0.25, 0.5, 0.75, 0.95)),
                loss=el.losses.MMD2(kernel="energy"),
                weight=1.0
            ) for i in [25, 75]
        ]

        optimizer3 = el.optimizer(
            optimizer=tf.keras.optimizers.Adam,
            learning_rate=0.01,
            clipnorm=1.0,
        )

        trainer3 = el.trainer(
            method="parametric_prior",
            seed=seed,
            epochs=600,
            progress=1
        )

        network3 = None

        initializer3 = el.initializer(
                method="sobol",
                loss_quantile=0,
                iterations=32,
                distribution=el.initialization.uniform(radius=2.0, mean=0.0),
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
        eliobj_v3.save(f"binomial-{approach}")

        #el.plots.hyperparameter(eliobj_v3)
        #el.plots.loss(eliobj_v3)
        #el.plots.elicits(eliobj_v3)
        #el.plots.prior_joint(eliobj_v3)
    else:
        raise ValueError("Unknown approach")

if __name__ == "__main__":
    seed = int(sys.argv[1])
    approach = str(sys.argv[2])

    case_study_binomial(seed, approach)


#python case_studies/case_study_1.py $i "parametric_prior"
#python case_studies/case_study_1.py $i "deep_prior-elicits"
#python case_studies/case_study_1.py $i "deep_prior-params"
