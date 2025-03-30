import elicito as el
import tensorflow as tf
import tensorflow_probability as tfp
from bayesflow.inference_networks import InvertibleNetwork
from copy import deepcopy
import patsy as pa

tfd = tfp.distributions

scenario = "independent"  # independent, correlated, skewed

class NormalModel:
    def __call__(self, prior_samples, design_matrix):
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


def design_matrix(n_group):
    # construct design matrix with a 3-level factor
    df = pa.dmatrix("a", pa.balanced(a=3, repeat=n_group),
                    return_type="dataframe")
    # save in correct format
    d_final = tf.cast(df, dtype=tf.float32)
    return d_final

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
    M = [[1.0, 0.95, -0.99], [0.95, 1.0, -0.95], [-0.99, -0.95, 1.0]]
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
    el.parameter(name="sigma")
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
    ) for j in ["y_gr1", "y_gr2", "y_gr3", "r2"]
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
    seed=0,
    epochs=600,
    progress=1
)

normalizing_flow=network=el.networks.NF(
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

# %% Update: Optimize on the parameter space
eliobj_v2 = deepcopy(eliobj)

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
]

eliobj_v2.update(targets=targets2)
eliobj_v2.fit()
