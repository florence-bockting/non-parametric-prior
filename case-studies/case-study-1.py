import elicito as el
import tensorflow as tf
import tensorflow_probability as tfp
from bayesflow.inference_networks import InvertibleNetwork
from copy import deepcopy
tfd = tfp.distributions


class BinomialModel:
    """
    specify generative Binomial model
    """
    def __call__(self, prior_samples: tf.Tensor, design_matrix: tf.Tensor, total_count: int, temp: float
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
        # combine parameter and model uncertainty
        n_batch, n_sample_prior, n_obs = ypred.shape
        ypred_reshaped = tf.reshape(ypred, (n_batch, n_sample_prior*n_obs))
        # select the 25th and 75th-quantile observation
        y_quantiles = tfp.stats.percentile(ypred_reshaped, [25, 75], axis=-1)
        y_quantiles = tf.transpose(y_quantiles, perm=[1,0])

        return dict(
            y_q25 = tf.expand_dims(y_quantiles[:,0], -1),
            y_q75 = tf.expand_dims(y_quantiles[:,1], -1),
        )

def design_matrix(n: int) -> tf.Tensor:
    """
    build design matrix with one continuous predictor

    Parameters
    ----------
    n
        number of observations

    Returns
    -------
    :
        design matrix
    """
    x_std = tfd.Normal(0., 1.).sample(n)
    return tf.stack([[1.]*len(x_std), x_std], axis=-1)

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
    design_matrix=design_matrix(n=50),
    total_count=30,
    temp=1.6
)

targets = [
        el.target(
        name=f"y_q{i}",
        query=el.queries.quantiles((0.25, 0.25, 0.5, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0
    ) for i in [25, 75]
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

#%% Update: Optimize on the parameter space
eliobj_v2 = eliobj.copy()

targets2 = [
    el.target(
        name=f"b{i}",
        query=el.queries.quantiles((0.25, 0.25, 0.5, 0.75, 0.95)),
        loss=el.losses.MMD2(kernel="energy"),
        weight=1.0
    ) for i in [0,1]
]

eliobj_v2.update(targets=targets2)
eliobj_v2.fit()