from typing import Any
from sklearn.metrics import root_mean_squared_error
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt


def flag_small_sd(
        eliobj: Any, min_sd: float
) -> list[int]:
    small_deviation = []

    for idx in range(len(eliobj.history)):
        B,S,P = eliobj.results[idx]["prior_samples"].shape
        sd_priors = np.std(
                tf.reshape(eliobj.results[idx]["prior_samples"], (B*S,P)),
            axis=0
        )
        for i in range(P):
            if sd_priors[i] < min_sd:
                small_deviation.append(idx)

    print(
        "The following replications have priors with a std. deviation less ",
          min_sd,": ", np.unique(small_deviation)
    )
    return np.unique(small_deviation)


def flag_high_rmse(
        eliobj: Any, max_rmse: float, plot: bool = False
) -> list[int]:
    """
    Extract index of replications with high RMSE

    rmse is computed for each elicited statistic (simulated vs. expert)
    and then averaged. If the average RMSE is higher than `max_rmse` the
    index of the corresponding replication is flagged.
     A list if all flagged replications is returned.

    Parameters
    ----------
    eliobj
        fitted eliobj

    max_rmse
        maximum RMSE that is still tolerated

    plot
        barplot with rmse per replication

    Returns
    -------
    :
        list with indices of replications that show high RMSE
    """
    # elicited statistics have not been learned accurately
    rmse = []
    for sim in range(len(eliobj.history)):
        rmse_sim = []
        for key in eliobj.results[sim]["elicited_statistics"].keys():
            B, quants = eliobj.results[sim]["elicited_statistics"][key].shape
            for quant in range(quants):
                rmse_sim.append(
                    root_mean_squared_error(
                        tf.broadcast_to(
                            eliobj.results[sim]["expert_elicited_statistics"][key],
                            (B,quants))[:,quant].numpy(),
                        eliobj.results[sim]["elicited_statistics"][key][:,quant].numpy()
                    )
                )
        rmse.append(np.mean(rmse_sim))

    high_rmse = np.where(np.array(rmse) > max_rmse)[0]

    print("The following replications have an rmse larger ",
          max_rmse,": ", high_rmse)

    if plot:
        _, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(rmse, ax=ax)
        sns.barplot(
            x=high_rmse,
            y=np.array(rmse)[np.array(rmse) > max_rmse],
            color="orange", ax=ax
        )
        ax.set_ylabel("avg. RMSE")
        ax.set_xlabel("replications")
        plt.show()

    return high_rmse
