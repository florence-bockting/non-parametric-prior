"""
run sensitivity analysis for binomial case study
results of sensitivity analysis as published in the manuscript
can be found here: PROVIDE OSF LINK
"""
from src.sensitivity_analysis.binomial_utils import (
    run_sensitivity,
    prep_sensitivity_res,
    plot_sensitivity
)

# input arguments
seed = 1
mu0_seq = [-0.4, -0.2, 0.0, 0.2, 0.4]
mu1_seq = [-0.4, -0.2, 0.0, 0.2, 0.4]
sigma0_seq = [0.01, 0.1, 0.3, 0.6, 1.0]
sigma1_seq = [0.01, 0.1, 0.3, 0.6, 1.0]

# Note: data used in manuscript are provided in OSF (see file header)
# if you want to rerun sensitivity analysis and save result files
# run the following function
run_sensitivity(
    seed=seed, mu0_seq=mu0_seq, mu1_seq=mu1_seq, sigma0_seq=sigma0_seq,
    sigma1_seq=sigma1_seq
)
# if you want to use data from manuscript and not rerun fitting
# use only the following function
df_sim_res = prep_sensitivity_res()

# plot results
plot_sensitivity(
    df_sim_res,
    save_fig="figures/binomial-sensitivity_analysis.png")
