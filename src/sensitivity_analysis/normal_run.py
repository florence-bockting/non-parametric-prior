"""
run sensitivity analysis for normal case study
results of sensitivity analysis as published in the manuscript
can be found on OSF: https://osf.io/xrzh6
"""

from src.sensitivity_analysis.normal_utils import (
    run_sensitivity,
    prep_sensitivity_res,
    plot_sensitivity
)

# select scenario
#selected_scenario = "independent"
selected_scenario = "correlated"
#selected_scenario = "skewed"

# input arguments
seed = 1
mu0_seq = [0, 5, 10, 15, 20]
sigma0_seq = [0.1, 1.5, 2, 3.0, 4]
mu1_seq = [0, 5, 10, 15, 20]
sigma1_seq = [0.1, 1.5, 2, 3.0, 4]
mu2_seq = [0, 5, 10, 15, 20]
sigma2_seq = [0.1, 1.5, 2, 3.0, 4]
a_seq = [1, 5, 10, 20, 25]
b_seq = [1, 5, 10, 20, 25]
cor_seq = [-0.3, -0.2, 0.2, 0.3]
skewness_seq = [0.1, 2.0, 4.0, 8.0, 12.0]

# Note: data used in manuscript are provided in OSF (see file header)
# if you want to rerun sensitivity analysis and save result files
# run the following function

run_sensitivity(
    scenario=selected_scenario, seed=seed, mu0_seq=mu0_seq, mu1_seq=mu1_seq,
    mu2_seq=mu2_seq, sigma0_seq=sigma0_seq, sigma1_seq=sigma1_seq,
    sigma2_seq=sigma2_seq, a_seq=a_seq, b_seq=b_seq, cor_seq=cor_seq,
    skewness_seq=skewness_seq
)
# if you want to use data from manuscript and not rerun fitting
# use only the following function
df_sim_res = prep_sensitivity_res(scenario=selected_scenario, path="results/sensitivity_analyses/")

# plot results
plot_sensitivity(
    df_sim_res,
    scenario=selected_scenario,
    save_fig=f"figures/normal-{selected_scenario}-sensitivity_analysis.png"
)
