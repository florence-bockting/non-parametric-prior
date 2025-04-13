"""
plot slope of total loss across the last epochs
Used as preliminary check for convergence
"""
from src.convergence_diagnostics.utils import plot_conv_diagnostics


plot_conv_diagnostics(path_sim= "results/deep_prior",
                      scenario="binomial",
                      start=400, end=500, last_vals=200, ymax_slope=0.02,
                      save_fig="figures/binomial-convergence-diagnostics.png")

plot_conv_diagnostics(path_sim= "results/deep_prior",
                      scenario="normal-independent",
                      start=700, end=800, last_vals=300, ymax_slope=0.02,
                      save_fig="figures/normal-independent-convergence-diagnostics.png")

plot_conv_diagnostics(path_sim= "results/deep_prior",
                      scenario="normal-correlated",
                      start=700, end=800, last_vals=200, ymax_slope=0.02,
                      save_fig="figures/normal-correlated-convergence-diagnostics.png")

plot_conv_diagnostics(path_sim= "results/deep_prior",
                      scenario="normal-skewed",
                      start=700, end=800, last_vals=300, ymax_slope=0.02,
                      save_fig="figures/normal-correlated-convergence-diagnostics.png")
