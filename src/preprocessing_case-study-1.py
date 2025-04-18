import os
from copy import deepcopy
import elicito as el
import numpy as np
from src.flag_outliers import flag_small_sd, flag_high_rmse

# select scenario by uncommenting corresponding line
#selected_scenario = "binomial-deep_prior-params"
#selected_scenario = "binomial-parametric_prior"
selected_scenario = "binomial-deep_prior-elicits"

path = f"results/{selected_scenario.split('-')[1]}"
files = os.listdir(path)
counter = 0

for file in files:
    if selected_scenario in file:
        counter += 1
        if counter == 1:
            eliobj = el.utils.load(path + "/" + file)
        else:
            eliobj2 = el.utils.load(path + "/" + file)
            eliobj.results.append(eliobj2.results[0])
            eliobj.history.append(eliobj2.history[0])

##### Quality flag criteria
## Note: not used for simulations in paper (only for internal playing)
outlier_sd = flag_small_sd(eliobj, min_sd=0.07)
outlier_rmse = flag_high_rmse(eliobj, max_rmse=0.45, plot=True)

outliers = np.unique(np.concatenate([outlier_sd, outlier_rmse]))

eliobj_clean = deepcopy(eliobj)
eliobj_clean.results = [eliobj.results[i] for i in range(len(eliobj.results)) if i not in outliers]
eliobj_clean.history = [eliobj.history[i] for i in range(len(eliobj.history)) if i not in outliers]


el.plots.loss(
    eliobj_clean, figsize=(5,2),
    weighted=False,
    save_fig=f"figures/{selected_scenario}_loss.png")
el.plots.elicits(
    eliobj_clean, figsize=(5,2),
    save_fig=f"figures/{selected_scenario}_elicits.png")
el.plots.hyperparameter(
    eliobj_clean, figsize=(5,2),
    save_fig=f"figures/{selected_scenario}_hyperparameter.png")
el.plots.prior_marginals(
    eliobj_clean, figsize=(5,2),
    save_fig=f"figures/{selected_scenario}_prior_marginals.png")
el.plots.prior_joint(
    eliobj_clean, idx=list(range(len(eliobj.history))), figsize=(3,3),
    save_fig=f"figures/{selected_scenario}_prior_joint.png")
el.plots.prior_averaging(
    eliobj_clean, height_ratio=[1,1], xlim_weights=0.1,
    save_fig=f"figures/{selected_scenario}_prior_averaging.png")

# compute average training time
# remove first iteration as it contains compiling time etc.
time_per_replication = []
for i in range(len(eliobj.history)):
    time_per_replication.append(np.sum(eliobj.history[i]["time"]))

print("time avg. :", np.round(np.median(time_per_replication)/60, 2))
print("time std. :", np.round(np.std(time_per_replication)/60, 2))