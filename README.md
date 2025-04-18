# Expert-elicitation method for non-parametric joint priors using normalizing flows
(*Code and Supplementary material*)

+ **preprint**: [see arXiv](https://doi.org/10.48550/arXiv.2411.15826)
+ **status**: currently in peer-review
+ **additional information**: simulation results are available through [OSF]()

### About this repository
This repository provides all code for our simulation setup underlying the reported results in our paper.
Furthermore, it includes additional simulation results that are part of the supplementary material. 

The structure of this repository is as follows:

- `\figures\`: include the plots generated for the supplementary material
- `\src\`
  - `\case_studies\`: specification of the elicitation method for each corresponding case study using 
  the [`elicito` package](https://github.com/florence-bockting/elicito)
  - `\convergence_diagnostics\`: preliminary convergence diagnostics in which we analyse the slope of the last 100 epochs
  for each replication (incl. preprocessing and plotting)
  - `\sensitivity_analysis\`: sensitivity analysis for each case study (incl. simulation setup, preprocessing, and plotting)
- `\supplementary-material\`:
  - Notebooks providing an overview of results for each case study
  - incl. additional simulation results for 
    - (1) learning prior distributions using the `deep-prior` method based on the
    parameter space (thus providing the method with full information; acts as validation set for our method)
    - (2) re-analysis of each case study using the `parametric-prior` method (except for the correlation case as optimizing for 
    covariance matrices is not yet implemented in `elicito`)

