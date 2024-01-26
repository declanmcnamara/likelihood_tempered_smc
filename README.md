# Likelihood-Tempered SMC

This repository contains an implementation of likelihood-tempered SMC based on Algorithm 17.1 of [1] ([link here](https://link.springer.com/book/10.1007/978-3-030-47845-2)). Adaptive temperature selection is performed by the procedure outlined in $\S$ 17.2.3 of [1], with some adjustments for numerical stability.  

Please read the example notebooks to get started, first `Gaussian.ipynb` followed by `TwoMoons.ipynb`. The implementation requires the user to specify three main functions: `log_prior`, `log_target`, and `proposal`. Our implementations use either Metropolis-Hastings random-walk (MHRW) or Metropolis-adjusted Langevin (MAL) dynamics, always with Gaussian noise. 

![Sample empirical approximations to posterior.](https://github.com/declanmcnamara/likelihood_tempered_smc/blob/main/examples/cover_fig.png)

> [1] Chopin, Nicolas and Papaspiliopoulos, Omiros. *An Introduction to Sequential Monte Carlo*. Springer, 2020. 
