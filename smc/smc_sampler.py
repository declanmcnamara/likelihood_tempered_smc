import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from scipy.optimize import brentq, fsolve
from scipy.special import logsumexp as lse


class EmpiricalDistribution(object):
    def __init__(self, items, weights, log_weights):
        """
        items: Tensor (n_items, ...)
        weights: Tensir (n_items, )
        self.log_raw_weights: Tensor (n_items, ...)

        """
        self.items = items
        self.weights = weights
        self.log_weights = log_weights

    def sample(self, n=10):
        indices = torch.multinomial(self.weights, num_samples=n, replacement=True)
        return self.items[indices]


class Sampler(object):
    def __init__(self, init_objs, init_weights, init_log_weights):
        """
        current_ed: empirical distribution of current iteration (time t)
        eds: list of past Empirical Distributions
        """
        self.eds = []
        first_ed = EmpiricalDistribution(init_objs, init_weights, init_log_weights)
        self.eds.append(first_ed)
        self.current_ed = first_ed

    def update(self, new_objs, new_weights, new_log_weights):
        new_ed = EmpiricalDistribution(new_objs, new_weights, new_log_weights)
        self.eds.append(new_ed)
        self.current_ed = new_ed


class LikelihoodTemperedSMC(object):
    def __init__(
        self,
        init_objs,
        init_weights,
        init_raw_log_weights,
        final_target_fcn,
        prior,
        log_prior,
        log_target_fcn,
        proposal_fcn,
        max_mc_steps=100,
        context=None,
        z_min=None,
        z_max=None,
        kwargs=None,
    ):
        """
        init_objs, init_weight as in Sampler class
        target_fcn: callable, given Tensor of particles z, return log p(x | z) for data x
        prior: callable, given Tensor of particles z, returns log p(z)
        """
        self.sampler = Sampler(init_objs, init_weights, init_raw_log_weights)
        self.num_particles = self.sampler.current_ed.items.shape[0]
        self.prior = prior
        if hasattr(prior, "support"):
            self.z_min = (
                prior.support.lower_bound
                if hasattr(prior.support, "lower_bound")
                else -np.inf
            )
            self.z_max = (
                prior.support.upper_bound
                if hasattr(prior.support, "upper_bound")
                else np.inf
            )
        else:
            self.z_min = z_min
            self.z_max = z_max
        self.log_prior = log_prior
        self.log_target_fcn = log_target_fcn
        self.proposal_fcn = proposal_fcn
        self.curr_tau = 0.0
        self.final_target_fcn = final_target_fcn
        self.ESS_min_prop = 0.5
        self.ESS_min = math.floor(self.num_particles * self.ESS_min_prop)
        self.curr_stage = 1
        self.max_mc_steps = max_mc_steps
        self.softmax = nn.Softmax(0)
        self.tau_list = [self.curr_tau]
        self.context = context
        self.kwargs = kwargs
        self.cached_log_targets = None

    def _aux_solver_func(self, delta, curr_particles):
        if self.cached_log_targets is not None:
            log_targets = self.cached_log_targets
        else:
            log_targets = self.final_target_fcn(curr_particles)
            log_targets = log_targets.detach().cpu().numpy()
            self.cached_log_targets = log_targets
        log_numerator = 2 * lse(np.nan_to_num(delta * log_targets, nan=-np.inf))
        log_denominator = lse(2 * np.nan_to_num(delta * log_targets, nan=-np.inf))
        result = log_numerator - log_denominator - np.log(self.ESS_min)
        return result

    def concatenate(self, old_objs, new_objs):
        """
        old_objs: Tensor n x (K-1) x ...
        new_objs Tensor n x ...

        Output: Tensor n x K x ...
        """
        return torch.cat([old_objs, new_objs.unsqueeze(1)], 1)

    def chi_square_dist(self, curr_particles):
        def func(delta):
            return self._aux_solver_func(delta, curr_particles)

        x0 = 0.0
        sign0 = np.sign(func(x0))
        bs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        signs = np.array([np.sign(func(x)) for x in bs])
        diffs = sign0 - signs
        if np.sum(diffs != 0) > 0:
            my_b = bs[np.where(diffs != 0)[0][0]]
            solutions = brentq(func, a=0.0, b=my_b)
            delta = np.sort(
                np.clip(np.array([solutions]), a_min=0.0, a_max=1.0 - self.curr_tau)
            )[0]
        else:
            solutions = fsolve(func=func, x0=x0, maxfev=1000)
            delta = np.sort(
                np.clip(np.array([solutions]), a_min=0.0, a_max=1.0 - self.curr_tau)
            )[0][0]

        self.cached_log_targets = None
        if delta <= 1e-20:
            if len(self.tau_list) >= 2:
                prev_diff = self.tau_list[-1] - self.tau_list[-2]
                return min(max(0.0, prev_diff), 1 - self.curr_tau)
            else:
                return 1e-20
        else:
            return delta

    def log_weight_helper(self, curr_zs, prev_log_target, curr_log_target):
        """
        curr_zs: sampled (latent) particles from current iteration
        prev_log_target: previous iteration's unnormalized target
        curr_log_target: this iteration's unnormalized target
        """
        z_to_use = curr_zs[:, -1, ...]
        num = curr_log_target(z_to_use)
        denom = prev_log_target(z_to_use)
        return num, denom

    def ess(self):
        """
        Check ESS of current set of particles and weights.
        """
        weights = self.sampler.current_ed.weights
        return weights.square().sum() ** (-1)

    def one_step(self):
        # Check ESS
        ess = self.ess()

        # Optionally resample
        if ess <= self.ESS_min:
            # Resample
            samples = self.sampler.current_ed.sample(self.num_particles)
            log_w_hat = torch.zeros(self.sampler.current_ed.weights.shape)
        else:
            samples = self.sampler.current_ed.items
            log_w_hat = self.sampler.current_ed.log_weights[:, -1]

        # Compute next tau in schedule
        z_to_use = samples[:, -1, ...]
        delta = self.chi_square_dist(z_to_use)
        if delta <= 1e-6:
            delta = 1e-6
        next_tau = self.curr_tau + delta
        print(next_tau)

        # Construct targets
        def curr_target(z):
            return (
                self.log_prior(z, **self.kwargs).cpu()
                + self.log_target_fcn(z, self.context, **self.kwargs).cpu() * next_tau
            )

        def prev_target(z):
            return self.log_prior(z, **self.kwargs).cpu() + torch.nan_to_num(
                self.log_target_fcn(z, self.context, **self.kwargs).cpu()
                * self.curr_tau,
                -torch.inf,
            )

        # Propagate
        new_zs = self.proposal_fcn(
            z_to_use, self.context, curr_target, **self.kwargs
        )  # context can be anything
        new_zs = new_zs.clamp(min=self.z_min, max=self.z_max)

        # Concatenate
        new_histories = self.concatenate(samples, new_zs)

        # Compute weights
        log_curr_target, log_prev_target = self.log_weight_helper(
            new_histories, prev_target, curr_target
        )
        log_weights = log_w_hat + log_curr_target.view(-1) - log_prev_target.view(-1)
        weights = self.softmax(torch.nan_to_num(log_weights, -torch.inf))

        # Concatenate history of weights
        new_log_weights = self.concatenate(
            self.sampler.current_ed.log_weights, log_weights
        )

        # Update state of the SMC system
        self.sampler.update(new_histories, weights, new_log_weights)
        self.curr_stage += 1

        # Update rolling quanitities
        self.curr_tau = next_tau
        self.curr_log_targets = log_curr_target
        self.tau_list.append(self.curr_tau)

    def run(self):
        while (self.curr_tau < 1.0) and (self.curr_stage < self.max_mc_steps):
            self.one_step()
