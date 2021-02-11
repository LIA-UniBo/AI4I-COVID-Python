# Bayesian melding library.

from sympy import Symbol, stats
from scipy.stats import gaussian_kde

import numpy as np

"""
Joint probability density function class.
At the moment uses sympy to represent marginals and each random variable is assumed independent.

Marginals are a dictionary of {"Variable name": sympy.stats distribution} pairs.
"""
class Joint_pdf:
    # Creates a joint pdf from marginals.
    def __init__(self, marginals):
        self.marginals = marginals

    # Returns a random sample for each of the random variables.
    def sample(self):
        return {k: next(stats.sample(v)) for k, v in self.marginals.items()}

    # Evaluates the PDF on the given realization of the random variables (all of which must be instantiated).
    def eval(self, realization):
        out = 1

        for k, v in self.marginals.items():
            out *= stats.density(v)(np.nan_to_num(realization[k])).evalf()  # Assuming marginals are independent...

        return np.nan_to_num(out)


"""
Bayesian melding algorithm. Takes a deterministic model, its constant arguments (as dictionary),
the prior and likelihood pdfs (of class Joint_pdf), and the number of samples to extract.
Returns a list of samples (values of input parameters) from the (approximate) real distribution, a dictionary of means
for each input parameter and a dictionary of standard deviations for each input parameter.
"""
def melding(model, model_args, input_prior, output_prior, input_likelihood, output_likelihood, n_samples, n_resampling, debug=False):
    means = {}
    variances = {}

    # Sampling phase:
    samples = []
    for i in range(n_samples):
        samples.append(input_prior.sample())

    # Weight computation phase:
    posteriors = []
    posteriors_np = np.zeros((len(samples), len(model.get_output_keys())), dtype=np.float)
    for i in range(len(samples)):
        model.input_params = samples[i]
        tmp = model.eval_last(**model_args)
        posteriors.append(tmp)
        posteriors_np[i] = [v for _, v in tmp.items()]

    posteriors_np = np.nan_to_num(posteriors_np)
    q_ind = gaussian_kde(posteriors_np.T)

    weights = np.zeros(len(samples), dtype=float)
    for i in range(len(samples)):
        if debug:
            print("Posterior sampled: {}".format(posteriors[i]))
            print("Output prior at posterior: {}".format(output_prior.eval(posteriors[i])))
            print("Induced prior at posterior: {}".format(q_ind(posteriors_np[i].T)))
            print("Input likelihood at posterior: {}".format(input_likelihood.eval(samples[i])))
            print("Output likelihood at posterior: {}".format(output_likelihood.eval(posteriors[i])))
            print("##########")

        weights[i] = np.nan_to_num((output_prior.eval(posteriors[i]) / q_ind(posteriors_np[i].T)) ** 0.5 * input_likelihood.eval(samples[i]) * output_likelihood.eval(posteriors[i]))

    weights = np.nan_to_num(weights)  # Due to numerical approximations, it's very easy to obtain NaNs for probabilities.
    weights /= np.sum(weights)

    # Importance-resampling phase:
    resampled_idx = np.zeros(n_resampling, dtype=int)
    if (np.sum(weights) > 0):
        for i in range(resampled_idx.shape[0]):
            resampled_idx[i] = np.random.choice(range(len(samples)), p=weights)

        # Mean and std deviation for each input parameter:
        for k in samples[0].keys():
            means[k] = 0
            for i in resampled_idx:
                means[k] += samples[i][k] / resampled_idx.shape[0]

        for k in samples[0].keys():
            variances[k] = 0
            for i in resampled_idx:
                variances[k] += (samples[i][k] - means[k]) ** 2 / resampled_idx.shape[0]
    else:
        for k in samples[0].keys():
            means[k] = np.nan

        for k in samples[0].keys():
            variances[k] = np.nan

    return [samples[i] for i in resampled_idx], means, variances