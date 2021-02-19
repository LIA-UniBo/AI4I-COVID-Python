# AI4I-COVID-Python
 Artificial Intelligence for Industry's project on Italian COVID-19 dataset.

In this project we explored the potential and limitations of Bayesian melding, a statistical technique which fits the input parameters of a deterministic function, according to stochastic observations.

The general idea behind the method is to merge different "opinions" about an observed phenomenon via statistical pooling:
- A prior probability on the outputs of the model ("what may be reasonable to happen")
- An induced probability computed by applying the deterministic model to some input prior distribution ("what we expect to observe according to the model")
- A likelihood probability on the inputs ("what we know has happened")
- A likelihood probability on the outputs ("what we actually observe").

In order to correctly apply the pooling operation,  the model needs to be inverted. Since this is seldom possible, pooling is approximated with the SIR (sampling importance-resampling, not to be confused with the susceptible-infected-removed model, also used in this repository) algorithm:
1. Extract a large number of random samples from the input prior distribution
2. Weight each sample ![formula](https://render.githubusercontent.com/render/math?math=\Theta_i) according to ![formula](https://render.githubusercontent.com/render/math?math=w_i%20=%20(\frac{q_2(M(\Theta_i))}{q_1^*(M(\Theta_i))})^{1-\alpha}%20L_1(\Theta_i)%20L_2(M(\Theta_i))), where:
   - ![formula](https://render.githubusercontent.com/render/math?math=M(\Theta_i)) is the output of the model applied to ![formula](https://render.githubusercontent.com/render/math?math=\Theta_i)
   - ![formula](https://render.githubusercontent.com/render/math?math=\alpha) is the pooling factor (usually 0.5)
   - ![formula](https://render.githubusercontent.com/render/math?math=q_2(M(\Theta_i))) is the output prior
   - ![formula](https://render.githubusercontent.com/render/math?math=q_1^*(M(\Theta_i))) is the induced output posterior, ie. the output distribution computed applying the input distribution to the model; it can be estimated by applying the model to each sample and then performing a kernel density estimation with a Gaussian kernel
   - ![formula](https://render.githubusercontent.com/render/math?math=L_1(\Theta_i)) is the input likelihood
   - ![formula](https://render.githubusercontent.com/render/math?math=L_2(M(\Theta_i))) is the output likelihood
3. Extract a small subset of samples, but this time use the computed weights instead of the prior distribution
4. The distribution on the resampled weights is an approximation of the true input distribution and the usual operations can be performed on it (eg. extract mean to fit the model to the data and variance to determine confidence).

Bayesian melding was applied to three different epidemiological models:
- SIR: Susceptible-infected-removed
- SIRD: Susceptible-infected-recovered-deceased
- SEIRD: Susceptible-exposed-infected-recovered-deceased, extended with hidden E compartment and reinfection rate.

Due to step 1. being very slow and the curse of dimensionality (especially for SEIRD), we also tried to perform deterministic seeding in order to reduce the search space, with limited success.

Authors: [G. Tsiotas](https://tsiotas.com), [L.S. Lorello](https://github.com/HashakGik).
