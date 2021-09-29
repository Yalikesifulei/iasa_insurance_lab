import numpy as np
from scipy.stats import distributions, rv_continuous, rv_discrete
from scipy.stats import poisson, uniform
from scipy.misc import derivative
from scipy.integrate import quad
from pynverse import inversefunc

class rv_mixed(object):
    """
        Define SciPy-like mixed random variable, which has CDF in form
        of convex combination with coefficients in 'weights' of CDF's 
        of random variables in 'distributions'.

        Has SciPy-like cdf, sd, pdf, rvs and mean methods.
    """
    def __init__(self, weights, distributions):
            if not np.allclose(sum(weights), 1):
                raise ValueError('weights must sum up to 1')
            else:
                assert len(weights) == len(distributions)
                self.weights = weights
                self.distributions = distributions
        
    def cdf(self, x):
        if type(x) == np.ndarray:
            res = np.zeros_like(x)
        else:
            res = 0.0
        for w, distr in zip(self.weights, self.distributions):
            res = res + w * distr.cdf(x)
        return res
    
    def sf(self, x):
        return 1 - self.cdf(x)

    def pdf(self, x):
        for distr in self.distributions:
            if hasattr(distr, 'dist') and isinstance(distr.dist, rv_discrete):
                raise ValueError(f'PDF is unavailable: there is a discrete distribution "{distr.dist.name}"')
            if not hasattr(distr, 'dist') and isinstance(distr, rv_discrete):
                raise ValueError(f'PDF is unavailable: there is a discrete distribution "{distr.name}"')
        if type(x) == np.ndarray:
            res = np.zeros_like(x)
        else:
            res = 0.0
        for w, distr in zip(self.weights, self.distributions):
            res = res + w * distr.pdf(x)
        return res

    def rvs(self, size, seed=None):
        rng = np.random.RandomState(seed)
        ind = rng.choice(range(len(self.weights)), p=self.weights, size=size)
        ind = np.eye(len(self.weights))[ind].T
        values = np.array([distr.rvs(size=size, random_state=seed) for distr in self.distributions])
        return (ind * values).sum(axis=0)

    def mean(self):
        res = 0
        for w, distr in zip(self.weights, self.distributions):
            res = res + w * distr.mean()
        return res

class rv_from_cdf(object):
    """
        Define SciPy-like random variable with given support and CDF.
        Inverse CDF (used for sampling) should be given, otherwise computed numerically
        and may lead to imprecise and slower sampling.
        PDF and mean values can also be specified, otherwise are computated numerically.

        Has SciPy-like cdf, sd, pdf, rvs and mean methods.
    """
    def __init__(self, support, cdf, inv_cdf=None, pdf=None, mean=None):
        self.support = np.sort(support)

        if np.allclose(cdf(support[1]), 1) and np.allclose(cdf(support[0]-1e-8), 0):
            self._cdf = cdf
        else:
            raise ValueError("Wrong support bounds or wrong CDF")

        if inv_cdf:
            if np.allclose(inv_cdf(1), support[1]) and np.allclose(cdf(0), support[0]-1e-8):
                self._inv_cdf = inv_cdf
            else:
                raise ValueError("Wrong support bounds or wrong inverse CDF")
        else:
            self._inv_cdf = None
        self._pdf = pdf
        self._mean = mean

    def cdf(self, x):
        return self._cdf(x)

    def sf(self, x):
        return 1 - self.cdf(x)
    
    def pdf(self, x):
        if self._pdf:
            return self._pdf(x)
        else:
            if type(x) == np.ndarray:
                res = np.array([derivative(self._cdf, x0, dx=1e-6) for x0 in x])
            else:
                res = derivative(self._cdf, x, dx=1e-6)
            return res

    def rvs(self, size, seed=None):
        unif_sample = uniform.rvs(size=size, random_state=seed)
        if self._inv_cdf:
            return self._inv_cdf(unif_sample)
        else:
            return inversefunc(self._cdf, y_values=unif_sample, domain=self.support)

    def mean(self):
        if self._mean:
            return self._mean
        else:
            return quad(lambda x: x*self.pdf(x), self.support[0], self.support[1])[0]
        


def N(t, lmb=1, seed=None):
    """
        Sample homogeneous Poisson process with intensity lmb on [0; t]
    """
    n = poisson.rvs(mu=lmb*t, random_state=seed)
    T = uniform.rvs(scale=t, size=n, random_state=seed)
    return np.sort(T)


def U(t, u0, c, lmb, X, seed=None):
    """
        Sample Cramér–Lundberg process with initial surplus u0,
        premiums rate c, claims with intensity lmb and distribution X
        on time interval [0; t]. 
    """
    T = N(t, lmb, seed)
    x = X.rvs(len(T), seed)
    time = np.linspace(0, t, t*10)
    time = np.sort(np.concatenate((time, T)))
    u = u0 + c * time
    claims_cumsum = np.zeros_like(time)
    for tau, cl in zip(T, x):
        claims_cumsum[time >= tau] += cl
    u = u - claims_cumsum

    return T, time, u