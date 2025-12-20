import math
from scipy.stats import binom


def subspace_size(num_noise, weight):
    """
    Calculate the size of the subspace
    """
    return math.comb(num_noise, weight)


# def binomial_weight(N, W, p):
#     if N<5000:
#         return math.comb(N, W) * ((p)**W) * ((1 - p)**(N - W))
#     else:
#         lam = N * p
#         # PMF(X=W) = e^-lam * lam^W / W!
#         # Evaluate in logs to avoid overflow for large W, then exponentiate
#         log_pmf = (-lam) + W*math.log(lam) - math.lgamma(W+1)
#         return math.exp(log_pmf)


def binomial_weight(N, W, p):
    return binom.pmf(W, N, p)
