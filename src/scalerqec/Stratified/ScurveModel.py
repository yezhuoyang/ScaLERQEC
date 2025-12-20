import numpy as np
from scipy.stats import norm

def scurve_function(x, center, sigma):
    return 0.5/(1+np.exp(-(x - center) / sigma))
    #return 0*x



def scurve_function_with_distance(x, cd, mu, sigma):
    """
    Piece-wise S-curve:
        0                          for x < cd
        0.5 * Φ((x - μ) / σ)       for x ≥ cd
    where Φ is the standard normal CDF.
    """
    x = np.asarray(x)                      # ensure array
    y = 0.5 * norm.cdf(x, loc=mu, scale=sigma)
    return np.where(x < cd, 0.0, y)        # vectorised “if”


# Define the inverse transform: y → 1/2 * 1 / (1 + e^y)
def inv_logit_half(y):
    return 0.5 / (1 + np.exp(y))


def linear_function(x, a, b):
    """
    Linear function for curve fitting.
    """
    return a * x + b



def modified_linear_function_with_d(x, a, b, c, d):
    eps   = 1e-12
    delta = (x - d)**0.5
    delta = np.where(np.abs(delta) < eps, np.sign(delta)*eps, delta)
    return a * x + b + c / delta



# Strategy A: keep the model safe near the pole
def modified_linear_function(d):
    def tempfunc(x,a,b,c,d=d):
        return modified_linear_function_with_d(x, a, b, c, d)
    return tempfunc


def modified_sigmoid_function(x, a, b,c,d):
    """
    Modified sigmoid function for curve fitting.
    This function is used to fit the S-curve.
    """
    z = a*x + b + c/((x - d)**0.5)
    # ignore overflows in exp → exp(z) becomes np.inf, so 0.5/(1+inf) = 0.0
    with np.errstate(over='ignore'):
        y = 0.5 / (1 + np.exp(z))
    return y

def quadratic_function(x, a, b,c):
    """
    Linear function for curve fitting.
    """
    return a * x**2+b*x + c


def poly_function(x, a, b,c,d):
    """
    Linear function for curve fitting.
    """
    return a * x**3+b*x**2 + c*x+d



# Redefine turning point where the 2nd term is still significant in dy/dw
def refined_sweat_spot(alpha, beta, t, ratio=0.05):
    # We define turning point by solving: 1/alpha = ratio * (1/2) * beta / (w - t)^{3/2}
    # => (w - t)^{3/2} = (ratio * beta * alpha) / 2
    # => w = t + [(ratio * beta * alpha / 2)]^{2/3}
    return t + ((ratio * beta * alpha / 2) ** (2 / 3))


"""
Return the estimated sigma of y(w)
"""
def sigma_estimator(N,M):
    return np.sqrt(N**2*(N-M)/(M*(N-1)*(N-2*M)**2))


"""
Return the estimated sigma of Pw
"""
def subspace_sigma_estimator(N,M):
    return np.sqrt(M*(N-M)/(N-1))/N

def bias_estimator(N, M):
    """
    Bias = E[y(w)] - y(w)
    Estimated by: (1/2) * f''(P_w) * Var(P_w)
    where f(x) = ln(1/(2x) - 1)
    """
    # Pw = M / N
    # var_Pw = sigma_estimator(N, M)**2
    # f2 = 4 / (1 - 2 * Pw)**2 + 1 / Pw**2
    # bias = 0.5 * f2 * var_Pw
    # return 0
    return 1/2*(N/M)*(N-4*M)/(N-2*M)**2*(N-M)/(N-1)


def show_bias_estimator(N, M):
    """
    Bias = E[y(w)] - y(w)
    Estimated by: (1/2) * f''(P_w) * Var(P_w)
    where f(x) = ln(1/(2x) - 1)
    """
    Pw = M / N
    return (1 - Pw) / (2 * Pw * N)




def evenly_spaced_ints(minw, maxw, N):
    if N == 1:
        return [minw]
    if N > (maxw - minw + 1):
        return list(range(minw, maxw + 1))
     
    # Use high-resolution linspace, round, then deduplicate
    raw = np.linspace(minw, maxw, num=10 * N)
    rounded = sorted(set(map(int, raw)))
    
    # Pick N evenly spaced indices from the unique set
    indices = np.linspace(0, len(rounded) - 1, num=N, dtype=int)
    return [rounded[i] for i in indices]
