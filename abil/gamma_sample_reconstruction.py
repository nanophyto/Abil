import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

def fit_gamma(p1, p2, x1, x2, upper_bound=1e24):
    """
    Find shape and scale parameters for a gamma distribution, given 
    percentiles p1 and p2 (p1 < p2) and values at those percentiles 
    x1 and x2 (x1 < x2). 

    Parameters
    ----------
    p1 : float
        Lower percentile **as a probability in (0, 1)** (e.g., 0.05 for 5%).
        Must be strictly between 0 and 1.
    p2 : float
        Upper percentile **as a probability in (0, 1)** (e.g., 0.95 for 95%).
        Must be strictly between 0 and 1.
    x1 : float
        Value at percentile `p1`. Must be positive.
    x2 : float
        Value at percentile `p2`. Must be positive.
    upper_bound : float, optional
        Upper bound on the search space for the shape `alpha` during
        scalar minimization. Defaults to 1e24.

    Returns
    -------
    (alpha, theta) : tuple of float
        Estimated shape and scale parameters. Returns `(np.nan, np.nan)` if
        inputs are invalid or optimization fails.

    """
    try:
        # Check assertions and return NA if any fail
        if not (0 < p1 < 1):
            return np.nan, np.nan
        if not (0 < p2 < 1):
            return np.nan, np.nan
        if not (x1 > 0):
            return np.nan, np.nan
        if not (x2 > 0):
            return np.nan, np.nan
            
        def score(a):
            """use a quadratic loss to find the point this goes negative"""
            return ((stats.gamma.ppf(p2, a)/stats.gamma.ppf(p1, a)) - (x2/x1))**2

        opt = minimize_scalar(score, bounds=(0, upper_bound))
        alpha = opt.x
        theta = x1 / stats.gamma.ppf(p1, alpha, scale=1)
        return alpha, theta
        
    except:
        # Return NA for any other errors that might occur during computation
        return np.nan, np.nan
    
    
def verify_fit(alpha, theta, n_samples=100, n_repeats=1, q=(5, 95)):
    """
    Given a shape and scale parameter for the gamma distribution,
    construct n_repeats batches of n_samples-sized samples
    from the gamma distribution, and report its qth percentiles. 

    Parameters
    ----------
    alpha : float
        Shape parameter of the Gamma distribution. Must be > 0.
    theta : float
        Scale parameter of the Gamma distribution. Must be > 0.
    n_samples : int, optional
        Number of draws per repeat. Defaults to 100.
    n_repeats : int, optional
        Number of independent replicates (columns). Defaults to 1.
    q : sequence of float, optional
        Percentiles (in **[0, 100]** units) to compute along each repeat.
        Defaults to (5, 95).

    Returns
    -------
    percentiles : np.ndarray, shape (len(q), n_repeats)
        Matrix of requested percentiles across repeats.
    """
    test = np.random.gamma(shape=alpha, scale=theta, size=(n_samples, n_repeats))
    percentiles = np.percentile(test, q=q, axis=0)
    assert percentiles.shape == (2, n_repeats)
    return percentiles


def generate_gamma_samples(x1, x2, p1=0.05, p2=0.95, sample_size=10000, upper_bound=1e3, species=None):
    """
    Given two values x1 and x2 at percentiles p1 and p2 respectively,
    approximate a gamma distribution that fits those percentiles, and
    return a sample of size sample_size from that distribution. 

    Parameters
    ----------
    x1 : float
        Value at percentile `p1`. Must be positive.
    x2 : float
        Value at percentile `p2`. Must be positive.
    p1 : float, optional
        Lower percentile **as a probability in (0, 1)**. Default is 0.05.
    p2 : float, optional
        Upper percentile **as a probability in (0, 1)**. Default is 0.95.
    sample_size : int, optional
        Size of the sample to return from the fitted Gamma. Default 10000.
    upper_bound : float, optional
        Upper bound for the shape search in `fit_gamma`. Default 1e3.
    species : str or None, optional
        Optional label included in the diagnostic printout.

    Returns
    -------
    sample : np.ndarray, shape (sample_size,)
        Random sample drawn from the fitted Gamma distribution.

    """
    a_, t_ = fit_gamma(p1, p2, x1, x2, upper_bound)

    x1_, x2_ = verify_fit(a_, t_, n_samples=10_000, n_repeats=1000)

    #print("alpha: " + str(a_))
    if species:
        print(
            f"For species {species}: "
            f"X1 is in the {(x1_ < x1).mean()*100:.0f}th percentile of replicate X1s, "
            f"and X2 is in the {(x2_ < x2).mean()*100:.0f}th percentile of replicate X2s. "
            "These should be close to the 50th if the results are accurate."
        )
    else:
        print(
            f"X1 is in the {(x1_ < x1).mean()*100:.0f}th percentile of replicate X1s, "
            f"and X2 is in the {(x2_ < x2).mean()*100:.0f}th percentile of replicate X2s. "
            "These should be close to the 50th if the results are accurate."
        )

    sample = np.random.gamma(shape=a_, scale=t_, size=sample_size)

    return(sample)

