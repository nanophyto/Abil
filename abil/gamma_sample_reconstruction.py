import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

def fit_gamma(p1, p2, x1, x2, upper_bound=1e24):
    """
    Find shape and scale parameters for a gamma distribution, given 
    percentiles p1 and p2 (p1 < p2) and values at those percentiles 
    x1 and x2 (x1 < x2). 
    Returns (alpha, theta) or (NA, NA) if inputs are invalid.
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
    """
    test = np.random.gamma(shape=alpha, scale=theta, size=(n_samples, n_repeats))
    percentiles = np.percentile(test, q=q, axis=0)
    assert percentiles.shape == (2, n_repeats)
    return percentiles


def approximate_Gamma(x1, x2, p1=0.05, p2=0.95, sample_size=10000, upper_bound=1e3, species=None):
    """
    Given two values x1 and x2 at percentiles p1 and p2 respectively,
    approximate a gamma distribution that fits those percentiles, and
    return a sample of size sample_size from that distribution. 
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



