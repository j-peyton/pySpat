from scipy.stats import poisson, chisquare

def poisson_test(observed, expected, alpha=0.05):
    """
    Perform a Poisson test for spatial randomness.

    Parameters:
    ----------
    observed : int
        The observed number of points in the spatial pattern.
    expected : float
        The expected number of points based on intensity.
    alpha : float, optional
        Significance level for the test (default is 0.05).

    Returns:
    -------
    bool
        True if the pattern is significantly random, False otherwise.
    """
    # Calculate the Poisson probability
    lambda_ = expected
    p_value = poisson.cdf(observed - 1, lambda_)

    # Determine if the p-value is less than alpha
    return p_value > alpha


def chi_square_test(observed_counts, expected_counts, alpha=0.05):
    """
    Perform a Chi-Square test for spatial stratification.

    Parameters:
    ----------
    observed_counts : list of int
        Observed counts of points in different strata.
    expected_counts : list of float
        Expected counts of points in different strata based on uniform distribution.
    alpha : float, optional
        Significance level for the test (default is 0.05).

    Returns:
    -------
    bool
        True if the pattern is significantly different from uniform, False otherwise.
    """
    # Calculate Chi-Square statistic and p-value
    statistic, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    # Determine if the p-value is less than alpha
    return p_value < alpha
