import numpy as np

__title__ = "Basis Functions"
__author__ = "D. I. G. Bennett, B. Citty"
__version__ = "1.2"


def determine_error_thresh(sorted_error, max_error, offset=0.0):
    """
    Determines which error value becomes the error threshold such
    that the sum of all errors below the threshold remains less than max_error.

    Parameters
    ----------
    1. sorted_error : np.array
                      List of error values, sorted least to greatest.

    2. max_error : float
                   Maximum error value.

    3. offset : float
                Pre-calculated squared-error value that all values in sorted_error
                must be modulated by.

    Returns
    -------
    1. error_thresh : float
                      Error value at which the threshold is established.
    """
    error_thresh = 0.0
    if len(sorted_error) > 0:
        index_thresh = np.argmax(np.cumsum(sorted_error) + offset >= max_error)
        if index_thresh > 0:
            error_thresh = sorted_error[index_thresh - 1]

        # If argmax = 0, either all cumulatively summed elements are less than
        # max_error (and should all be truncated from the basis), or all elements are
        # greater than/equal to max_error (and should all be added to the basis).
        elif sorted_error[0] + offset < max_error:
            error_thresh = np.inf
    return error_thresh
    
def calculate_delta_bound(delta_sq, stable_error_sq):

    """
    Calculates the error remaining after the stable step.  The total error E 
    satisfies E^2 = E_S^2 + E_B^2 < delta_sq where E_S and E_B are the stable 
    and boundary errors, respectively. Therefore, the remaining squared boundary error 
    threshold delta_bound_sq is given by delta_bound_sq = delta_sq - E_S^2.
    
    Parameters
    ----------
    1. delta_sq : float
                  Total squared error tolerance.
    
    2. stable_error_sq : float
                         Squared stable error.
                         
    Returns
    -------
    1. delta_bound_sq : float
                        Boundary squared error tolerance.
    """
    delta_bound_sq = delta_sq - stable_error_sq
    return delta_bound_sq
