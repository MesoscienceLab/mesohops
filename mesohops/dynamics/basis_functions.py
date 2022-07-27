import numpy as np

__title__ = "Basis Functions"
__author__ = "D. I. G. Bennett, B. Citty"
__version__ = "1.2"

def determine_error_thresh(sorted_error, max_error):
    """
    This function determines which error value becomes the error threshold such
    that the sum of all errors below the threshold remains less then max_error.

    PARAMETERS
    ----------
    1. sorted_error : np.array
                      a list of error values
    2. max_error : float
                   the maximum error value

    RETURNS
    -------
    1. error_thresh : float
                      the error value at which the threshold is established

    """
    error_thresh = 0.0
    if len(sorted_error) > 0:
        index_thresh = np.argmax(np.sqrt(np.cumsum(sorted_error ** 2)) > max_error)

        if index_thresh > 0:
            error_thresh = sorted_error[index_thresh - 1]
        elif sorted_error[0] <= max_error:
            error_thresh = np.inf
    return error_thresh
