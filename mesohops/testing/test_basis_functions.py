from pyhops.dynamics.basis_functions import determine_error_thresh
import numpy as np


def test_determine_error_thresh():
    """
    Test to ensure that determine_error_thresh returns the first index at which the
    cumulative sum of the pythagorean of error is greater than some threshold error.
    """

    # Cumulative sum of error will be [1, sqrt(5), sqrt(14), sqrt(30), sqrt(55)]
    test_error = np.array([1,2,3,4,5])  # change test error to be floats and non sequential values

    assert determine_error_thresh([], 100) == 0.0
    assert determine_error_thresh(test_error, 8) == np.inf
    assert determine_error_thresh(test_error, 6) == 4
    assert determine_error_thresh(test_error, np.sqrt(30)) == 4
    assert determine_error_thresh(test_error, np.sqrt(29.99)) == 3
    assert determine_error_thresh(test_error, 1) == 1
    assert determine_error_thresh(test_error, 0.5) == 0
