# utils.py
# Utility functions for testing in the mesohops library.
# Provides deep comparison utilities for complex data structures such as numpy arrays, sparse matrices, lists, and dictionaries.

import numpy as np  # Import numpy for array operations and testing utilities
import scipy as sp  # Import scipy for sparse matrix support


def _compare_values(v1, v2):
    """
    Recursively compares two values for equality, supporting a variety of data types.
    Handles numpy arrays, scipy sparse matrices, lists, dictionaries, and scalars.
    Raises an AssertionError if the values are not considered equal.

    Inputs
    ------
    1. v1 : Any
           First value to compare.
    2. v2 : Any
           Second value to compare.

    Returns
    -------
    None
    """
    # Compare sparse matrices by converting to dense arrays
    if sp.sparse.issparse(v1) or sp.sparse.issparse(v2):
        # Both must be sparse matrices of compatible shape and type
        assert sp.sparse.issparse(v1) and sp.sparse.issparse(v2), "Both values must be sparse matrices."
        np.testing.assert_allclose(v1.toarray(), v2.toarray())

    # Compare numpy arrays
    elif isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
        # Handle object arrays element-wise (recursively)
        if isinstance(v1, np.ndarray) and v1.dtype == object:
            assert isinstance(v2, np.ndarray) and v2.dtype == object, "Both arrays must be object dtype."
            assert v1.shape == v2.shape, "Object arrays must have the same shape."
            for a, b in zip(v1.flat, v2.flat):
                _compare_values(a, b)
        else:
            # Use numpy's allclose for numerical arrays
            np.testing.assert_allclose(v1, v2)

    # Recursively compare lists
    elif isinstance(v1, list) and isinstance(v2, list):
        assert len(v1) == len(v2), "Lists must have the same length."
        for a, b in zip(v1, v2):
            _compare_values(a, b)

    # Recursively compare dictionaries
    elif isinstance(v1, dict) and isinstance(v2, dict):
        assert set(v1.keys()) == set(v2.keys()), "Dictionaries must have the same keys."
        for k in v1:
            _compare_values(v1[k], v2[k])

    else:
        # Fallback for scalars and other comparable objects
        assert v1 == v2, f"Values {v1} and {v2} are not equal."


def compare_dictionaries(dict1, dict2):
    """
    Assert that two parameter dictionaries are equivalent, deeply comparing all values.
    Useful for testing that two sets of parameters or results are identical in structure and content.

    Parameters
    ----------
    1. dict1 : dict
               First dictionary for comparison.
    2. dict2 : dict
               Second dictionary for comparison.

    Returns
    -------
    None
    """
    # Ensure both dictionaries have the same set of keys
    assert set(dict1.keys()) == set(dict2.keys()), "Dictionaries must have the same keys."
    for key in dict1:
        print(f'Comparing {key}')  # Print the key being compared for debugging
        _compare_values(dict1[key], dict2[key])
