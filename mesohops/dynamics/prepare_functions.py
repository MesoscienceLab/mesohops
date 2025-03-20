from mesohops.dynamics.hops_noise import HopsNoise
from collections import Counter
from mesohops.util.helper_functions import array_to_tuple

__title__ = "preparation functions"
__author__ = "D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"

def prepare_hops_noise(noise_param, system_param, flag=1):
    """
    Returns the proper noise class given the user inputs.

    Parameters
    ----------
    1. noise_param : dict
                     Dictionary of noise parameters.

    2. system_param : dict
                      Dictionary of system parameters.

    3. flag : int
              1-NOISE1 parameters, 2-NOISE2 parameters.

    Returns
    -------
    1. noise : instance(HopsNoise)

    """
    # DETERMINE CORRELATION PARAMETERS
    if flag == 1:
        # Creates list of unique l2 tuples in order they appear in "L_NOISE1"
        l2_as_tuples = [array_to_tuple(L2) for L2 in system_param["L_NOISE1"]]
        list_unique_l2_as_tuples = list(Counter(l2_as_tuples))
        noise_corr = {
            "CORR_FUNCTION": system_param["ALPHA_NOISE1"],
            "N_L2": len(set(list_unique_l2_as_tuples)),
            "LIND_BY_NMODE": system_param["LIST_INDEX_L2_BY_NMODE1"],
            "CORR_PARAM": system_param["PARAM_NOISE1"],
        }
    elif flag == 2:
        # Creates list of unique l2 tuples in order they appear in "L_NOISE2"
        l2_as_tuples = [array_to_tuple(L2) for L2 in system_param["L_NOISE2"]]
        list_unique_l2_as_tuples = list(Counter(l2_as_tuples))
        noise_corr = {
            "CORR_FUNCTION": system_param["ALPHA_NOISE2"],
            "N_L2": len(set(list_unique_l2_as_tuples)),
            "LIND_BY_NMODE": system_param["LIST_INDEX_L2_BY_NMODE2"],
            "CORR_PARAM": system_param["PARAM_NOISE2"],
        }
    else:
        Exception("Unknown flag value in prepare_hops_noise")

    # Instantiate a HopsNoise subclass
    # --------------------------------
    if noise_param["MODEL"] == "FFT_FILTER" or noise_param["MODEL"] == "ZERO" or \
            noise_param["MODEL"] == "PRE_CALCULATED":
        return HopsNoise(noise_param, noise_corr)
    else:
        raise Exception("MODEL of NoiseDict is not known!")
