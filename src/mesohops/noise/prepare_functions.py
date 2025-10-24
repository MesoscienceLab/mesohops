import warnings
from collections import Counter
from mesohops.noise.hops_noise import HopsNoise
from mesohops.util.helper_functions import array_to_tuple

__title__ = "preparation functions"
__author__ = "D. I. G. B. Raccah, J. K. Lynd"
__version__ = "1.6"

def prepare_hops_noise(noise_param, system_param, noise_type=1):
    """
    Returns the proper noise class given the user inputs.

    Parameters
    ----------
    1. noise_param : dict
                     Dictionary of noise parameters.

    2. system_param : dict
                      Dictionary of system parameters.

    3. noise_type : int
                    Defines whether the noise is noise 1 (the noise associated with
                    the bath correlation function modes of the hierarchy) or noise 2
                    (an explicit time-dependence in the system Hamiltonian), which is
                    typically purely real (options: 1 or 2).

    Returns
    -------
    1. noise : instance(HopsNoise)

    """
    # DETERMINE CORRELATION PARAMETERS
    # Skips checking for all the noise 2 params if noise 2 is defaulted to a zero model.
    if noise_type == 1 or noise_param["MODEL"] == "ZERO":
        # Creates list of unique l2 tuples in order they appear in "L_NOISE1"
        l2_as_tuples = [array_to_tuple(L2) for L2 in system_param["L_NOISE1"]]
        list_unique_l2_as_tuples = list(Counter(l2_as_tuples))
        noise_corr = {
            "CORR_FUNCTION": system_param["ALPHA_NOISE1"],
            "N_L2": len(set(list_unique_l2_as_tuples)),
            "LIND_BY_NMODE": system_param["LIST_INDEX_L2_BY_NMODE1"],
            "NMODE_BY_LIND": system_param["LIST_NMODE1_BY_INDEX_L2"],
            "CORR_PARAM": system_param["PARAM_NOISE1"],
        }

        if not "FLAG_REAL" in noise_param.keys():
            noise_param["FLAG_REAL"] = False
        if noise_param["FLAG_REAL"] == True:
            warnings.warn("Noise 1 should never be flagged real. For a purely "
                          "real noise, set PARAM_NOISE1 to ensure that noise 1 "
                          "goes to zero and use noise 2 for the real noise "
                          "instead.")
            noise_param["FLAG_REAL"] = False

    elif noise_type == 2:
        # Creates list of unique l2 tuples in order they appear in "L_NOISE2"
        l2_as_tuples = [array_to_tuple(L2) for L2 in system_param["L_NOISE2"]]
        list_unique_l2_as_tuples = list(Counter(l2_as_tuples))
        noise_corr = {
            "CORR_FUNCTION": system_param["ALPHA_NOISE2"],
            "N_L2": len(set(list_unique_l2_as_tuples)),
            "LIND_BY_NMODE": system_param["LIST_INDEX_L2_BY_NMODE2"],
            "NMODE_BY_LIND": system_param["LIST_NMODE2_BY_INDEX_L2"],
            "CORR_PARAM": system_param["PARAM_NOISE2"],
        }
        # Noise 2 defaults to purely-real noise, in accordance with the most common
        # use cases: time-dependence of the Hermitian system Hamiltonian and the
        # noise 2 laid out in "Exact open quantum system dynamics using the Hierarchy
        # of Pure States (HOPS)." Richard Hartmann and Walter T. Strunz J. Chem.
        # Theory Comput. 13, p. 5834-5845 (2017).
        if "FLAG_REAL" not in noise_param.keys():
            noise_corr["FLAG_REAL"] = True
            warnings.warn("Noise 2 FLAG_REAL not specified: setting to True. If noise 2 "
                          "should not be purely real, specify FLAG_REAL as False in "
                          "the noise 2 parameters during initialization of the HOPS "
                          "trajectory.")

    else:
        Exception("Unknown noise_type value in prepare_hops_noise")

    # Instantiate a HopsNoise subclass
    # --------------------------------
    if noise_param["MODEL"] == "FFT_FILTER" or noise_param["MODEL"] == "ZERO" or \
            noise_param["MODEL"] == "PRE_CALCULATED":
        return HopsNoise(noise_param, noise_corr)
    else:
        raise Exception("MODEL of NoiseDict is not known!")
