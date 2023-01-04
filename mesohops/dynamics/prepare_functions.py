from mesohops.dynamics.noise_fft import FFTFilterNoise
from mesohops.dynamics.noise_zero import ZeroNoise

__title__ = "preparation functions"
__author__ = "D. I. G. Bennett"
__version__ = "1.2"


def prepare_noise(noise_param, system_param, flag=1):
    """
    Return the proper noise class given the user inputs.

    Parameters
    ----------
    1. noise_param : dict
                     Dictionary of noise parameter.

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
        noise_corr = {
            "CORR_FUNCTION": system_param["ALPHA_NOISE1"],
            "N_L2": system_param["N_L2"],
            "LIND_BY_NMODE": system_param["LIST_INDEX_L2_BY_NMODE1"],
            "CORR_PARAM": system_param["PARAM_NOISE1"],
        }
    elif flag == 2:
        noise_corr = {
            "CORR_FUNCTION": system_param["ALPHA_NOISE2"],
            "N_L2": system_param["N_L2"],
            "LIND_BY_NMODE": system_param["LIST_INDEX_L2_BY_NMODE2"],
            "CORR_PARAM": system_param["PARAM_NOISE2"],
        }
    else:
        Exception("Unknown flag value in prepare_noise")

    # Instantiate a HopsNoise subclass
    # --------------------------------
    if noise_param["MODEL"] == "FFT_FILTER":
        return FFTFilterNoise(noise_param, noise_corr)
    elif noise_param["MODEL"] == "ZERO":
        return ZeroNoise(noise_param, noise_corr)
    else:
        raise Exception("MODEL of NoiseDict is not known!")
