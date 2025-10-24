import numpy as np
from mesohops.util.physical_constants import hbar


def bcf_exp(t_axis, g, w):
    r"""
    This is the form of the correlation function:
    alpha(t) = \displastyle\ g exp(-w*t)

    Parameters
    ----------
    1. t_axis : np.array(float)
                Time axis [units: fs].

    2. g : complex
           Exponential prefactor [units: cm^-2].

    3. w : complex
           Exponent [units: cm^-1].

    Returns
    -------
    1. bcf : np.array(complex)
             Exponential bath correlation function sampled on t_axis.
    """
    return g * np.exp(-w * t_axis / hbar)