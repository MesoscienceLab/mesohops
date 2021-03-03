import numpy as np
from mesohops.util.physical_constants import kB, hbar

__title__ = "bath_corr_functions"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"

# Bath Correlation Functions
# --------------------------
# Note: Remember that the function arguments determine the
# parameters that are in the 'GW_SYSBATH' slot in the system
# dictionary.


def bcf_exp(t_axis, g, w):
    """
    This is the form of the correlation function
    alpha(t) = \displastyle\ g exp(-w*t)

    PARAMETERS
    ----------
    1. t_axis : array
                array of time points
    2. g : complex floating point number
           the exponential prefactor [cm^-2]
    3. w : complex floating point number
           the exponent [cm^-1]

    RETURNS
    -------
    1. bcf : array
             the exponential bath correlation function sampled on t_axis
    """
    return g * np.exp(-w * t_axis / hbar)


def bcf_convert_sdl_to_exp(lambda_sdl, gamma_sdl, omega_sdl, temp):
    """
    This function converts a shifted drude-lorentz spectral density
    parameters to the exponential equivalent.

    NOTE: THIS WILL NEED TO BE REPLACED WITH A MORE ROBUST FITTING
          ROUTINE SIMILAR TO WHAT EISFELD HAS DONE PREVIOUSLY.

    PARAMETERS
    ----------
    1. lamnda_sdl : float
                    the reorganization energy [cm^-1]
    2. gamma_sdl : float
                   the reorganization time scale [cm^-1]
    3. omega_sdl : float
                   the vibrational frequency [cm^-1]
    4. temp : float
              the temperature [K]

    RETURNS
    -------
    1. g_exp : complex
               the exponential prefactor [cm^-2]
    2. w_exp : complex
               the exponent [cm^-1]
    """
    beta = 1 / (kB * temp)
    g_exp = 2 * lambda_sdl / beta - 1j * lambda_sdl * gamma_sdl
    w_exp = gamma_sdl - 1j * omega_sdl

    return (g_exp, w_exp)


def bcf_convert_Ishizaki_sdl_to_exp(lambda_sdl, gamma_sdl, temp, epsilon_sdl,
                                    k_matsubara):
    """
    This function converts a Drude-Lorentz special density, altered so that Matsubara
    frequencies cause fewer issues as described in "Prerequisites for Relevant Spectral
    Density and Convergence of Reduced Density Matrices at Low Temperatures" by Akihito
    Ishizaki, to exponential form.

    NOTE: THOUGH THIS SPECTRAL DENSITY IS MORE ACCURATE AT LOWER TEMPERATURES THAN
    THE OTHER SDL_TO_EXP FUNCTION, IT IS *NOT* EQUIVALENT TO THE HEOM LOW-TEMPERATURE
    CORRECTION: AN EQUIVALENT TO THAT METHOD IS STILL PENDING IN HOPS.

    PARAMETERS
    ----------
     1. lambda_sdl : float
                    the reorganization energy [cm^-1]
    2. gamma_sdl : float
                   the reorganization time scale [cm^-1]. IMPORTANT NOTE: THIS
                   CORRESPONDS TO 2X THE GAMMA IN OUR PREVIOUS DRUDE-LORENTZ MODEL!
    3. temp : float
              the temperature [K]
    4. epsilon_sdl: float
                    an ad-hoc parameter for conversion of linear dependence on time.
                    See the paper. This should never be more than gamma_sdl/4 [cm^-1]
    5. k_matsubara: int
                    the number of Matsubara frequencies to be included [number]

    RETURNS
    -------
    1. list_exp_factors : list(tuple(complex))
                          a list of all exponential prefactors and exponents that
                          will be needed. The order is [(g_exp, w_exp of 1st
                          classical mode), (2nd classical mode), (1st Matsubara
                          mode), (2nd Matsubara mode) ... (kth Matsubara mode)]. The
                          prefactors are in units of [cm^-2] and the exponents are in
                          units of [cm^-1]

    """
    # This function returns 2 exponential modes + 1 per Matsubara frequency to be
    # included. First, we define central parameters.
    GAMMA_plus = gamma_sdl + 1j*epsilon_sdl
    GAMMA_minus = gamma_sdl - 1j*epsilon_sdl
    beta = 1/(temp*kB)


    # The overall correlation function is D_cl(t) - (i/2)Phi(t) + sum[D_k(t)]. This
    # will break down into 2 + k_matsubara exponential modes.
    w_exp_plus_mode = GAMMA_plus
    w_exp_minus_mode = GAMMA_minus

    # The response function is found via...
    phi_t = (1j/epsilon_sdl)*lambda_sdl*GAMMA_plus*GAMMA_minus
    g_exp_plus_mode = -(1j/2)*phi_t
    g_exp_minus_mode = -(1j/2)*np.conj(phi_t)

    # The classical component of the symmetrized correlation function is...
    g_exp_plus_mode += (1j/(beta*epsilon_sdl))*lambda_sdl*GAMMA_minus
    g_exp_minus_mode += np.conj((1j/(beta*epsilon_sdl))*lambda_sdl*GAMMA_minus)

    # Unit check: cm^-1*cm*cm^-1*cm^-1 is in units of cm^-2.

    # We build the spectral density and spectral energy functions below to calculate
    # the quantum corrections to the symmetrized correlation function.
    def J(w):
        return 4*lambda_sdl*gamma_sdl*GAMMA_plus*GAMMA_minus*w/\
               ((w**2 + GAMMA_plus**2)*(w**2 + GAMMA_minus**2))

    def E(k_freq, w):
        return (2*w**2)/(beta*(w**2-k_freq**2))


    # An empty list of Matsubara frequency prefactors and exponential frequencies
    list_mats_factors = []

    # Define the base Matsubara frequency where nu_k = k*base.
    matsubara_base = 2 * np.pi / beta

    for k in np.arange(k_matsubara)+1:
        # Find the prefactor-frequency pair of the exponential form of the kth Matsubara
        # frequency.
        matsubara_w = matsubara_base*k
        matsubara_g = (2j/beta)*J(1j*matsubara_w)

        # Find the kth quantum correction term to the non-Matsubara terms.
        quantum_corr = (1j*lambda_sdl*GAMMA_minus/epsilon_sdl)*E(matsubara_w, GAMMA_plus)
        g_exp_plus_mode += quantum_corr
        g_exp_minus_mode += np.conj(quantum_corr)
        list_mats_factors.append((matsubara_g, matsubara_w))

    # The return structure is a list of length k_matsubara + 2 containing tuples of
    # the g_exp and w_exp prefactor-exponent couplets.
    list_exp_factors = [(g_exp_plus_mode, w_exp_plus_mode), (g_exp_minus_mode,
                                                w_exp_minus_mode)] + list_mats_factors

    return list_exp_factors