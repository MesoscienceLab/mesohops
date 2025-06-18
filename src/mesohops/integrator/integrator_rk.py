import copy
import numpy as np
from mesohops.util.physical_constants import hbar

__title__ = "Integrators, Runge-Kutta"
__author__ = "D. I. G. Bennett"
__version__ = "1.2"


def runge_kutta_step(dsystem_dt, phi, z_mem, z_rnd, z_rnd2, tau):
    """
    Performs a single Runge-Kutta step from the current time to a time tau forward.

    Parameters
    ----------
    1. dsystem_dt : function
                    Calculates the system derivatives.

    2. phi : np.array(complex)
             Full hierarchy vector.

    3. z_mem : np.array(complex)
               Noise memory drift terms for the bath [units: cm^-1].

    4. z_rnd : np.array(complex)
               Random numbers for the bath (at three time points) [units: cm^-1].

    5. z_rnd2 : np.array(complex)
                Secondary real contribution to the noise (at three time points).
                Imaginary portion discarded in dsystem_dt [units: cm^-1].
                For primary use-case, see:

                "Exact open quantum system dynamics using the Hierarchy of Pure States
                (HOPS)."
                Richard Hartmann and Walter T. Strunz J. Chem. Theory Comput. 13,
                p. 5834-5845 (2017)

    6. tau : float
             Timestep of the calculation [units: fs].

    Returns
    -------
    1. phi : np.array(complex)
             Updated hierarchy vector.

    2. z_mem : np.array(complex)
               Updated noise memory drift terms for the bath [units: cm^-1].
    """
    # Calculation constants
    # ---------------------
    k = [[] for i in range(4)]
    kz = [[] for i in range(4)]
    c_rk = [0.0, 0.5, 0.5, 1.0]
    i_zrnd = [0, 1, 1, 2]

    for i in range(4):
        # Update system values: phi_tmp, z_mem_tmp
        if i == 0:
            z_mem_tmp = copy.deepcopy(z_mem)
            phi_tmp = copy.deepcopy(phi)
        else:
            z_mem_tmp = z_mem + c_rk[i] * kz[i - 1] * tau / hbar
            phi_tmp = phi + c_rk[i] * k[i - 1] * tau / hbar

        # Calculate system derivatives
        k[i], kz[i] = dsystem_dt(
            phi_tmp, z_mem_tmp, z_rnd[:, i_zrnd[i]], z_rnd2[:, i_zrnd[i]]
        )

    # Actual Integration Step
    phi = phi + tau / hbar * (k[0] + 2.0 * k[1] + 2.0 * k[2] + k[3]) / 6.0
    z_mem = z_mem + tau / hbar * (kz[0] + 2.0 * kz[1] + 2.0 * kz[2] + kz[3]) / 6.0

    return phi, z_mem


def runge_kutta_variables(phi,z_mem, t, noise, noise2, tau, storage,
                          list_absindex_L2,effective_noise_integration=False):
    """
    Accepts a storage and noise objects and returns the pre-requisite variables for
    a runge-kutta integration step in a list that can be unraveled to correctly feed
    into runge_kutta_step.

    Parameters
    ----------
    1. phi : np.array(complex)
             Full hierarchy vector.

    2. z_mem : list(complex)
               List of memory terms [units: cm^-1].

    3. t : int
           Integration time point.

    4. noise : instance(HopsNoise)

    5. noise2 : instance(HopsNoise)

    6. tau : float
             Integration time step [units: fs].
             
    7. storage : instance(HopsStorage)

    8. effective_noise_integration: bool
                                    True indicates that the effective noise
                                    integration is used to take a moving average over
                                    the noise while False indicates otherwise.

    Returns
    -------
    1. variables : dict
                   Dictionary of variables needed for Runge Kutta.
    """
    if effective_noise_integration:
        tau_ratio = round(tau/noise.param["TAU"])
        tau_ratio2 = round(tau / noise2.param["TAU"])
        z_rnd_raw = noise.get_noise([t + (i/tau_ratio)*tau for i in
                                     range(round(tau_ratio*1.5))],list_absindex_L2)
        z_rnd2_raw = noise2.get_noise([t + (i / tau_ratio2) * tau for i in
                                       range(round(tau_ratio2 * 1.5))],list_absindex_L2)
        z_rnd = np.array([np.mean(z_rnd_raw[:,:round(tau_ratio/2)], axis=1),
                          np.mean(z_rnd_raw[:,round(tau_ratio/2):tau_ratio], axis=1),
                          np.mean(z_rnd_raw[:, tau_ratio:], axis=1)]).T
        z_rnd2 = np.array([np.mean(z_rnd2_raw[:, :round(tau_ratio2 / 2)], axis=1),
                           np.mean(z_rnd2_raw[:, round(tau_ratio2 / 2):tau_ratio2],
                                   axis=1),
                           np.mean(z_rnd2_raw[:, tau_ratio2:], axis=1)]).T

    else:
        z_rnd = noise.get_noise([t, t + tau * 0.5, t + tau],list_absindex_L2)
        z_rnd2 = noise2.get_noise([t, t + tau * 0.5, t + tau],list_absindex_L2)
        
    return {"phi": phi, "z_mem": z_mem, "z_rnd": z_rnd, "z_rnd2": z_rnd2, "tau": tau}
