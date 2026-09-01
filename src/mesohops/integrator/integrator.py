"""
Time-integration routines for vector-based HOPS.

Each integrator operates on the flat hierarchy vector (``phi``) used by
``HopsTrajectory``.

Functions
---------
runge_kutta_step(dsystem_dt, phi, z_mem, z_rnd, z_rnd2, tau)
    Classic RK4 step for the flat hierarchy vector.

runge_kutta_variables(phi, z_mem, t, noise, noise2, tau, storage, ...)
    Gathers noise samples at the three RK4 time points and returns a dict
    ready to unpack into runge_kutta_step.
"""
from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np

from mesohops.noise.hops_noise import HopsNoise
from mesohops.storage.hops_storage import HopsStorage
from mesohops.util.physical_constants import hbar

__title__ = 'Integrators'
__author__ = 'D. I. G. Bennett, B. Z. Citty'
__version__ = '1.6'


def runge_kutta_step(
    dsystem_dt: Callable,
    phi: np.ndarray,
    z_mem: np.ndarray,
    z_rnd: np.ndarray,
    z_rnd2: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
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
                Secondary, (typically) real contribution to the noise (at three time
                points). Imaginary portion discarded by the FLAG_REAL key of the
                noise object's parameter dictionary [units: cm^-1].
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


def runge_kutta_variables(
    phi: np.ndarray,
    z_mem: np.ndarray,
    t: float,
    noise: HopsNoise,
    noise2: HopsNoise,
    tau: float,
    storage: HopsStorage,
    list_l2idx_abs: list[int],
    effective_noise_integration: bool = False,
) -> dict:
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
        # Number of fine noise sub-steps per integration step
        tau_ratio = round(tau/noise.param["TAU"])
        tau_ratio2 = round(tau / noise2.param["TAU"])
        # Sample noise at fine resolution over [t, t + 1.5*tau)
        z_rnd_raw = noise.get_noise([t + (i/tau_ratio)*tau for i in
                                     range(round(tau_ratio*1.5))],list_l2idx_abs)
        z_rnd2_raw = noise2.get_noise([t + (i / tau_ratio2) * tau for i in
                                       range(round(tau_ratio2 * 1.5))],list_l2idx_abs)
        # Average fine noise into 3 bins matching RK4 time-points:
        #   bin 0: [t, t+tau/2)         -> noise at t
        #   bin 1: [t+tau/2, t+tau)     -> noise at t+tau/2
        #   bin 2: [t+tau, t+1.5*tau)   -> noise at t+tau
        z_rnd = np.array([np.mean(z_rnd_raw[:,:round(tau_ratio/2)], axis=1),
                          np.mean(z_rnd_raw[:,round(tau_ratio/2):tau_ratio], axis=1),
                          np.mean(z_rnd_raw[:, tau_ratio:], axis=1)]).T
        z_rnd2 = np.array([np.mean(z_rnd2_raw[:, :round(tau_ratio2 / 2)], axis=1),
                           np.mean(z_rnd2_raw[:, round(tau_ratio2 / 2):tau_ratio2],
                                   axis=1),
                           np.mean(z_rnd2_raw[:, tau_ratio2:], axis=1)]).T

    else:
        z_rnd = noise.get_noise([t, t + tau * 0.5, t + tau],list_l2idx_abs)
        z_rnd2 = noise2.get_noise([t, t + tau * 0.5, t + tau],list_l2idx_abs)

    return {"phi": phi, "z_mem": z_mem, "z_rnd": z_rnd, "z_rnd2": z_rnd2, "tau": tau}
