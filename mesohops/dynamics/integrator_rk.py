import copy
from pyhops.util.physical_constants import hbar

__title__ = "Integrators, Runge-Kutta"
__author__ = "D. I. G. Bennett"
__version__ = "1.2"


def runge_kutta_step(dsystem_dt, phi, z_mem, z_rnd, z_rnd2, tau):
    """
    This function performs a single Runge-Kutta step from the current
    time to a time tau forward.

    PARAMETERS
    ----------
    1. dsystem_dt : function
                    a function that calculates the system derivatives
    2. phi : array
             the full hierarchy vector
    3. z_mem : array
              the memory terms for the bath
    4. z_rnd : array
               the random numbers for the bath (at three time points)
    5. tau : float
             the timestep of the calculation

    RETURNS
    -------
    1. phi : array
             the updated hierarchy vector
    2. z_mem : array
               the updated memory term
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


def runge_kutta_variables(phi,z_mem, t, noise, noise2, tau, storage
                          ):
    """
    This is a function that accepts a storage and noise objects and returns the
    pre-requisite variables for a runge-kutta integration step in a list
    that can be unraveled to correctly feed into runge_kutta_step.

    PARAMETERS
    ----------
    1. phi : array
             the full hierarchy vector
    2. z_mem : HopsNoise object
               an instantiation of HopsNoise associated with the trajectory
    3. t : int
           the time point
    4. noise : HopsNoise object
               an instantiation of HopsNoise associated with the trajectory
    5. noise2 : HopsNoise object
                an instantiation of HopsNoise associated with the trajectory
    6. tau : float
             the time step

    RETURNS
    -------
    1. variables : dict
                   a dictionary of variables needed for Runge Kutta
    """
    z_rnd = noise.get_noise([t, t + tau * 0.5, t + tau])
    z_rnd2 = noise2.get_noise([t, t + tau * 0.5, t + tau])

    return {"phi": phi, "z_mem": z_mem, "z_rnd": z_rnd, "z_rnd2": z_rnd2, "tau": tau}
