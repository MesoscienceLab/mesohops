import copy

import numpy as np

from mesohops.util.tensor_operations import contract_down_exact, extract_gs_amp

__title__ = "storage functions"
__author__ = "L. Varvelo, D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"


def save_psi_traj(phi_new, state_list, **kwargs):
    """
    Saves the real wave function.

    Parameters
    ----------
    1. phi_new : list(complex)
                 Physical wave function.

    2. state_list : list(int)
                    List of the states in the current basis.

    Returns
    -------
    1. psi : list(complex)
             Physical wave function.
    """
    psi = np.zeros_like(phi_new[:len(state_list)])
    psi[:] = phi_new[:len(state_list)]
    return psi


def save_phi_traj(phi_new, **kwargs):
    """
    Returns the full wave function.

    Parameters
    ----------
    1. phi_new : list(complex)
                 Full wave function.

    Returns
    -------
    1. phi_new : list(complex)
                 Full wave function.
    """
    return phi_new


def save_phi_norm(phi_new, **kwargs):
    """
    Returns the L2-norm of the full wave function.

    Parameters
    ----------
    1. phi_new : list(complex)
                 Full wave function.

    Returns
    -------
    1. phi_norm : float
                  L2-norm of the full wave function.
    """
    return np.linalg.norm(phi_new)


def save_t_axis(t_new, **kwargs):
    """
    Returns the new time point.

    Parameters
    ----------
    1. t_new : float
               Time point.

    Returns
    -------
    1. t_new : float
               Time point.
    """
    return t_new


def save_aux_list(aux_list, **kwargs):
    """
    Returns the list of auxiliaries in new basis.

    Parameters
    ----------
    1. aux_list : list(instance(AuxiliaryVector))
                  List of auxiliaries.

    Returns
    -------
    1. aux_list : list(np.array(int))
                  List of auxiliaries in new basis.
    """
    return [aux.array_aux_vec for aux in aux_list]

def save_state_list(state_list, **kwargs):
    """
    Returns the list of states in the current basis.

    Parameters
    ----------
    1. state_list : list(int)
                    List of the states in the current basis.

    Returns
    -------
    1. state_list : list(int)
                    List of the states in the current basis.
    """
    return state_list


def save_list_nstate(state_list, **kwargs):
    """
    Returns the number of states in the current basis.

    Parameters
    ----------
    1. state_list : list(int)
                    List of the states in the current basis.

    Returns
    -------
    1. nstate : int
                Number of states in the current basis.
    """
    return len(state_list)


def save_list_nhier(aux_list, **kwargs):
    """
    Returns the number of auxiliary wave functions.

    Parameters
    ----------
    1. aux_list : list(instance(AuxiliaryVector))
                  List of the current auxiliaries in the hierarchy basis.

    Returns
    -------
    1. nhier : int
               Number of auxiliary wave functions in the current basis.
    """
    return len(aux_list)

def save_list_aux_norm(phi_new, aux_list, **kwargs):
    """
    Returns the total population of each auxiliary wave function in the basis.

    Parameters
    ----------
    1. phi_new : list(complex)
                 Full wave function.
    2. aux_list : list(instance(AuxiliaryVector))
                  List of the current auxiliaries in the hierarchy basis.
    Returns
    -------
    1. list_aux_norm : np.array(float)
                       The population of each auxiliary wave function in the current
                       basis.
    """
    n_aux = len(aux_list)
    n_state = len(phi_new) // len(aux_list)
    try:
        return np.sqrt(np.sum(np.abs(phi_new).reshape(n_aux,n_state)**2,axis=1))
    except ValueError:
        print("WARNING: phi is not cleanly divisible into auxiliary wave functions "
              "and states. The save_list_aux_norm function has no meaningful output.")
        return 0

def save_z_mem(z_mem_new, **kwargs):
    """
    Returns the noise memory drift.
    """
    return z_mem_new


def save_list_zmemmodeidx_abs(list_zmemmodeidx_abs, **kwargs):
    """
    Returns the list of absolute indices for the z_mem modes used at this time.
    """
    return list_zmemmodeidx_abs


# ============================================================
# Tensor-specific storage functions
# ============================================================
# These are registered by HopsTensorTrajectory.__init__ to replace
# the default functions for keys that need MPS-aware handling.
# They receive a 'wavefunction' kwarg (HopsTensorWavefunction instance)
# passed by the tensor trajectory's store_step calls.


def save_phi_traj_tensor(wavefunction, **kwargs):
    """
    Returns the full hierarchy state as MPS cores (deep copy).

    Parameters
    ----------
    1. wavefunction: HopsTensorWavefunction
                     Tensor wavefunction instance.

    Returns
    -------
    1. list_cores: list(np.ndarray)
                   Deep copy of the MPS cores.
    """
    return copy.deepcopy(wavefunction.list_cores_phi)


def save_max_tensor_complexity(max_tensor_complexity, **kwargs):
    """
    Returns the peak uncompressed-MPS complexity observed during this timestep.

    The scalar is the max across all tensor_matvec_prod calls in one
    integration step (4 evaluations for RK4, 0 for TDVP).  See
    util.tensor_operations.calc_mps_complexity for the per-core formula.

    Parameters
    ----------
    1. max_tensor_complexity: int
                                Peak complexity for this step.

    Returns
    -------
    1. max_tensor_complexity: int
                                Same value, appended to the storage trace.
    """
    return max_tensor_complexity


def save_phi_norm_tensor(wavefunction, **kwargs):
    """
    Returns the L2-norm of the full hierarchy state via MPS contraction.

    Parameters
    ----------
    1. wavefunction: HopsTensorWavefunction
                     Tensor wavefunction instance.

    Returns
    -------
    1. phi_norm: float
                 L2-norm of the full hierarchy state.
    """
    n_state = len(wavefunction.M1_modes_per_site)
    V1_norm_sq = contract_down_exact(
        wavefunction.list_cores_phi,
        wavefunction.method,
        n_state,
    )
    return np.sqrt(np.sum(V1_norm_sq).real)


def save_psi_g_traj_tensor(wavefunction, **kwargs):
    """
    Returns the amplitude of the all-zeros configuration of the MPS.

    Counterpart to save_psi_traj for the vacuum-convention spectroscopy
    code path: under the vacuum convention extract_psi sees only the
    excited-state slots, so the physical |g> amplitude (the all-zeros
    configuration of the MPS) is not in psi_traj.  This hook captures
    it per storage step so detection-phase propagation can be a single
    traj.propagate(t3_max, t_step) call rather than a per-step Python
    loop calling extract_gs_amp.

    Parameters
    ----------
    1. wavefunction: HopsTensorWavefunction
                     Tensor wavefunction instance.

    Returns
    -------
    1. gs_amp: complex
               <0,...,0|psi> at the current timestep.
    """
    return extract_gs_amp(
        wavefunction.list_cores_phi, wavefunction.method,
    )


storage_default_func = {'psi_traj':save_psi_traj, 'phi_traj':save_phi_traj,
                        'phi_norm':save_phi_norm, 't_axis':save_t_axis,
                        'aux_list':save_aux_list, 'state_list':save_state_list,
                        'list_nstate':save_list_nstate, 'list_nhier':save_list_nhier,
                        'list_aux_norm':save_list_aux_norm, 'z_mem':save_z_mem,
                        'list_zmemmodeidx_abs': save_list_zmemmodeidx_abs,
                        'psi_g_traj': save_psi_g_traj_tensor}
