import numpy as np

__title__ = "storage functions"
__author__ = "L. Varvelo, D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"


def save_psi_traj(phi_new, state_list, **kwargs):
    """
    Saves the real wave function

    Parameters
    ----------
    1. phi_new : list
                 Physical wave function.

    2. state_list : list
                    List of the states in the current basis.

    Returns
    -------
    1. psi : list
             Physical wave function.
    """
    psi = np.zeros_like(phi_new[:len(state_list)])
    psi[:] = phi_new[:len(state_list)]
    return psi


def save_phi_traj(phi_new, **kwargs):
    """
    Returns the full wave function

    Parameters
    ----------
    1. phi_new : list
                 Full wave function.

    Returns
    -------
    1. phi_new : list
                 Full wave function.
    """
    return phi_new


def save_phi_norm(phi_new, **kwargs):
    """
    Returns the L2-norm of the full wave function

    Parameters
    __________
    1. phi_new : list
                 Full wave function.

    Returns
    -------
    1. phi_norm : float
                  The L2-norm of the full wave function.
    """
    return np.linalg.norm(phi_new)


def save_t_axis(t_new, **kwargs):
    """
    Returns the new time point

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


def save_aux_new(aux_new, **kwargs):
    """
    Returns the list of auxiliaries in new basis

    Parameters
    ----------
    1. aux_new : list
                 List of auxiliaries.

    Returns
    -------
    1. aux_new : list
                 List of auxiliaries in new basis.
    """
    return aux_new

def save_state_list(state_list, **kwargs):
    """
    Returns the list of states in the current basis

    Parameters
    ----------
    1. state_list : list
                    List of the states in the current basis.

    Returns
    -------
    1. state_list : list
                    Lst of the states in the current basis.
    """
    return state_list


def save_list_nstate(state_list, **kwargs):
    """
    Returns the number of states in the current basis

    Parameters
    ----------
    1. state_list : list
                    List of the states in the current basis.

    Returns
    -------
    1. nstate : int
                Number of states in the current basis.
    """
    return len(state_list)


def save_list_nhier(aux_new, **kwargs):
    """
    Returns the number of auxiliary wave functions

    Parameters
    ----------
    1. aux_new : list
                 List of auxiliaries components [aux_new, aux_stable, aux_bound].

    Returns
    -------
    1. nhier : int
               Number of auxiliary wave functions in the current basis.
    """
    return len(aux_new)


storage_default_func = {'psi_traj':save_psi_traj, 'phi_traj':save_phi_traj,
                        'phi_norm':save_phi_norm, 't_axis':save_t_axis,
                        'aux_new':save_aux_new, 'state_list':save_state_list,
                        'list_nstate':save_list_nstate, 'list_nhier':save_list_nhier}