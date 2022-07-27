import numpy as np

__title__ = "storage functions"
__author__ = "L. Varvelo, D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"


def save_psi_traj(phi_new, state_list, **kwargs):
    """
    This function saves the real wave function

    PARAMETERS
    ----------
    1. phi_new : list
                 the physical wave function
    2. state_list : list
                    the list of the states in the current basis

    RETURNS
    -------
    1. psi : list
             the physical wave function
    """
    psi = np.zeros_like(phi_new[:len(state_list)])
    psi[:] = phi_new[:len(state_list)]
    return psi


def save_phi_traj(phi_new, **kwargs):
    """
    This function returns the full wave function

    PARAMETERS
    ----------
    1. phi_new : list
                 the full wave function

    RETURNS
    -------
    1. phi_new : list
                 the full wave function
    """
    return phi_new


def save_phi_norm(phi_new, **kwargs):
    """
    This function returns the L2-norm of the full wave function

    PARAMETERS
    __________
    1. phi_new : list
                 The full wave function

    RETURNS
    -------
    1. phi_norm : float
                  The L2-norm of the full wave function
    """
    return np.linalg.norm(phi_new)


def save_t_axis(t_new, **kwargs):
    """
    This function returns the new time point

    PARAMETERS
    ----------
    1. t_new : float
               the time point

    RETURNS
    -------
    1. t_new : float
               the time point
    """
    return t_new


def save_aux_new(aux_new, **kwargs):
    """
    This function returns the list of auxiliaries in new basis

    PARAMETERS
    ----------
    1. aux_new : list
                 list of auxiliaries components [aux_new, aux_stable, aux_bound]

    RETURNS
    -------
    1. aux_new : list
                 list of auxiliaries in new basis (H_1)
    """
    return aux_new[0]


def save_aux_stable(aux_new, **kwargs):
    """
    This function returns the list of stable auxiliaries in the new basis (H_S)

    PARAMETERS
    ----------
    1. aux_new : list
                 list of auxiliaries components [aux_new, aux_stable, aux_bound]

    RETURNS
    -------
    1. aux_stable : list
                    list of stable auxiliaries in the new basis (H_S)
    """
    return aux_new[1]


def save_aux_bound(aux_new, **kwargs):
    """
    This function returns the list of boundary auxiliaries in the new basis (H_B)

    PARAMETERS
    ----------
    1. aux_new : list
                 list of auxiliaries components [aux_new, aux_stable, aux_bound]

    RETURNS
    -------
    1. aux_bound : list
                   list of boundary auxiliaries in the new basis (H_B)
    """

    return aux_new[2]


def save_state_list(state_list, **kwargs):
    """
    This function returns the list of states in the current basis

    PARAMETERS
    ----------
    1. state_list : list
                    the list of the states in the current basis

    RETURNS
    -------
    1. state_list : list
                    the list of the states in the current basis
    """
    return state_list


def save_list_nstate(state_list, **kwargs):
    """
    This function returns the number of states in the current basis

    PARAMETERS
    ----------
    1. state_list : list
                    the list of the states in the current basis

    RETURNS
    -------
    1. nstate : int
                the numnber of states in the current basis
    """
    return len(state_list)


def save_list_nhier(aux_new, **kwargs):
    """
    This function returns the number of auxiliary wave functions

    PARAMETERS
    ----------
    1. aux_new : list
                 list of auxiliaries components [aux_new, aux_stable, aux_bound]

    RETURNS
    -------
    1. nhier : int
               the number of auxiliary wave functions in the current basis
    """
    return len(aux_new[0])


storage_default_func = {'psi_traj':save_psi_traj, 'phi_traj':save_phi_traj,
                        'phi_norm':save_phi_norm, 't_axis':save_t_axis,
                        'aux_new':save_aux_new, 'aux_stable':save_aux_stable, 'aux_bound':save_aux_bound,
                        'state_list':save_state_list, 'list_nstate':save_list_nstate, 'list_nhier':save_list_nhier}