import numpy as np

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


storage_default_func = {'psi_traj':save_psi_traj, 'phi_traj':save_phi_traj,
                        'phi_norm':save_phi_norm, 't_axis':save_t_axis,
                        'aux_list':save_aux_list, 'state_list':save_state_list,
                        'list_nstate':save_list_nstate, 'list_nhier':save_list_nhier,
                        'list_aux_norm':save_list_aux_norm, 'z_mem':save_z_mem}
