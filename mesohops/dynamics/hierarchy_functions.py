import numpy as np

__title__ = "Hierarchy Functions"
__author__ = "D. I. G. Bennett, L. Varvelo"
__version__ = "1.4"

def filter_aux_triangular(list_aux, list_boolean_by_mode, kmax_2):
    """
    Filters a list of auxiliaries such that the total depth of a given auxiliary in
    the specified modes is no greater than a user-provided triangular truncation depth.

    Parameters
    ----------
    1. list_aux : list(instance(AuxiliaryVector))
                  List of auxiliaries.

    2. list_boolean_by_mode : list(bool)
                              True indicates the associated mode is filtered while False
                              indicates otherwise.

    3. kmax_2 : int
                Triangular truncation depth for the subset of modes specified in
                list_boolean_by_mode.

    Returns
    -------
    1. list_new_aux : list
                      Filtered list of auxiliaries.
    """
    aux_list_orig = list(list_aux)
    list_aux = np.array([aux.todense() for aux in aux_list_orig])
    test_triangle = np.sum(list_aux[:, list_boolean_by_mode], axis=1) <= kmax_2
    return [aux for (i, aux) in enumerate(aux_list_orig) if test_triangle[i]]

def filter_aux_longedge(list_aux, list_boolean_by_mode, kdepth):
    """
    Filters a list of auxiliaries such that any auxiliary with depth in the
    specified modes and total depth greater than kdepth must be an edge term with
    depth only in one mode.

    Parameters
    ----------
    1. list_aux : list(instance(AuxiliaryVector))
                  List of auxiliaries.

    2. list_boolean_by_mode : list(bool)
                              True indicates the associated mode is filtered while
                              False indicates otherwise.

    3. kdepth : int
                Depth beyond which only edge members of hierarchy are kept for
                filtered modes.

    Returns
    -------
    1. list_new_aux : list(tuple)
                      Filtered list of auxiliaries.
    """
    return[
        aux
    for aux in list_aux
    if check_long_edge(aux, list_boolean_by_mode, kdepth)
    ]

def check_long_edge(aux_vec, list_boolean, kdepth):
    """
    Checks if an individual auxiliary should be filtered out by the longedge filter.

    Parameters
    ----------
    1. aux : instance(AuxiliaryVector)

    2. list_boolean : list(bool)
                      The modes to filter. True indicates that the individual auxiliary
                      is filtered while False indicates otherwise.

    3. kdepth : int
                Depth beyond which only edge members of hierarchy are kept for
                filtered modes.

    RETURNS
    -------
    1. check_aux : bool
                   True indicates that the aux should be kept while False
                   indicates otherwise.
    """
    aux = aux_vec.get_values_nonzero(np.arange(len(list_boolean))[list_boolean])
    return (
        (np.sum(aux_vec) <= kdepth)
        or (np.sum(aux) == 0)
        or (len(aux_vec.keys()) <= 1)
    )


def check_markovian(aux, list_boolean):
    """
    Checks whether an auxiliary is an allowed markovian mode or a non-markovian mode.

    Parameters
    ----------
    1. aux : instance(AuxiliaryVector)

    2. list_boolean : list(bool)
                      True indicates the associated mode is filtered while False
                      indicates otherwise.

    Returns
    -------
    1. allowed : bool
                 True indicates that the auxiliary is allowed while False indicates
                 otherwise.
    """
    flag_adaptive = True
    if np.sum(aux) == 0:
        flag_adaptive = True
    elif any(np.array(list_boolean)[aux.toarray()[:, 0]]) and np.sum(aux) > 1:
        flag_adaptive = False
    return flag_adaptive


def filter_markovian(list_aux, list_boolean):
    """
    Filters a list of auxiliaries based on whether it is a non-markovian
    auxiliary or an allowed markovian auxiliary.

    Parameters
    ----------
    1. list_aux : list(instance(AuxiliaryVector))
                  List of auxiliaries.

    2. list_boolean : list(bool)
                      True indicates the associated mode is filtered while False
                      indicates otherwise.

    Returns
    -------
    list_aux : list
               Filtered auxiliary list.
    """
    return [aux for aux in list_aux if check_markovian(aux, list_boolean)]
