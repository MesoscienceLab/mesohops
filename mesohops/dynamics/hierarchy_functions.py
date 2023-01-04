import numpy as np

__title__ = "Hierarchy Functions"
__author__ = "D. I. G. Bennett, L. Varvelo"
__version__ = "1.2"


def filter_aux_triangular(list_aux, list_boolean_by_mode, kmax, kdepth):
    """
    If the mode has a non-zero value, then it is kept only if
    the value is less than kmax and the total auxillary is less
    than kdepth. This essentially truncates some modes at a lower
    order than the other modes.

    Parameters
    ----------
    1. list_aux : list
                  List of auxiliary tuples.

    2. list_boolean_by_mode : list
                              List of booleans: True-Filter, False-No Filter.

    3. kmax : int
              The largest value a filtered mode can have in an auxiliary.

    4. kdepth : int
                The largest depth of an auxiliary at which a filtered mode can have a
                non-zero value.

    Returns
    -------
    1. list_new_aux : list
                      Filtered list of auxiliaries.
    """
    aux_list_orig = list(list_aux)
    list_aux = np.array([aux.todense() for aux in aux_list_orig])
    test1 = np.sum(list_aux[:, list_boolean_by_mode], axis=1) == 0
    test2 = np.sum(list_aux[:, list_boolean_by_mode] > kmax, axis=1) == 0
    test3 = np.sum(list_aux, axis=1) <= kdepth
    test_triangle = test1 | (test2 & test3)
    return [aux for (i, aux) in enumerate(aux_list_orig) if test_triangle[i]]


def filter_aux_longedge(list_aux, list_boolean_by_mode, kmax, kdepth):
    """
    Beyond kdepth, only keep the edge terms upto kmax.

    Parameters
    ----------
    1. list_aux : list
                  List of auxiliaries.

    2. list_boolean_by_mode : list
                              List of booleans: True-Filter, False-No Filter.

    3. kmax : int
              Maximum depth of the hierarchy.

    4. kdepth : int
                Depth beyond which only edge members of hierarchy are kept for
                filtered modes.

    Returns
    -------
    1. list_new_aux : list
                      Filtered list of auxiliaries.
    """
    return [
        aux
        for aux in list_aux
        if check_long_edge(aux, list_boolean_by_mode, kmax, kdepth)
    ]


def check_long_edge(aux_vec, list_boolean, kmax, kdepth):
    """
    Checks if an individual auxiliary should be filtered out by the longedge filter.

    Parameters
    ----------
    1. aux : tuple
             Auxiliary to check.

    2. list_boolean : list
                      The modes to filter True-filter, False-No filter.

    3. kmax : int
              Maximum depth of the hierarchy.

    4. kdepth : int
                Depth beyond which only edge members of hierarchy are kept for
                filtered modes.

    RETURNS
    -------
    1. check_aux : bool
                   True-keep aux, False-remove aux.
    """
    aux = aux_vec.get_values_nonzero(np.arange(len(list_boolean))[list_boolean])
    return (
        np.sum(aux_vec) <= kdepth
        or (np.sum(aux) == 0 and np.sum(aux_vec) <= kmax)
        or (np.sum(aux_vec) <= kmax and len(aux_vec.keys()) <= 1 and np.sum(aux) > 0)
    )


def check_markovian(aux, list_boolean):
    """
    Checks whether an auxiliary is an allowed markovian mode or a non-markovian mode.

    Parameters
    ----------
    1. aux : instance(AuxiliaryVector)

    2. list_boolean : list
                      List of boolean values True: markovian mode, False:
                      non-markovian mode.

    Returns
    -------
    1. allowed : boolean
                 True: non-markovian/ allowed markovian, False: disallowed markovian.
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
    1. list_aux : list
                  List of unfiltered auxiliaries.

    2. list_boolean : list
                      List of boolean values
                      True: markovian mode, False: non-markovian mode

    Returns
    -------
    list_aux : list
               Filtered auxiliary list.
    """
    return [aux for aux in list_aux if check_markovian(aux, list_boolean)]
