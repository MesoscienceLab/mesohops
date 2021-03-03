import numpy as np

__title__ = "Hierarchy Functions"
__author__ = "D. I. G. Bennett, Leo Varvelo"
__version__ = "1.0"


def filter_aux_triangular(list_aux, list_boolean_by_mode, kmax, kdepth):
    """
    If the mode has a non-zero value, then it is kept only if
    the value is less than kmax and the total auxillary is less
    than kdepth. This essentially truncates some modes at a lower
    order than the other modes.

    PARAMETERS
    ----------
    1. list_aux : list
                  a list of auxiliary tuples
    2. list_boolean_by_mode : list
                              a list of booleans: True-Filter, False-No Filter
    3. kmax : int
              the largest value a filtered mode can have in an auxiliary
    4. kdepth : int
                the largest depth of an auxiliary at which a filtered mode can have a
                non-zero value

    RETURNS
    -------
    1. list_new_aux : list
                      a filtered list of auxiliaries
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

    PARAMETERS
    ----------
    1. list_aux : list
                  list of auxiliaries
    2. list_boolean_by_mode : list
                              a list of booleans: True-Filter, False-No Filter
    3. kmax : int
              the maximum depth of the hierarchy
    4. kdepth : int
                the depth beyond which only edge members of hierarchy are kept for
                filtered modes

    RETURNS
    -------
    1. list_new_aux : list
                      a filtered list of auxiliaries
    """
    return [
        aux
        for aux in list_aux
        if check_long_edge(aux, list_boolean_by_mode, kmax, kdepth)
    ]


def check_long_edge(aux_vec, list_boolean, kmax, kdepth):
    """

    PARAMETERS
    ----------
    1. aux : tuple
             an auxiliary to check
    2. list_boolean : list
                      the modes to filter True-filter, False-No filter
    3. kmax : int
              the maximum depth of the hierarchy
    4. kdepth : int
                the depth beyond which only edge members of hierarchy are kept for
                filtered modes

    RETURNS
    -------
    1. check_aux : bool
                   True-keep aux, False-remove aux
    """
    aux = aux_vec.get_values_nonzero(np.arange(len(list_boolean))[list_boolean])
    return (
        np.sum(aux_vec) <= kdepth
        or (np.sum(aux) == 0 and np.sum(aux_vec) <= kmax)
        or (np.sum(aux_vec) <= kmax and len(aux_vec.keys()) <= 1 and np.sum(aux) > 0)
    )


def check_markovian(aux, list_boolean):
    """
    This function checks whether a auxiliary is a an allowed markovian mode or a
    non-markovian mode

    PARAMETERS
    ----------
    1. aux : AuxiliaryVector object
    2. list_boolean : list
                      a list of boolean values
                      True: markovian mode, False: non-markovian mode

    RETURNS
    -------
    1. allowed : boolean
                 True: non-markovian/ allowed markovian, False: disallowed markovian
    """
    flag_adaptive = True
    if np.sum(aux) == 0:
        flag_adaptive = True
    elif any(np.array(list_boolean)[aux.toarray()[:, 0]]) and np.sum(aux) > 1:
        flag_adaptive = False
    return flag_adaptive


def filter_markovian(list_aux, list_boolean):
    """
    This function filter's a list of auxiliaries based on whether it is a non markovian
    auxiliary or an allowed markovian auxiliary

    Parameters
    ----------
    1. list_aux : list
                  list of unfiltered auxiliaries
    2. list_boolean : list
                      a list of boolean values
                      True: markovian mode, False: non-markovian mode

    Returns
    -------
    list_aux : list
               filtered auxiliary list
    """
    return [aux for aux in list_aux if check_markovian(aux, list_boolean)]
