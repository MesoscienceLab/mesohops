import numpy as np
from pyhops.util.physical_constants import precision


__title__ = "EOM Functions"
__author__ = "D. I. G. Bennett"
__version__ = "1.2"


def operator_expectation(oper, vec, flag_gcorr = False):
    """
    This is a function that calculates the expectation value of a sparse
    operator

    PARAMETERS
    ----------
    1. oper : np.array
             dense or sparse operator
    2. vec : np.array
             an nx1 matrix
    3. flag_gcorr : bool
                    whether or not to correct the normalization for the inclusion of
                    a ground state (in the dyadic equation governing absorption)

    RETURNS
    -------
    1. expectation_value : float
                           expectation value of the operator, <vec|Oper|vec> /(<vec|vec>)
    """
    if not flag_gcorr:
        return (np.conj(vec) @ (oper @ vec)) / (np.conj(vec) @ vec)
    else:
        return (np.conj(vec) @ (oper @ vec)) / (1+np.conj(vec) @ vec)


def compress_zmem(z_mem, list_index_L2_by_mode, list_absindex_mode):
    """
    This function compresses all of the memory terms into their
    respective slots for each L_operator.

    PARAMETERS
    ----------
    1. z_mem : list
               a list of all the memory terms in absolute basis
    2. list_index_L2_by_mode : list
                               list of length equal to the number of modes in the
                               current hierarchy basis
    3. list_absindex_mode : list
                            list of absolute mode indices in relative mode order

    RETURNS
    -------
    1. dz_hat : list
                a list of compressed memory terms indexed by list_index_L2_by_mode
    """
    dz_hat = [0 for i in set(list_index_L2_by_mode)]
    for (i, lind) in enumerate(list_index_L2_by_mode):
        dz_hat[lind] += z_mem[list_absindex_mode[i]]

    return dz_hat


def calc_delta_zmem(
    z_mem, list_avg_L2, list_g, list_w, list_index_L2_by_mode, list_absindex_mode
):
    """
    This updates the memory term. The form of the equation depends on expanding the
    memory integral assuming an exponential expansion.

    NOTE: THIS ASSUMES THE NOISE HAS EXPONENTIAL FORM.

    NOTE: Again this function mixes relative and absolute
          indexing which makes it much more confusing.
          z_mem : absolute
          list_avg_L2 : relative
          list_g : absolute
          list_w : absolute
          list_absindex_L2_by_mode : relative
          list_absindex_mode : mapping from relative-->absolute

    PARAMETERS
    ----------
    1. z_mem : list
               a list of memory terms in absolute indices
    2. list_avg_L2 : list
                     relative list of the expectation values of the L operators
    3. list_g : list
                list of pre exponential factors for bath correlation functions [absolute]
    3. list_w :  list
                 list of exponents for bath correlation functions (w = γ+iΩ) [absolute]
    4. list_absindex_L2_by_mode : list
                                  list of length equal to the number of 'modes' in the
                                  current hierarchy basis and each entry is an index for
                                  the absolute list_L2.
    5. list_absindex_mode : list
                          list of the absolute indices  of the modes in current basis

    RETURNS
    -------
    1. delta_z_mem : list
                     a list of updated memory terms
    """
    delta_z_mem = np.zeros(len(z_mem), dtype=np.complex128)

    # Determine modes where z_mem > precision but not in current basis
    list_nonzero_zmem = list(
        set(np.where(z_mem > precision)[0]) - set(list_absindex_mode)
    )

    # Loop over modes in the current basis
    for (relindex_mode, absindex_mode) in enumerate(list_absindex_mode):
        l_avg = list_avg_L2[list_index_L2_by_mode[relindex_mode]]
        delta_z_mem[absindex_mode] = (
            l_avg * np.conj(list_g[absindex_mode])
            - np.conj(list_w[absindex_mode]) * z_mem[absindex_mode]
        )

    for absindex_mode in list_nonzero_zmem:
        delta_z_mem[absindex_mode] -= (
            np.conj(list_w[absindex_mode]) * z_mem[absindex_mode]
        )

    return delta_z_mem


def calc_norm_corr(
    phi, z_hat, list_avg_L2, list_L2, nstate, list_index_phi_L2_mode, list_g, list_w
):
    """ 
    This function computes the correction factor for propagating the
    normalized wavefunction.

    PARAMETERS
    ----------
    1. phi : np.array
             the full hierarchy
    2. z_hat : list
               the memory term with both with random noise
    3. list_avg_L2 : list
                     relative list of the expectation values of the L operators
    4. list_L2 : list
                 list of L operators
    5. nstate : int
                the current dimension (size) of the system
    6. list_index_L2_by_mode : list
                               list of length equal to the number of modes in the current
                               hierarchy basis: each entry is an index for the
                               relative list_L2.
    7. list_g : list
                list of pre exponential factors for bath correlation functions [absolute]
    8. list_w : list
                list of exponents for bath correlation functions (w = γ+iΩ) [absolute]

    RETURNS
    -------
    1. delta : float
               the norm correction factor
    """
    delta = np.dot(z_hat, list_avg_L2)
    phi_0 = phi[0:nstate]

    for (i_aux, l_ind, nmode) in list_index_phi_L2_mode:
        phi_1 = (list_g[nmode] / list_w[nmode]) * phi[
            nstate * (i_aux) : nstate * (i_aux + 1)
        ]
        # ASSUMING: L = L^*
        delta -= (np.conj(phi_0) @ (list_L2[l_ind] @ phi_1))
        delta += (np.conj(phi_0) @ phi_1) * list_avg_L2[l_ind]

    return np.real(delta)
