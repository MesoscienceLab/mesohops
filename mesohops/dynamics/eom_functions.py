import numpy as np
from mesohops.util.physical_constants import precision


__title__ = "EOM Functions"
__author__ = "D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"


def operator_expectation(oper, vec, flag_gcorr = False):
    """
    Calculates the expectation value of a sparse operator

    Parameters
    ----------
    1. oper : np.array
              Dense or sparse operator.

    2. vec : np.array
             An nx1 matrix.

    3. flag_gcorr : bool
                    Whether to correct the normalization for the inclusion of
                    a ground state (in the dyadic equation governing absorption).

    Returns
    -------
    1. expectation_value : float
                           Expectation value of the operator, <vec|Oper|vec> /(
                           <vec|vec>).
    """
    if not flag_gcorr:
        return (np.conj(vec) @ (oper @ vec)) / (np.conj(vec) @ vec)
    else:
        return (np.conj(vec) @ (oper @ vec)) / (1+np.conj(vec) @ vec)


def compress_zmem(z_mem, list_index_L2_by_mode, list_absindex_mode):
    """
    Compresses all of the memory terms into their respective slots for each L_operator.

    Parameters
    ----------
    1. z_mem : list(complex)
               List of all the memory terms in absolute basis.

    2. list_index_L2_by_mode : list(complex)
                               List of length equal to the number of modes in the
                               current hierarchy basis.

    3. list_absindex_mode : list(int)
                            List of absolute mode indices in relative mode order.

    Returns
    -------
    1. dz_hat : list(complex)
                List of compressed memory terms indexed by list_index_L2_by_mode.
    """
    dz_hat = [0 for i in set(list_index_L2_by_mode)]
    for (i, lind) in enumerate(list_index_L2_by_mode):
        dz_hat[lind] += z_mem[list_absindex_mode[i]]

    return dz_hat


def calc_delta_zmem(z_mem, list_avg_L2, list_g, list_w, list_index_L2_by_mode,
                    list_absindex_mode, list_index_L2_active):
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
          list_index_L2_by_mode : relative
          list_absindex_mode : mapping from relative-->absolute
          list_index_L2_active : relative
          
    Parameters
    ----------
    1. z_mem : list(complex)
               List of memory terms in absolute indices.

    2. list_avg_L2 : list(complex)
                     Relative list of the expectation values of the L operators.

    3. list_g : list(complex)
                List of pre exponential factors for bath correlation functions [
                absolute].

    4. list_w :  list(complex)
                 List of exponents for bath correlation functions (w = γ+iΩ) [absolute].

    5. list_index_L2_by_mode : list(int)
                               List of length equal to the number of 'modes' in the
                               current hierarchy basis and each entry is an index for
                               the absolute list_L2.

    6. list_absindex_mode : list(int)
                            List of the absolute indices  of the modes in current basis.

    7. list_index_L2_active : list(int)
                              List of relative indices of L-operators that have any
                              non-zero values.

    Returns
    -------
    1. delta_z_mem : list(complex)
                     List of updated memory terms.
    """
    delta_z_mem = np.zeros(len(z_mem), dtype=np.complex128)

    # Determine modes where z_mem > precision but not in current basis
    list_nonzero_zmem = list(
        set(np.where(z_mem > precision)[0]) - set(list_absindex_mode)
    )

    # Loop over modes in the current basis
    for (relindex_mode, absindex_mode) in enumerate(list_absindex_mode):
        index_L2 = list_index_L2_by_mode[relindex_mode]
        if index_L2 in list_index_L2_active:
            l_avg = list_avg_L2[list_index_L2_active.index(index_L2)]
        else:
            l_avg = 0
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
    Computes the correction factor for propagating the normalized wave function.

    Parameters
    ----------
    1. phi : np.array(complex)
             Full hierarchy.

    2. z_hat : list(complex)
               List of memory term with both with random noise.

    3. list_avg_L2 : list(complex)
                     Relative list of the expectation values of the L operators.

    4. list_L2 : list(sparse matrix)
                 List of L operators.

    5. nstate : int
                Current dimension (size) of the system.

    6. list_index_L2_by_mode : list(int)
                               List of length equal to the number of modes in the
                               current hierarchy basis: each entry is an index for the
                               relative list_L2.
    7. list_g : list(complex)
                List of pre exponential factors for bath correlation functions [
                absolute].

    8. list_w : list(complex)
                List of exponents for bath correlation functions (w = γ+iΩ) [absolute].

    Returns
    -------
    1. delta : float
               Norm correction factor.
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

def calc_LT_corr(
    list_LT_coeff, list_L2, list_avg_L2, physical = False
):
    """
    This function computes the low-temperature correction factor associated with each
    member of the hierarchy in the nonlinear equation of motion. The factor is given by
    the sum over the low-temperature correction coefficients and associated
    L-operators c_n and L_n:
    \sum_n (2<L_n>Re[c_n] - L_nc_n)L_n
    Where c_n is the nth low-temperature correction factor, and L_n is the nth
    L-operator associated with that factor.

    PARAMETERS
    ----------
    1. list_LT_coeff : list
                       relative list of low-temperature coefficients
    2. list_L2 : list
                 relative list of active L operators
    3. list_avg_L2 : list
                     relative list of the expectation values of the active L operators
    4. physical : boolean
                  Determines if this correction is to be applied to the physical
                  wavefunction or elsewhere
    RETURNS
    -------
    1. C2_corr : np.array
                 the low-temperature correction self-derivative term
    """
    # Adds the terminated flux from above to the physical wavefunction
    if physical:
        return np.sum(np.array([(list_LT_coeff[n]*list_avg_L2[n]*np.eye(
        list_L2[n].shape[0]) - list_LT_coeff[n]*list_L2[n])@list_L2[n] for n in
                            range(len(list_LT_coeff))]), axis=0)
    # Adds the delta-approximated noise memory drift to all members of the hierarchy
    else:
        return np.sum(np.array([(np.conj(list_LT_coeff[n])*list_avg_L2[n]*np.eye(
        list_L2[n].shape[0]))@list_L2[n] for n in range(len(list_LT_coeff))]), axis=0)

def calc_LT_corr_to_norm_corr(
    list_LT_coeff, list_avg_L2, list_avg_L2_sq
):
    """
    This function computes the low-temperature correction to the normalization factor in
    the normalized nonlinear equation of motion. The correction is given by the sum
    over the low-temperature correction coefficients and associated L-operators c_n
    and L_n:
    \sum_n Re[c_n](2<L_n>^2 - <L_n^2>)
    Where c_n is the nth low-temperature correction factor, and L_n is the nth
    L-operator associated with that factor.

    PARAMETERS
    ----------
    1. list_LT_coeff : list
                       relative list of low-temperature coefficients
    2. list_avg_L2 : list
                     relative list of the expectation values of the L operators
    3. list_avg_L2_sq : list
                        relative list of the expectation values of the squared L
                       operators
    RETURNS
    -------
    1. delta_corr : float
                    the low-temperature correction to the normalization correction
                    factor
    """
    return np.sum(np.array([
    np.real(list_LT_coeff[j])*(2*list_avg_L2[j]**2 - list_avg_L2_sq[j])
        for j in range(len(list_LT_coeff))]))

def calc_LT_corr_linear(
    list_LT_coeff, list_L2
):
    """
    This function computes the low-temperature correction factor associated with each
    member of the hierarchy in the linear equation of motion. The factor is given by
    the sum over the low-temperature correction coefficients and associated
    L-operators c_n and L_n:
    -\sum_n c_nL_n^2
    Where c_n is the nth low-temperature correction factor, and L_n is the nth
    L-operator associated with that factor.
    NOTE: this correction should only be applied to the physical wavefunction.

    PARAMETERS
    ----------
    1. list_LT_coeff : list
                       relative list of low-temperature coefficients
    2. list_L2 : list
                 relative list of L operators
    RETURNS
    -------
    1. C2_corr : np.array
                 the low-temperature correction self-derivative term
    """
    return -1*np.sum(np.array([(list_LT_coeff[n])*list_L2[n]@list_L2[n] for n in
                            range(len(list_LT_coeff))]),axis=0)