import numpy as np
from mesohops.util.physical_constants import precision


__title__ = "EOM Functions"
__author__ = "D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"


def operator_expectation(oper, vec, flag_gcorr = False):
    """
    Calculates the expectation value of a sparse operator.

    Parameters
    ----------
    1. oper : np.array
              Dense or sparse operator.

    2. vec : np.array
             nx1 matrix.

    3. flag_gcorr : bool
                    True indictates that the normalization of the expectation value
                    should be corrected  for the inclusion of a ground state (e.g.,
                    in the linear absorption equation of motion) while False 
                    indicates otherwise.

    Returns
    -------
    1. expectation_value : float
                           Expectation value of the operator, <vec|Oper|vec> /(
                           <vec|vec>).
    """
    if not flag_gcorr:
        return (np.conj(vec) @ (oper @ vec)) / (np.conj(vec) @ vec)
    else:
        return (np.conj(vec) @ (oper @ vec)) / (1 + np.conj(vec) @ vec)


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


def calc_delta_zmem(z_mem, list_avg_L2, list_g, list_w, list_absindex_L2_by_mode,
                    list_absindex_mode, list_absindex_L2_active):
    """
    Updates the memory term. The form of the equation depends on expanding the
    memory integral assuming an exponential expansion.

    NOTE: This asumes the noise has exponential form.

    NOTE: This function mixes relative and absolute indexing.
          z_mem : absolute
          list_avg_L2 : relative
          list_g : absolute
          list_w : absolute
          list_absindex_L2_by_mode : absolute
          list_absindex_mode : mapping from relative-->absolute
          list_absindex_L2_active : absolute
          
    Parameters
    ----------
    1. z_mem : list(complex)
               List of memory terms in absolute indices.

    2. list_avg_L2 : list(complex)
                     Relative list of the expectation values of the L operators.

    3. list_g : list(complex)
                List of pre exponential factors for bath correlation functions [units:
                cm^-2].

    4. list_w :  list(complex)
                 List of exponents for bath correlation functions (w = γ+iΩ) [units:
                 cm^-1].

    5. list_absindex_L2_by_mode : list(int)
                                  List of indices for the absolute list of L-operators
                                  to match L-operators to the associated absolute mode
                                  index.

    6. list_absindex_mode : list(int)
                            List of the absolute indices  of the modes in current basis.

    7. list_absindex_L2_active : list(int)
                                 List of absolute indices of L-operators that have any 
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
    for absindex_mode in list_absindex_mode:
        absindex_L2 = list_absindex_L2_by_mode[absindex_mode]
        try:
            relindex_L2 = list(list_absindex_L2_active).index(absindex_L2)
            l_avg = list_avg_L2[relindex_L2]
        except:
            l_avg = 0
        delta_z_mem[absindex_mode] = (
            l_avg * np.conj(list_g[absindex_mode])
            - np.conj(list_w[absindex_mode]) * z_mem[absindex_mode]
        )

    delta_z_mem[list_nonzero_zmem] -= np.conj(list_w[list_nonzero_zmem]) * z_mem[
        list_nonzero_zmem]

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
        delta += ((np.conj(phi_0) @ phi_1))* list_avg_L2[l_ind]

    return np.real(delta)

def calc_LT_corr(
    list_LT_coeff, list_L2, list_avg_L2, list_L2_sq):
    """
    Computes the low-temperature correction factor associated with each member of the
    hierarchy in the nonlinear equation of motion. The factor is given by the sum over
    the low-temperature correction coefficients and associated L-operators c_n and L_n:
    \sum_n conj(c_n)<L_n>L_n
    to all auxiliary wave functions and
    \sum_n c_n(<L_n> - L_n)L_n
    to the physical wave function, where c_n is the nth low-temperature correction
    factor, and L_n is the nth L-operator associated with that factor.

    Parameters
    ----------
    1. list_LT_coeff : list(complex)
                       Relative list of low-temperature coefficients [units: cm^-1].
    2. list_L2 : list(sparse matrix)
                 Relative list of active L operators.
    3. list_avg_L2 : list(float)
                     Relative list of the expectation values of the active L operators.
    4. list_L2_sq: list(sparse matrix)
                   Relative list of active L operators squared.

    Returns
    -------
    1. C2_LT_corr_physical : np.array(complex)
                             Low-temperature correction self-derivative term applied to the
                             physical wave function only to account for the terminator
                             correction [units: cm^-1].
    2. C2_LT_corr_hier : np.array(complex)
                         Low-temperature correction self-derivative term applied to
                         all auxiliary wave functions to account for the noise memory
                         drift [units: cm^-1].

    """
    # Cast everything to arrays for easy multiplication
    G1_LT_coeff = np.array(list_LT_coeff)
    L1_avg_L2 = np.array(list_avg_L2)
    return np.sum((G1_LT_coeff * L1_avg_L2) * list_L2 -
                  G1_LT_coeff * list_L2_sq, axis=0), \
           np.sum((np.conj(G1_LT_coeff)*L1_avg_L2)*list_L2, axis=0)

def calc_LT_corr_to_norm_corr(
    list_LT_coeff, list_avg_L2, list_avg_L2_sq
):
    """
    Computes the low-temperature correction to the normalization factor in the
    normalized nonlinear equation of motion. The correction is given by the sum over
    the low-temperature correction coefficients and associated L-operators c_n and L_n:
    \sum_n Re[c_n](2<L_n>^2 - <L_n^2>),
    where c_n is the nth low-temperature correction
    factor, and L_n is the nth L-operator associated with that factor.

    Parameters
    ----------
    1. list_LT_coeff : list(complex)
                       Relative list of low-temperature coefficients [units: cm^-1].
    2. list_avg_L2 : list(float)
                     Relative list of the expectation values of the L operators.
    3. list_avg_L2_sq : list(float)
                        Relative list of the expectation values of the squared L
                        operators.
    Returns
    -------
    1. delta_corr : float
                    Low-temperature correction to the normalization correction factor.
    """
    return np.sum(np.real(np.array(list_LT_coeff))*(2*np.array(list_avg_L2)**2 -
                                                    np.array(list_avg_L2_sq)))

def calc_LT_corr_linear(
    list_LT_coeff, list_L2_sq
):
    """
    Computes the low-temperature correction factor associated with each member of the
    hierarchy in the linear equation of motion. The factor is given by the sum over the
    low-temperature correction coefficients and associated L-operators c_n and L_n:
    -\sum_n c_nL_n^2,
    where c_n is the nth low-temperature correction factor, and L_n is
    the nth L-operator associated with that factor.
    NOTE: This correction should only be applied to the physical wavefunction.

    Parameters
    ----------
    1. list_LT_coeff : list(complex)
                       Relative list of low-temperature coefficients [units: cm^-1].
    2. list_L2_sq : np.array(sparse matrix)
                    Relative list of squared L operators, cast to array.
    Returns
    -------
    1. C2_LT_corr : np.array(complex)
                    Low-temperature correction self-derivative term [units: cm^-1].
    """
    return -1 * np.sum(np.array(list_LT_coeff)*list_L2_sq, axis=0)
