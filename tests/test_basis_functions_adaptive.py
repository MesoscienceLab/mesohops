import numpy as np
import pytest
import scipy as sp
from scipy import sparse
from mesohops.basis.basis_functions_adaptive import *
from mesohops.basis.hops_aux import AuxiliaryVector
from mesohops.util.physical_constants import hbar

__title__ = "Test of adaptive basis functions"
__author__ = "J. K. Lynd, D. I. G. B. Raccah, B. Citty"
__version__ = "1.4"

hbar2 = hbar * hbar

def test_error_deriv():
    """
    test for the error associated with losing flux to a component
    """
    phi = np.ones(6, dtype=np.complex128)
    dt = 2
    n_state = 2
    n_hier = 3
    z_step = [[0, 0], [0, 0], [1, 1, 1, 1]]
    list_index_aux_stable = [0, 1]

    def dsystem_dt_if(phi, z1, z2, z3):
        return [np.array([1, 1, 1, 1, 0, 0], dtype=np.complex128)]

    def dsystem_dt_else(phi, z1, z2, z3):
        return [np.ones(6, dtype=np.complex128)]

    # Code path with explicit stable auxiliary filtering.
    E2_del_phi = error_deriv(dsystem_dt_if, phi, z_step, n_state, n_hier, dt,
                             list_index_aux_stable=list_index_aux_stable)
    known_deriv_error = np.array([[1 / hbar, 1 / hbar], [1 / hbar, 1 / hbar]])
    known_del_flux = phi.reshape([n_state, n_hier], order="F")[:,
                     list_index_aux_stable] / dt
    known_error = np.abs(known_deriv_error + known_del_flux) ** 2
    assert np.allclose(E2_del_phi, known_error)

    # Code path without stable auxiliary filtering.
    E2_del_phi = error_deriv(dsystem_dt_else, phi, z_step, n_state, n_hier, dt)
    known_deriv_error = [[1 / hbar, 1 / hbar, 1 / hbar], [1 / hbar, 1 / hbar, 1 / hbar]]
    known_del_flux = phi.reshape([n_state, n_hier], order="F") / dt
    known_error = np.abs(known_deriv_error + known_del_flux) ** 2
    assert np.allclose(E2_del_phi, known_error)

def test_error_sflux_hier():
    """
    test for the error associated with losing all flux terms inside the kth auxiliary to
    states not contained in S_t
    """
    nstate_extd = 5
    hs = np.zeros([nstate_extd, nstate_extd])
    hs[4, 3] = 100
    hs[3, 4] = 100
    hs[1, 2] = 50
    hs[2, 1] = 50

    phi = np.ones(9)
    n_state = 3
    n_hier = 3
    state_list = [1, 2, 3]
    hamiltonian = sparse.csr_array(hs)
    list_fullbndidx_abs = [4]
    E2_flux_state = error_sflux_hier(phi, state_list, list_fullbndidx_abs, n_state, n_hier,
                                           hamiltonian)
    hbar2 = hbar * hbar
    known_error = [10000 / hbar2, 10000 / hbar2, 10000 / hbar2]
    assert np.allclose(E2_flux_state, known_error)

    ltc_phys = np.zeros_like(hs, dtype=np.complex128)
    ltc_hier = np.zeros_like(hs, dtype=np.complex128)
    ltc_phys[0,3] = 20
    ltc_phys[3,0] = 20
    ltc_phys[4,3] = -50
    ltc_phys[3,4] = -50
    ltc_phys[4,4] = 500000000
    ltc_phys = sparse.csr_array(ltc_phys)

    ltc_hier[0,2] = -10
    ltc_hier[2,0] = -10
    ltc_hier[4,0] = 10 - 100000000j
    ltc_hier[0,4] = 10 + 100000000j
    ltc_hier = sparse.csr_array(ltc_hier)
    list_fullbndidx_abs = [0,4]
    E2_flux_state = error_sflux_hier(phi, state_list, list_fullbndidx_abs, n_state, n_hier,
                                                 hamiltonian, ltc_phys, ltc_hier)
    known_error = [(2500+400) / hbar2, (10000 + 100) / hbar2, (10000 + 100) / hbar2]
    assert np.allclose(E2_flux_state, known_error)

# ------------------------------------------------------------
# TEST: non-uniform phi verifies correct column selection
# ------------------------------------------------------------
def test_error_sflux_hier_nonuniform_phi():
    """
    Verify auxiliary-to-column mapping when each auxiliary has distinct phi values.
    """
    # No-LTC code path with distinct phi per auxiliary. Here:
    # s0/state_list = active basis states, sc/list_fullbndidx_abs = destination/boundary states.
    # This verifies that each auxiliary k uses the correct Fortran-order phi column.
    #
    # phi = [1, 2, 3, 4] reshapes (Fortran order) to:
    #   C2_phi = [[1, 3],    psi_0 = [1, 2],  psi_1 = [3, 4]
    #             [2, 4]]
    #
    # H_sub = H[sc, s0] = [[3, 4],
    #                       [1, 2]]
    #
    # E[k] = sum_d |sum_s H[d,s] * psi_k[s]|^2 / hbar^2
    # E[0]: H_sub @ psi_0 = [3+8, 1+4] = [11, 5]  -> (121 + 25) / hbar^2 = 146 / hbar^2
    # E[1]: H_sub @ psi_1 = [9+16, 3+8] = [25, 11] -> (625 + 121) / hbar^2 = 746 / hbar^2
    nstate_extd = 4
    hs = np.zeros([nstate_extd, nstate_extd])
    hs[2, 0] = 3
    hs[2, 1] = 4
    hs[3, 0] = 1
    hs[3, 1] = 2
    
    # Add entries to the Hamiltonian that should not matter because they are sliced out.
    # (See state_list, list_fullbndidx_abs below).
    hs[0,1] = 7
    hs[1,2] = 11
    hs[3,3] = 26
    hs[3,2] = 999
    hs[1,3] = 666666
    
    hamiltonian = sparse.csr_array(hs)

    phi = np.array([1, 2, 3, 4], dtype=np.complex128)
    n_state = 2
    n_hier = 2
    state_list = [0, 1]
    list_fullbndidx_abs = [2, 3]

    E2_flux_state = error_sflux_hier(phi, state_list, list_fullbndidx_abs, n_state, n_hier,
                                     hamiltonian)
    known_error = [146 / hbar2, 746 / hbar2]
    assert np.allclose(E2_flux_state, known_error)

    # LTC-enabled code path with distinct phi per auxiliary. This verifies that
    # the physical wave function (k=0) uses T2_phys while k>0 uses T2_hier.
    #
    # H[2,0] = 5, T2_phys[2,0] = 3, T2_hier[2,0] = -1
    # -> H_eff_phys[2,0] = 8,  H_eff_hier[2,0] = 4
    #
    # phi = [1, 0, 10, 0, 3, 0] reshapes (Fortran order) to:
    #   C2_phi = [[1, 10, 3],    psi_0 = [1, 0],  psi_1 = [10, 0],  psi_2 = [3, 0]
    #             [0,  0, 0]]
    #
    # E[0] (phys): |8*1 + 0*0|^2 / hbar^2 = 64
    # E[1] (aux):  |4*10 + 0*0|^2 / hbar^2 = 1600
    # E[2] (aux):  |4*3 + 0*0|^2 / hbar^2 = 144
    nstate_extd = 4
    hs = np.zeros([nstate_extd, nstate_extd])
    hs[2, 0] = 5
    hamiltonian = sparse.csr_array(hs)

    ltc_phys = np.zeros([nstate_extd, nstate_extd], dtype=np.complex128)
    ltc_phys[2, 0] = 3
    # Adding random values that should be sliced out
    ltc_phys[0, 1] = 6
    ltc_phys[1, 2] = 333
    ltc_phys = sparse.csr_array(ltc_phys)

    ltc_hier = np.zeros([nstate_extd, nstate_extd], dtype=np.complex128)
    ltc_hier[2, 0] = -1
    # Adding random values that should be sliced out
    ltc_hier[0, 1] = 6
    ltc_hier[1, 2] = 333
    ltc_hier = sparse.csr_array(ltc_hier)

    phi = np.array([1, 0, 10, 0, 3, 0], dtype=np.complex128)
    n_state = 2
    n_hier = 3
    state_list = [0, 1]
    list_fullbndidx_abs = [2]

    E2_flux_state = error_sflux_hier(phi, state_list, list_fullbndidx_abs, n_state, n_hier,
                                     hamiltonian, ltc_phys, ltc_hier)
    known_error = [64 / hbar2, 1600 / hbar2, 144 / hbar2]
    assert np.allclose(E2_flux_state, known_error)
# ------------------------------------------------------------
# TEST: complex phi verifies abs/power arithmetic on sparse
# ------------------------------------------------------------
def test_error_sflux_hier_complex_phi():
    """
    Verify complex-valued phi/Hamiltonian handling in the no-LTC code path.
    """
    # This case exercises the no-LTC code path with complex entries in both H and phi.
    # and the wave function, confirming that np.abs and .power(2) on sparse complex
    # results produce the correct squared modulus.
    #
    # H[2,0] = 1j, H[2,1] = 1
    # phi = [1+1j, 2, 3, 4] reshapes (Fortran order) to:
    #   C2_phi = [[1+1j, 3],    psi_0 = [1+1j, 2],  psi_1 = [3, 4]
    #             [2,    4]]
    #
    # H_sub = [[1j, 1]]
    #
    # E[0]: 1j*(1+1j) + 1*2 = (1j - 1) + 2 = 1 + 1j  -> |1+1j|^2 / hbar^2 = 2 / hbar^2
    # E[1]: 1j*3 + 1*4 = 4 + 3j                        -> |4+3j|^2 / hbar^2 = 25 / hbar^2
    nstate_extd = 3
    hs = np.zeros([nstate_extd, nstate_extd], dtype=np.complex128)
    hs[2, 0] = 1j
    hs[2, 1] = 1
    # Add values that should be sliced out, and should not matter
    hs[0, 1] = 6
    hs[0, 2] = 33
    hs[1, 2] = 4
    hamiltonian = sparse.csr_array(hs)

    phi = np.array([1 + 1j, 2, 3, 4], dtype=np.complex128)
    n_state = 2
    n_hier = 2
    state_list = [0, 1]
    list_fullbndidx_abs = [2]

    E2_flux_state = error_sflux_hier(phi, state_list, list_fullbndidx_abs, n_state, n_hier,
                                     hamiltonian)
    known_error = [2 / hbar2, 25 / hbar2]
    assert np.allclose(E2_flux_state, known_error)

# ------------------------------------------------------------
# TEST: single auxiliary with LTC verifies empty aux array path
# ------------------------------------------------------------
def test_error_sflux_hier_single_aux():
    """
    Verify n_hier=1 behavior when only the physical wave function exists.
    """
    # This case tests the LTC branch when the auxiliary slice C2_phi_aux[:, 1:] is
    # empty, so only the physical-wavefunction contribution remains.
    #
    # H[2,0] = 5, T2_phys[2,0] = 3 -> H_eff_phys[2,0] = 8
    # phi = [2, 3] reshapes to C2_phi = [[2], [3]], only column is the physical WF
    #
    # The aux part C2_phi_aux[:, 1:] is empty (n_hier=1), contributing zero.
    # E[0] (phys): |8*2 + 0*3|^2 / hbar^2 = 256 / hbar^2
    nstate_extd = 3
    hs = np.zeros([nstate_extd, nstate_extd])
    hs[2, 0] = 5
    # Add values that should be sliced out, and should not matter
    hs[0, 1] = 6
    hs[0, 2] = 33
    hs[1, 2] = 4 
    hamiltonian = sparse.csr_array(hs)

    ltc_phys = np.zeros([nstate_extd, nstate_extd], dtype=np.complex128)
    ltc_phys[2, 0] = 3
    # Adding random values that should be sliced out
    ltc_phys[0, 1] = 6
    ltc_phys[1, 2] = 333
    ltc_phys = sparse.csr_array(ltc_phys)

    ltc_hier = sparse.csr_array(np.zeros([nstate_extd, nstate_extd], dtype=np.complex128))
    # Adding random values that should be sliced out
    ltc_hier[0, 1] = 6
    ltc_hier[1, 2] = 333
    phi = np.array([2, 3], dtype=np.complex128)
    n_state = 2
    n_hier = 1
    state_list = [0, 1]
    list_fullbndidx_abs = [2]

    E2_flux_state = error_sflux_hier(phi, state_list, list_fullbndidx_abs, n_state, n_hier,
                                     hamiltonian, ltc_phys, ltc_hier)
    known_error = [256 / hbar2]
    assert np.allclose(E2_flux_state, known_error)

def get_error_term_hier_flux_up(list_k_vec, list_w_bymode, list_lop_bymode, P2_phi,
                                state_list=None, dest_state_list=None):
    """
    Get the flux up hierarchy error matrix manually.

    Parameters
    ----------
    1. list_k_vec : list(np.array(int)) [nhier [nmode]]
                    The list of auxiliary indexing vectors
    2. list_w_bymode : list(complex) [nmode]
                The list of correlation function mode decay rate constants
    3. list_lop_bymode : list(np.array(complex)) [nmode, [N_state, N_state]]
                   The list of L2 system-bath coupling operators organized by the
                   same indexing as list of modes (in the full state basis).
    4. P2_phi : np.array(complex) [nstate, nhier]
                The full wave function
    5. state_list : list(int)
                    List of current states in the basis
    6. dest_state_list : list(int)
                         List of states that can accept flux from the current basis

    Returns
    -------
    1. E2_error : np.array(complex) [nmodes, nhier]
                  The flux up error from each aux along each mode
    """
    n_state = P2_phi.shape[0]
    n_aux = P2_phi.shape[1]
    n_mode = len(list_lop_bymode)
    if state_list is None:
        state_list = np.arange(n_state)
    if dest_state_list is None:
        dest_state_list = state_list
    E2_err = np.zeros([n_mode, n_aux])
    for mode_index in range(n_mode):
        L2_lop = list_lop_bymode[mode_index][np.ix_(dest_state_list, state_list)]
        w_mode = list_w_bymode[mode_index]
        for aux_index in range(n_aux):
            k_mode = list_k_vec[aux_index][mode_index]
            # Note - we have another absolute squared that makes the adaptive
            # error less tight than what is presented below: this may change in
            # the future depending on code performance analysis.
            #  E[m,k] = |w_m * (k_m+1)|^2 * \sum_d |(L_m @ \psi_k)[d]|^2
            E_mode_aux = np.abs(L2_lop) ** 2 @ np.abs(P2_phi[:, aux_index]) ** 2
            E_mode_aux *= np.abs((1 + k_mode) * w_mode) ** 2
            E2_err[mode_index, aux_index] = float(np.sum(E_mode_aux))
    return E2_err / hbar2


def get_error_term_state_flux_up(list_k_vec, list_w_bymode, list_lop_bymode, P2_phi,
                                 F2_filter, state_list=None, dest_state_list=None):
    """
    Get the flux up state error matrix manually.

    Parameters
    ----------
    1. list_k_vec : list(np.array(int)) [nhier [nmode]]
                    The list of auxiliary indexing vectors. BE CLEARER ABOUT WHAT THESE ARE: THEY ARE SOME MANNER OF "REDUCED" K VECTOR.
    2. list_w_bymode : list(complex) [nmode]
                The list of correlation function mode decay rates
    3. list_lop_bymode : list(np.array(complex)) [nmode, [Nstate, Nstate]]
                   The list of L2 system-bath coupling operators organized by the
                   same indexing as list of modes - EXIST IN THE FULL STATE BASIS
    4. P2_phi : np.array(complex) [nstate, nhier]
                The full wavefunction
    5. F2_filter : np.array(int or bool) [nmode, nhier]
                   Determines if flux up exists for a given mode, aux pair
    6. state_list : list(int)
                    List of current states in the basis
    7. dest_state_list : list(int)
                         List of states that can accept flux from the current basis.

    Returns
    -------
    1. E2_error : np.array(complex) [nstate, nhier]
                  The error in each aux by state
    """
    n_state = P2_phi.shape[0]
    n_aux = P2_phi.shape[1]
    n_mode = len(list_lop_bymode)

    if state_list is None:
        state_list = np.arange(n_state)
    if dest_state_list is None:
        dest_state_list = state_list
    if F2_filter is None:
        F2_filter = np.ones([n_mode,n_aux])
    E2_err = np.zeros([n_state, n_aux])
    n_dest = len(dest_state_list)
    list_known_err_by_d = np.zeros(n_dest)
    for state_index in range(n_state):
        state = state_list[state_index]
        for aux_index in range(n_aux):
            E_state_aux = 0
            for mode_index in range(len(list_lop_bymode)):
                if F2_filter[mode_index, aux_index]:
                    L2_lop = list_lop_bymode[mode_index]
                    w_mode = list_w_bymode[mode_index]
                    k_mode = list_k_vec[aux_index][mode_index]
                    # E[s,k] = \sum_m |w_m * (k_m+1)|^2 * \sum_d|L_m[d,
                    # s] * \psi_k[s]|^2
                    for d in range(n_dest):
                        dest = dest_state_list[d]
                        err_element = float(
                            np.abs((1 + k_mode) * w_mode) ** 2
                            * np.abs(
                                L2_lop[dest, state]
                                * P2_phi[state_index, aux_index]
                            )
                            ** 2
                        )
                        E_state_aux += err_element
                        list_known_err_by_d[d] += err_element
            E2_err[state_index, aux_index] = float(E_state_aux)
    return E2_err / hbar2, list_known_err_by_d / hbar2


def get_list_M2_mode_from_state(list_lop, list_states, list_dest_states):
    """
    Gets the M2_mode_from_state matrix from a list of L-operators.

    Parameters
    ----------
    1. list_lop : list(np.array(complex)), [Nstate, Nstate]
                  A list of L2 system-bath coupling operators in the full state
                  basis.

    2. list_states : list(int)
                     All states in the current basis

    3. list_dest_states : list(int)
                          All the destination states flux might enter

    Returns
    -------
    1. list_M2_mode_from_state_off : list(sparse matrix(complex))
                                     The matrix connecting each mode to source states
                                     for each destination state for the off-diagonal
                                     entries of the L-operators.
    2. M2_diag : sparse matrix(complex)
                 The matrix connecting each mode to source states for the diagonal
                 entries of the L-operators.
    """
    nstate = len(list_states)
    nmodes = len(list_lop)

    list_M2_mode_from_state_off = []
    list_data_diag = []
    list_row_diag = []
    list_col_diag = []
    for s_rel in range(nstate):
        for m in range(nmodes):
            L2_lop = list_lop[m]
            s_abs = list_states[s_rel]
            if s_abs in list_dest_states:
                list_row_diag += [m]
                list_col_diag += [s_rel]
                list_data_diag += [L2_lop[s_abs, s_abs]]
    for d in list_dest_states:
        list_data = []
        list_row = []
        list_col = []
        for m in range(nmodes):
            L2_lop = list_lop[m]
            for s_rel in range(nstate):
                s_abs = list_states[s_rel]
                if np.abs(L2_lop[d, s_abs]):
                    if not d == s_abs:
                        list_row += [m]
                        list_col += [s_rel]
                        list_data += [L2_lop[d, s_abs]]
        list_M2_mode_from_state_off += [sparse.coo_matrix((list_data, (list_row,
                                                                   list_col)),
                                                      shape=(nmodes, nstate))]
    M2_diag = sparse.coo_matrix((list_data_diag, (list_row_diag, list_col_diag)),
                                                      shape=(nmodes, nstate))
    return list_M2_mode_from_state_off, M2_diag

def test_get_list_M2_mode_from_state():
    """
    Tests that the get_list_M2_mode_from_state returns the correct list of M2
    mode from state matrices.
    """
    lop_1 = np.array([[1, 0, -1j],
                      [0, 1, 0,],
                      [1j, 0, -2]])
    lop_2 = np.array([[0, 1, 0],
                      [1, 0, 1,],
                      [1, 0, 1]])
    lop_list = [lop_1, lop_2]
    state_list = [0,2]
    dest_list = [0,1,2]
    known_M2_to_0 = np.array([[0, -1j],
                              [0, 0]])
    known_M2_to_1 = np.array([[0, 0],
                              [1, 1]])
    known_M2_to_2 = np.array([[1j, 0],
                              [1, 0]])
    known_M2_diag = np.array([[1, -2],
                              [0, 1]])
    list_M2_off, M2_diag = get_list_M2_mode_from_state(lop_list, state_list, dest_list)
    np.testing.assert_allclose(known_M2_to_0, list_M2_off[0].toarray())
    np.testing.assert_allclose(known_M2_to_1, list_M2_off[1].toarray())
    np.testing.assert_allclose(known_M2_to_2, list_M2_off[2].toarray())
    np.testing.assert_allclose(known_M2_diag, M2_diag.toarray())

    # Same case when the destination state list is very limited - not realistic but
    # does test our test cases properly
    state_list = [0, 2]
    dest_list = [0]
    known_M2_to_0 = np.array([[0, -1j],
                              [0, 0]])
    known_M2_diag = np.array([[1, 0],
                              [0, 0]])
    list_M2_off, M2_diag = get_list_M2_mode_from_state(lop_list, state_list, dest_list)
    np.testing.assert_allclose(known_M2_to_0, list_M2_off[0].toarray())
    np.testing.assert_allclose(known_M2_diag, M2_diag.toarray())

def case_test_error_flux_up_general(F2):
    """
    Tests that the error associated with neglecting flux from lower auxiliaries to
    higher auxiliaries is calculated correctly for general L-operators for a single
    flux up filter.  Includes the stable hierarchy, stable state, and destination
    state cases.

    Parameters
    ----------
    1. F2 : np.array(complex) [nmode, nhier]
            Filters all forbidden flux terms from a given auxiliary up along a given
            mode: k --> k+e_m.
    Returns
    -------
    None
    """
    phi = np.array([0.1j, -0.5j, -0.3j, 0.2, -1.5j, 0.4])
    nstate = 2
    nhier = 3
    nmodes = 4
    P2_phi = phi.reshape([nstate, nhier], order='F')
    # Note that list_w is relative: these 4 modes may belong to a larger set.
    list_w = [50. + 0.j, 500. + 0.j, 50 + 0.j, 500. + 0.j]
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(2, 1)], 4),
                AuxiliaryVector([(2, 2)], 4)]
    list_k_vec = [aux.todense() for aux in aux_list]
    K2_aux_bymode = np.array(list_k_vec).T

    # Peierls L-operators only - Hierarchy Adaptivity Test
    list_l_op = [np.array([[0, 1], [1, 0]])] * 2 + [np.array([[0, -1j], [1j, 0]])] * 2
    state_list = [0, 1]
    dest_list = [0, 1]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    error = error_flux_up_hier_stable(phi, nstate, nhier, nmodes, list_w,
                                      K2_aux_bymode, M2_diag, list_M2_off)
    known_error = get_error_term_hier_flux_up(list_k_vec, list_w, list_l_op, P2_phi)
    assert np.allclose(error, known_error)

    # State Adaptivity Test
    error = error_flux_up_state_stable(phi, nstate, nhier, nmodes, list_w,
                                       K2_aux_bymode, M2_diag, list_M2_off, F2, F2)
    err_by_d = error_flux_up_by_dest_state(phi, nstate, nhier, nmodes, list_w,
                                           K2_aux_bymode, list_M2_off, None, F2, None)
    known_error, known_err_by_d = get_error_term_state_flux_up(list_k_vec, list_w,
                                                               list_l_op, P2_phi, F2)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Test when auxiliaries have been removed from the stable basis
    list_k_vec_red = [list_k_vec[s] for s in [0,2]]
    error = error_flux_up_state_stable(phi, nstate, nhier, nmodes, list_w,
                                       K2_aux_bymode, M2_diag, list_M2_off, F2, F2)
    err_by_d = error_flux_up_by_dest_state(phi, nstate, nhier, nmodes, list_w,
                                           K2_aux_bymode, list_M2_off, [0,2], F2, None)
    known_error, known_err_by_d = get_error_term_state_flux_up(list_k_vec_red, list_w,
                                                               list_l_op,
                                                               P2_phi[:, [0, 2]], F2)
    assert np.allclose(error[:, [0, 2]], known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Peierls and Holstein L-operators - Hierarchy Adaptivity Test
    list_l_op = [np.diag([0.3, -0.7])]*2 + [np.array([[0,-1j],[1j,0]])]*2
    state_list = [0, 1]
    dest_list = [0, 1]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    error = error_flux_up_hier_stable(phi, nstate, nhier, nmodes, list_w,
                                      K2_aux_bymode, M2_diag, list_M2_off)
    known_error = get_error_term_hier_flux_up(list_k_vec, list_w, list_l_op,
                                                   P2_phi)
    assert np.allclose(error, known_error)

    # State Adaptivity Test
    error = error_flux_up_state_stable(phi, nstate, nhier, nmodes, list_w,
                                       K2_aux_bymode, M2_diag, list_M2_off, F2, F2)
    err_by_d = error_flux_up_by_dest_state(phi, nstate, nhier, nmodes, list_w,
                                           K2_aux_bymode, list_M2_off, None, F2, None)
    known_error, known_err_by_d = get_error_term_state_flux_up(list_k_vec, list_w,
                                                               list_l_op, P2_phi, F2)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Mismatched destination and source states - Hierarchy Adaptivity Test
    list_l_op = [np.diag([0.3, -0.7, 0.5])] * 2 + [np.array([[0, -1j, 1],
                                                             [1j, 0, 1],
                                                             [1, 1, 0]])] * 2
    state_list = [0,1]
    dest_list = [0,1,2]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    error = error_flux_up_hier_stable(phi, nstate, nhier, nmodes, list_w,
                                      K2_aux_bymode, M2_diag, list_M2_off)
    known_error = get_error_term_hier_flux_up(list_k_vec, list_w, list_l_op, P2_phi,
                                              state_list, dest_list)
    assert np.allclose(error, known_error)
    # State Adaptivity Test
    error = error_flux_up_state_stable(phi, nstate, nhier, nmodes, list_w,
                                       K2_aux_bymode, M2_diag, list_M2_off, F2, F2)
    err_by_d = error_flux_up_by_dest_state(phi, nstate, nhier, nmodes, list_w,
                                           K2_aux_bymode, list_M2_off, None, F2, None)
    known_error, known_err_by_d = get_error_term_state_flux_up(list_k_vec, list_w,
                                                               list_l_op, P2_phi,
                                                               F2, state_list,
                                                               dest_list)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Source states are not identical to their relative indices - Hierarchy Adaptivity
    # Test
    state_list = [1, 2]
    dest_list = [0, 1, 2]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    error = error_flux_up_hier_stable(phi, nstate, nhier, nmodes, list_w,
                                      K2_aux_bymode, M2_diag, list_M2_off)
    known_error = get_error_term_hier_flux_up(list_k_vec, list_w, list_l_op, P2_phi,
                                              state_list, dest_list)
    assert np.allclose(error, known_error)
    # State Adaptivity Test
    error = error_flux_up_state_stable(phi, nstate, nhier, nmodes, list_w,
                                       K2_aux_bymode, M2_diag, list_M2_off, F2, F2)
    err_by_d = error_flux_up_by_dest_state(phi, nstate, nhier, nmodes, list_w,
                                           K2_aux_bymode, list_M2_off, None, F2, None)
    known_error, known_err_by_d = get_error_term_state_flux_up(list_k_vec, list_w,
                                                               list_l_op, P2_phi, F2,
                                                               state_list, dest_list)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Test the case when states have been removed from the stable basis
    P2_phi_stable_state = np.zeros_like(P2_phi)
    P2_phi_stable_state[1] = P2_phi[1]
    phi_stable_state = np.zeros_like(phi)
    for aux in range(nhier):
        phi_stable_state[nstate*aux+1] = phi[nstate*aux+1]

    error = error_flux_up_state_stable(phi_stable_state, nstate, nhier, nmodes,
                                       list_w, K2_aux_bymode, M2_diag, list_M2_off,
                                       F2, F2)
    err_by_d = error_flux_up_by_dest_state(phi_stable_state, nstate, nhier, nmodes,
                                           list_w, K2_aux_bymode, list_M2_off, None,
                                           F2, [1])
    # Test that including a list of stable states achieves the same result as
    # manually removing population from excluded states.
    err_by_d_prime = error_flux_up_by_dest_state(phi, nstate, nhier, nmodes,
                                                 list_w, K2_aux_bymode, list_M2_off,
                                                 None, F2, [1])
    assert np.allclose(err_by_d, err_by_d_prime)
    known_error, known_err_by_d = get_error_term_state_flux_up(list_k_vec, list_w,
                                                               list_l_op,
                                                               P2_phi_stable_state,
                                                               F2, state_list,
                                                               dest_list)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Test when auxiliaries have been removed from the stable basis
    list_k_vec_red = [list_k_vec[s] for s in [0, 2]]
    err_by_d = error_flux_up_by_dest_state(phi, nstate, nhier, nmodes, list_w,
                                           K2_aux_bymode, list_M2_off, [0,2], F2, [1])
    _, known_err_by_d = get_error_term_state_flux_up(list_k_vec_red, list_w, list_l_op,
                                                     P2_phi_stable_state[:, [0, 2]], F2)
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Fewer destination than source states - Hierarchy Adaptivity Test
    state_list = [0, 1]
    dest_list = [0]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    error = error_flux_up_hier_stable(phi, nstate, nhier, nmodes, list_w,
                                      K2_aux_bymode, M2_diag, list_M2_off)
    known_error = get_error_term_hier_flux_up(list_k_vec, list_w, list_l_op, P2_phi,
                                              state_list, dest_list)
    assert np.allclose(error, known_error)
    # State Adaptivity Test
    error = error_flux_up_state_stable(phi, nstate, nhier, nmodes, list_w,
                                       K2_aux_bymode, M2_diag, list_M2_off, F2, F2)
    err_by_d = error_flux_up_by_dest_state(phi, nstate, nhier, nmodes, list_w,
                                           K2_aux_bymode, list_M2_off, None, F2, None)
    known_error, known_err_by_d = get_error_term_state_flux_up(list_k_vec, list_w,
                                                               list_l_op, P2_phi, F2,
                                                               state_list, dest_list)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

def test_error_flux_up_general():
    """
    Tests that the error associated with neglecting flux from lower auxiliaries to
    higher auxiliaries is calculated correctly for general L-operators.
    """
    F2_initial = None
    case_test_error_flux_up_general(F2_initial)
    F2_exclude = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    case_test_error_flux_up_general(F2_exclude)
def get_error_term_hier_flux_down(list_gw_bymode, list_lop_bymode, list_l_exp_bymode,
                                   P2_phi, state_list=None, dest_state_list = None):
    """
    Gets the flux down hierarchy error matrix manually.

    Parameters
    ----------
    1. list_gw_bymode : list(complex) [nmode]
                The list of correlation function mode coefficients divided by
                their respective decay rates (that is, g/w).
    2. list_lop_bymode : list(np.array(complex)) [nmode, [nstate, nstate]]
                   The list of L2 system-bath coupling operators organized by the
                   same indexing as list of modes, given in the full basis.
    3. list_l_exp_bymode : list(float) [nmode]
                    The list of expectation values of L2 system-bath coupling
                    operators organized by the same indexing as list of modes.
    4. P2_phi : np.array(complex) [nstate, nhier]
                The full wavefunction
    5. state_list : list(int)
                    List of current states in the basis.
    6. dest_state_list : list(int)
                         List of states that can accept flux from the current basis.

    Returns
    -------
    1. E2_error : np.array(complex) [nmodes, nhier]
                  The error in each aux by mode.
    """
    n_state = P2_phi.shape[0]
    n_aux = P2_phi.shape[1]
    n_mode = len(list_lop_bymode)
    if state_list is None:
        state_list = np.arange(n_state)
    if dest_state_list is None:
        dest_state_list = state_list
    n_dest = len(dest_state_list)
    # Generate the identity matrix in the source-to-destination state space.
    I2 = np.zeros([n_dest, n_state])
    for d in range(n_dest):
        for s in range(n_state):
            if dest_state_list[d] == state_list[s]:
                I2[d, s] = 1

    E2_err = np.zeros([n_mode, n_aux])

    # Accounts for the flux down from expectation values of L-operators in all states
    # for states not represented in the destination states.
    list_s_not_in_d = list(set(state_list).difference(set(dest_state_list)))
    if len(list_s_not_in_d) > 0:
        list_rel_s_not_in_d = [np.where(np.array(state_list) == s)[0] for s in
                               list_s_not_in_d]

    for mode_index in range(n_mode):
        L2_lop = (list_lop_bymode[mode_index][np.ix_(dest_state_list,
                                                     state_list)] -
                  I2*list_l_exp_bymode[mode_index])
        gw_mode = list_gw_bymode[mode_index]
        for aux_index in range(n_aux):
            E_mode_aux = np.abs(L2_lop)**2 @ np.abs(P2_phi[:, aux_index]) ** 2
            E_mode_aux *= np.abs(gw_mode)**2
            E2_err[mode_index, aux_index] = float(np.sum(E_mode_aux))
            if len(list_s_not_in_d) > 0:
                for s in list_rel_s_not_in_d:
                    val = (
                        np.abs(gw_mode) ** 2
                        * np.abs(list_l_exp_bymode[mode_index]) ** 2
                        * np.abs(P2_phi[s, aux_index]) ** 2
                    )
                    E2_err[mode_index, aux_index] += float(np.sum(val))
    return E2_err/hbar2

def get_error_term_state_flux_down(list_gw_bymode, list_lop_bymode, list_l_exp_bymode,
                                   P2_phi, F2_filter, state_list=None,
                                  dest_state_list = None):
    """
    Gets the flux up hierarchy error matrix manually.

    Parameters
    ----------
    1. list_gw_bymode : list(complex) [nmode]
                The list of correlation function mode coefficients divided by
                their respective decay rates.
    2. list_lop_bymode : list(np.array(complex)) [nmode, [nstate, nstate]]
                   The list of L2 system-bath coupling operators organized by the
                   same indexing as list of modes, given in the full basis.
    3. list_l_exp_bymode : list(float) [nmode]
                    The list of expectation values of L2 system-bath coupling
                    operators organized by the same indexing as list of modes.
    4. P2_phi : np.array(complex) [nstate, nhier]
                The full wavefunction
    5. F2_filter : np.array(int or bool) [nmode, nhier]
                   Determines whether flux down along a given mode is allowed for a
                   given auxiliary.
    6. state_list : list(int)
                    List of current states in the basis.
    7. dest_state_list : list(int)
                         List of states that can accept flux from the current basis.

    Returns
    -------
    1. E2_error : np.array(complex) [nstate, nhier]
                  The error in each aux by state
    """
    n_state = P2_phi.shape[0]
    n_aux = P2_phi.shape[1]
    n_mode = len(list_lop_bymode)
    if F2_filter is None:
        F2_filter = np.ones([n_mode,n_aux])
    if state_list is None:
        state_list = np.arange(n_state)
    if dest_state_list is None:
        dest_state_list = state_list
    n_dest = len(dest_state_list)

    # Accounts for the flux down from expectation values of L-operators in all states
    # for states not represented in the destination states.
    list_s_not_in_d = list(set(state_list).difference(set(dest_state_list)))
    if len(list_s_not_in_d) > 0:
        list_rel_s_not_in_d = [np.where(np.array(state_list) == s)[0] for s in
                               list_s_not_in_d]

    list_known_err_by_d = np.zeros(n_dest)
    E2_err = np.zeros([n_state, n_aux])
    for state_index in range(n_state):
        for aux_index in range(n_aux):
            E_state_aux = 0
            for mode_index in range(n_mode):
                if F2_filter[mode_index, aux_index]:
                    L2_lop = (list_lop_bymode[mode_index] -
                              np.eye(len(list_lop_bymode[mode_index])) *
                              list_l_exp_bymode[mode_index])
                    for d in range(n_dest):
                        dest = dest_state_list[d]
                        err_element = np.abs(
                            list_gw_bymode[mode_index]
                            * L2_lop[dest, state_list[state_index]]
                            * P2_phi[state_index, aux_index]
                        ) ** 2
                        err_element = float(np.sum(err_element))
                        E_state_aux += err_element
                        list_known_err_by_d[d] += err_element
                    if len(list_s_not_in_d) > 0:
                        if state_index in list_rel_s_not_in_d:
                            err_add = np.abs(
                                list_gw_bymode[mode_index]
                                * list_l_exp_bymode[mode_index]
                                * P2_phi[state_index, aux_index]
                            ) ** 2
                            E_state_aux += float(np.sum(err_add))
            E2_err[state_index, aux_index] = float(E_state_aux)
    return E2_err/hbar2, list_known_err_by_d/hbar2

def get_X2_exp_mode_state(list_lop, list_states, list_exp_lop):
    """
    Gets the L-operator expectation value matrix manually.

    Parameters
    ----------
    1. list_lop : list(np.array(complex)), [nstate, nstate]
                  A list of L2 system-bath coupling operators, given in the full basis.

    2. list_states : list(int)
                     All states in the current basis.

    3. list_exp_lop : list(float)
                      The list of expectation values of L2 system-bath coupling
                        operators organized by the same indexing as list of L2
                        operators.

    Returns
    -------
    1. X2_exp_lop_mode_state : sparse matrix
                               The expectation value contriubtion to flux down, <L_m> *
                               I[s,s], reshaped into the space of [mode, s].
    """
    nstate = len(list_states)
    nmodes = len(list_lop)
    list_data = []
    list_row = []
    list_col = []

    for s in range(len(list_states)):
        list_data += list_exp_lop
        list_col += [s] * len(list_exp_lop)
        list_row += [m for m in range(len(list_exp_lop))]

    return sparse.coo_matrix((list_data, (list_row, list_col)), shape=(nmodes, nstate))

def test_get_X2_exp_mode_from_state():
    """
    Tests that the test_get_X2_exp_mode_from_state returns the expectation value matrix.
    """
    lop_1 = np.array([[1, 0, -1j],
                      [0, 1, 0, ],
                      [1j, 0, -2]])
    lop_2 = np.array([[0, 1, 0],
                      [1, 0, 1, ],
                      [1, 0, 1]])
    list_lop = [lop_1,lop_2]
    exp_lop_1 = 1.0
    exp_lop_2 = -0.5
    list_states = [1,2,3]
    list_exp_lop = [exp_lop_1, exp_lop_2]
    X2_test = get_X2_exp_mode_state(list_lop, list_states, list_exp_lop)
    X2_known = np.array([[1.0, 1.0, 1.0],
                        [-0.5, -0.5, -0.5]])
    np.testing.assert_allclose(X2_test.toarray(), X2_known)

def case_test_error_flux_down_general(F2):
    """
    Tests that the error associated with neglecting flux from higher auxiliaries to
    lower auxiliaries is calculated correctly for general L-operators for a single
    flux down filter. Includes the stable hierarchy, stable state, and destination
    state cases.

    Parameters
    ----------
    1. F2 : np.array(complex) [nmode, nhier]
            Filters all forbidden flux terms from a given auxiliary up along a given
            mode: k --> k+e_m.
    Returns
    -------
    None
    """
    phi = np.array([0.1j, -0.5j, -0.3j, 0.2, -1.5j, 0.4])
    nstate = 2
    nhier = 3
    nmodes = 4
    P2_phi = phi.reshape([nhier,nstate]).T
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(1, 1)], 4),
                AuxiliaryVector([(0, 1), (1, 1)], 4)]

    list_g = np.array(
        [[1000. + 1000j], [1000. + 1000j], [1000. + 1000j], [1000. + 1000j]])
    list_w = np.array([[50. + 0.j], [500. + 0.j], [50. + 0.j], [500. + 0.j]])
    list_gw = list_g/list_w

    # Peierls L-operators only - Hierarchy Adaptivity Test
    list_l_op = [np.array([[0, 1], [1, 0]])] * 2 + [np.array([[0, -1j], [1j, 0]])] * 2
    state_list = [0, 1]
    dest_list = [0, 1]
    list_l_exp = [np.conj(P2_phi[:, 0]) @ lop[np.ix_(state_list,state_list)] @
                  P2_phi[:, 0] for lop in list_l_op]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    X2_exp_mode_state = get_X2_exp_mode_state(list_l_op, state_list, list_l_exp)
    known_error = get_error_term_hier_flux_down(list_gw, list_l_op, list_l_exp, P2_phi)
    error = error_flux_down_hier_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                        M2_diag, list_M2_off, X2_exp_mode_state)
    assert np.allclose(error, known_error)
    # State Adaptivity Test
    known_error, known_err_by_d = get_error_term_state_flux_down(list_gw, list_l_op,
                                                          list_l_exp, P2_phi, F2)
    error = error_flux_down_state_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                         M2_diag, list_M2_off, X2_exp_mode_state, F2,
                                         F2)
    err_by_d = error_flux_down_by_dest_state(phi, nstate, nhier, nmodes, list_g,
                                             list_w, list_M2_off, None, F2, None)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Test when auxiliaries have been removed from the stable basis
    known_error, known_err_by_d = get_error_term_state_flux_down(list_gw, list_l_op,
                                                                 list_l_exp,
                                                                 P2_phi[:,[0,2]], F2)
    error = error_flux_down_state_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                         M2_diag, list_M2_off, X2_exp_mode_state, F2,
                                         F2)
    err_by_d = error_flux_down_by_dest_state(phi, nstate, nhier, nmodes, list_g,
                                             list_w, list_M2_off, [0,2], F2, None)

    assert np.allclose(error[:,[0,2]], known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Peierls and Holstein L-operators - Hierarchy Adaptivity Test
    list_l_op = [np.diag([0.3, -0.7])] * 2 + [np.array([[0, -1j], [1j, 0]])] * 2
    state_list = [0, 1]
    dest_list = [0, 1]
    list_l_exp = [np.conj(P2_phi[:, 0]) @ lop[np.ix_(state_list, state_list)] @
                  P2_phi[:, 0] for lop in list_l_op]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    X2_exp_mode_state = get_X2_exp_mode_state(list_l_op, state_list, list_l_exp)
    known_error = get_error_term_hier_flux_down(list_gw, list_l_op, list_l_exp, P2_phi)
    error = error_flux_down_hier_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                        M2_diag, list_M2_off, X2_exp_mode_state)
    assert np.allclose(error, known_error)
    # State Adaptivity Test
    known_error, known_err_by_d = get_error_term_state_flux_down(list_gw, list_l_op,
                                                                 list_l_exp, P2_phi, F2)
    error = error_flux_down_state_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                         M2_diag, list_M2_off, X2_exp_mode_state, F2,
                                         F2)
    err_by_d = error_flux_down_by_dest_state(phi, nstate, nhier, nmodes, list_g,
                                             list_w, list_M2_off, None, F2, None)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Mismatched destination and source states - Hierarchy Adaptivity Test
    list_l_op = [np.diag([0.3, -0.7, 0.5])] * 2 + [np.array([[0, 1, 1j],
                                                             [1, 0, 1],
                                                             [-1j, 1, 0]])] * 2
    state_list = [0, 1]
    dest_list = [0, 1,2]
    list_l_exp = [np.conj(P2_phi[:, 0]) @ lop[np.ix_(state_list, state_list)] @
                  P2_phi[:, 0] for lop in list_l_op]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    X2_exp_mode_state = get_X2_exp_mode_state(list_l_op, state_list, list_l_exp)
    known_error = get_error_term_hier_flux_down(list_gw, list_l_op, list_l_exp,
                                                P2_phi, state_list, dest_list)
    error = error_flux_down_hier_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                        M2_diag, list_M2_off, X2_exp_mode_state)
    assert np.allclose(error, known_error)
    # State Adaptivity Test
    known_error, known_err_by_d = get_error_term_state_flux_down(list_gw, list_l_op,
                                                                 list_l_exp, P2_phi,
                                                                 F2, state_list,
                                                                 dest_list)
    error = error_flux_down_state_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                         M2_diag, list_M2_off, X2_exp_mode_state, F2,
                                         F2)
    err_by_d = error_flux_down_by_dest_state(phi, nstate, nhier, nmodes, list_g,
                                             list_w, list_M2_off, None, F2, None)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Source states are not identical to their relative indices - Hierarchy Adaptivity
    # Test
    state_list = [1, 2]
    dest_list = [0, 1, 2]
    list_l_exp = [np.conj(P2_phi[:, 0]) @ lop[np.ix_(state_list, state_list)] @
                  P2_phi[:, 0] for lop in list_l_op]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    X2_exp_mode_state = get_X2_exp_mode_state(list_l_op, state_list, list_l_exp)
    known_error = get_error_term_hier_flux_down(list_gw, list_l_op, list_l_exp,
                                                P2_phi, state_list, dest_list)
    error = error_flux_down_hier_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                        M2_diag, list_M2_off, X2_exp_mode_state)
    assert np.allclose(error, known_error)
    # State Adaptivity Test
    known_error, known_err_by_d = get_error_term_state_flux_down(list_gw, list_l_op,
                                                                list_l_exp, P2_phi,
                                                                 F2, state_list,
                                                                 dest_list)
    error = error_flux_down_state_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                         M2_diag, list_M2_off, X2_exp_mode_state, F2,
                                         F2)
    err_by_d = error_flux_down_by_dest_state(phi, nstate, nhier, nmodes, list_g,
                                             list_w, list_M2_off, None, F2, None)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Test the case when states have been removed from the stable basis
    P2_phi_stable_state = np.zeros_like(P2_phi)
    P2_phi_stable_state[1] = P2_phi[1]
    phi_stable_state = np.zeros_like(phi)
    for aux in range(nhier):
        phi_stable_state[nstate * aux + 1] = phi[nstate * aux + 1]

    known_error, known_err_by_d = get_error_term_state_flux_down(list_gw, list_l_op,
                                                                 list_l_exp,
                                                                 P2_phi_stable_state,
                                                                 F2, state_list,
                                                                 dest_list)
    error = error_flux_down_state_stable(phi_stable_state, nstate, nhier, nmodes,
                                         list_g, list_w, M2_diag, list_M2_off,
                                         X2_exp_mode_state, F2, F2)
    err_by_d = error_flux_down_by_dest_state(phi_stable_state, nstate, nhier, nmodes,
                                             list_g, list_w, list_M2_off, None, F2, [1])
    # Test that including a list of stable states achieves the same result as
    # manually removing population from excluded states.
    err_by_d_prime = error_flux_down_by_dest_state(phi, nstate, nhier, nmodes, list_g,
                                                   list_w, list_M2_off, None, F2, [1])
    assert np.allclose(err_by_d, err_by_d_prime)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Test when auxiliaries have been removed from the stable basis
    err_by_d = error_flux_down_by_dest_state(phi, nstate, nhier, nmodes, list_g,
                                             list_w, list_M2_off, [0,2], F2, [1])
    _, known_err_by_d = get_error_term_state_flux_down(list_gw, list_l_op, list_l_exp,
                                                       P2_phi_stable_state[:,[0,2]], F2)
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

    # Fewer destination than source states - Hierarchy Adaptivity Test
    state_list = [0, 1]
    dest_list = [0]
    list_l_exp = [np.conj(P2_phi[:, 0]) @ lop[np.ix_(state_list, state_list)] @
                  P2_phi[:, 0] for lop in list_l_op]
    list_M2_off, M2_diag = get_list_M2_mode_from_state(list_l_op, state_list, dest_list)
    X2_exp_mode_state = get_X2_exp_mode_state(list_l_op, state_list, list_l_exp)
    known_error = get_error_term_hier_flux_down(list_gw, list_l_op, list_l_exp,
                                                P2_phi, state_list, dest_list)
    error = error_flux_down_hier_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                        M2_diag, list_M2_off, X2_exp_mode_state)
    assert np.allclose(error, known_error)
    # State Adaptivity Test
    known_error, known_err_by_d = get_error_term_state_flux_down(list_gw, list_l_op,
                                                                 list_l_exp, P2_phi,
                                                                 F2, state_list,
                                                                 dest_list)
    error = error_flux_down_state_stable(phi, nstate, nhier, nmodes, list_g, list_w,
                                         M2_diag, list_M2_off, X2_exp_mode_state, F2,
                                         F2)
    err_by_d = error_flux_down_by_dest_state(phi, nstate, nhier, nmodes, list_g,
                                             list_w, list_M2_off, None, F2, None)
    assert np.allclose(error, known_error)
    index_out = list(set(dest_list).difference(state_list))
    assert np.allclose(err_by_d[index_out], known_err_by_d[index_out])

def test_error_flux_down_general():
    """
    Tests that the error associated with neglecting flux from higher auxiliaries to
    lower auxiliaries is calculated correctly for general L-operators.
    """
    F2_initial = None
    case_test_error_flux_down_general(F2_initial)
    F2_exclude = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    case_test_error_flux_down_general(F2_exclude)

def test_error_sflux_state():
    """
    Tests that the state flux error for the stable states is correctly calculated.
    """
    
    phi = np.ones(12)
    nstate = 3
    nhier = 4
    list_index_aux_stable = [0, 1, 2]
    list_states = [1, 2, 3]
    
    nstate_extd = 5
    hs = np.zeros([nstate_extd, nstate_extd],dtype=np.complex128)
    hs[1,1] = -50000
    hs[1, 2] = 100
    hs[2, 1] = 100
    hs[4, 2] = 100
    hs[2, 4] = 100
    hs[0,4] = 10000
    hs[4,0] = 10000
    hamiltonian = sparse.csr_array(hs)
    
    
    
    ltc_phys = np.zeros_like(hs)
    ltc_phys[0,0] = 20000
    ltc_phys[1,1] = 30000
    ltc_phys[0,4] = -1000j
    ltc_phys[4,0] = 1000j
    ltc_phys[1,4] = 100
    ltc_phys[4,1] = 100
    ltc_phys[1, 2] = -50
    ltc_phys[2, 1] = -50
    ltc_phys = sparse.csr_array(ltc_phys)

    ltc_hier = np.zeros_like(hs)
    ltc_hier[0, 0] = 20000
    ltc_hier[1, 1] = 30000
    ltc_hier[0, 4] = -1000j
    ltc_hier[4, 0] = 1000j
    ltc_hier[3, 0] = -200
    ltc_hier[0, 3] = -200
    ltc_hier = sparse.csr_array(ltc_hier)

    

    E1_state_flux = error_sflux_stable_state(phi, nstate, nhier,
                                                         hamiltonian,
                                                         list_index_aux_stable,
                                                         list_states)
    known_error = [30000 / hbar ** 2, 60000 / hbar ** 2, 0]

    assert np.allclose(E1_state_flux, known_error)

    E1_state_flux = error_sflux_stable_state(phi, nstate, nhier,
                                                         hamiltonian,
                                                         list_index_aux_stable,
                                                         list_states, ltc_phys,
                                                         ltc_hier)
    known_error = [32500 / hbar ** 2, 52500 / hbar ** 2, 80000 / hbar ** 2]

    assert np.allclose(E1_state_flux, known_error)


# ------------------------------------------------------------
# TEST: Peierls off-diagonal noise (no LTC)
# ------------------------------------------------------------
def test_error_sflux_stable_state_peierls_no_ltc():
    """
    Tests the state flux error with a Peierls-style Hamiltonian input:
    H2_hamiltonian_ext = -1j * H_sys + Z_noise, where off-diagonal noise creates
    complex boundary-basis coupling. Exercises the no-LTC code path with complex Phi.
    """
    
    nstate = 2
    nhier = 2
    phi = np.array([1 + 1j, 2, 1j, 1], dtype=np.complex128)
    list_index_aux_stable = [0, 1]
    list_states = [1, 2]
    
    # 4-state extended space: basis = {1, 2}, boundary = {0, 3}
    hs = np.zeros([4, 4])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 20
    hs[2, 1] = 20
    hs[2, 3] = 30
    hs[3, 2] = 30

    # Off-diagonal noise couples boundary to basis (Peierls modes)
    z_noise = np.zeros([4, 4], dtype=np.complex128)
    z_noise[0, 1] = 5 + 3j
    z_noise[3, 2] = -2 + 4j

    H2_hamiltonian_extd = sparse.csr_array(-1j * hs + z_noise)

    

    E1_state_flux = error_sflux_stable_state(phi, nstate, nhier,
                                             H2_hamiltonian_extd,
                                             list_index_aux_stable,
                                             list_states)

    # H_ext = -1j*H_sys + Z_noise (diagonal is zero):
    #   [[0,      5-7j,     0,       0],
    #    [-10j,   0,        -20j,    0],
    #    [0,      -20j,     0,       -2-26j],
    #    [0,      0,        -2-26j,  0]]
    #
    # H_couplings[:, [1,2]] (basis columns):
    #   [[5-7j,     0      ],
    #    [0,        -20j    ],
    #    [-20j,     0       ],
    #    [0,        -2-26j  ]]
    #
    # V1[s] = sum_d |H[d,s]|^2:
    #   V1[0] = |5-7j|^2 + |-20j|^2 = 74 + 400 = 474
    #   V1[1] = |-20j|^2 + |-2-26j|^2 = 400 + 680 = 1080
    #
    # C2_phi reshaped [2,2] order="F": [[1+1j, 1j], [2, 1]]
    # C1_norm[s] = sum_k |psi_k[s]|^2:
    #   C1[0] = |1+1j|^2 + |1j|^2 = 2 + 1 = 3
    #   C1[1] = |2|^2 + |1|^2 = 4 + 1 = 5
    known_error = [474 * 3 / hbar ** 2, 1080 * 5 / hbar ** 2]
    assert np.allclose(E1_state_flux, known_error)


# ------------------------------------------------------------
# TEST: Peierls + LTC with separate physical/auxiliary operators
# ------------------------------------------------------------
def test_error_sflux_stable_state_peierls_ltc():
    """
    Tests the full Peierls scenario for the stable state flux: complex Hamiltonian
    with off-diagonal noise and separate LTC operators for the physical and
    auxiliary wave functions. Exercises the LTC-enabled code path.
    """
    hs = np.zeros([4, 4])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 20
    hs[2, 1] = 20
    hs[2, 3] = 30
    hs[3, 2] = 30

    z_noise = np.zeros([4, 4], dtype=np.complex128)
    z_noise[0, 1] = 5 + 3j
    z_noise[3, 2] = -2 + 4j

    H2_hamiltonian_extd = sparse.csr_array(-1j * hs + z_noise)

    ltc_phys = np.zeros([4, 4], dtype=np.complex128)
    ltc_phys[0, 1] = 2
    ltc_phys[3, 2] = -5
    ltc_phys = sparse.csr_array(ltc_phys)

    ltc_hier = np.zeros([4, 4], dtype=np.complex128)
    ltc_hier[0, 1] = 1
    ltc_hier[3, 2] = 3
    ltc_hier = sparse.csr_array(ltc_hier)

    nstate = 2
    nhier = 3
    phi = np.array([1 + 1j, 2, 1j, 1, 3, -1 + 1j], dtype=np.complex128)
    list_index_aux_stable = [0, 1, 2]
    list_states = [1, 2]

    E1_state_flux = error_sflux_stable_state(phi, nstate, nhier,
                                             H2_hamiltonian_extd,
                                             list_index_aux_stable,
                                             list_states,
                                             ltc_phys, ltc_hier)

    # H_ext (diagonal is zero, H_couplings = H_ext):
    #   [[0,      5-7j,     0,        0],
    #    [-10j,   0,        -20j,     0],
    #    [0,      -20j,     0,        -2-26j],
    #    [0,      0,        -2-26j,   0]]
    #
    # --- Physical (k=0) ---
    # T_phys: (diagonal is zero)
    #   [[0,      2,     0,   0],
    #    [0,      0,     0,   0],
    #    [0,      0,     0,   0],
    #    [0,      0,    -5,   0]]
    # H_phys = H_couplings + T_phys_couplings:
    #   H_phys[:,[1,2]] = [[7-7j, 0], [0, -20j], [-20j, 0], [0, -7-26j]]
    #   V1_phys[0] = |7-7j|^2 + |-20j|^2 = 98 + 400 = 498
    #   V1_phys[1] = |-20j|^2 + |-7-26j|^2 = 400 + 725 = 1125
    #
    # --- Auxiliary (k>0) ---
    # T_hier: (diagonal is zero)
    #   [[0,      1,     0,   0],
    #    [0,      0,     0,   0],
    #    [0,      0,     0,   0],
    #    [0,      0,     3,   0]]
    # H_hier = H_couplings + T_hier_couplings:
    #   H_hier[:,[1,2]] = [[6-7j, 0], [0, -20j], [-20j, 0], [0, 1-26j]]
    #   V1_hier[0] = |6-7j|^2 + |-20j|^2 = 85 + 400 = 485
    #   V1_hier[1] = |-20j|^2 + |1-26j|^2 = 400 + 677 = 1077
    #
    # Phi reshaped [2,3] order="F": [[1+1j, 1j, 3], [2, 1, -1+1j]]
    # C_phys = [[1+1j], [2]]
    #   |C_phys|^2 = [2, 4]
    # C_aux = [[0, 1j, 3], [0, 1, -1+1j]]
    #   |C_aux|^2 summed = [0+1+9, 0+1+2] = [10, 3]
    #
    # E1 = (V_phys * |C_phys|^2 + V_hier * |C_aux|^2) / hbar^2
    known_error = [(498 * 2 + 485 * 10) / hbar ** 2,
                   (1125 * 4 + 1077 * 3) / hbar ** 2]
    assert np.allclose(E1_state_flux, known_error)


# ------------------------------------------------------------
# TEST: Peierls + LTC with single auxiliary (no aux wave functions)
# ------------------------------------------------------------
def test_error_sflux_stable_state_peierls_ltc_single_aux():
    """
    Tests the LTC branch when only the physical wave function is stable
    (single auxiliary). The auxiliary contribution C2_phi_aux[:, 1:] is empty,
    so only the physical wave function contributes to the error.
    """
    hs = np.zeros([4, 4])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 20
    hs[2, 1] = 20
    hs[2, 3] = 30
    hs[3, 2] = 30

    z_noise = np.zeros([4, 4], dtype=np.complex128)
    z_noise[0, 1] = 5 + 3j
    z_noise[3, 2] = -2 + 4j

    H2_hamiltonian_extd = sparse.csr_array(-1j * hs + z_noise)

    ltc_phys = np.zeros([4, 4], dtype=np.complex128)
    ltc_phys[0, 1] = 2
    ltc_phys[3, 2] = -5
    ltc_phys = sparse.csr_array(ltc_phys)

    ltc_hier = sparse.csr_array(np.zeros([4, 4], dtype=np.complex128))

    nstate = 2
    nhier = 1
    phi = np.array([1 + 1j, 2], dtype=np.complex128)
    list_index_aux_stable = [0]
    list_states = [1, 2]

    E1_state_flux = error_sflux_stable_state(phi, nstate, nhier,
                                             H2_hamiltonian_extd,
                                             list_index_aux_stable,
                                             list_states,
                                             ltc_phys, ltc_hier)
    # H_ext (diagonal is zero, H_couplings = H_ext):
    #   [[0,      5-7j,     0,        0],
    #    [-10j,   0,        -20j,     0],
    #    [0,      -20j,     0,        -2-26j],
    #    [0,      0,        -2-26j,   0]]
    #
    # --- Physical (k=0) ---
    # T_phys: (diagonal is zero)
    #   [[0,      2,     0,   0],
    #    [0,      0,     0,   0],
    #    [0,      0,     0,   0],
    #    [0,      0,    -5,   0]]
    # H_phys = H_couplings + T_phys_couplings:
    # H_phys[:,[1,2]] = [[7-7j, 0], [0, -20j], [-20j, 0], [0, -7-26j]]
    # V1_phys[0] = |7-7j|^2 + |-20j|^2 = 98 + 400 = 498
    # V1_phys[1] = |-20j|^2 + |-7-26j|^2 = 400 + 725 = 1125
    # V1_phys = [498, 1125]
    # Phi reshaped [2,3] order="F": [[1+1j, 1j, 3], [2, 1, -1+1j]]
    # C_phys = [[1+1j], [2]]
    # |C_phys|^2 = [2, 4]
    # C_aux is empty -> |C_aux|^2 = [0, 0]
    # E1 = V_phys * |C_phys|^2 / hbar^2 (auxiliary term is zero)
    known_error = [498 * 2 / hbar ** 2, 1125 * 4 / hbar ** 2]
    assert np.allclose(E1_state_flux, known_error)


def test_error_sflux_boundary_state_general():
    """
    Tests that the error for the boundary states is correctly calculated when there
    is flux up and flux down that could lead into those states.
    """
    nsite = 10
    hs = np.zeros([nsite, nsite])
    # hs[0, 4] = 1000
    # hs[4, 0] = 1000
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 20
    hs[2, 1] = 20
    hs[2, 3] = 30
    hs[3, 2] = 30
    hs[3, 4] = 40
    hs[4, 3] = 40
    hs[4, 5] = 50
    hs[5, 4] = 50
    hs[5, 6] = 60
    hs[6, 5] = 60
    hs[6, 7] = 70
    hs[7, 6] = 70
    hs[7, 8] = 80
    hs[3, 8] = 1000
    hs[8, 3] = 1000
    hs[8, 7] = 80
    hs[8, 9] = 90
    hs[9, 8] = 90

    # 5th site DNE
    phi = np.ones(20)

    # Extended basis ordering used by H2_hamiltonian_extd.
    state_plus_bound = np.array([1, 2, 3, 4, 5, 6, 7, 9])
    hamiltonian = sparse.csr_array(hs)
    H2_hamiltonian_extd = sparse.csr_array(hs[np.ix_(state_plus_bound, state_plus_bound)])

    # Boundary states in absolute indexing (state 8 excluded: just removed from basis).
    list_fullbndidx_abs = [1, 2, 7, 9]

    # Indices are relative to state_plus_bound ordering.
    list_stblstateidx_extd = [2, 3, 4, 5]
    list_bndstateidx_extd = [0, 1, 6, 7]
    # Destination states that can receive flux up/down from hierarchy terms.
    list_d = [1, 2, 3, 4, 8, 9]
    # Sums of flux up and down going into each state in list_d
    list_flux_by_dest = [0.1, 0.2, 0.3, 0.4, 0, 0]

    # Mimics the code in _define_state_basis in hops_basis.py that produces the lists
    # of relative sc states that receive flux and their respective total fluxes.
    list_fullbndidx_abs_dest = []
    list_flux_updown = []
    for d_ind in range(len(list_d)):
        d = list_d[d_ind]
        if d in list_fullbndidx_abs:
            list_fullbndidx_abs_dest.append(np.where(np.array(list_fullbndidx_abs) == d)[0][0])
            list_flux_updown.append(list_flux_by_dest[d_ind])
    # list_fullbndidx_abs destinations are states [1, 2, 9] -> positions [0, 1, 3].
    # State 8 is intentionally excluded because it is in list_sremove.
    assert np.allclose(list_fullbndidx_abs_dest, np.array([0, 1, 3]))
    assert np.allclose(list_flux_updown, np.array([0.1, 0.2, 0]))

    # This setup gives nonzero boundary coupling between (2,3) and (6,7).
    nstate = 5
    nhier = 4
    list_index_aux_stable = [0, 1, 2]

    # Early-return code path: no boundary states are omitted from stable set.
    list_index_state_stable = np.arange(0, 10)
    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd, nstate,
                                               nhier, H2_hamiltonian_extd,
                                               list_index_state_stable,
                                               list_index_aux_stable,
                                               list_fullbndidx_abs_dest, list_flux_updown))
    known_error = []
    known_indices = []
    assert np.allclose(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)

    # Normal boundary-flux code path: some boundary states are omitted from stable set.
    list_index_state_stable = [0, 1, 2, 3]
    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd, nstate,
                                               nhier, H2_hamiltonian_extd,
                                               list_index_state_stable,
                                               list_index_aux_stable,
                                               list_fullbndidx_abs_dest, list_flux_updown))
    known_indices = [1, 2, 7]
    known_error = [0.1,
                   (30 ** 2 / hbar ** 2 + 30 ** 2 / hbar ** 2
                    + 30 ** 2 / hbar ** 2) + 0.2,
                   (70 ** 2 / hbar ** 2 + 70 ** 2 / hbar ** 2 + 70 ** 2 / hbar ** 2)]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)

    # Test with no flux up or down
    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd, nstate,
                                               nhier, H2_hamiltonian_extd,
                                               list_index_state_stable,
                                               list_index_aux_stable, [], [])
    )
    known_indices = [2, 7]
    known_error = [(30 ** 2 / hbar ** 2 + 30 ** 2 / hbar ** 2 + 30 ** 2 / hbar ** 2),
                   (70 ** 2 / hbar ** 2 + 70 ** 2 / hbar ** 2 + 70 ** 2 / hbar ** 2)]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)


    # Test with low-temperature correction
    ltc_phys = np.zeros_like(hs,dtype=np.complex128)
    ltc_hier = np.zeros_like(hs,dtype=np.complex128)
    ltc_phys[0,3] = 20
    ltc_phys[3,0] = 20
    ltc_phys[3,4] = 500000000
    ltc_phys[4,3] = 500000000
    ltc_phys[2,3] = -5
    ltc_phys[3,2] = -5
    ltc_hier[0,4] = -10
    ltc_hier[4,3] = 10-100000000j
    ltc_hier[3,4] = 10+100000000j

    state_plus_bound = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9])
    ltc_phys_extd = sparse.csr_array(ltc_phys[np.ix_(state_plus_bound, state_plus_bound)])
    ltc_hier_extd = sparse.csr_array(ltc_hier[np.ix_(state_plus_bound, state_plus_bound)])
    H2_hamiltonian_extd = sparse.csr_array(hs[np.ix_(state_plus_bound, state_plus_bound)])

    # Boundary states in absolute indexing (state 8 excluded: just removed from basis).
    list_fullbndidx_abs = [0, 1, 2, 7, 9]
    # Indices are relative to state_plus_bound ordering.
    list_stblstateidx_extd = [3, 4, 5, 6]
    list_bndstateidx_extd = [0, 1, 2, 7, 8]
    # States that can receive flux up or down
    list_d = [1, 2, 3, 4, 8, 9]
    # Sums of flux up and down going into each state in list_d
    list_flux_by_dest = [0.1, 0.2, 0.3, 0.4, 0, 0]

    # Mimics the code in _define_state_basis in hops_basis.py that produces the lists
    # of relative sc states that receive flux and their respective total fluxes.
    list_fullbndidx_abs_dest = []
    list_flux_updown = []
    for d_ind in range(len(list_d)):
        d = list_d[d_ind]
        if d in list_fullbndidx_abs:
            list_fullbndidx_abs_dest.append(np.where(np.array(list_fullbndidx_abs) == d)[0][0])
            list_flux_updown.append(list_flux_by_dest[d_ind])
    # list_fullbndidx_abs destinations are states [1, 2, 9] -> positions [1, 2, 4].
    # State 8 is intentionally excluded because it is in list_sremove.
    assert np.allclose(list_fullbndidx_abs_dest, np.array([1, 2, 4]))
    assert np.allclose(list_flux_updown, np.array([0.1, 0.2, 0]))

    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd, nstate,
                                               nhier, H2_hamiltonian_extd,
                                               list_index_state_stable,
                                               list_index_aux_stable,
                                               list_fullbndidx_abs_dest, list_flux_updown,
                                               ltc_phys_extd, ltc_hier_extd))
    known_indices = [0,1,2,7]
    known_error = [(20 ** 2 /hbar ** 2 + 10 ** 2 / hbar **2 + 10 ** 2 / hbar **2), 0.1,
                   (25 ** 2 / hbar ** 2 + 30 ** 2 / hbar ** 2 + 30 ** 2 / hbar ** 2) +
                   0.2,
                   (70 ** 2 / hbar ** 2 + 70 ** 2 / hbar ** 2 + 70 ** 2 / hbar ** 2)]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error, rtol=1e-8)


# ------------------------------------------------------------
# TEST: Complex Hamiltonian (production-like input)
# ------------------------------------------------------------
def test_error_sflux_boundary_state_complex_hamiltonian():
    """
    Tests that boundary state error is correctly calculated when the extended
    Hamiltonian is complex, as in production where H_ext = -1j * H_sys + Z_noise.
    """
    # 4-site system: basis {1,2}, boundary {0,3}
    hs = np.zeros([4, 4])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 20
    hs[2, 1] = 20
    hs[2, 3] = 30
    hs[3, 2] = 30

    nstate = 2
    nhier = 2
    phi = np.array([1 + 1j, 2 + 0j, 0 + 1j, 1 + 0j], dtype=np.complex128)
    list_stblstateidx_extd = [1, 2]
    list_fullbndidx_abs = [0, 3]
    list_bndstateidx_extd = [0, 3]
    list_index_state_stable = [0, 1]
    list_index_aux_stable = [0, 1]

    # This case tests complex H_ext = -1j * H_sys (no noise, no LTC)
    H2_hamiltonian_extd = sparse.csr_array(-1j * hs)

    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd, nstate,
                                   nhier, H2_hamiltonian_extd,
                                   list_index_state_stable,
                                   list_index_aux_stable, [], []))

    # H_ext[boundary, basis] = [[-10j, 0], [0, -30j]]
    # C2_phi = [[1+1j, 1j], [2, 1]]
    # Row 0: |(-10j)(1+1j)|^2 + |(-10j)(1j)|^2 = 200 + 100
    # Row 1: |(-30j)(2)|^2 + |(-30j)(1)|^2 = 3600 + 900
    known_indices = [0, 3]
    known_error = [300 / hbar ** 2, 4500 / hbar ** 2]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)

    # This case tests complex H with Peierls-style flux up/down
    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd, nstate,
                                   nhier, H2_hamiltonian_extd,
                                   list_index_state_stable,
                                   list_index_aux_stable,
                                   [0], [0.3]))
    known_indices = [0, 3]
    known_error = [300 / hbar ** 2 + 0.3, 4500 / hbar ** 2]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)

def test_error_sflux_boundary_state_partial_aux_stable_selection():
    """
    Verify only list_index_aux_stable columns contribute when not all auxiliaries are stable.
    """
    hs = np.zeros([4, 4])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 20
    hs[2, 1] = 20
    hs[2, 3] = 30
    hs[3, 2] = 30

    nstate = 2
    nhier = 3
    # Fortran-order columns:
    # psi_0 = [1, 2], psi_1 = [3, 4], psi_2 = [100, 200] (should be ignored)
    phi = np.array([1, 2, 3, 4, 100, 200], dtype=np.complex128)
    list_stblstateidx_extd = [1, 2]
    list_fullbndidx_abs = [0, 3]
    list_bndstateidx_extd = [0, 3]
    list_index_state_stable = [0, 1]
    list_index_aux_stable = [0, 1]
    H2_hamiltonian_extd = sparse.csr_array(-1j * hs)

    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(
            phi,
            list_stblstateidx_extd,
            list_fullbndidx_abs,
            list_bndstateidx_extd,
            nstate,
            nhier,
            H2_hamiltonian_extd,
            list_index_state_stable,
            list_index_aux_stable,
            [],
            [],
        )
    )

    # H_ext[boundary, basis] = [[-10j, 0], [0, -30j]]
    # Row 0: |(-10j)*1|^2 + |(-10j)*3|^2 = 100 + 900 = 1000
    # Row 1: |(-30j)*2|^2 + |(-30j)*4|^2 = 3600 + 14400 = 18000
    known_indices = [0, 3]
    known_error = [1000 / hbar ** 2, 18000 / hbar ** 2]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)


# ------------------------------------------------------------
# TEST: Off-diagonal noise creates boundary flux (Peierls)
# ------------------------------------------------------------
def test_error_sflux_boundary_state_offdiag_noise():
    """
    Tests that off-diagonal noise terms from Peierls modes create flux into
    boundary states even when the system Hamiltonian has no boundary-basis coupling.
    """
    # H_sys has ONLY intra-basis coupling (no boundary-basis coupling)
    nstate_extd = 4
    nstate = 2
    nhier = 2
    phi = np.array([1 + 1j, 2 + 0j, 0 + 1j, 1 + 0j], dtype=np.complex128)
    list_stblstateidx_extd = [1, 2]
    list_fullbndidx_abs = [0, 3]
    list_bndstateidx_extd = [0, 3]
    list_index_state_stable = [0, 1]
    list_index_aux_stable = [0, 1]
    hs = np.zeros([nstate_extd, nstate_extd])
    hs[1, 2] = 20
    hs[2, 1] = 20
    # Add more random entries which should be sliced out and don't matter
    hs[0, 3] = 7
    hs[1, 0] = 33
    hs[2, 0] = 666666
    # Off-diagonal noise creates boundary-basis coupling
    z_noise = np.zeros([4, 4], dtype=np.complex128)
    z_noise[0, 1] = 5 + 3j
    z_noise[3, 2] = -2 + 4j

    

    H2_hamiltonian_extd = sparse.csr_array(-1j * hs + z_noise)

    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd, nstate,
                                   nhier, H2_hamiltonian_extd,
                                   list_index_state_stable,
                                   list_index_aux_stable, [], []))

    # H_ext[boundary, basis] = [[5+3j, 0], [0, -2+4j]]
    # (noise-only, no H_sys contribution to boundary-basis block)
    # C2_phi = [[1+1j, 1j], [2, 1]]
    # Row 0: |(5+3j)(1+1j)|^2 + |(5+3j)(1j)|^2 = |2+8j|^2 + |-3+5j|^2 = 68 + 34
    # Row 1: |(-2+4j)(2)|^2 + |(-2+4j)(1)|^2 = |-4+8j|^2 + |-2+4j|^2 = 80 + 20
    known_indices = [0, 3]
    known_error = [102 / hbar ** 2, 100 / hbar ** 2]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)


# ------------------------------------------------------------
# TEST: LTC branch with no auxiliary wave functions
# ------------------------------------------------------------
def test_error_sflux_boundary_state_ltc_no_aux():
    """
    Tests the LTC branch when n_hier=1 (only the physical wave function).
    Exercises the len(C2_phi_aux) == 0 path that skips auxiliary computation.
    """
    nstate_extd = 4
    hs = np.zeros([nstate_extd, nstate_extd])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 20
    hs[2, 1] = 20
    hs[2, 3] = 30
    hs[3, 2] = 30

    nstate = 2
    nhier = 1
    phi = np.array([1 + 1j, 2 + 0j], dtype=np.complex128)
    list_stblstateidx_extd = [1, 2]
    list_fullbndidx_abs = [0, 3]
    list_bndstateidx_extd = [0, 3]
    list_index_state_stable = [0, 1]
    list_index_aux_stable = [0]

    H2_hamiltonian_extd = sparse.csr_array(hs, dtype=np.complex128)
    ltc_phys = np.zeros([4, 4], dtype=np.complex128)
    ltc_phys[0, 1] = 5
    ltc_phys[3, 2] = -3
    ltc_phys_extd = sparse.csr_array(ltc_phys)
    # T2_hier is provided but never accessed (no aux wave functions)
    ltc_hier_extd = sparse.csr_array(np.zeros([4, 4], dtype=np.complex128))

    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd, nstate,
                                   nhier, H2_hamiltonian_extd,
                                   list_index_state_stable,
                                   list_index_aux_stable, [], [],
                                   ltc_phys_extd, ltc_hier_extd))

    # H2_sparse_phys = (H_ext + T2_phys)[boundary, basis] = [[15, 0], [0, 27]]
    # C1_phi_phys = [[1+1j], [2]]
    # Row 0: |15*(1+1j)|^2 = |15+15j|^2 = 450
    # Row 1: |27*2|^2 = 2916
    # C2_phi_aux is empty -> no auxiliary contribution
    known_indices = [0, 3]
    known_error = [450 / hbar ** 2, 2916 / hbar ** 2]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)


# ------------------------------------------------------------
# TEST: Full Peierls + LTC combination
# ------------------------------------------------------------
def test_error_sflux_boundary_state_peierls_ltc():
    """
    Tests the full Peierls scenario: complex Hamiltonian with off-diagonal noise,
    separate LTC operators for physical and auxiliary wave functions, and flux
    up/down contributions from off-diagonal modes.
    """
    nstate_extd = 4
    hs = np.zeros([nstate_extd, nstate_extd])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 20
    hs[2, 1] = 20
    hs[2, 3] = 30
    hs[3, 2] = 30

    z_noise = np.zeros([nstate_extd, nstate_extd], dtype=np.complex128)
    z_noise[0, 2] = 7j

    nstate = 2
    nhier = 3
    phi = np.array([1 + 1j, 2, 1j, 1, 3, -1 + 1j], dtype=np.complex128)
    list_stblstateidx_extd = [1, 2]
    list_fullbndidx_abs = [0, 3]
    list_bndstateidx_extd = [0, 3]
    list_index_state_stable = [0, 1]
    list_index_aux_stable = [0, 1, 2]

    H2_hamiltonian_extd = sparse.csr_array(-1j * hs + z_noise)

    ltc_phys = np.zeros([4, 4], dtype=np.complex128)
    ltc_phys[0, 1] = 2
    ltc_phys[3, 2] = -5
    ltc_phys_extd = sparse.csr_array(ltc_phys)

    ltc_hier = np.zeros([4, 4], dtype=np.complex128)
    ltc_hier[0, 1] = 1
    ltc_hier[3, 2] = 3
    ltc_hier_extd = sparse.csr_array(ltc_hier)

    # Flux up/down into state 3 (index 1 in list_fullbndidx_abs)
    list_fullbndidx_abs_dest = [1]
    list_flux_updown = [0.5]

    E1_sum_indices, E1_sum_error = (
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd, nstate,
                                   nhier, H2_hamiltonian_extd,
                                   list_index_state_stable,
                                   list_index_aux_stable,
                                   list_fullbndidx_abs_dest, list_flux_updown,
                                   ltc_phys_extd, ltc_hier_extd))

    # Phi reshaped [2,3] order="F":
    #   [[1+1j, 1j,  3     ],
    #    [2,    1,   -1+1j  ]]
    #
    # H_ext = -1j*H_sys + Z_noise:
    #   [[0,    -10j,  7j,    0],
    #    [-10j,  0,    -20j,  0],
    #    [0,    -20j,  0,    -30j],
    #    [0,     0,   -30j,   0]]
    #
    # --- Physical (k=0) ---
    # H_phys = (H_ext+T_phys)[boundary, basis] = [[2-10j, 7j], [0, -5-30j]]
    # C_phys = [[1+1j], [2]]
    # Row 0: |(2-10j)(1+1j) + 7j*2| = |12+6j| -> 180
    # Row 1: |(-5-30j)*2| = |-10-60j| -> 3700
    #
    # --- Auxiliary (k>0) ---
    # H_aux = (H_ext+T_hier)[boundary, basis] = [[1-10j, 7j], [0, 3-30j]]
    # C_aux = [[1j, 3], [1, -1+1j]]
    # Row 0: |10+8j|^2 + |-4-37j|^2 = 164 + 1385 = 1549
    # Row 1: |3-30j|^2 + |27+33j|^2 = 909 + 1818 = 2727
    #
    # Total = phys + aux + flux:
    # State 0: (180 + 1549) / hbar^2 = 1729 / hbar^2
    # State 3: (3700 + 2727) / hbar^2 + 0.5 = 6427 / hbar^2 + 0.5
    known_indices = [0, 3]
    known_error = [1729 / hbar ** 2, 6427 / hbar ** 2 + 0.5]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error, rtol=1e-8)


# ------------------------------------------------------------
# TEST: LTC operator mismatch raises error
# ------------------------------------------------------------
def test_error_sflux_boundary_state_ltc_mismatch():
    """
    Tests that providing T2_phys without T2_hier raises a TypeError when
    n_hier > 1, since the function attempts H_ext + None.
    """
    hs = np.zeros([4, 4])
    hs[0, 1] = 10
    hs[1, 0] = 10

    nstate = 2
    nhier = 2
    phi = np.array([1, 2, 3, 4], dtype=np.complex128)
    list_stblstateidx_extd = [1, 2]
    list_fullbndidx_abs = [0, 3]
    list_bndstateidx_extd = [0, 3]
    list_index_state_stable = [0, 1]
    list_index_aux_stable = [0, 1]

    H2_hamiltonian_extd = sparse.csr_array(hs, dtype=np.complex128)
    ltc_phys_extd = sparse.csr_array(np.zeros([4, 4], dtype=np.complex128))

    with pytest.raises(TypeError):
        error_sflux_boundary_state(phi, list_stblstateidx_extd, list_fullbndidx_abs, list_bndstateidx_extd,
                                   nstate, nhier, H2_hamiltonian_extd,
                                   list_index_state_stable,
                                   list_index_aux_stable, [], [],
                                   ltc_phys_extd, None)
