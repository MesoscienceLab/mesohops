import numpy as np
import pytest
import scipy as sp
from scipy import sparse
from mesohops.dynamics.basis_functions_adaptive import *
from mesohops.dynamics.hops_aux import AuxiliaryVector
from mesohops.util.physical_constants import hbar


def test_error_sflux_hier():
    """
    test for the error associated with losing all flux terms inside the kth auxiliary to
    states not contained in S_t
    """
    nsite = 5
    hs = np.zeros([nsite, nsite])
    hs[4, 3] = 100
    hs[3, 4] = 100
    hs[1, 2] = 50
    hs[2, 1] = 50

    phi = np.ones(9)
    n_state = 3
    n_hier = 3
    state_list = [1, 2, 3]
    hamiltonian = sparse.csc_matrix(hs)

    E2_flux_state = error_sflux_hier(phi, state_list, n_state, n_hier, hamiltonian)
    hbar2 = hbar * hbar
    known_error = [10000 / hbar2, 10000 / hbar2, 10000 / hbar2]
    assert np.allclose(E2_flux_state, known_error)


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

    # if test
    E2_del_phi = error_deriv(dsystem_dt_if, phi, z_step, n_state, n_hier, dt,
                             list_index_aux_stable=list_index_aux_stable)
    known_deriv_error = np.array([[1 / hbar, 1 / hbar], [1 / hbar, 1 / hbar]])
    known_del_flux = phi.reshape([n_state, n_hier], order="F")[:,
                     list_index_aux_stable] / dt
    known_error = np.abs(known_deriv_error + known_del_flux) ** 2
    assert np.allclose(E2_del_phi, known_error)

    # else test
    E2_del_phi = error_deriv(dsystem_dt_else, phi, z_step, n_state, n_hier, dt)
    known_deriv_error = [[1 / hbar, 1 / hbar, 1 / hbar], [1 / hbar, 1 / hbar, 1 / hbar]]
    known_del_flux = phi.reshape([n_state, n_hier], order="F") / dt
    known_error = np.abs(known_deriv_error + known_del_flux) ** 2
    assert np.allclose(E2_del_phi, known_error)


def test_error_flux_up():
    """
    the error associated with neglecting flux from members of A_t to auxiliaries in
    A_t^C that arise due to flux from lower auxiliaries to higher auxiliaries.
    """
    def get_error_term_hier_flux_up(list_k_vec, w_list, l_op_list, P2_phi):
        """
        Get the flux up hierarchy error matrix manually.
        Parameters
        ----------
        1. list_k_vec : list(np.array(int)) [nhier [nmode]]
                        The list of auxiliary indexing vectors
        2. w_list : list(complex) [nmode]
                    The list of correlation function mode decay rates
        3. l_op_list : list(np.array(complex)) [nmode, [nstate, nstate]]
                       The list of L2 system-bath coupling operators organized by the
                       same indexing as list of modes
        4. P2_phi : np.array(complex) [nstate, nhier]
                    The full wavefunction

        Returns
        -------
        1. E2_error : np.array(complex) [nmodes, nhier]
                      The error in each aux by mode
        """
        E2_err = np.zeros([len(l_op_list), len(P2_phi[0])])
        for mode_index in range(len(l_op_list)):
            L2_lop = l_op_list[mode_index]
            w_mode = list_w[mode_index]
            for aux_index in range(len(P2_phi[0])):
                E_mode_aux = 0
                k_mode = list_k_vec[aux_index][mode_index]
                for state_index in range(len(P2_phi)):
                    E_mode_aux += np.abs(P2_phi[state_index, aux_index] * \
                                  L2_lop[state_index, state_index])**2
                E_mode_aux *= np.abs((1 + k_mode) * w_mode)**2
                E2_err[mode_index, aux_index] = E_mode_aux
        return E2_err/hbar2

    def get_error_term_state_flux_up(list_k_vec, w_list, l_op_list, P2_phi, F2_filter):
        """
        Get the flux up state error matrix manually.
        Parameters
        ----------
        1. list_k_vec : list(np.array(int)) [nhier [nmode]]
                        The list of auxiliary indexing vectors
        2. w_list : list(complex) [nmode]
                    The list of correlation function mode decay rates
        3. l_op_list : list(np.array(complex)) [nmode, [nstate, nstate]]
                       The list of L2 system-bath coupling operators organized by the
                       same indexing as list of modes
        4. P2_phi : np.array(complex) [nstate, nhier]
                    The full wavefunction
        5. F2_filter : np.array(int or bool) [nmode, nhier]
                       Determines if flux down exists for a given mode, aux pair

        Returns
        -------
        1. E2_error : np.array(complex) [nstate, nhier]
                      The error in each aux by state
        """
        E2_err = np.zeros([len(P2_phi), len(P2_phi[0])])
        for state_index in range(len(P2_phi)):
            for aux_index in range(len(P2_phi[0])):
                E_state_aux = 0
                for mode_index in range(len(l_op_list)):
                    if F2_filter[mode_index, aux_index]:
                        L2_lop = l_op_list[mode_index]
                        w_mode = list_w[mode_index]
                        k_mode = list_k_vec[aux_index][mode_index]
                        E_state_aux += np.abs(L2_lop[state_index, state_index]) ** 2 \
                                       * np.abs((1 + k_mode) * w_mode)**2
                E_state_aux *= np.abs(P2_phi[state_index, aux_index]) ** 2
                E2_err[state_index, aux_index] = E_state_aux
        return E2_err/hbar2

    def get_M2_mode_from_state(list_lop):
        """
        Gets the M2_mode_from_state matrix from a list of L-operators
        Parameters
        ----------
        1. list_lop : list(np.array(complex)), [nstate, nstate]
                      A list of L2 system-bath coupling operators

        Returns
        -------
        1. M2_mode_from_state : sparse matrix(complex)
                                The matrix connecting mode to state
        """
        list_data = []
        list_row = []
        list_col = []
        nstate = list_lop[0].shape[0]
        nmodes = len(list_lop)
        for row_ind in range(len(list_lop)):
            L2_lop = list_lop[row_ind]
            list_data += list(sparse.coo_matrix(L2_lop).data)
            list_row += len((sparse.coo_matrix(L2_lop).row))*[row_ind]
            list_col += list(sparse.coo_matrix(L2_lop).col)
        return sparse.coo_matrix((list_data, (list_row, list_col)),
                                           shape=(nmodes, nstate))
    phi = np.array([0.1j, -0.5, -0.3j, 0.2, -1.5j, 0.4])
    nstate = 2
    nhier = 3
    n_hmodes = 4
    P2_phi = phi.reshape([nhier, nstate]).T
    list_w = [50. + 0.j, 500. + 0.j, 50 + 0.j, 500. + 0.j]
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(2, 1)], 4),
                AuxiliaryVector([(2, 2)], 4)]
    list_k_vec = [aux.todense() for aux in aux_list]
    list_l_op = [np.diag([1, 0])]*2 + [np.diag([0, 1])]*2
    K2_aux_bymode = np.array(
        [[0, 0, 0], [0, 0, 0], [0, 1, 2],
         [0, 0, 0]])  # These objects are now constructed elsewhere

    # Hierarchy Adaptivity Test (No filter required) - single-entry L-operators
    M2_mode_from_state = get_M2_mode_from_state(list_l_op)
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode,
                          M2_mode_from_state, "H")
    hbar2 = hbar * hbar
    known_error = get_error_term_hier_flux_up(list_k_vec, list_w, list_l_op, P2_phi)
    print(error)
    print(known_error)
    assert np.allclose(error, known_error)

    # Hierarchy Adaptivity Test (No filter required) - multiple-entry L-operators.
    list_l_op = [np.diag([0.3, -0.7])]*2 + [np.diag([0.5, -0.5])]*2
    M2_mode_from_state = get_M2_mode_from_state(list_l_op)
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode,
                          M2_mode_from_state, "H")
    known_error = get_error_term_hier_flux_up(list_k_vec, list_w, list_l_op, P2_phi)
    assert np.allclose(error, known_error)

    # State Adaptivity Test 1 (No filter) - single-entry L-operators
    phi = np.array([0.1j, -0.5, -0.3j, 0.2, -1.5j, 0.4])
    nstate = 2
    nhier = 3
    n_hmodes = 4
    P2_phi = phi.reshape([nhier, nstate]).T
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(2, 1)], 4),
                AuxiliaryVector([(2, 2)], 4)]
    list_k_vec = [aux.todense() for aux in aux_list]
    list_l_op = [np.diag([1, 0])] * 2 + [np.diag([0, 1])] * 2
    M2_mode_from_state = get_M2_mode_from_state(list_l_op)
    F2_filter = np.ones(
        [n_hmodes, nhier])  # First we test with no filter (all connections included)
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode,
                          M2_mode_from_state, "S", F2_filter)
    known_error = get_error_term_state_flux_up(list_k_vec, list_w, list_l_op, P2_phi, F2_filter)
    assert np.allclose(error, known_error)

    # State Adaptivity Test 1 (No filter) - multiple-entry L-operators
    list_l_op = [np.diag([0.3, -0.7])] * 2 + [np.diag([0.5, -0.5])] * 2
    M2_mode_from_state = get_M2_mode_from_state(list_l_op)
    F2_filter = np.ones(
        [n_hmodes, nhier])  # First we test with no filter (all connections included)
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode,
                          M2_mode_from_state, "S", F2_filter)
    known_error = get_error_term_state_flux_up(list_k_vec, list_w, list_l_op, P2_phi, F2_filter)
    assert np.allclose(error, known_error)

    # State Adaptivity Test 2 (With Filter) - single-entry L-operators
    phi = np.array([0.1j, -0.5, -0.3j, 0.2, -1.5j, 0.4])
    nstate = 2
    nhier = 3
    n_hmodes = 4
    P2_phi = phi.reshape([nhier, nstate]).T
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(2, 1)], 4),
                AuxiliaryVector([(2, 2)], 4)]
    list_k_vec = [aux.todense() for aux in aux_list]
    list_l_op = [np.diag([1, 0])] * 2 + [np.diag([0, 1])] * 2
    M2_mode_from_state = get_M2_mode_from_state(list_l_op)
    F2_filter = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode,
                          M2_mode_from_state, "S", F2_filter)
    known_error = get_error_term_state_flux_up(list_k_vec, list_w, list_l_op, P2_phi,
                                               F2_filter)
    assert np.allclose(error, known_error)

    # State Adaptivity Test 2 (With Filter) - multiple-entry L-operators
    list_l_op = [np.diag([0.3, -0.7])] * 2 + [np.diag([0.5, -0.5])] * 2
    M2_mode_from_state = get_M2_mode_from_state(list_l_op)
    F2_filter = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode,
                          M2_mode_from_state, "S", F2_filter)
    known_error = get_error_term_state_flux_up(list_k_vec, list_w, list_l_op, P2_phi,
                                               F2_filter)
    assert np.allclose(error, known_error)

    try:
        error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode,
                          M2_mode_from_state, "T", F2_filter)
    except UnsupportedRequest as excinfo:
        assert "type T in the error_flux_up function" in str(excinfo)


def test_error_flux_down():
    """
    test for the error associated with neglecting flux from members of
    A_t to auxiliaries in A_t^C
    """
    hbar2 = hbar * hbar
    def get_error_term_hier_flux_down(g_list, w_list, l_op_list, P2_phi):
        """
        Get the flux up hierarchy error matrix manually.
        Parameters
        ----------
        1. g_list : list(complex) [nmode]
                    The list of correlation function mode prefactors
        2. w_list : list(complex) [nmode]
                    The list of correlation function mode decay rates
        3. l_op_list : list(np.array(complex)) [nmode, [nstate, nstate]]
                       The list of L2 system-bath coupling operators organized by the
                       same indexing as list of modes
        4. P2_phi : np.array(complex) [nstate, nhier]
                    The full wavefunction

        Returns
        -------
        1. E2_error : np.array(complex) [nmodes, nhier]
                      The error in each aux by mode
        """
        E2_err = np.zeros([len(l_op_list), len(P2_phi[0])])
        list_lop_exp = [np.conj(P2_phi[:,0])@L2_lop@P2_phi[:,0] for L2_lop in l_op_list]
        for mode_index in range(len(l_op_list)):
            L2_lop = l_op_list[mode_index]
            exp_L2 = list_lop_exp[mode_index]
            g_mode = list_g[mode_index]
            w_mode = list_w[mode_index]
            for aux_index in range(len(P2_phi[0])):
                E_mode_aux = 0
                for state_index in range(len(P2_phi)):
                    E_mode_aux += np.abs(P2_phi[state_index, aux_index])**2 * \
                                  np.abs(L2_lop[state_index, state_index] - exp_L2)**2
                E_mode_aux *= np.abs(g_mode/w_mode)**2
                E2_err[mode_index, aux_index] = E_mode_aux
        return E2_err/hbar2

    def get_error_term_state_flux_down(g_list, w_list, l_op_list, P2_phi, F2_filter):
        """
        Get the flux up state error matrix manually.
        Parameters
        ----------
        1. g_list : list(complex) [nmode]
                    The list of correlation function mode prefactors
        2. w_list : list(complex) [nmode]
                    The list of correlation function mode decay rates
        3. l_op_list : list(np.array(complex)) [nmode, [nstate, nstate]]
                       The list of L2 system-bath coupling operators organized by the
                       same indexing as list of modes
        4. P2_phi : np.array(complex) [nstate, nhier]
                    The full wavefunction
        5. F2_filter : np.array(int or bool) [nmode, nhier]
                       Determines if flux down exists for a given mode, aux pair

        Returns
        -------
        1. E2_error : np.array(complex) [nstate, nhier]
                      The error in each aux by state
        """
        E2_err = np.zeros([len(P2_phi), len(P2_phi[0])])
        list_lop_exp = [np.conj(P2_phi[:,0])@L2_lop@P2_phi[:,0] for L2_lop in l_op_list]
        for state_index in range(len(P2_phi)):
            for aux_index in range(len(P2_phi[0])):
                E_state_aux = 0
                for mode_index in range(len(l_op_list)):
                    if F2_filter[mode_index, aux_index]:
                        L2_lop = l_op_list[mode_index]
                        exp_L2 = list_lop_exp[mode_index]
                        g_mode = list_g[mode_index]
                        w_mode = list_w[mode_index]
                        E_state_aux +=  np.abs(L2_lop[state_index, state_index] - \
                                         exp_L2)**2 * np.abs(g_mode/w_mode)**2
                E_state_aux *= np.abs(P2_phi[state_index, aux_index])**2
                E2_err[state_index, aux_index] = E_state_aux
        return E2_err/hbar2

    def get_M2_mode_from_state(list_lop):
        """
        Gets the M2_mode_from_state matrix from a list of L-operators
        Parameters
        ----------
        1. list_lop : list(np.array(complex)), [nstate, nstate]
                      A list of L2 system-bath coupling operators

        Returns
        -------
        1. M2_mode_from_state : sparse matrix(complex)
                                The matrix connecting mode to state
        """
        list_data = []
        list_row = []
        list_col = []
        nstate = list_lop[0].shape[0]
        nmodes = len(list_lop)
        for row_ind in range(len(list_lop)):
            L2_lop = list_lop[row_ind]
            list_data += list(sparse.coo_matrix(L2_lop).data)
            list_row += len((sparse.coo_matrix(L2_lop).row))*[row_ind]
            list_col += list(sparse.coo_matrix(L2_lop).col)
        return sparse.coo_matrix((list_data, (list_row, list_col)),
                                           shape=(nmodes, nstate))



    # One-Particle Case
    phi = np.array([0.1j, -0.5, -0.3j, 0.2, -1.5j, 0.4])
    n_state = 2
    n_hier = 3
    n_hmodes = 4
    P2_phi = phi.reshape([n_hier,n_state]).T
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(1, 1)], 4),
                AuxiliaryVector([(0, 1), (1, 1)], 4)]

    # Test Hierarchy Case (No Filter)
    list_g = np.array(
        [[1000. + 1000j], [1000. + 1000j], [1000. + 1000j], [1000. + 1000j]])
    list_w = np.array([[50. + 0.j], [500. + 0.j], [50. + 0.j], [500. + 0.j]])
    list_lop = [np.array([[1, 0], [0,0]])]*2 + [np.array([[0, 0], [0,1]])]*2
    error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g, list_w,
                            get_M2_mode_from_state(list_lop), "H")
    # The following comments are intermediate calculations to help understand the known_error
    # E1_lm = [1,1,1,1]
    # D2_mode_from_state = [[0,-1],[0,-1],[-1,0],[-1,0]]
    known_error = get_error_term_hier_flux_down(list_g, list_w, list_lop, P2_phi)
    assert np.allclose(error, known_error)

    # Test State Case (Filter)
    boundary_aux = [AuxiliaryVector([(1, 2)], 4)]
    F2_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
    M2_mode_from_state = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g, list_w,
                            get_M2_mode_from_state(list_lop), "S", False, F2_filter)
    # The following comments are intermediate calculations to help understand the known_error
    # M2_state_from_mode = [[1,1,0,0],[0,0,1,1]]
    # D2_state_from_mode = [[0,0,-1,-1],[-1,-1,0,0]]
    # E2_fluxes*F2_filter = [[0,0,0],[0,8*1/hbar2,0],[0,0,0],[0,0,0]]
    # E2_error = [[0,0,0],[0,8*1/hbar2,0]]
    known_error = get_error_term_state_flux_down(list_g, list_w, list_lop, P2_phi, F2_filter)
    assert np.allclose(error, known_error)

    # Two-Particle Case
    # Hierarchy Case
    phi = np.array([0.1j, -0.5, -0.3j, 0.2, -1.5j, 0.4, 0.1, -0.5j, -0.3, 0.2j,
                    -1.5, 0.4j])
    n_state = 4
    n_hier = 3
    n_hmodes = 4
    P2_phi = phi.reshape([n_hier, n_state]).T
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(1, 1)], 4),
                AuxiliaryVector([(0, 1), (1, 1)], 4)]
    list_g = np.array(
        [[1000. + 1000j], [1000. + 1000j], [1000. + 1000j], [1000. + 1000j]])
    list_w = np.array([[50. + 0.j], [500. + 0.j], [50. + 0.j], [500. + 0.j]])
    list_lop = [np.diag([0.5, 0.5, 0, 0])]*2 + [np.diag([0, 0, 0.5, 0.5])]*2
    # M2_mode_from_state = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0,
    # 0, 1, 1]])

    error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g, list_w,
                            get_M2_mode_from_state(list_lop), "H")
    # E1_lm = [1,1,1,1]
    # D1_mode_from_state = [[-0.5, -0.5, -1, -1], [-0.5, -0.5, -1, -1], [-1, -1,
    # -0.5, -0.5], [-1, -1, -0.5, -1]]
    known_error = get_error_term_hier_flux_down(list_g, list_w, list_lop, P2_phi)
    assert np.allclose(error, known_error)

    # Test State Case (Filter)
    boundary_aux = [AuxiliaryVector([(1, 2)], 4)]
    F2_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
    error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g, list_w,
                            get_M2_mode_from_state(list_lop), "S", False, F2_filter)

    # The following comments are intermediate calculations to help understand the known_error
    # E1_lm = [2,2,2,2]
    # D2_mode_from_state = [[-1, -1, -2, -2], [-1, -1, -2, -2], [-2, -2, -1, -1],
    # [-2, -2, -1, -1]]
    # E2_fluxes*F2_filter = [[0,0,0],[0,8*1/hbar2,0],[0,0,0],[0,0,0]]
    # Error element corresponding state 0, to k = \vec{0}: no flux down from the
    # boundary
    # Error element corresponding state 1, to k = \vec{1} 1 * a sum over modes:
    # \sum_m \delta_{m,1} abs(g_0/w_0)^2 abs(L_m[s, s] - 1)^2 =
    # abs(L_1[s, s]-1)^2 abs(g1/m1)^2
    # Note this works as written because <L_m> = 0.5 + 0.5 = 1 in all cases
    known_error = get_error_term_state_flux_down(list_g, list_w, list_lop, P2_phi, F2_filter)
    assert np.allclose(error, known_error)


    # Two-Particle Case - negative values in L-operators
    # Hierarchy Case
    phi = np.array([0.1j, -0.5, -0.3j, 0.2, -1.5j, 0.4, 0.1, -0.5j, -0.3, 0.2j,
                    -1.5, 0.4j])
    n_state = 4
    n_hier = 3
    n_hmodes = 4
    P2_phi = phi.reshape([n_hier, n_state]).T
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(1, 1)], 4),
                AuxiliaryVector([(0, 1), (1, 1)], 4)]
    list_g = np.array(
        [[1000. + 1000j], [1000. + 1000j], [1000. + 1000j], [1000. + 1000j]])
    list_w = np.array([[50. + 0.j], [500. + 0.j], [50. + 0.j], [500. + 0.j]])
    list_lop = [np.diag([0.5, -0.3, 0, 0])] * 2 + [np.diag([0, 0, -0.7, 0.5])] * 2
    # M2_mode_from_state = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0,
    # 0, 1, 1]])

    error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g, list_w,
                            get_M2_mode_from_state(list_lop), "H")
    # E1_lm = [1,1,1,1]
    # D1_mode_from_state = [[-0.5, -0.5, -1, -1], [-0.5, -0.5, -1, -1], [-1, -1,
    # -0.5, -0.5], [-1, -1, -0.5, -1]]
    known_error = get_error_term_hier_flux_down(list_g, list_w, list_lop, P2_phi)
    assert np.allclose(error, known_error)

    # Test State Case (Filter)
    boundary_aux = [AuxiliaryVector([(1, 2)], 4)]
    F2_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
    error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g, list_w,
                            get_M2_mode_from_state(list_lop), "S", False, F2_filter)

    # The following comments are intermediate calculations to help understand the known_error
    # E1_lm = [2,2,2,2]
    # D2_mode_from_state = [[-1, -1, -2, -2], [-1, -1, -2, -2], [-2, -2, -1, -1],
    # [-2, -2, -1, -1]]
    # E2_fluxes*F2_filter = [[0,0,0],[0,8*1/hbar2,0],[0,0,0],[0,0,0]]
    # Error element corresponding state 0, to k = \vec{0}: no flux down from the
    # boundary
    # Error element corresponding state 1, to k = \vec{1} 1 * a sum over modes:
    # \sum_m \delta_{m,1} abs(g_0/w_0)^2 abs(L_m[s, s] - 1)^2 =
    # abs(L_1[s, s]-1)^2 abs(g1/m1)^2
    # Note this works as written because <L_m> = 0.5 + 0.5 = 1 in all cases
    known_error = get_error_term_state_flux_down(list_g, list_w, list_lop, P2_phi,
                                                 F2_filter)
    assert np.allclose(error, known_error)

    try:
        error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g, list_w,
                            get_M2_mode_from_state(list_lop), "T", False, F2_filter)
    except UnsupportedRequest as excinfo:
        assert "type T in the error_flux_down function" in str(excinfo)


def test_error_sflux_state():
    """
    test for the error associated with flux out of each state in S_0
    """
    nsite = 5
    hs = np.zeros([nsite, nsite])
    hs[1, 2] = 100
    hs[2, 1] = 100

    phi = np.ones(9)
    nstate = 3
    nhier = 3
    hamiltonian = sparse.csc_matrix(hs)
    list_index_aux_stable = [0, 1]
    list_states = [1, 2, 3]

    E1_state_flux = error_sflux_stable_state(phi, nstate, nhier, hamiltonian,
                                             list_index_aux_stable, list_states)
    known_error = [20000 / hbar ** 2, 20000 / hbar ** 2, 0]

    assert np.allclose(E1_state_flux, known_error)
    """
    another test for the error associated with flux out of each state in S_0
    """
    nsite = 5
    hs = np.zeros([nsite, nsite])
    hs[0, 2] = 2
    hs[2, 0] = 2
    hs[1, 2] = 100
    hs[2, 1] = 100
    hs[1, 3] = 13
    hs[3, 1] = 13
    hs[1, 4] = 14
    hs[4, 1] = 14
    hs[2, 4] = 24
    hs[4, 2] = 24
    hs[3, 4] = 34
    hs[4, 3] = 34
    hs[0, 4] = 4
    hs[4, 0] = 4

    phi = np.ones(9)
    nstate = 3
    nhier = 3
    hamiltonian = sparse.csc_matrix(hs)
    list_index_aux_stable = [0, 1]
    list_states = [1, 2, 3]

    E1_state_flux = error_sflux_stable_state(phi, nstate, nhier, hamiltonian,
                                             list_index_aux_stable, list_states)

    known_error = np.array(
        [2 * (100 ** 2 + 13 ** 2 + 14 ** 2), 2 * (100 ** 2 + 2 ** 2 + 24 ** 2),
         2 * (13 ** 2 + 34 ** 2)]) / hbar ** 2
    assert np.allclose(E1_state_flux, known_error)


def test_error_sflux_boundary_state():
    """
    test of the error values for the boundary states
    """
    nsite = 10
    hs = np.zeros([nsite, nsite])
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
    hs[8, 7] = 80
    hs[8, 9] = 90
    hs[9, 8] = 90

    psi_0 = np.ones(12)

    list_s0 = [3, 4, 5, 6]
    list_sc = [0, 1, 2, 7, 8, 9]
    # this choice of list_s0 and list_sc along with nearest neighbor couplings will lead to nonzero terms between
    # state 2&3 and 6&7
    n_state = len(list_s0)
    n_hier = 3
    hamiltonian = sparse.csc_matrix(hs)
    list_index_aux_stable = [0, 1]

    # if test
    list_index_state_stable = np.arange(0, 10)
    E1_sum_indices, E1_sum_error = error_sflux_boundary_state(psi_0, list_s0, list_sc,
                                                              n_state, n_hier,
                                                              hamiltonian,
                                                              list_index_state_stable,
                                                              list_index_aux_stable)
    known_error = []
    known_indices = []
    assert np.allclose(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)

    # else test
    list_index_state_stable = [0, 1, 2, 3]
    E1_sum_indices, E1_sum_error = error_sflux_boundary_state(psi_0, list_s0, list_sc,
                                                              n_state, n_hier,
                                                              hamiltonian,
                                                              list_index_state_stable,
                                                              list_index_aux_stable)
    known_indices = [2, 7]
    known_error = [(30 ** 2 / hbar ** 2 + 30 ** 2 / hbar ** 2),
                   (70 ** 2 / hbar ** 2 + 70 ** 2 / hbar ** 2)]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)

