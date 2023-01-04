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
    hs[1,2] = 50
    hs[2,1] = 50

    phi = np.ones(9)
    n_state = 3
    n_hier = 3
    state_list = [1,2,3]
    hamiltonian = sparse.csc_matrix(hs)

    E2_flux_state = error_sflux_hier(phi, state_list, n_state, n_hier, hamiltonian)
    hbar2 = hbar*hbar
    known_error = [10000/hbar2, 10000/hbar2, 10000/hbar2]
    assert np.allclose(E2_flux_state, known_error)


def test_error_deriv():
    """
    test for the error associated with losing flux to a component
    """
    phi = np.ones(6,dtype=np.complex128)
    dt = 2
    n_state = 2
    n_hier = 3
    z_step = [[0,0],[0,0],[1,1,1,1]]
    list_index_aux_stable = [0,1]

    def dsystem_dt_if(phi,z1,z2,z3):
        return [np.array([1,1,1,1,0,0],dtype=np.complex128)]

    def dsystem_dt_else(phi,z1,z2,z3):
        return [np.ones(6,dtype=np.complex128)]

    # if test
    E2_del_phi = error_deriv(dsystem_dt_if, phi, z_step, n_state, n_hier,dt,list_index_aux_stable=list_index_aux_stable)
    known_deriv_error = np.array([[1/hbar, 1/hbar], [1/hbar, 1/hbar]])
    known_del_flux = phi.reshape([n_state,n_hier],order="F")[:, list_index_aux_stable] / dt
    known_error = np.abs(known_deriv_error + known_del_flux)**2
    assert np.allclose(E2_del_phi, known_error)

    # else test
    E2_del_phi = error_deriv(dsystem_dt_else, phi, z_step, n_state,n_hier,dt)
    known_deriv_error = [[1/hbar, 1/hbar, 1/hbar],[1/hbar, 1/hbar, 1/hbar]]
    known_del_flux = phi.reshape([n_state,n_hier],order="F") / dt
    known_error = np.abs(known_deriv_error + known_del_flux)**2
    assert np.allclose(E2_del_phi, known_error)


def test_error_flux_up():
    """
    the error associated with neglecting flux from members of A_t to auxiliaries in
    A_t^C that arise due to flux from lower auxiliaries to higher auxiliaries.
    """
    phi = np.ones(6)
    nstate = 2
    nhier = 3
    n_hmodes = 4
    list_w = [[50. + 0.j], [500. + 0.j], [50. + 0.j], [500. + 0.j]]
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(2, 1)], 4), AuxiliaryVector([(2, 2)], 4)]
    K2_aux_bymode = np.array(
        [[0, 0, 0], [0, 0, 0], [0, 1, 2], [0, 0, 0]])  # These objects are now constructed elsewhere
    M2_mode_from_state = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])  # These objects are now constructed elsewhere

    # Hierarchy Adaptivity Test (No filter required)
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode, M2_mode_from_state, "H")
    hbar2 = hbar * hbar
    known_error = [[50 * 50 / hbar2, 50 * 50 / hbar2, 50 * 50 / hbar2],
                   [500 * 500 / hbar2, 500 * 500 / hbar2, 500 * 500 / hbar2],
                   [50 * 50 / hbar2, 2 * 2 * 50 * 50 / hbar2, 3 * 3 * 50 * 50 / hbar2],
                   [500 * 500 / hbar2, 500 * 500 / hbar2, 500 * 500 / hbar2]]
    assert np.allclose(error, known_error)

    # State Adaptivity Test 1 (No filter)
    F2_filter = np.ones([n_hmodes, nhier])  # First we test with no filter (all connections included)
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode, M2_mode_from_state, "S", F2_filter)
    known_error = [[(50 * 50 + 500 * 500) / hbar2, (50 * 50 + 500 * 500) / hbar2, (50 * 50 + 500 * 500) / hbar2],
                   [(50 * 50 + 500 * 500) / hbar2, (2 * 2 * 50 * 50 + 500 * 500) / hbar2,
                    (3 * 3 * 50 * 50 + 500 * 500) / hbar2]]
    assert np.allclose(error, known_error)
    # State Adaptivity Test 2 (With Filter)
    phi = np.ones(6)
    nstate = 2
    nhier = 3
    list_w = [[50. + 0.j], [500. + 0.j], [50. + 0.j], [500. + 0.j]]
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(2, 1)], 4), AuxiliaryVector([(2, 2)], 4)]
    bound_aux = [AuxiliaryVector([(1, 1)], 4)]
    F2_filter = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w, K2_aux_bymode, M2_mode_from_state, "S", F2_filter)
    known_error = [[(500 * 500) / hbar2, 0, 0], [0, 0, 0]]
    assert np.allclose(error, known_error)


def test_error_flux_down():
    """
    test for the error associated with neglecting flux from members of
    A_t to auxiliaries in A_t^C
    """
    phi = np.ones(6)
    n_state = 2
    n_hier = 3
    n_hmodes = 4
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([(1, 1)], 4),
                AuxiliaryVector([(0, 1), (1, 1)], 4)]

    # Test Hierarchy Case (No Filter)
    list_g = np.array(
        [[1000. + 1000j], [1000. + 1000j], [1000. + 1000j], [1000. + 1000j]])
    list_w = np.array([[50. + 0.j], [500. + 0.j], [50. + 0.j], [500. + 0.j]])
    M2_mode_from_state = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

    error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g, list_w,
                            M2_mode_from_state, "H")
    # The following comments are intermediate calculations to help understand the known_error
    # E1_lm = [1,1,1,1]
    # D2_mode_from_state = [[0,-1],[0,-1],[-1,0],[-1,0]]
    hbar2 = hbar * hbar
    known_error = np.zeros([n_hmodes, n_hier], dtype=np.float64)
    known_error[0, :] = [800 * 1 / hbar2, 800 * 1 / hbar2, 800 * 1 / hbar2]
    known_error[1, :] = [8 * 1 / hbar2, 8 * 1 / hbar2, 8 * 1 / hbar2]
    known_error[2, :] = [800 * 1 / hbar2, 800 * 1 / hbar2, 800 * 1 / hbar2]
    known_error[3, :] = [8 * 1 / hbar2, 8 * 1 / hbar2, 8 * 1 / hbar2]
    assert np.allclose(error, known_error)

    # Test State Case (Filter)

    boundary_aux = [AuxiliaryVector([(1, 2)], 4)]
    F2_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]])
    M2_mode_from_state = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g, list_w,
                            M2_mode_from_state, "S", False, F2_filter)

    # The following comments are intermediate calculations to help understand the known_error
    # M2_state_from_mode = [[1,1,0,0],[0,0,1,1]]
    # D2_state_from_mode = [[0,0,-1,-1],[-1,-1,0,0]]
    # E2_fluxes*F2_filter = [[0,0,0],[0,8*1/hbar2,0],[0,0,0],[0,0,0]]
    # E2_error = [[0,0,0],[0,8*1/hbar2,0]]
    known_error = np.zeros([n_state, n_hier], dtype=np.float64)
    known_error[0, :] = [0, 0, 0]
    known_error[1, :] = [0, 8 * 1 / hbar2, 0]
    assert np.allclose(error, known_error)

    with pytest.raises(Exception):
        error = error_flux_down(phi, n_state, n_hier, n_hmodes, list_g,
                                list_w, "T", aux_list)


def test_error_sflux_state():
    """
    test for the error associated with flux out of each state in S_0
    """
    nsite = 5
    hs = np.zeros([nsite, nsite])
    hs[1,2] = 100
    hs[2,1] = 100

    phi = np.ones(9)
    nstate = 3
    nhier = 3
    hamiltonian = sparse.csc_matrix(hs)
    list_index_aux_stable = [0,1]
    list_states = [1,2,3]

    E1_state_flux = error_sflux_stable_state(phi, nstate, nhier, hamiltonian,
                                      list_index_aux_stable, list_states)
    known_error = [20000/hbar**2, 20000/hbar**2, 0]
    
    assert np.allclose(E1_state_flux, known_error)
    """
    another test for the error associated with flux out of each state in S_0
    """
    nsite = 5
    hs = np.zeros([nsite, nsite])
    hs[0,2] = 2
    hs[2,0] = 2
    hs[1,2] = 100
    hs[2,1] = 100
    hs[1,3] = 13
    hs[3,1] = 13
    hs[1,4] = 14
    hs[4,1] = 14
    hs[2,4] = 24
    hs[4,2] = 24
    hs[3,4] = 34
    hs[4,3] = 34
    hs[0,4] = 4
    hs[4,0] = 4

    phi = np.ones(9)
    nstate = 3
    nhier = 3
    hamiltonian = sparse.csc_matrix(hs)
    list_index_aux_stable = [0,1]
    list_states = [1,2,3]

    E1_state_flux = error_sflux_stable_state(phi, nstate, nhier, hamiltonian,
                                      list_index_aux_stable, list_states)

    known_error = np.array([2*(100**2 + 13**2 + 14**2), 2*(100**2 + 2**2 + 24**2), 2*(13**2 + 34**2)])/hbar**2
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

    list_s0 = [3,4,5,6]
    list_sc = [0,1,2,7,8,9]
    # this choice of list_s0 and list_sc along with nearest neighbor couplings will lead to nonzero terms between
    # state 2&3 and 6&7
    n_state = len(list_s0)
    n_hier = 3
    hamiltonian = sparse.csc_matrix(hs)
    list_index_aux_stable = [0, 1]

    # if test
    list_index_state_stable = np.arange(0,10)
    E1_sum_indices, E1_sum_error = error_sflux_boundary_state(psi_0,list_s0,list_sc,n_state,n_hier,hamiltonian,
                                                              list_index_state_stable,list_index_aux_stable)
    known_error = []
    known_indices = []
    assert np.allclose(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error,known_error)

    # else test
    list_index_state_stable = [0,1,2,3]
    E1_sum_indices, E1_sum_error = error_sflux_boundary_state(psi_0, list_s0, list_sc, n_state, n_hier, hamiltonian,
                                                              list_index_state_stable, list_index_aux_stable)
    known_indices = [2,7]
    known_error = [(30**2/hbar**2 + 30**2/hbar**2), (70**2/hbar**2 + 70**2/hbar**2)]
    assert np.array_equal(E1_sum_indices, known_indices)
    assert np.allclose(E1_sum_error, known_error)

