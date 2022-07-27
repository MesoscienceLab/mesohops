import pytest
import scipy as sp
from scipy import sparse
from pyhops.dynamics.basis_functions_adaptive import *
from pyhops.dynamics.hops_aux import AuxiliaryVector


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
    known_error = [100/hbar, 100/hbar, 100/hbar]
    assert np.allclose(E2_flux_state, known_error)


def test_error_deriv():
    """
    test for the error associated with losing flux to a component
    """
    phi = np.ones(6)
    n_state = 2
    n_hier = 3
    z_step = [[0,0],[0,0],[1,1,1,1]]
    list_index_aux_stable = [0,1]

    def dsystem_dt_if(phi,z1,z2,z3):
        return [np.array([1,1,1,1,0,0])]

    def dsystem_dt_else(phi,z1,z2,z3):
        return [np.ones((2,3))]

    # if test
    E2_del_phi = error_deriv(dsystem_dt_if, phi, z_step, n_state, n_hier,list_index_aux_stable=list_index_aux_stable)
    known_error = [[1 / hbar, 1 / hbar, 0], [1 / hbar, 1 / hbar, 0]]
    assert np.allclose(E2_del_phi, known_error)

    # else test
    E2_del_phi = error_deriv(dsystem_dt_else, phi, z_step, n_state,n_hier)
    known_error = [[1/hbar, 1/hbar, 1/hbar],[1/hbar, 1/hbar, 1/hbar]]
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
    list_w = [[50.+0.j], [500.+0.j],[50.+0.j], [500.+0.j]]
    list_state_indices_by_hmode = np.array([[0],[0],[1],[1]])
    list_absindex_mode = [0,1,2,3]
    aux_list = [AuxiliaryVector([(2,1)],4), AuxiliaryVector([(2,2)],4)]
    kmax = 4
    static_filter = []

    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w,list_state_indices_by_hmode,
                          list_absindex_mode, aux_list, kmax, static_filter)

    known_error = [[50/hbar, 50/hbar, 50/hbar], [500/hbar, 500/hbar, 500/hbar],
                   [150/hbar, 150/hbar, 150/hbar], [500/hbar, 500/hbar, 500/hbar]]
    assert np.allclose(error, known_error)

    #static filter test with Markovian mode
    static_filter = [['Markovian',[False, True, False, True]]]
    error = error_flux_up(phi, nstate, nhier, n_hmodes, list_w,
                          list_state_indices_by_hmode,
                          list_absindex_mode, aux_list, kmax, static_filter)
    known_error = [[50 / hbar, 50 / hbar, 50 / hbar],
                   [500 / hbar, 0, 0],
                   [150 / hbar, 150 / hbar, 150 / hbar],
                   [500 / hbar, 0, 0]]
    assert np.allclose(error, known_error)


def test_error_flux_down():
    """
    test for the error associated with neglecting flux from members of
    A_t to auxiliaries in A_t^C
    """
    phi = np.ones(6)
    nstate = 2
    nhier = 3
    n_hmodes = 2
    list_state_indices_by_hmode = np.array([[0], [0]])
    list_absindex_mode = np.array([0,1])
    aux_list = [AuxiliaryVector([],2),AuxiliaryVector([(1, 1)], 2), AuxiliaryVector([(0,1),(1, 1)], 2)]

    # Need to set aux._index
    for (index, aux) in enumerate(aux_list):
        aux._index = index
    
    aux00 = aux_list[0]
    aux01 = aux_list[1]
    aux01.add_aux_connect(1,AuxiliaryVector([],2),-1)
    aux11 = aux_list[2]
    aux11.add_aux_connect(0,AuxiliaryVector([(1, 1)], 2),-1)

    #Aux 0 has no connections below, Aux 1's only connection is Aux 0, which is in basis.
    #Aux 2 has two connections below.  Its connection via mode 0 is already in basis.  Its connection via mode 1 is not in basis, therefore it will be counted here
    #Therefore, the filter should be [[0,0,0],[0,0,1]]

    known_filter = np.array([[0,0,0],[0,0,1]])
    list_g = np.array([[1000. + 1000j], [1000. + 1000j]])
    list_w = [[50. + 0.j], [500. + 0.j]]

    error = error_flux_down(phi,nstate, nhier,n_hmodes,list_state_indices_by_hmode,
                            list_absindex_mode, aux_list, list_g, list_w, "H")
    g_w = np.abs(list_g/list_w)
    known_P2_pop_modes_down_1 = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0]])
    known_P1_aux_norm = np.array(np.sqrt(2),np.sqrt(2))
    known_P2_pop_modes = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0]])
    
    known_error = known_filter * g_w * (known_P1_aux_norm * known_P2_pop_modes_down_1 + known_P2_pop_modes) / hbar
    assert np.allclose(error, known_error)
    
    error = error_flux_down(phi, nstate, nhier, n_hmodes, list_state_indices_by_hmode,
                            list_absindex_mode, aux_list, list_g, list_w, "S")
    known_E2_lm = np.tile(
                    np.sum(
                    known_filter * g_w * known_P2_pop_modes_down_1,
                    axis=0,
                    ),
                [nstate, 1],
                )
    known_M2_state_from_mode = np.array([[1,1],[0,0]])
    known_P2_pop_site = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0]])
    known_error = known_M2_state_from_mode @ (known_filter * g_w * known_P2_pop_modes) / hbar
    known_error += known_E2_lm * known_P2_pop_site / hbar
    assert np.allclose(error, known_error)

    with pytest.raises(Exception):
        error = error_flux_down(phi, nstate, nhier, n_hmodes,list_state_indices_by_hmode,
                                list_absindex_mode, aux_list, list_g, list_w, "T")


def test_error_deletion():
    """
    test for the error induced by removing components of Phi
    """
    phi = np.zeros(6)
    phi[1] = 1
    delta_t = 2.0
    nstate = 2
    nhier = 3

    E2_site_aux = error_deletion(phi, delta_t, nstate, nhier)
    known_error = [[0, 0, 0], [0.5, 0, 0]]
    assert np.allclose(E2_site_aux, known_error)


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

    E1_state_flux = error_sflux_state(phi, nstate, nhier, hamiltonian,
                                      list_index_aux_stable, list_states)
    known_error = [0, 0, 0]
    
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

    E1_state_flux = error_sflux_state(phi, nstate, nhier, hamiltonian,
                                      list_index_aux_stable, list_states)
    known_error = np.sqrt([2*(14*14),2*(2*2+24*24), 2*(34*34)])/hbar

    assert np.allclose(E1_state_flux, known_error)
    
