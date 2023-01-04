# Title: Test of HopsBasis Class
# Author: Doran I. G. Bennett

import os
import numpy as np
import scipy as sp
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxiliaryVector
from mesohops.dynamics.hops_hierarchy import HopsHierarchy as HHier
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp
from mesohops.util.physical_constants import hbar


def map_to_auxvec(list_aux):
    """
        This function takes a list of auxiliaries and outputs the associated
        auxiliary-objects.

        PARAMETERS
        ----------
        1. list_aux :  list
                       list of values corresponding to the auxiliaries in a basis.
        RETURNS
        -------
        1. list_aux_vec :  list
                           list of auxiliary-objects corresponding to these auxiliaries.
        """
    list_aux_vec = []
    for aux_values in list_aux:
        aux_key = np.where(aux_values)[0]
        list_aux_vec.append(
            AuxiliaryVector([tuple([key, aux_values[key]]) for key in aux_key], 4)
        )
    return list_aux_vec

path_data = os.path.realpath(__file__)
path_data1 = path_data[: -len("test_hops_basis.py")] + "/hops_basis_noise_10site.npy"
path_data2 = path_data[: -len("test_hops_basis.py")] + "/hops_basis_noise_2site.npy"
basis_noise_10site = np.load(path_data1)
basis_noise_2site = np.load(path_data2)


def test_initialize():
    """
    Test for the hops_basis initialize function
    """
    noise_param = {
        "SEED": basis_noise_10site,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 10
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

    loperator = np.zeros([10, 10, 10], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite])
    hs[0, 1] = 40
    hs[1, 0] = 40
    hs[1, 2] = 10
    hs[2, 1] = 10
    hs[2, 3] = 40
    hs[3, 2] = 40
    hs[3, 4] = 10
    hs[4, 3] = 10
    hs[4, 5] = 40
    hs[5, 4] = 40
    hs[5, 6] = 10
    hs[6, 5] = 10
    hs[6, 7] = 40
    hs[7, 6] = 40
    hs[7, 8] = 10
    hs[8, 7] = 10
    hs[8, 9] = 40
    hs[9, 8] = 40

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Adaptive Hops
    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    # Non-adaptive Hops
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 1},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.initialize(psi_0)

    # non adaptive
    hier_param = {"MAXHIER": 4}
    sys_param = {"N_HMODES": 4}
    HH = HHier(hier_param, sys_param)
    HH.initialize(False)
    aux_list = HH.auxiliary_list
    list_hier = hops.basis.hierarchy.auxiliary_list
    assert type(list_hier) == type(aux_list)

    known_state = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert np.array_equal(hops.basis.system.state_list, known_state)

    # adaptive
    hier_param = {"MAXHIER": 4}
    sys_param = {"N_HMODES": 4}
    HH = HHier(hier_param, sys_param)
    HH.initialize(True)
    aux_list = HH.auxiliary_list
    list_hier = hops_ad.basis.hierarchy.auxiliary_list
    assert type(list_hier) == type(aux_list)

    known_state = [4, 5, 6]
    assert np.array_equal(hops_ad.basis.system.state_list, known_state)


def test_define_basis_state():
    """
    Test to check whether the correct state basis is being calculated
    """
    noise_param = {
        "SEED": basis_noise_10site,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 10
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

    loperator = np.zeros([10, 10, 10], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite])
    hs[0, 1] = 40
    hs[1, 0] = 40
    hs[1, 2] = 10
    hs[2, 1] = 10
    hs[2, 3] = 40
    hs[3, 2] = 40
    hs[3, 4] = 10
    hs[4, 3] = 10
    hs[4, 5] = 40
    hs[5, 4] = 40
    hs[5, 6] = 10
    hs[6, 5] = 10
    hs[6, 7] = 40
    hs[7, 6] = 40
    hs[7, 8] = 10
    hs[8, 7] = 10
    hs[8, 9] = 40
    hs[9, 8] = 40

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    z_step = hops_ad._prepare_zstep(hops_ad.z_mem)
    state_update, _ = hops_ad.basis.define_basis(hops_ad.phi, 2.0, z_step)

    # initial state
    state_new = state_update
    known_new = [4, 5, 6]
    assert state_new == known_new

    # state after propagation
    phi_new = 0*hops_ad.phi
    phi_new[0:hops_ad.n_state] = 1/np.sqrt(hops_ad.n_state)
    state_update, _ = hops_ad.basis.define_basis(phi_new, 2.0, z_step)
    state_new = state_update
    known_new = [3, 4, 5, 6, 7]
    assert state_new == known_new

def test_define_basis_hier():
    """
    Test to check whether the correct hierarchy basis is being calculated
     """
    noise_param = {
        "SEED": basis_noise_2site,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }

    nsite = 2
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

    loperator = np.zeros([2, 2, 2], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite], dtype=np.float64)
    hs[0, 1] = 40
    hs[1, 0] = 40

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[1] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    z_step = hops_ad._prepare_zstep(hops_ad.z_mem)
    _, hier_update = hops_ad.basis.define_basis(hops_ad.phi, 2.0, z_step)
    # initial Hierarchy
    hier_new = hier_update
    known_new = [
        AuxiliaryVector([], 4),
        AuxiliaryVector([(3, 1)], 4),
        AuxiliaryVector([(2, 1)], 4),
    ]
    assert set(hier_new) == set(known_new)

    # Hierachy after expansion
    known_phi = np.array([np.sqrt(0.1),np.sqrt(0.9),np.sqrt(0.05),np.sqrt(0.95),0,np.sqrt(1)])
    _, hier_update = hops_ad.basis.define_basis(known_phi, 2.0, z_step)
    hier_new = hier_update
    known_new = [
        AuxiliaryVector([(2, 1), (3, 1)], 4),
        AuxiliaryVector([(0, 1)], 4),
        AuxiliaryVector([(1, 1)], 4),
        AuxiliaryVector([(1, 1), (2, 1)], 4),
        AuxiliaryVector([(2, 1)], 4),
        AuxiliaryVector([(2, 2)], 4),
        AuxiliaryVector([(3, 2)], 4),
        AuxiliaryVector([], 4),
        AuxiliaryVector([(3, 1)], 4),
        AuxiliaryVector([(0, 1), (2, 1)], 4),
    ]

    assert set(hier_new) == set(known_new)

def test_update_basis():
    """
    Test to make sure the basis is getting properly updated and returning the updated
    phi and dsystem_dt
    """
    noise_param = {
        "SEED": basis_noise_10site,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 10
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

    loperator = np.zeros([10, 10, 10], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite])
    hs[0, 1] = 40
    hs[1, 0] = 40
    hs[1, 2] = 10
    hs[2, 1] = 10
    hs[2, 3] = 40
    hs[3, 2] = 40
    hs[3, 4] = 10
    hs[4, 3] = 10
    hs[4, 5] = 40
    hs[5, 4] = 40
    hs[5, 6] = 10
    hs[6, 5] = 10
    hs[6, 7] = 40
    hs[7, 6] = 40
    hs[7, 8] = 10
    hs[8, 7] = 10
    hs[8, 9] = 40
    hs[9, 8] = 40

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad1 = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad1.make_adaptive(1e-3, 1e-3)
    hops_ad1.initialize(psi_0)
    hops_ad1.propagate(2, 2)

    hops_ad2 = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad2.make_adaptive(1e-3, 1e-3)
    hops_ad2.initialize(psi_0)
    hops_ad2.propagate(2, 2)

    # state update
    state_new = [3, 4, 5, 6, 7]
    state_stable = [4, 5, 6]
    state_bound = [3, 7]
    state_update = state_new

    # hierarchy update
    hier_stable = hops_ad1.auxiliary_list[:4]
    assert AuxiliaryVector([(0, 4)], 20) not in hops_ad1.auxiliary_list
    hier_bound = [
        AuxiliaryVector([(0, 4)], 20)
    ]

    hier_new = hier_stable+hier_bound
    hier_update = hier_new

    phi, _ = hops_ad1.basis.update_basis(hops_ad1.phi, state_update, hier_update)
    assert len(phi) == hops_ad1.n_state * hops_ad1.n_hier
    P2 = hops_ad2.phi.view().reshape([hops_ad2.n_state, hops_ad2.n_hier], order="F")
    P2_new = phi.view().reshape([hops_ad1.n_state, hops_ad1.n_hier], order="F")

    states = set(hops_ad2.state_list) & set(hops_ad1.state_list)
    aux_list = set(hops_ad2.auxiliary_list) & set(hops_ad1.auxiliary_list)
    states = list(states)
    aux_list = list(aux_list)

    for state in states:
        for aux in aux_list:
            state_list_ad2 = list(hops_ad2.state_list)
            state_list_ad1 = list(hops_ad1.state_list)
            aux_ad2 = hops_ad2.auxiliary_list.index(aux)
            state_ad2 = state_list_ad2.index(state)
            aux_ad1 = hops_ad1.auxiliary_list.index(aux)
            state_ad1 = state_list_ad1.index(state)
            assert np.allclose(P2[state_ad2, aux_ad2], P2_new[state_ad1, aux_ad1])


def test_define_state_basis():
    """
    Test to make sure _define_state_basis is giving out correct stable and bound states
    """
    noise_param = {
        "SEED": basis_noise_10site,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 10
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

    loperator = np.zeros([10, 10, 10], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite])
    hs[0, 1] = 40
    hs[1, 0] = 40
    hs[1, 2] = 10
    hs[2, 1] = 10
    hs[2, 3] = 40
    hs[3, 2] = 40
    hs[3, 4] = 10
    hs[4, 3] = 10
    hs[4, 5] = 40
    hs[5, 4] = 40
    hs[5, 6] = 10
    hs[6, 5] = 10
    hs[6, 7] = 40
    hs[7, 6] = 40
    hs[7, 8] = 10
    hs[8, 7] = 10
    hs[8, 9] = 40
    hs[9, 8] = 40

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    # before propagation
    z_step = hops_ad._prepare_zstep(hops_ad.z_mem)
    list_index_aux_stable = [0, 1, 2]
    list_stable_state, list_state_bound = hops_ad.basis._define_state_basis(
        hops_ad.phi, 2.0, z_step, list_index_aux_stable, []
    )
    known_states = [4, 5, 6]
    assert np.array_equal(list_stable_state, known_states)
    assert np.array_equal(list_state_bound, [])

    # propagate
    phi_new = 0*hops_ad.phi
    phi_new[0:hops_ad.n_state] = 1/np.sqrt(hops_ad.n_state)
    list_stable_state, list_state_bound = hops_ad.basis._define_state_basis(
        phi_new, 2.0, z_step, list_index_aux_stable, []
    )
    known_states = [4, 5, 6]
    assert np.array_equal(list_stable_state, known_states)
    known_boundary = [3, 7]
    assert np.array_equal(list_state_bound, known_boundary)


def test_define_hierarchy_basis():
    """
    Test to make sure define_hierarchy_basis is giving out correct stable and bound hierarchy
    members
    """
    noise_param = {
        "SEED": basis_noise_2site,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }

    nsite = 2
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

    loperator = np.zeros([2, 2, 2], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite], dtype=np.float64)
    hs[0, 1] = 40
    hs[1, 0] = 40

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[1] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    # inital hierarchy
    z_step = hops_ad._prepare_zstep(hops_ad.z_mem)
    list_aux_stable, list_aux_boundary = hops_ad.basis._define_hierarchy_basis(
        hops_ad.phi, 2.0, z_step
    )
    known_stable = [
        AuxiliaryVector([], 4),
        AuxiliaryVector([(2, 1)], 4),
        AuxiliaryVector([(3, 1)], 4),
    ]
    assert np.array_equal(list_aux_stable, known_stable)
    known_boundary = []
    assert tuple(known_boundary) == tuple(list_aux_boundary)

    # hierarchy after propagate
    phi_test = np.array([np.sqrt(1/2), np.sqrt(1/2)]*hops_ad.n_hier)
    list_aux_stable, list_aux_boundary = hops_ad.basis._define_hierarchy_basis(
        phi_test, 2.0, z_step
    )

    assert set(list_aux_stable) == set(known_stable)

    known_boundary = [
        AuxiliaryVector([(1, 1), (2, 1)], 4), AuxiliaryVector([(3, 2)], 4),
        AuxiliaryVector([(0, 1), (3, 1)], 4),  AuxiliaryVector([(0, 1)], 4),
        AuxiliaryVector([(1, 1)], 4), AuxiliaryVector([(2, 2)], 4),
        AuxiliaryVector([(2, 1), (3, 1)], 4), AuxiliaryVector([(0, 1), (2, 1)], 4),
        AuxiliaryVector([(1, 1), (3, 1)], 4)
    ]

    assert set(list_aux_boundary) == set(known_boundary)


def test_determine_boundary_hier():
    """
    Tests the selection of boundary hierarchy members
    """
    noise_param = {
        "SEED": basis_noise_2site,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }

    nsite = 2
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

    loperator = np.zeros([2, 2, 2], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite], dtype=np.float64)
    hs[0, 1] = 40
    hs[1, 0] = 40

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[1] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    hops_ad.basis.hierarchy.auxiliary_list = [AuxiliaryVector([],4)]
    hops_ad.basis.system.state_list = [1]
    hops_ad.basis.mode.list_absindex_mode = [2,3]
    # Creating flux up and flux down matrices for initial hierarchy
    flux_down = np.zeros((2, 1))
    flux_up = np.zeros((2, 1))
    # Set the error associated with neglecting flux into the boundary via the 0th
    # mode to be larger than is acceptable, and the error for the 1st mode to be
    # acceptably small.
    flux_up[0, 0] = 0.1**2
    flux_up[1, 0] = 0.001**2
    list_e2_kflux = np.array((flux_up, flux_down))
    list_index_stable = np.array([0])
    list_aux_bound = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, 0.001
    )
    known_list_aux = [AuxiliaryVector([(2, 1)], 4)]
    assert set(list_aux_bound) == set(known_list_aux)

    # Creating flux up and flux down matrices for hierarchy after propagation with a
    # more-complicated basis
    hops_ad.basis.hierarchy.auxiliary_list = [AuxiliaryVector([],4),AuxiliaryVector([(1, 1)],4),AuxiliaryVector([(2, 1)],4),AuxiliaryVector([(3, 1)],4),
                                               AuxiliaryVector([(0, 2)],4),AuxiliaryVector([(1, 2)],4),AuxiliaryVector([(2, 1),(3, 1)],4),
                                               AuxiliaryVector([(2, 1),(3, 2)],4),AuxiliaryVector([(3, 3)],4),AuxiliaryVector([(2, 1),(3, 3)],4),AuxiliaryVector([(3, 4)],4)]
    hops_ad.basis.system.state_list = [0,1]
    hops_ad.basis.mode.list_absindex_mode = [0,1,2,3]

    flux_up = np.zeros((4, 11))
    flux_up[0, 4] = 0.00003**2
    flux_up[0, 6] = 0.01**2
    flux_up[1, 7] = 0.00004**2
    flux_up[1, 4] = 0.01**2
    flux_up[2, 4] = 0.01**2

    flux_down = np.zeros((4, 11))
    flux_down[0, 4] = 0.1**2

    list_e2_kflux = np.array((flux_up, flux_down))
    list_index_stable = np.array([6, 4, 10, 5, 7, 1, 9, 0, 3, 8, 2])
    list_index_stable.sort()
    list_aux_bound = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, 0.001
    )
    known_aux_bound = [
        AuxiliaryVector([(0, 1),(2, 1),(3, 1)], 4),
        AuxiliaryVector([(0, 2), (1, 1)], 4),
        AuxiliaryVector([(0, 2), (2, 1)], 4),
        AuxiliaryVector([(0, 1)], 4)
    ]
    assert set(list_aux_bound) == set(known_aux_bound)


def test_determine_basis_from_list():
    """
    Test to determines the members of a list that must be kept in order
    for the total error to be below the max_error value.
    """
    noise_param = {
        "SEED": basis_noise_10site,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 10
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

    loperator = np.zeros([10, 10, 10], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite])
    hs[0, 1] = 40
    hs[1, 0] = 40
    hs[1, 2] = 10
    hs[2, 1] = 10
    hs[2, 3] = 40
    hs[3, 2] = 40
    hs[3, 4] = 10
    hs[4, 3] = 10
    hs[4, 5] = 40
    hs[5, 4] = 40
    hs[5, 6] = 10
    hs[6, 5] = 10
    hs[6, 7] = 40
    hs[7, 6] = 40
    hs[7, 8] = 10
    hs[8, 7] = 10
    hs[8, 9] = 40
    hs[9, 8] = 40

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    # sorted list
    error_by_member = np.array([0.1, 0.1, 0.3, 0.2, 0.3])
    max_error = 0.2
    list_member = np.array([1, 2, 3, 4, 5])
    list_index, list_new_member = hops_ad.basis._determine_basis_from_list(
        error_by_member, max_error, list_member
    )
    known_index = [2, 3, 4]
    assert np.array_equal(list_index, known_index)
    known_members = [3, 4, 5]
    assert np.array_equal(list_new_member, known_members)

    # unsorted list
    error_by_member = np.array([0.2, 0.4, 0.1, 0.3, 0.2])
    max_error = 0.3
    list_member = np.array([1, 2, 3, 4, 5])
    list_index, list_new_member = hops_ad.basis._determine_basis_from_list(
        error_by_member, max_error, list_member
    )
    known_index = [1, 3]
    assert np.array_equal(list_index, known_index)
    known_members = [2, 4]
    assert np.array_equal(list_new_member, known_members)


def test_state_stable_error():
    """
    test of the error values for the stable states
    """
    noise_param = {
        "SEED": basis_noise_10site,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 10
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

    loperator = np.zeros([10, 10, 10], dtype=np.float64)
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator[i, i, i] = 1.0
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite])
    hs[0, 1] = 40
    hs[1, 0] = 40
    hs[1, 2] = 10
    hs[2, 1] = 10
    hs[2, 3] = 40
    hs[3, 2] = 40
    hs[3, 4] = 10
    hs[4, 3] = 10
    hs[4, 5] = 40
    hs[5, 4] = 40
    hs[5, 6] = 10
    hs[6, 5] = 10
    hs[6, 7] = 40
    hs[7, 6] = 40
    hs[7, 8] = 10
    hs[8, 7] = 10
    hs[8, 9] = 40
    hs[9, 8] = 40

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1, 1e-3)
    hops_ad.initialize(psi_0)

    z_step = hops_ad._prepare_zstep(hops_ad.z_mem)
    hops_ad.basis.hierarchy.auxiliary_list = [AuxiliaryVector([], 20),
                                              AuxiliaryVector([(10, 1), (11, 1)], 20)]
    hops_ad.phi = np.array([0, 0.9 + 0j, 0, 0, 0.8 - 0j, 0], dtype=np.complex128)
    list_index_aux_stable = [0, 1]
    list_aux_bound = [AuxiliaryVector([(10, 1)], 20), AuxiliaryVector([(11, 1)], 20)]

    hops_ad_dsystem_dt = hops_ad.basis.eom._prepare_derivative(hops_ad.basis.system,
                                                               hops_ad.basis.hierarchy,
                                                               hops_ad.basis.mode)

    # Get all error terms
    gw_10 = gw_sysbath[10]
    gw_11 = gw_sysbath[11]
    dsystem_dt = (hops_ad_dsystem_dt(hops_ad.phi, z_step[2],
                                     z_step[0], z_step[1])[0].reshape([3,2],
                     order = "F")/hbar)
    deletion = (hops_ad.phi.reshape([3,2], order = "F")/2.0)
    analytic_error_deriv_deletion = np.abs(dsystem_dt + deletion)**2
    analytic_sflux_deriv = np.array([0, (0.9**2 + 0.8**2)*(10**2 + 40**2)/hbar**2, 0])
    analytic_flux_up = np.array([[0, 0],
                                 [(0.9 ** 2) * (np.abs(gw_10[1]) ** 2 + np.abs(
                                     gw_11[1]) ** 2), 0],
                                 [0, 0]], dtype=np.complex128) / hbar ** 2
    analytic_flux_down = np.array([[0, 0],
                                   [0, (0.8 ** 2 * (1.0 - 0.9 ** 2) ** 2) * (
                                               np.abs(gw_10[0] / gw_10[1]) ** 2 +
                                               np.abs(gw_11[0] / gw_11[1]) ** 2)],
                                   [0, 0]], dtype=np.complex128) / hbar ** 2

    known_error = np.sqrt(np.sum(analytic_error_deriv_deletion,axis=1) +
                          analytic_sflux_deriv +
                          np.sum(analytic_flux_up, axis=1) +
                          np.sum(analytic_flux_down, axis=1))
    error = hops_ad.basis.state_stable_error(
        hops_ad.phi, 2.0, z_step, list_index_aux_stable, list_aux_bound
    )
    assert np.allclose(error, known_error)
