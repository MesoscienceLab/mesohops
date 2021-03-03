# Title: Test of HopsBasis Class
# Author: Doran I. G. Bennett

import numpy as np
import scipy as sp
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxiliaryVector
from mesohops.dynamics.hops_hierarchy import HopsHierarchy as HHier
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp


def map_to_auxvec(list_aux):
    list_aux_vec = []
    for aux_values in list_aux:
        aux_key = np.where(aux_values)[0]
        list_aux_vec.append(
            AuxiliaryVector([tuple([key, aux_values[key]]) for key in aux_key], 4)
        )
    return list_aux_vec


def test_initialize():
    """
    Test for the hops_basis initialize function
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    z_step = np.zeros(10)
    state_update, _ = hops_ad.basis.define_basis(hops_ad.phi, 2.0, z_step)

    # initial state
    state_new, state_stable, state_bound = state_update
    known_new = [4, 5, 6]
    assert state_new == known_new
    known_stable = [4, 5, 6]
    assert tuple(state_stable) == tuple(known_stable)
    known_bound = []
    assert np.array_equal(state_bound, known_bound)

    # state after propagation
    hops_ad.propagate(10, 2)
    state_update, _ = hops_ad.basis.define_basis(hops_ad.phi, 2.0, z_step)
    state_new, state_stable, state_bound = state_update
    known_new = [3, 4, 5, 6, 7]
    assert state_new == known_new
    known_stable = [4, 5, 6]
    assert tuple(state_stable) == tuple(known_stable)
    known_bound = [3, 7]
    assert np.array_equal(state_bound, known_bound)


def test_define_basis_hier():
    """
    Test to check whether the correct hierarchy basis is being calculated
     """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    z_step = np.zeros(10)
    _, hier_update = hops_ad.basis.define_basis(hops_ad.phi, 2.0, z_step)
    # initial Hierarchy
    hier_new, hier_stable, hier_bound = hier_update
    known_new = [
        AuxiliaryVector([], 4),
        AuxiliaryVector([(3, 1)], 4),
        AuxiliaryVector([(2, 1)], 4),
    ]
    assert hier_new == known_new
    known_stable = [
        AuxiliaryVector([], 4),
        AuxiliaryVector([(2, 1)], 4),
        AuxiliaryVector([(3, 1)], 4),
    ]
    assert tuple(hier_stable) == tuple(known_stable)
    known_bound = []
    assert np.array_equal(hier_bound, known_bound)

    # Hierachy after propagation
    hops_ad.propagate(12.0, 2.0)
    _, hier_update = hops_ad.basis.define_basis(hops_ad.phi, 2.0, z_step)
    hier_new, hier_stable, hier_bound = hier_update
    known_new = [
        AuxiliaryVector([(2, 1), (3, 1)], 4),
        AuxiliaryVector([(1, 2)], 4),
        AuxiliaryVector([(0, 1)], 4),
        AuxiliaryVector([(2, 1), (3, 3)], 4),
        AuxiliaryVector([(3, 4)], 4),
        AuxiliaryVector([(1, 1)], 4),
        AuxiliaryVector([(2, 2), (3, 2)], 4),
        AuxiliaryVector([(1, 1), (3, 1)], 4),
        AuxiliaryVector([(3, 3)], 4),
        AuxiliaryVector([(1, 1), (2, 1)], 4),
        AuxiliaryVector([(2, 1), (3, 2)], 4),
        AuxiliaryVector([(1, 4)], 4),
        AuxiliaryVector([(2, 1)], 4),
        AuxiliaryVector([(1, 2), (3, 2)], 4),
        AuxiliaryVector([(2, 2)], 4),
        AuxiliaryVector([(3, 2)], 4),
        AuxiliaryVector([(2, 2), (3, 1)], 4),
        AuxiliaryVector([(1, 3)], 4),
        AuxiliaryVector([(1, 3), (3, 1)], 4),
        AuxiliaryVector([(1, 2), (3, 1)], 4),
        AuxiliaryVector([], 4),
        AuxiliaryVector([(1, 1), (3, 2)], 4),
        AuxiliaryVector([(3, 1)], 4),
        AuxiliaryVector([(0, 1), (1, 1)], 4),
        AuxiliaryVector([(1, 1), (3, 3)], 4),
    ]
    assert hier_new == known_new
    known_stable = [
        AuxiliaryVector([], 4),
        AuxiliaryVector([(0, 1)], 4),
        AuxiliaryVector([(1, 1)], 4),
        AuxiliaryVector([(2, 1)], 4),
        AuxiliaryVector([(3, 1)], 4),
        AuxiliaryVector([(0, 1), (1, 1)], 4),
        AuxiliaryVector([(1, 2)], 4),
        AuxiliaryVector([(1, 1), (2, 1)], 4),
        AuxiliaryVector([(1, 1), (3, 1)], 4),
        AuxiliaryVector([(2, 2)], 4),
        AuxiliaryVector([(2, 1), (3, 1)], 4),
        AuxiliaryVector([(3, 2)], 4),
        AuxiliaryVector([(1, 3)], 4),
        AuxiliaryVector([(1, 2), (3, 1)], 4),
        AuxiliaryVector([(1, 1), (3, 2)], 4),
        AuxiliaryVector([(2, 2), (3, 1)], 4),
        AuxiliaryVector([(2, 1), (3, 2)], 4),
        AuxiliaryVector([(3, 3)], 4),
        AuxiliaryVector([(1, 4)], 4),
        AuxiliaryVector([(1, 3), (3, 1)], 4),
        AuxiliaryVector([(2, 2), (3, 2)], 4),
        AuxiliaryVector([(2, 1), (3, 3)], 4),
        AuxiliaryVector([(3, 4)], 4),
    ]
    assert hier_stable == known_stable
    known_bound = [
        AuxiliaryVector([(1, 1), (3, 3)], 4),
        AuxiliaryVector([(1, 2), (3, 2)], 4),
    ]
    assert hier_bound == known_bound


def test_update_basis():
    """
    Test to make sure the basis is getting properly updated and returning the updated
    phi and dsystem_dt
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[5] = 1.0
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
    hops_ad.propagate(2, 2)

    # state update
    state_new = [3, 4, 5, 6, 7]
    state_stable = [4, 5, 6]
    state_bound = [3, 7]
    state_update = (state_new, state_stable, state_bound)

    # hierarchy update
    hier_stable = [
        AuxiliaryVector([], 20),
        AuxiliaryVector([(9, 1)], 20),
        AuxiliaryVector([(10, 1)], 20),
        AuxiliaryVector([(11, 1)], 20),
        AuxiliaryVector([(10, 1), (11, 1)], 20),
        AuxiliaryVector([(11, 2)], 20),
        AuxiliaryVector([(10, 1), (11, 2)], 20),
        AuxiliaryVector([(11, 3)], 20),
        AuxiliaryVector([(11, 4)], 20),
    ]
    hier_bound = [
        AuxiliaryVector([(5, 1)], 20),
        AuxiliaryVector([(13, 1)], 20),
        AuxiliaryVector([(14, 1)], 20),
    ]
    
    hier_new = hier_stable+hier_bound
    hier_update = (hier_new, hier_stable, hier_bound)

    phi, _ = hops_ad.basis.update_basis(hops_ad.phi, state_update, hier_update)
    P2 = hops_ad.phi.view().reshape([3, 9], order="F")
    P2_new = phi.view().reshape([hops_ad.n_state, hops_ad.n_hier], order="F")
    assert np.shape(P2) == np.shape(np.zeros((3, 9)))
    assert np.shape(P2_new) == np.shape(np.zeros((5, 12)))
    assert P2[1, 5] == P2_new[2, 8]
    assert P2[1, 1] == P2_new[2, 2]
    assert P2[1, 7] == P2_new[2, 10]
    assert P2[2, 3] == P2_new[3, 4]


def test_check_state_list():
    """
    Test to make sure check_state_list is giving out correct stable and bound states
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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
    z_step = np.zeros(10)
    list_index_aux_stable = [0, 1, 2]
    list_stable_state, list_state_bound = hops_ad.basis._check_state_list(
        hops_ad.phi, 2.0, z_step, list_index_aux_stable
    )
    known_states = [4, 5, 6]
    assert np.array_equal(list_stable_state, known_states)
    assert np.array_equal(list_state_bound, [])

    # propagate
    hops_ad.propagate(10.0, 2.0)
    list_stable_state, list_state_bound = hops_ad.basis._check_state_list(
        hops_ad.phi, 2.0, z_step, list_index_aux_stable
    )
    known_states = [4, 5, 6]
    assert np.array_equal(list_stable_state, known_states)
    known_boundary = [3, 7]
    assert np.array_equal(list_state_bound, known_boundary)


def test_check_hierarchy_list():
    """
    Test to make sure check_state_list is giving out correct stable and bound hierarchy
    members
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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
    z_step = [0, 0]
    list_aux_stable, list_aux_boundary = hops_ad.basis._check_hierarchy_list(
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
    hops_ad.propagate(12.0, 2.0)
    list_aux_stable, list_aux_boundary = hops_ad.basis._check_hierarchy_list(
        hops_ad.phi, 2.0, z_step
    )
    known_stable = [
        AuxiliaryVector([], 4),
        AuxiliaryVector([(0, 1)], 4),
        AuxiliaryVector([(1, 1)], 4),
        AuxiliaryVector([(2, 1)], 4),
        AuxiliaryVector([(3, 1)], 4),
        AuxiliaryVector([(0, 1), (1, 1)], 4),
        AuxiliaryVector([(1, 2)], 4),
        AuxiliaryVector([(1, 1), (2, 1)], 4),
        AuxiliaryVector([(1, 1), (3, 1)], 4),
        AuxiliaryVector([(2, 2)], 4),
        AuxiliaryVector([(2, 1), (3, 1)], 4),
        AuxiliaryVector([(3, 2)], 4),
        AuxiliaryVector([(1, 3)], 4),
        AuxiliaryVector([(1, 2), (3, 1)], 4),
        AuxiliaryVector([(1, 1), (3, 2)], 4),
        AuxiliaryVector([(2, 2), (3, 1)], 4),
        AuxiliaryVector([(2, 1), (3, 2)], 4),
        AuxiliaryVector([(3, 3)], 4),
        AuxiliaryVector([(1, 4)], 4),
        AuxiliaryVector([(1, 3), (3, 1)], 4),
        AuxiliaryVector([(2, 2), (3, 2)], 4),
        AuxiliaryVector([(2, 1), (3, 3)], 4),
        AuxiliaryVector([(3, 4)], 4),
    ]
    assert list_aux_stable == known_stable
    known_boundary = [
        AuxiliaryVector([(1, 1), (3, 3)], 4),
        AuxiliaryVector([(1, 2), (3, 2)], 4),
    ]
    assert list_aux_boundary == known_boundary


def test_determine_boundary_hier():
    """
    Test to make sure correct boundary hierarchy members are accounted for
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    # Creating flux up and flux down matrices for initial hierarchy
    # flux down
    flux_down = np.zeros((4, 3))
    # flux up
    flux_up = np.zeros((4, 3))
    flux_up[2, 0] = 0.1
    flux_up[3, 0] = 0.001
    list_e2_kflux = np.array((flux_up, flux_down))
    list_index_stable = [0, 1, 2]
    list_aux_up, list_aux_down = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, 0.001
    )
    known_aux_up = [AuxiliaryVector([(2, 1)], 4)]
    assert list_aux_up == known_aux_up
    assert list_aux_down == []

    # Creating flux up and flux down matrices for hierarchy after propagation
    hops_ad.propagate(4.0, 2.0)
    # flux up
    flux_up = np.zeros((4, 12))
    flux_up[0, 0] = 0.01
    flux_up[0, 6] = 0.01
    flux_up[1, 7] = 0.00004
    flux_up[2, 3] = 0.1
    flux_up[2, 10] = 0.00007
    flux_up[3, 9] = 0.0011
    # flux down
    flux_down = np.zeros((4, 12))
    flux_down[1, 1] = 0.1
    flux_down[2, 2] = 0.1
    flux_down[2, 10] = 0.1
    flux_down[2, 6] = 0.00007
    list_e2_kflux = np.array((flux_up, flux_down))
    list_index_stable = [6, 4, 10, 11, 5, 7, 1, 9, 0, 3, 8, 2]
    list_index_stable.sort()
    list_aux_up, list_aux_down = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, 0.001
    )
    known_aux_up = [
        AuxiliaryVector([(0, 1)], 4),
        AuxiliaryVector([(0, 1), (2, 1), (3, 1)], 4),
        AuxiliaryVector([(2, 1), (3, 1)], 4),
        AuxiliaryVector([(3, 4)], 4),
    ]
    assert list_aux_up == known_aux_up
    assert list_aux_down == [
        AuxiliaryVector([], 4),
        AuxiliaryVector([], 4),
        AuxiliaryVector([(3, 3)], 4),
    ]


def test_determine_basis_from_list():
    """
    Test to determines the members of a list that must be kept in order
    for the total error to be below the max_error value.
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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


def test_determine_error_thresh():
    """
    test to determine correct error value is set as the error threshold
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    # if test
    sorted_error = np.array([0, 1, 2, 3, 5])
    max_error = 3.0
    error = hops_ad.basis._determine_error_thresh(sorted_error, max_error)
    assert error == 2

    # else test
    sorted_error = np.array([0, 1, 1, 3, 5])
    max_error = 20.0
    error = hops_ad.basis._determine_error_thresh(sorted_error, max_error)
    assert error == 0.0


def test_error_boundary_states():
    """
    test of the error values for the boundary states
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    # initial boundary states
    list_index_stable = [0, 1, 2]
    list_index_aux_stable = [0, 1]
    list_index_nonzero, list_error_nonzero = hops_ad.basis.error_boundary_state(
        hops_ad.phi, list_index_stable, list_index_aux_stable
    )
    assert np.array_equal(list_index_nonzero, [])
    assert np.array_equal(list_error_nonzero, [])

    # boundary states after propagation
    hops_ad.propagate(10.0, 2.0)
    list_index_nonzero, list_error_nonzero = hops_ad.basis.error_boundary_state(
        hops_ad.phi, list_index_stable, list_index_aux_stable
    )
    known_bound = [3, 7]
    assert np.array_equal(list_index_nonzero, known_bound)
    known_error = [0.00014827, 0.00014875]
    assert np.allclose(list_error_nonzero, known_error)


def test_error_stable_state():
    """
    test of the error values for the stable states
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    z_step = np.ones(10)
    list_index_aux_stable = [0, 1, 2]
    error = hops_ad.basis.error_stable_state(
        hops_ad.phi, 2.0, z_step, list_index_aux_stable
    )
    known_error = [0.00753443 + 0.0j, 0.51766558 + 0.0j, 0.00188361 + 0.0j]
    assert np.allclose(error, known_error)


def test_hier_stable_error():
    """
    test of the error values for the stable hierarchy members
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    z_step = [0, 0]
    error, flux_error = hops_ad.basis.hier_stable_error(hops_ad.phi, 2.0, z_step)
    known_error = [0.50893557, 0.00941804, 0.09418041]
    assert np.allclose(error, known_error)


def test_error_sflux_state():
    """
    test for the error associated with flux out of each state in S_0
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    list_index_aux_stable = [0, 1]
    list_states = np.arange(10)
    E1_state_flux = hops_ad.basis.error_sflux_state(
        hops_ad.phi, list_index_aux_stable, list_states
    )
    known_error = [0.0 + 0.0j, 0.00753443 + 0.0j, 0.0 + 0.0j]
    assert np.allclose(E1_state_flux, known_error)


def test_error_sflux_hier():
    """
    test for the error introduced by losing flux within k from S_0 to S_0^C
    for each k in H_0
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    E2_flux_state = hops_ad.basis.error_sflux_hier(hops_ad.phi, [4, 5, 6])
    known_error = [0, 0, 0]
    assert np.allclose(E2_flux_state, known_error)

    hops_ad.propagate(10, 2)
    E2_flux_state = hops_ad.basis.error_sflux_hier(hops_ad.phi, [4, 5, 6])
    known_error = [2.10004255e-04,
                   2.50146052e-06,
                   5.28986969e-05,
                   9.14077348e-06,
                   5.32245330e-05,
                   5.17421294e-05,
                   2.33186603e-05,
                   8.56003704e-06,
                   3.53943880e-07,
                   3.25690823e-06,
                   1.98318688e-05,
                   1.95396184e-05]
    assert np.allclose(E2_flux_state, known_error)


def test_error_deriv():
    """
    test for the error associated with losing flux to a component
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    E2_del_phi = hops_ad.basis.error_deriv(hops_ad.phi, [0, 0])
    known_error = [[0.00753443, 0.0, 0.0], [0.0, 0.00941804, 0.09418041]]
    assert np.allclose(E2_del_phi, known_error)


def test_error_deletion():
    """
    test for the error induced by removing components of Phi
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[1] = 1.0
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

    E2_site_aux = hops_ad.basis.error_deletion(hops_ad.phi, 2.0)
    known_error = [[0, 0, 0], [0.5, 0, 0]]
    assert np.allclose(E2_site_aux, known_error)


def test_error_flux_down():
    """
    test for the error induced by neglecting flux from H_0 (or H_S)
    to auxiliaries with higher summed index in H_0^C.
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    error = hops_ad.basis.error_flux_down(hops_ad.phi, "H")
    known_error = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert np.allclose(error, known_error)

    hops_ad.propagate(4, 2)
    known_error = [
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            0.00000000e00,
            1.77300996e-06,
            0.00000000e00,
            0.00000000e00,
            3.42275816e-07,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            1.11871052e-03,
            0.00000000e00,
            0.00000000e00,
            3.12152420e-05,
            3.51225702e-04,
            0.00000000e00,
            1.09725980e-04,
            0.00000000e00,
            3.31052922e-05,
            0.00000000e00,
        ],
        [
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            2.36349134e-04,
            0.00000000e00,
            0.00000000e00,
            8.74004868e-06,
            7.42495639e-05,
            2.73046762e-06,
            2.30899011e-05,
            8.23806071e-07,
            7.44294797e-06,
        ],
    ]
    error = hops_ad.basis.error_flux_down(hops_ad.phi, "H")
    print(error)
    assert np.allclose(error, known_error)


def test_error_flux_up():
    """
    test for the error induced by neglecting flux from H_0 (or H_S)
    to auxiliaries with lower summed index in H_0^C.
    """
    noise_param = {
        "SEED": 0,
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
        "INCHWORM": False,
        "INCHWORM_MIN": 5,
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

    error = hops_ad.basis.error_flux_up(hops_ad.phi)
    known_error = [[0, 0, 0], [0, 0, 0], [0.00941804, 0, 0], [0.09418041, 0, 0]]
    assert np.allclose(error, known_error)
