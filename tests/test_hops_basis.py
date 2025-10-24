import os

import numpy as np
import pytest
import scipy as sp

from mesohops.basis.hops_aux import AuxiliaryVector as AuxiliaryVector
from mesohops.basis.hops_hierarchy import HopsHierarchy as HHier
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.physical_constants import hbar

__title__ = "Test of HopsBasis class"
__author__ = "D. I. G. B. Raccah, J. K. Lynd"
__version__ = "1.4"

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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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

    with pytest.raises(UnsupportedRequest, match="does not support non-normalized Φ in "
                                                 "the _define_state"):
        hops_ad.basis._define_state_basis(phi_new/5, 2.0, z_step,
                                          list_index_aux_stable, [])

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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
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
    assert tuple(list_aux_stable) == tuple(known_stable)
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

    with pytest.raises(UnsupportedRequest, match="does not support non-normalized Φ in "
                                                 "the _define_hierarchy"):
        list_aux_stable, list_aux_boundary = hops_ad.basis._define_hierarchy_basis(
            phi_test/5, 2.0, z_step
        )


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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[1] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 6},
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
    
    error_bound = 0.0015 ** 2
    f_discard = 0.01
    
    list_aux_bound = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, error_bound, f_discard
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

    error_bound = 0.001 ** 2
    f_discard = 0.01

    list_e2_kflux = np.array((flux_up, flux_down))
    list_index_stable = np.array([6, 4, 10, 5, 7, 1, 9, 0, 3, 8, 2])
    list_index_stable.sort()
    list_aux_bound = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, error_bound, f_discard
    )
    known_aux_bound = [
        AuxiliaryVector([(0, 1),(2, 1),(3, 1)], 4),
        AuxiliaryVector([(0, 2), (1, 1)], 4),
        AuxiliaryVector([(0, 2), (2, 1)], 4),
        AuxiliaryVector([(0, 1)], 4)
    ]
    assert set(list_aux_bound) == set(known_aux_bound)
    
    hops_ad.basis.hierarchy.auxiliary_list = [hops_ad.basis.hierarchy.auxiliary_list[0]]
    # Creating flux up and flux down matrices for hierarchy after propagation with a
    # more-complicated basis involving multiple fluxes into one boundary
    hops_ad.basis.hierarchy.auxiliary_list = [hops_ad.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(1, 1)],4),AuxiliaryVector([(2, 1)],4),AuxiliaryVector([(3, 1)],4),
                                               AuxiliaryVector([(0, 2)],4),AuxiliaryVector([(1, 2)],4),AuxiliaryVector([(2, 1),(3, 1)],4),
                                               AuxiliaryVector([(1,1),(2,2)],4),AuxiliaryVector([(1,1),(2,1),(3,1)],4),AuxiliaryVector([(2,2),(3,1)],4),
                                               AuxiliaryVector([(1,2),(2,1),(3,1)],4),AuxiliaryVector([(1,2),(3,2)],4),
                                               AuxiliaryVector([(1,1),(2,1),(3,2)],4),AuxiliaryVector([(1,3),(2,1),(3,1)],4),AuxiliaryVector([(1,2),(2,2),(3,1)],4),
                                               AuxiliaryVector([(1,2),(2,1),(3,2)],4)]
    hops_ad.basis.system.state_list = [0,1]
    hops_ad.basis.mode.list_absindex_mode = [0,1,2,3]

    mainaux = 0
    aux_1 = 1
    aux_2 = 2
    aux_3 = 3
    aux_00 = 4
    aux_11 = 5
    aux_23 = 6
    aux_122 = 7
    aux_123 = 8
    aux_223 = 9
    aux_1123 = 10
    aux_1133 = 11
    aux_1233 = 12
    aux_11123 = 13
    aux_11223 = 14
    aux_11233 = 15

    # The comments represent the boundary auxiliary receiving the flux. The
    # parenthetical numbers are the total flux that boundary auxiliary receives (all
    # individual flux terms are 1).
    flux_up = np.zeros((4, 16))
    flux_up[0, mainaux] = 1.0 #aux_0 (2)
    flux_up[0, aux_1] = 1.0 #aux_01 (1)
    flux_up[0, aux_2] = 1.0 #aux_02 (1)
    flux_up[0, aux_3] = 1.0 #aux_03 (1)
    flux_up[0, aux_00] = 1.0 #aux_000 (1)
    flux_up[0, aux_11] = 1.0 #aux_011 (1)
    flux_up[0, aux_23] = 1.0 #aux_023 (1)
    flux_up[0, aux_122] = 1.0 #aux_0112 (1)
    flux_up[0, aux_123] = 1.0 #aux_0123 (1)
    flux_up[0, aux_223] = 1.0 #aux_0223 (1)
    flux_up[0, aux_1123] = 1.0 #aux_01123 (1)
    flux_up[0, aux_1133] = 1.0 #aux_01133 (1)
    flux_up[0, aux_1233] = 1.0 #aux_01233 (1)
    
    flux_up[1, aux_2] = 1.0 #aux_12 (4)
    flux_up[1, aux_3] = 1.0 #aux_13 (3)
    flux_up[1, aux_00] = 1.0 #aux_001 (1)
    flux_up[1, aux_11] = 1.0 #aux_111 (1)
    flux_up[1, aux_122] = 1.0 #aux_1122 (2)
    flux_up[1, aux_223] = 1.0 #aux_1223  (4)
    flux_up[1, aux_1133] = 1.0 #aux_11133 (1)
    
    flux_up[2, aux_1] = 1.0 #aux_12 (4)
    flux_up[2, aux_2] = 1.0 #aux_22 (3)
    flux_up[2, aux_00] = 1.0 #aux_002 (1)
    flux_up[2, aux_11] = 1.0 #aux_112 (2)
    flux_up[2, aux_122] = 1.0 #aux_1222 (1)
    flux_up[2, aux_123] = 1.0 #aux_1223 (4)
    flux_up[2, aux_223] = 1.0 #aux_2223 (1)
    flux_up[2, aux_1233] = 1.0 #aux_12233 (1)
    
    
    flux_up[3, aux_1] = 1.0 #aux_13 (3)
    flux_up[3, aux_3] = 1.0 #aux_33 (1)
    flux_up[3, aux_00] = 1.0 #aux_003 (1)
    flux_up[3, aux_11] = 1.0 #aux_113 (3)
    flux_up[3, aux_23] = 1.0 #aux_233 (2)
    flux_up[3, aux_122] = 1.0 #aux_1223 (4)
    flux_up[3, aux_223] = 1.0 #aux_2233 (1)
    flux_up[3, aux_1133] = 1.0 #aux_11333 (1)
    flux_up[3, aux_1233] = 1.0 #aux_12333 (1)
    
    flux_down = np.zeros((4, 16))
    flux_down[0, aux_00] = 1.0 #aux_0 (2)
    flux_down[1, aux_122] = 1.0 #aux_22 (3)
    flux_down[1, aux_1133] = 1.0 #aux_133 (2)
    flux_down[1, aux_1233] = 1.0 #aux_233 (2)
    flux_down[1, aux_11223] = 1.0 #aux_1223 (4)
    flux_down[2, aux_122] = 1.0 #aux_12 (4)
    flux_down[2, aux_123] = 1.0 #aux_13 (3)
    flux_down[2, aux_1123] = 1.0 #aux_113 (3)
    flux_down[2, aux_1233] = 1.0 #aux_133 (2)
    flux_down[2, aux_11123] = 1.0 #aux_1113 (1)
    flux_down[3, aux_123] = 1.0 #aux_12 (4)
    flux_down[3, aux_223] = 1.0 #aux_22 (3)
    flux_down[3, aux_1123] = 1.0 #aux_112 (2)
    flux_down[3, aux_1133] = 1.0 #aux_113 (3)
    flux_down[3, aux_11123] = 1.0 #aux_1112 (1)
    flux_down[3, aux_11223] = 1.0 #aux_1122 (2)
    
    #Boundary Auxes with one flux:  
    
    #aux_01, aux_02, aux_03, aux_000, aux_011
    #aux_023, aux_0112, aux_0123, aux_0223, aux_01123
    #aux_01133, aux_01233, aux_001, aux_111, aux_12233
    #aux_11133, aux_002, aux_1222, aux_2223, aux_33,
    #aux_003, aux_2233, aux_11333, aux_12333, aux_1113,
    #aux_1112
    
    num_1flux = 26
    #Boundary Auxes with two fluxes:
    #aux_0, aux_1122, aux_112, aux_233, aux_133
    
    num_2flux = 5
    #Boundary Auxes with three fluxes:
    #aux_13, aux_22, aux_113
    
    num_3flux = 3
    #Boundary Auxes with four fluxes:
    
    #aux_12, aux_1223
    num_4flux = 2
    
    #All but flux 4 terms can be removed
    bound_error = 1.0 * num_1flux + 2.0 * num_2flux + 3.0 * num_3flux + 0.001
    
    list_e2_kflux = [flux_up, flux_down]
    list_index_stable = np.array([6, 4, 10, 5, 7, 1, 9, 0, 3, 8, 2,11,12,13,14,15])
    list_index_stable.sort()
    list_aux_bound = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, bound_error, 0.00
    )
    
    known_aux_bound = [
        AuxiliaryVector([(1, 1),(2, 1)], 4),
        AuxiliaryVector([(1, 1), (2, 2), (3, 1)], 4),
    ]
    assert set(list_aux_bound) == set(known_aux_bound)
    
    #All but flux 4 and 3 terms can be removed
    bound_error = 1.0 * num_1flux + 2.0 * num_2flux + 0.001
    
    list_aux_bound = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, bound_error, 0.00
    )
    
    known_aux_bound = [
        AuxiliaryVector([(1, 1),(2, 1)], 4),
        AuxiliaryVector([(1, 1), (2, 2), (3, 1)], 4),
        AuxiliaryVector([(1, 1), (3, 1)], 4),
        AuxiliaryVector([(2, 2)], 4),
        AuxiliaryVector([(1, 2), (3, 1)], 4),
    ]
    assert set(list_aux_bound) == set(known_aux_bound)
    
    #All but flux 4, 3, and 2 terms can be removed
    
    bound_error = 1.0 * num_1flux + 0.001
    
    list_aux_bound = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, bound_error, 0.00
    )
    
    known_aux_bound = [
        AuxiliaryVector([(1, 1),(2, 1)], 4),
        AuxiliaryVector([(1, 1), (2, 2), (3, 1)], 4),
        AuxiliaryVector([(1, 1), (3, 1)], 4),
        AuxiliaryVector([(2, 2)], 4),
        AuxiliaryVector([(1, 2), (3, 1)], 4),
        AuxiliaryVector([(0, 1)], 4),
        AuxiliaryVector([(1, 2), (2, 2)], 4),
        AuxiliaryVector([(1, 2), (2, 1)], 4),
        AuxiliaryVector([(2, 1), (3, 2)], 4),
        AuxiliaryVector([(1, 1), (3, 2)], 4),
    ]
    assert set(list_aux_bound) == set(known_aux_bound)

def test_fraction_discard():
    """
    Tests that f_discard is stored properly and affects the calculation of the
    auxiliary boundary correctly.
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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[1] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    with pytest.raises(UnsupportedRequest) as excinfo:
        hops_ad.make_adaptive(1e-3, 1e-3, f_discard=-1)
    assert "acceptable range of [0,1]" in str(excinfo.value)

    with pytest.raises(UnsupportedRequest) as excinfo:
        hops_ad.make_adaptive(1e-3, 1e-3, f_discard=2)
    assert "acceptable range of [0,1]" in str(excinfo.value)

    # test that f_discard is properly assigned
    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(0.15, 0, f_discard=0.99)
    assert hops_ad.basis.f_discard == 0.99
    hops_ad.initialize(psi_0)
    hops_ad.basis.hierarchy.auxiliary_list = [AuxiliaryVector([], 4),
                                              AuxiliaryVector([(2,1)],4),
                                              AuxiliaryVector([(3,1)],4)]
    hops_ad.basis.system.state_list = [1]
    hops_ad.basis.mode.list_absindex_mode = [2, 3]
    # Creating flux up and flux down matrices for initial hierarchy
    flux_down = np.zeros((2, 3))
    flux_up = np.zeros((2, 3))
    flux_up[0, 1] = 0.14 ** 2 # flux into <0, 0, 2, 0>
    flux_up[1, 1] = 0.1 ** 2 # flux into <0, 0, 1, 1>
    flux_up[0, 2] = 0.099 ** 2 # flux into <0, 0, 1, 1>
    flux_up[1, 2] = 0.201 ** 2 # flux into <0, 0, 0, 2>
    list_e2_kflux = np.array((flux_up, flux_down))
    list_index_stable = np.array([0, 1, 2])
    list_aux_bound = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, hops_ad.basis.delta_a ** 2, hops_ad.basis.f_discard
    )
    # Explanation: the total squared-flux associated with <0, 0, 1, 1> is just over
    # 0.14^2. However, the individual components of flux are the smallest in the flux
    # list. Their sum is less than (0.15*0.99)^2 and as such both are discarded.
    # With the ensuing offset in place, <0, 0, 2, 0> is included in the basis.
    known_list_aux = [AuxiliaryVector([(2, 2)], 4), AuxiliaryVector([(3, 2)], 4)]
    assert set(list_aux_bound) == set(known_list_aux)

    # Test with f_discard set to 0
    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(0.15, 1e-3, f_discard=0)
    # Note: we should guarantee that integer 0, because it's simple to enter,
    # yields the same results as 0.0.
    assert hops_ad.basis.f_discard == 0.0
    hops_ad.initialize(psi_0)
    hops_ad.basis.hierarchy.auxiliary_list = [AuxiliaryVector([], 4),
                                              AuxiliaryVector([(2,1)],4),
                                              AuxiliaryVector([(3,1)],4)]
    hops_ad.basis.system.state_list = [1]
    hops_ad.basis.mode.list_absindex_mode = [2, 3]
    # Creating flux up and flux down matrices for initial hierarchy
    flux_down = np.zeros((2, 3))
    flux_up = np.zeros((2, 3))
    flux_up[0, 1] = 0.14 ** 2 # flux into <0, 0, 2, 0>
    flux_up[1, 1] = 0.1 ** 2 # flux into <0, 0, 1, 1>
    flux_up[0, 2] = 0.099 ** 2 # flux into <0, 0, 1, 1>
    flux_up[1, 2] = 0.201 ** 2 # flux into <0, 0, 0, 2>
    list_e2_kflux = np.array((flux_up, flux_down))
    list_index_stable = np.array([0, 1, 2])
    list_aux_bound = hops_ad.basis._determine_boundary_hier(
        list_e2_kflux, list_index_stable, hops_ad.basis.delta_a ** 2, hops_ad.basis.f_discard
    )
    # Explanation: the total squared-flux associated with <0, 0, 1, 1> is just over
    # 0.14^2. Thus, <0, 0, 2, 0> is not included in the basis, but <0, 0, 1, 1> is.
    known_list_aux = [AuxiliaryVector([(2, 1), (3, 1)], 4), AuxiliaryVector([(3, 2)],
                                                                            4)]
    assert set(list_aux_bound) == set(known_list_aux)

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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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
    known_index = [0, 1, 3, 4]
    assert np.array_equal(list_index, known_index)
    known_members = [1, 2, 4, 5]
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
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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
    list_index_aux_stable = [0, 1]
    list_aux_bound = [AuxiliaryVector([(10, 1)], 20), AuxiliaryVector([(11, 1)], 20)]

    hops_ad_dsystem_dt = hops_ad.basis.eom._prepare_derivative(hops_ad.basis.system,
                                                               hops_ad.basis.hierarchy,
                                                               hops_ad.basis.mode)

    # Get all error terms
    gw_10 = gw_sysbath[10]
    gw_11 = gw_sysbath[11]
    hops_ad.phi = np.array([0, 1.0 + 0j, 0, 0, 0.8 - 0j, 0], dtype=np.complex128)
    dsystem_dt = (hops_ad_dsystem_dt(hops_ad.phi, z_step[2],
                                     z_step[0], z_step[1])[0].reshape([3,2],
                     order = "F")/hbar)
    deletion = (hops_ad.phi.reshape([3,2], order = "F")/2.0)
    analytic_error_deriv_deletion = np.abs(dsystem_dt + deletion)**2

    analytic_sflux_deriv = np.array([0, (1.0**2 + 0.8**2)*(10**2 + 40**2)/hbar**2, 0])

    analytic_flux_up = np.array([[0, 0],
                                 [(1.0 ** 2) * (np.abs(gw_10[1]) ** 2 + np.abs(
                                     gw_11[1]) ** 2), 0],
                                 [0, 0]], dtype=np.complex128) / hbar ** 2

    # Note that expectation values are normalized to norm of physical wave function
    analytic_flux_down = np.array([[0, 0],
                                   [0, (0.8 ** 2 * (1.0 - 1.0 ** 2) ** 2) * (
                                               np.abs(gw_10[0] / gw_10[1]) ** 2 +
                                               np.abs(gw_11[0] / gw_11[1]) ** 2)],
                                   [0, 0]], dtype=np.complex128) / hbar ** 2

    known_error = (np.sum(analytic_error_deriv_deletion,axis=1) +
                          analytic_sflux_deriv +
                          np.sum(analytic_flux_up, axis=1) +
                          np.sum(analytic_flux_down, axis=1))

    error = hops_ad.basis.state_stable_error(
        hops_ad.phi, 2.0, z_step, list_index_aux_stable, list_aux_bound
    )

    np.testing.assert_allclose(np.array(error,dtype=np.complex128), known_error)

def test_list_M2_by_dest():
    """
    Tests that the matrix that finds the attachment between modes and states for
    each possible destination state that can be fluxed into from the current basis is
    correct for general L-operators. In addition, tests that the adjustment to flux
    down from the expectation values of the L-operators, given by
    X2_exp_lop_mode_state, is also correct.
    """
    noise_param = {
        "SEED": basis_noise_10site[:7, :],
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 4
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

    gw_sysbath = []
    def get_holstein(n):
        # Helper function to get site-projection L-operator
        lop = np.zeros([nsite,nsite], dtype=np.complex128)
        lop[n,n] = 1
        return lop
    def get_peierls(n):
        # Helper function to get nearest-neighbor coupling L-operators (with opposite
        # signs of coupling to make the M2_mode_by_state matrices more unique)
        lop = np.zeros([nsite, nsite], dtype=np.complex128)
        lop[n, n+1] = 1j
        lop[n+1, n] = -1j
        return lop
    lop_1 = get_holstein(0)
    lop_2 = get_holstein(1)
    lop_3 = get_holstein(2)
    lop_4 = get_holstein(3)
    lop_5 = get_peierls(0)
    lop_6 = get_peierls(1)
    lop_7 = get_peierls(2)


    lop_list_base = [lop_1, lop_2, lop_3, lop_4, lop_5, lop_6, lop_7]
    lop_list = []
    for lop in lop_list_base:
        gw_sysbath.append([g_0, w_0])
        lop_list.append(lop)
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(lop)

    hs = np.zeros([nsite, nsite])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 10
    hs[2, 1] = 10
    hs[2, 3] = 10
    hs[3, 2] = 10

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
    psi_0[0] = 1.0/np.sqrt(2)
    psi_0[3] = -1.0/np.sqrt(2)
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
    hops_ad.basis.system.state_list = [0, 3]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 6, 7, 8, 9, 12, 13]
    phi = hops_ad.phi
    hops_ad.phi = phi.reshape([len(hops_ad.basis.hierarchy.auxiliary_list),len(hs)])[:,
                  hops_ad.basis.system.state_list].flatten()

    # first column represents flux from state 0, second from state 3
    # For destination state 0
    M2_mode_from_state_known_dest_0 = np.array(
        [[1.0, 0.0],
         [1.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0]]
    )

    M2_mode_from_state_known_dest_0_off_diag = np.zeros_like(
        M2_mode_from_state_known_dest_0)
    M2_mode_from_state_known_dest_0_off_diag[:,1] = M2_mode_from_state_known_dest_0[:,1]

    # For destination state 1
    M2_mode_from_state_known_dest_1 = np.array(
        [[0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [-1.0j, 0.0],
         [-1.0j, 0.0],
         [0.0, 0.0],
         [0.0, 0.0]]
    )

    M2_mode_from_state_known_dest_1_off_diag = np.zeros_like(
        M2_mode_from_state_known_dest_1)
    M2_mode_from_state_known_dest_1_off_diag[:,:] = M2_mode_from_state_known_dest_1[:,:]

    # For destination state 2
    M2_mode_from_state_known_dest_2 = np.array(
        [[0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0j],
         [0.0, 1.0j]]
    )

    M2_mode_from_state_known_dest_2_off_diag = np.zeros_like(
        M2_mode_from_state_known_dest_2)
    M2_mode_from_state_known_dest_2_off_diag[:,:] = M2_mode_from_state_known_dest_2[:,:]

    # For destination state 3
    M2_mode_from_state_known_dest_3 = np.array(
        [[0.0, 0.0],
         [0.0, 0.0],
         [0.0, 1.0],
         [0.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0]]
    )

    M2_mode_from_state_known_dest_3_off_diag = np.zeros_like(
        M2_mode_from_state_known_dest_3)
    M2_mode_from_state_known_dest_3_off_diag[:,0] = M2_mode_from_state_known_dest_3[:,0]

    X2_exp_lop_mode_state_known = np.array(
        [[1.0, 1.0],
         [1.0, 1.0],
         [1.0, 1.0],
         [1.0, 1.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0]]
    )/2

    list_M2_by_dest_off_diag = hops_ad.basis.list_M2_by_dest_off_diag
    X2_exp_lop_mode_state = hops_ad.basis.X2_exp_lop_mode_state
    M2_diag = hops_ad.basis.M2_mode_from_state_diag
    list_M2_by_dest_known = [M2_mode_from_state_known_dest_0,
                             M2_mode_from_state_known_dest_1,
                             M2_mode_from_state_known_dest_2,
                             M2_mode_from_state_known_dest_3]
    list_M2_by_dest_off_diag_known = [M2_mode_from_state_known_dest_0_off_diag,
                                      M2_mode_from_state_known_dest_1_off_diag,
                                      M2_mode_from_state_known_dest_2_off_diag,
                                      M2_mode_from_state_known_dest_3_off_diag]
    M2_diag_known = np.sum(list_M2_by_dest_known, axis=0) - np.sum(
        list_M2_by_dest_off_diag_known, axis=0)
    for i in range(4):
        np.testing.assert_allclose(list_M2_by_dest_off_diag[i].todense(),
                                   list_M2_by_dest_off_diag_known[i])
    np.testing.assert_allclose(X2_exp_lop_mode_state_known,
                               X2_exp_lop_mode_state.toarray())
    np.testing.assert_allclose(M2_diag_known, M2_diag.toarray())

    # Test a dense-coupling case
    noise_param["SEED"] = basis_noise_10site[:1, :]
    gw_sysbath = [gw_sysbath[0]]

    lop = np.zeros([nsite,nsite], dtype=np.complex128)
    lop[0, 0] = 2
    lop[0, 1] = 1j
    lop[1, 0] = -1j
    lop[0, 2] = 1
    lop[2, 0] = 1
    lop[0, 3] = -1j
    lop[3, 0] = 1j

    lop_list = [lop]

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    hops_ad_dense_coupling = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad_dense_coupling.make_adaptive(1e-3, 1e-3)
    psi_0 = np.array([1, 0, 1, 0])/np.sqrt(2)
    hops_ad_dense_coupling.initialize(psi_0)
    hops_ad_dense_coupling.basis.system.state_list = [0, 2]
    hops_ad_dense_coupling.basis.mode.list_absindex_mode = [0]
    phi = hops_ad_dense_coupling.phi
    hops_ad_dense_coupling.phi = phi.reshape([len(
        hops_ad_dense_coupling.basis.hierarchy.auxiliary_list), len(hs)])[:,
                  hops_ad_dense_coupling.basis.system.state_list].flatten()

    # For destination state 0
    M2_mode_from_state_known_dest_0 = np.array(
        [[2.0, 1.0]],
    )

    M2_mode_from_state_known_dest_0_off_diag = np.array(
        [[0.0, 1.0]],
    )

    # For destination state 1
    M2_mode_from_state_known_dest_1 = np.array(
        [[-1.0j, 0.0]],
    )

    M2_mode_from_state_known_dest_1_off_diag = np.array(
        [[-1.0j, 0.0]],
    )

    # For destination state 2
    M2_mode_from_state_known_dest_2 = np.array(
        [[1.0, 0.0]],
    )

    M2_mode_from_state_known_dest_2_off_diag = np.array(
        [[1.0, 0.0]],
    )

    # For destination state 0
    M2_mode_from_state_known_dest_3 = np.array(
        [[1.0j, 0.0]],
    )

    M2_mode_from_state_known_dest_3_off_diag = np.array(
        [[1.0j, 0.0]],
    )

    X2_exp_lop_mode_state_known = np.array(
        [[2.0, 2.0]]
    )

    list_M2_by_dest_off_diag = hops_ad_dense_coupling.basis.list_M2_by_dest_off_diag
    M2_diag = hops_ad_dense_coupling.basis.M2_mode_from_state_diag
    X2_exp_lop_mode_state = hops_ad_dense_coupling.basis.X2_exp_lop_mode_state
    list_M2_by_dest_known = [M2_mode_from_state_known_dest_0,
                             M2_mode_from_state_known_dest_1,
                             M2_mode_from_state_known_dest_2,
                             M2_mode_from_state_known_dest_3]
    list_M2_by_dest_off_diag_known = [M2_mode_from_state_known_dest_0_off_diag,
                                      M2_mode_from_state_known_dest_1_off_diag,
                                      M2_mode_from_state_known_dest_2_off_diag,
                                      M2_mode_from_state_known_dest_3_off_diag]
    M2_diag_known = np.sum(list_M2_by_dest_known, axis=0) - np.sum(
        list_M2_by_dest_off_diag_known, axis=0)
    for i in range(4):
        np.testing.assert_allclose(list_M2_by_dest_off_diag[i].todense(),
                           list_M2_by_dest_off_diag_known[i])
    np.testing.assert_allclose(X2_exp_lop_mode_state.todense(),
                               X2_exp_lop_mode_state_known)
    np.testing.assert_allclose(M2_diag_known, M2_diag.toarray())

def test_get_Z2_noise_sparse():
    """
    Tests that the matrix that projects the noise onto the sparse Hamiltonian for
    the purposes of adaptivity is properly constructed.
    """
    noise_param = {
        "SEED": basis_noise_10site[:7, :],
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 4
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

    gw_sysbath = []

    def get_holstein(n):
        # Helper function to get site-projection L-operator
        lop = np.zeros([nsite, nsite], dtype=np.complex128)
        lop[n, n] = 1
        return lop

    def get_peierls(n):
        # Helper function to get nearest-neighbor coupling L-operators (with opposite
        # signs of coupling to make the M2_mode_by_state matrices more unique)
        lop = np.zeros([nsite, nsite], dtype=np.complex128)
        lop[n, n + 1] = 1j
        lop[n + 1, n] = -1j
        return lop

    lop_1 = get_holstein(0)
    lop_2 = get_holstein(1)
    lop_3 = get_holstein(2)
    lop_4 = get_holstein(3)
    lop_5 = get_peierls(0)
    lop_6 = get_peierls(1)
    lop_7 = get_peierls(2)

    # Associate each BCF mode with the appropriate L-operator.
    lop_list_base = [lop_1, lop_2, lop_3, lop_4, lop_5, lop_6, lop_7]
    lop_list = []
    for lop in lop_list_base:
        gw_sysbath.append([g_0, w_0])
        lop_list.append(lop)
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(lop)

    hs = np.zeros([nsite, nsite])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 10
    hs[2, 1] = 10
    hs[2, 3] = 10
    hs[3, 2] = 10

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    sys_param_holstein = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath[:8],
        "L_HIER": lop_list[:8],
        "L_NOISE1": lop_list[:8],
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath[:8],
    }

    noise_param_holstein = {
        "SEED": basis_noise_10site[:4, :],
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
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
    psi_0[0] = 1.0 / np.sqrt(2)
    psi_0[3] = -1.0 / np.sqrt(2)
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
    hops_ad.basis.system.state_list = [0, 3]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 6, 7, 8, 9, 12, 13]
    # determined manually - 2 modes per unique bath!
    list_lop_in_basis = [0, 3, 4, 6]
    list_lop_in_basis_off_diag = [4,6]
    list_mode_off_diag = [8, 9, 12, 13]

    # Noise arrays should be of a length equal to the list of unique L-operators.
    noise_1 = 1j*np.arange(len(lop_list_base))
    noise_2 = 2 * np.ones_like(noise_1)
    # Noise memory array has an entry for each BCF mode.
    noise_mem = np.arange(len(lop_list))
    z_step = [noise_1[list_lop_in_basis],
              noise_2[list_lop_in_basis],
              noise_mem]

    Z2_noise_sparse_known = np.sum((np.array([noise_mem[m]*lop_list[m] for m in
                list_mode_off_diag])), axis=0) + np.sum(np.array(
        [(np.conj(noise_1)-1j*noise_2)[m] * lop_list_base[m] for m in
         list_lop_in_basis_off_diag]), axis=0)
    Z2_noise_sparse = hops_ad.basis.get_Z2_noise_sparse(z_step)

    assert np.allclose(Z2_noise_sparse_known, Z2_noise_sparse.todense())

    # Test that if only diagonal L-operators are included in the basis, we get a an
    # empty noise matrix instead to save time.
    hops_ad = HOPS(
        sys_param_holstein,
        noise_param=noise_param_holstein,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    hops_ad.basis.system.state_list = [0,3]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 6, 7]
    z_step = [noise_1[list_lop_in_basis],
              noise_2[list_lop_in_basis],
              noise_mem]
    Z2_noise_sparse = hops_ad.basis.get_Z2_noise_sparse(z_step)

    assert np.allclose(Z2_noise_sparse_known*0, Z2_noise_sparse.todense())


def test_get_T2_ltc():
    """
    Tests that the matrix that projects the low-temperature correction onto the sparse
    Hamiltonian for the purposes of adaptivity is properly constructed.
    """
    noise_param = {
        "SEED": basis_noise_10site[:7, :],
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 4
    e_lambda = 20.0
    gamma = 50.0
    temp = 140.0
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

    gw_sysbath = []

    def get_holstein(n):
        # Helper function to get site-projection L-operator
        lop = np.zeros([nsite, nsite], dtype=np.complex128)
        lop[n, n] = 1
        return lop

    def get_peierls(n):
        # Helper function to get nearest-neighbor coupling L-operators (with opposite
        # signs of coupling to make the M2_mode_by_state matrices more unique)
        lop = np.zeros([nsite, nsite], dtype=np.complex128)
        lop[n, n + 1] = 1j
        lop[n + 1, n] = -1j
        return lop

    lop_1 = get_holstein(0)
    lop_2 = get_holstein(1)
    lop_3 = get_holstein(2)
    lop_4 = get_holstein(3)
    lop_5 = get_peierls(0)
    lop_6 = get_peierls(1)
    lop_7 = get_peierls(2)

    # Associate each BCF mode with the appropriate L-operator.
    lop_list_base = [lop_1, lop_2, lop_3, lop_4, lop_5+lop_1, lop_6, lop_7]
    lop_list = []
    for lop in lop_list_base:
        gw_sysbath.append([g_0, w_0])
        lop_list.append(lop)
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(lop)

    hs = np.zeros([nsite, nsite])
    hs[0, 1] = 10
    hs[1, 0] = 10
    hs[1, 2] = 10
    hs[2, 1] = 10
    hs[2, 3] = 10
    hs[3, 2] = 10

    G1_ltc = [10, 5+5j, -10, 5-5j, 10j, -5j, 10 + 10j]

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
        "L_LT_CORR" : lop_list_base,
        "PARAM_LT_CORR" : G1_ltc,
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
    psi_0[0] = 1.0 / np.sqrt(2)
    psi_0[3] = -1.0 / np.sqrt(2)
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    list_lop_avg = [np.conj(psi_0)@L@psi_0 for L in lop_list_base]

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    hops_ad.basis.system.state_list = [0, 3]
    # Make sure that it is the HopsModes object's list_absindex_mode that indexes
    # everything. The auxiliary with depth in mode 10 will  cause a dimension
    # mismatch if any piece of the T2 matrix is calculated with the HopsSystem's list
    # of absolute L-operator indices, because this is a mode the state basis simply
    # does not know about.
    hops_ad.basis.hierarchy.auxiliary_list = [hops_ad.basis.hierarchy.auxiliary_list[
                                                  0], AuxiliaryVector([(10, 1)], 14)]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 6, 7, 8, 9, 10, 12, 13]
    # Determined manually - 2 modes per unique bath!
    list_lop_in_basis = [0, 3, 4, 5, 6]
    hops_ad.phi = psi_0[[0,3]]

    # Ensure the expectation value of the L-operators is set based on the current
    # physical wave function (this is typically managed prior to adaptive error
    # calculation)
    hops_ad.basis.psi = psi_0[[0,3]]
    T2_phys, T2_hier = hops_ad.basis.get_T2_ltc()
    T2_phys_known = np.zeros([nsite,nsite],dtype=np.complex128)
    T2_hier_known = np.zeros([nsite, nsite], dtype=np.complex128)
    for l_ind in list_lop_in_basis:
        L = lop_list_base[l_ind]
        if not np.allclose(L, np.diag(np.diag(L))):
            G = G1_ltc[l_ind]
            X = list_lop_avg[l_ind]
            T2_phys_known += 2*np.real(G)*X*L - G*(L@L)
            T2_hier_known += np.conj(G)*X*L
    np.testing.assert_allclose(T2_phys_known, T2_phys.toarray())
    np.testing.assert_allclose(T2_hier_known, T2_hier.toarray())

    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
        "L_LT_CORR": lop_list_base,
        "PARAM_LT_CORR": [0, 100, 1000, 0, 0, 1000000, 0, 10000000000000],
    }

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    hops_ad.basis.system.state_list = [0, 3]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 6, 7, 8, 9, 12, 13]
    hops_ad.basis.hierarchy.auxiliary_list = [hops_ad.basis.hierarchy.auxiliary_list[0]]
    hops_ad.phi = psi_0[[0, 3]]

    # When the LTC factors associated with L-operators in the basis are all 0,
    # the low-temperature correction should default to None. The same should be true
    # if the L-operators are fully diagonal.
    T2_phys, T2_hier = hops_ad.basis.get_T2_ltc()
    assert T2_phys is None
    assert T2_hier is None

    sys_param_holstein = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath[:8],
        "L_HIER": lop_list[:8],
        "L_NOISE1": lop_list[:8],
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath[:8],
        "L_LT_CORR": lop_list_base[:4],
        "PARAM_LT_CORR": [10000, 10000, 10000, 10000],
    }

    noise_param_holstein = {
        "SEED": basis_noise_10site[:4, :],
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    hops_ad = HOPS(
        sys_param_holstein,
        noise_param=noise_param_holstein,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    hops_ad.basis.system.state_list = [0, 3]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 6, 7]

    T2_phys, T2_hier = hops_ad.basis.get_T2_ltc()
    assert T2_phys is None
    assert T2_hier is None

def test_hier_max_adaptive():
    """
    Tests that auxiliaries are constructed in full for calculations with small
    delta_A up to the maximum hierarchy depth of 255 using a monomer model.
    """
    nsite = 1
    e_lambda = 300
    gamma = 60
    temp = 300
    hs = np.zeros([1, 1], np.complex128)
    (g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)
    gw_sysbath = [[g_0, w_0]]
    lop_list = np.ones([1, 1, 1])

    sys_param = {
        "HAMILTONIAN": hs,
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    noise_param = {
        "SEED": None,
        "MODEL": "ZERO",
        "TLEN": 3000.0,
        "TAU": 0.5,
    }

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([1.0], dtype=np.complex128)
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 255},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    # Set delta_a to floating point minimum and run until all auxiliaries can be populated
    hops_ad.make_adaptive(1e-323, 0, 1)
    hops_ad.initialize(psi_0)
    assert hops_ad.basis.n_hier < 256

    hops_ad.propagate(300, 1.0)
    assert hops_ad.basis.n_hier == 256