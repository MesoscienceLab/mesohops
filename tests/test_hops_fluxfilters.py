import numpy as np
import scipy as sp
from mesohops.basis.hops_aux import AuxiliaryVector as AuxiliaryVector
from mesohops.util.bath_corr_functions import bcf_convert_sdl_to_exp
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS


def test_filter_hierarchy_stable_up():
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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
        gw_sysbath.append([g_0, w_0])
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite])

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Adaptive Hops
    hops_ad = HOPS(sys_param, noise_param=noise_param, hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param, integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    # The auxiliary list currently contains AuxVec([],20),AuxVec([(10,1)],20),AuxVec([(11,1)],20)
    # In order to keep the current connections intact, we must pass in the existing main auxiliary,
    # rather than creating a new one.
    aux_list = [hops_ad.auxiliary_list[0], AuxiliaryVector([(1, 1)], 20),
                AuxiliaryVector([(2, 1)], 20), AuxiliaryVector([(1, 1), (2, 1)], 20),
                AuxiliaryVector([(2, 3)], 20), AuxiliaryVector([(1, 4)], 20),
                AuxiliaryVector([(3, 4)], 20)]
    hops_ad.basis.hierarchy.auxiliary_list = aux_list
    hops_ad.basis.system.state_list = [0, 1]
    filter_hier_stable_up = hops_ad.basis.flux_filters.construct_filter_auxiliary_stable_up()

    known_filter_hier_stable_up = np.array(
        [[1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0]])
    assert np.all(filter_hier_stable_up == known_filter_hier_stable_up)


def test_filter_hierarchy_stable_down():
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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
        gw_sysbath.append([g_0, w_0])
        gw_sysbath.append([g_0, w_0])
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        lop_list.append(sp.sparse.coo_matrix(loperator[i]))
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
        lop_list.append(loperator[i])

    hs = np.zeros([nsite, nsite])

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Adaptive Hops
    hops_ad = HOPS(sys_param, noise_param=noise_param, hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param, integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    # The auxiliary list currently contains AuxVec([],20),AuxVec([(10,1)],20),AuxVec([(11,1)],20)
    # In order to keep the current connections intact, we must pass in the existing main auxiliary,
    # rather than creating a new one.
    aux_list = [hops_ad.auxiliary_list[0], AuxiliaryVector([(1, 1)], 20),
                AuxiliaryVector([(2, 1)], 20), AuxiliaryVector([(1, 1), (2, 1)], 20),
                AuxiliaryVector([(2, 3)], 20), AuxiliaryVector([(1, 4)], 20),
                AuxiliaryVector([(3, 4)], 20)]
    hops_ad.basis.hierarchy.auxiliary_list = aux_list
    hops_ad.basis.system.state_list = [0]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 2, 3]
    filter_hier_stable_down = hops_ad.basis.flux_filters.construct_filter_auxiliary_stable_down()
    known_filter_hier_stable_down = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 1, 0],
         [0, 0, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1]])
    assert np.all(filter_hier_stable_down == known_filter_hier_stable_down)


def test_filter_hierarchy_boundary_up():
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Adaptive Hops
    hops_ad = HOPS(sys_param, noise_param=noise_param, hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param, integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    hops_ad.basis.hierarchy.auxiliary_list = [hops_ad.basis.hierarchy.auxiliary_list[0]]
    # The auxiliary list currently contains AuxVec([],20)
    # In order to keep the current connections intact, we must pass in the existing main auxiliary,
    # rather than creating a new one.
    list_aux = [hops_ad.basis.hierarchy.auxiliary_list[0], AuxiliaryVector([(1, 1)], 20),
                AuxiliaryVector([(2, 1)], 20), AuxiliaryVector([(1, 1), (2, 1)], 20),
                AuxiliaryVector([(2, 3)], 20), AuxiliaryVector([(1, 4)], 20),
                AuxiliaryVector([(3, 4)], 20)]
    hops_ad.basis.hierarchy.auxiliary_list = list_aux
    hops_ad.basis.system.state_list = [0]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 2, 3]
    filter_hier_boundary_up = hops_ad.basis.flux_filters.construct_filter_auxiliary_boundary_up()

    known_filter_hier_boundary_up = np.array(
        [[1, 1, 1, 1, 1, 0, 0],
         [0, 1, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0, 0],
         [1, 1, 1, 1, 1, 0, 0]])
    assert np.all(filter_hier_boundary_up == known_filter_hier_boundary_up)


def test_filter_hierarchy_boundary_down():
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[0] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Adaptive Hops
    hops_ad = HOPS(sys_param, noise_param=noise_param, hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param, integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    hops_ad.basis.hierarchy.auxiliary_list = [hops_ad.basis.hierarchy.auxiliary_list[0]]
    # The auxiliary list currently contains AuxVec([],20)
    # In order to keep the current connections intact, we must pass in the existing main auxiliary,
    # rather than creating a new one.
    list_aux = [hops_ad.basis.hierarchy.auxiliary_list[0], AuxiliaryVector([(1, 1)], 20),
                AuxiliaryVector([(2, 1)], 20), AuxiliaryVector([(1, 1), (2, 1)], 20),
                AuxiliaryVector([(2, 3)], 20), AuxiliaryVector([(1, 4)], 20),
                AuxiliaryVector([(3, 4)], 20)]
    hops_ad.basis.hierarchy.auxiliary_list = list_aux
    hops_ad.basis.system.state_list = [0]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 2, 3]
    filter_hier_boundary_down = hops_ad.basis.flux_filters.construct_filter_auxiliary_boundary_down()
    known_filter_hier_boundary_down = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1]])
    assert np.all(filter_hier_boundary_down == known_filter_hier_boundary_down)


def test_filter_state_stable_up():
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Adaptive Hops
    hops_ad = HOPS(sys_param, noise_param=noise_param, hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param, integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    # The auxiliary list currently contains AuxVec([],20),AuxVec([(10,1)],20),AuxVec([(11,1)],20)
    # In order to keep the current connections intact, we must pass in the existing main auxiliary,
    # rather than creating a new one.
    aux_list = [hops_ad.basis.hierarchy.auxiliary_list[0], AuxiliaryVector([(1, 1)], 20),
                AuxiliaryVector([(2, 1)], 20), AuxiliaryVector([(1, 1), (2, 1)], 20),
                AuxiliaryVector([(2, 3)], 20), AuxiliaryVector([(1, 4)], 20),
                AuxiliaryVector([(3, 4)], 20)]
    hops_ad.basis.hierarchy.auxiliary_list = aux_list

    list_aux_bound = [AuxiliaryVector([(1, 2)], 20), AuxiliaryVector([(2, 2)], 20),
                      AuxiliaryVector([(3, 3)], 20),
                      AuxiliaryVector([(1, 2), (2, 1)], 20)]
    hops_ad.basis.system.state_list = [0, 1]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 2, 3]
    filter_state_stable_up = hops_ad.basis.flux_filters.construct_filter_state_stable_up(
        list_aux_bound)

    known_filter_state_stable_up = np.array(
        [[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])

    assert np.all(filter_state_stable_up == known_filter_state_stable_up)

    hops_ad = HOPS(sys_param, noise_param=noise_param, hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param, integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    aux_list = [hops_ad.basis.hierarchy.auxiliary_list[0], AuxiliaryVector([(1, 1)], 20),
                AuxiliaryVector([(2, 1)], 20), AuxiliaryVector([(1, 1), (2, 1)], 20),
                AuxiliaryVector([(2, 3)], 20), AuxiliaryVector([(1, 4)], 20),
                AuxiliaryVector([(3, 4)], 20)]
    hops_ad.basis.hierarchy.auxiliary_list = aux_list
    list_aux_bound = [AuxiliaryVector([(1, 2)], 20), AuxiliaryVector([(2, 2)], 20),
                      AuxiliaryVector([(3, 3)], 20),
                      AuxiliaryVector([(1, 2), (2, 1)], 20)]
    hops_ad.basis.system.state_list = [1]
    hops_ad.basis.mode.list_absindex_mode = [1, 2, 3]
    n_hmodes = 3
    filter_state_stable_up = hops_ad.basis.flux_filters.construct_filter_state_stable_up(
        list_aux_bound)

    known_filter_state_stable_up = np.array(
        [[0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])
    assert np.all(filter_state_stable_up == known_filter_state_stable_up)


def test_filter_state_stable_down():
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Adaptive Hops
    hops_ad = HOPS(sys_param, noise_param=noise_param, hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param, integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    # The auxiliary list currently contains AuxVec([],20),AuxVec([(10,1)],20),AuxVec([(11,1)],20)
    # In order to keep the current connections intact, we must pass in the existing main auxiliary,
    # rather than creating a new one.
    aux_list = [hops_ad.basis.hierarchy.auxiliary_list[0], AuxiliaryVector([(1, 1)], 20),
                AuxiliaryVector([(2, 1)], 20), AuxiliaryVector([(1, 1), (2, 1)], 20),
                AuxiliaryVector([(2, 3)], 20), AuxiliaryVector([(1, 4)], 20),
                AuxiliaryVector([(3, 4)], 20)]

    list_aux_bound = [AuxiliaryVector([(1, 2)], 20), AuxiliaryVector([(2, 2)], 20),
                      AuxiliaryVector([(3, 3)], 20),
                      AuxiliaryVector([(1, 2), (2, 1)], 20)]

    hops_ad.basis.hierarchy.auxiliary_list = aux_list
    hops_ad.basis.system.state_list = [0, 1]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 2, 3]
    filter_state_stable_down = hops_ad.basis.flux_filters.construct_filter_state_stable_down(
        list_aux_bound)

    known_filter_state_stable_down = np.array(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1]])
    assert np.all(filter_state_stable_down == known_filter_state_stable_down)


def test_filter_markovian_up():
    """
    Tests that the Markovian static filter that filters subset of modes M generates
    a filter matrix F such that, for mode m and aux k,
    F[m,k] = 0 if m in M AND k is NOT the physical wave function
    OR if k has depth in any m in M.
    and
    F[m,k] = 1 otherwise.

    Tests that filter generation supports multiple instances of the same filter.
    """
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Testing the case with no filter
    hops_nofilter = HOPS(sys_param,
                         noise_param=noise_param,
                         hierarchy_param={'MAXHIER': 6},
                         eom_param=eom_param,
                         integration_param=integrator_param, )
    hops_nofilter.make_adaptive(1e-3, 1e-3)
    hops_nofilter.initialize(psi_0)
    assert hops_nofilter.basis.flux_filters.construct_filter_markov_up() == 1

    # Odd-numbered modes are in M, the list of filtered modes. We split this out among
    # two filters to test for consistency.
    list_mark_filter = ([False] * (1) + [True] * (1)) * nsite
    list_mark_filter[3] = False
    list_mark_filter_2 = [False]*2*nsite
    list_mark_filter_2[3] = True
    # Adaptive Hops
    hops_ad = HOPS(sys_param,
                   noise_param=noise_param,
                   hierarchy_param={'MAXHIER': 6,
                         'STATIC_FILTERS': [['Markovian', list_mark_filter],
                                            ['Markovian', list_mark_filter_2]], },
                   eom_param=eom_param,
                   integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    # The auxiliary list currently contains AuxVec([],20),AuxVec([(10,1)],20),
    # AuxVec([(11,1)],20). In order to keep the current connections intact, we must
    # pass in the existing main auxiliary, rather than creating a new one.
    aux_list = [hops_ad.basis.hierarchy.auxiliary_list[0],                   #0
                AuxiliaryVector([(1, 1)], 20),                               #1
                AuxiliaryVector([(2, 1)], 20),                               #2
                AuxiliaryVector([(3, 1)], 20),                               #3
                AuxiliaryVector([(4, 1)], 20),                               #4
                AuxiliaryVector([(2, 3)], 20)]                               #5

    hops_ad.basis.hierarchy.auxiliary_list = aux_list
    hops_ad.basis.system.state_list = [0, 1, 2]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 2, 3, 4, 5]
    filter_markovian = hops_ad.basis.flux_filters.construct_filter_markov_up()

    known_filter_markovian = np.array([
        #0  1  2  3  4  5
        [1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0]
    ])

    assert np.all(filter_markovian == known_filter_markovian)

def test_filter_triangular_up():
    """
    Tests that the triangular static filter that filters subset of modes M with
    depth k2 generates a filter matrix F such that, for mode m and aux k,
    F[m,k] = 0 if m in M AND (sum_{m' in M} k[m']) >= k2
    and
    F[m,k] = 1 otherwise.

    Tests that filter generation supports multiple instances of the same filter.
    """
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Testing the case with no filter
    hops_nofilter = HOPS(sys_param,
                         noise_param=noise_param,
                         hierarchy_param={'MAXHIER': 6},
                         eom_param=eom_param,
                         integration_param=integrator_param, )
    hops_nofilter.make_adaptive(1e-3, 1e-3)
    hops_nofilter.initialize(psi_0)
    assert hops_nofilter.basis.flux_filters.construct_filter_triangular_up() == 1

    # Odd-numbered modes are in M, the list of filtered modes. We split this out among
    # two filters with two different depths to test for consistency.
    list_tri_filter = ([False] * (1) + [True] * (1)) * nsite
    list_tri_filter[5] = False
    list_tri_filter_2 = [False] * 5 + [True] * 2 + [False] * 13
    # Adaptive Hops
    hops_ad = HOPS(sys_param, noise_param=noise_param,
                   hierarchy_param={'MAXHIER': 6,
                         'STATIC_FILTERS': [['Triangular', [list_tri_filter, 2]],
                                            ['Triangular', [list_tri_filter_2, 3]]],},
                   eom_param=eom_param,
                   integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    # The auxiliary list currently contains AuxVec([],20),AuxVec([(10,1)],20),
    # AuxVec([(11,1)],20). In order to keep the current connections intact, we must
    # pass in the existing main auxiliary, rather than creating a new one.
    aux_list = [hops_ad.basis.hierarchy.auxiliary_list[0],   #0
                AuxiliaryVector([(0, 1)], 20),               #1
                AuxiliaryVector([(3, 1)], 20),               #2
                AuxiliaryVector([(1, 1), (3, 1)], 20),       #3
                AuxiliaryVector([(2, 2)], 20),               #4
                AuxiliaryVector([(2, 1), (3, 1)], 20),       #5
                AuxiliaryVector([(5, 2)], 20),               #6
                AuxiliaryVector([(5, 3)], 20),               #7
                AuxiliaryVector([(3,2), (5,3)], 20),         #8
                AuxiliaryVector([(2, 3), (3,1), (5,2)], 20), #9
                ]

    hops_ad.basis.hierarchy.auxiliary_list = aux_list
    hops_ad.basis.system.state_list = [0, 1, 2]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 2, 3, 4, 5]
    filter_triangular = hops_ad.basis.flux_filters.construct_filter_triangular_up()

    known_filter_triangular = np.array([
        #0` 1  2  3  4  5  6  7  8  9
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # This mode is never filtered
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # This mode is never filtered
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # This mode is never filtered
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    ])

    assert np.all(filter_triangular == known_filter_triangular)

def test_filter_longedge_up():
    """
    Tests that the longedge static filter that filters subset of modes M with
    depth k2 generates a filter matrix F such that, for mode m and aux k,
    F[m,k] = 0 if:
        k has nonzero depth along multiple modes (is not on an edge in the hierarchy)
        AND |k| > k2 (total depth in all modes greater than k2)
        AND (sum_{m' in M} k[m']) >= 1 (nonzero depth in any of the filtered modes)
    and
    F[m,k] = 1 otherwise.

    Tests that filter generation supports multiple instances of the same filter.
    """
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    # Testing the case with no filter
    hops_nofilter = HOPS(sys_param,
                   noise_param=noise_param,
                   hierarchy_param={'MAXHIER': 6},
                   eom_param=eom_param,
                   integration_param=integrator_param, )
    hops_nofilter.make_adaptive(1e-3, 1e-3)
    hops_nofilter.initialize(psi_0)
    assert hops_nofilter.basis.flux_filters.construct_filter_longedge_up() == 1

    # Odd-numbered modes are in M, the list of filtered modes. We split this out
    # among two filters with two different depths to test for consistency.
    list_le_filter = ([False] * (1) + [True] * (1)) * nsite
    list_le_filter[5] = False
    list_le_filter_2 = [False] * 5 + [True] * 2 + [False] * 13
    list_le_filter_2[3] = True # Redundant filtration - should change nothing

    # Adaptive Hops
    hops_ad = HOPS(sys_param,
                   noise_param=noise_param,
                   hierarchy_param={'MAXHIER': 6,
                         'STATIC_FILTERS': [['LongEdge', [list_le_filter, 2]],
                                            ['LongEdge', [list_le_filter_2, 3]]], },
                   eom_param=eom_param,
                   integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    # The auxiliary list currently contains AuxVec([],20),AuxVec([(10,1)],20),
    # AuxVec([(11,1)],20). In order to keep the current connections intact, we must
    # pass in the existing main auxiliary, rather than creating a new one.
    aux_list = [hops_ad.basis.hierarchy.auxiliary_list[0],  #0
                AuxiliaryVector([(1, 1)], 20),              #1
                AuxiliaryVector([(0, 1), (2,1)], 20),       #2
                AuxiliaryVector([(1, 1), (2,1)], 20),       #3
                AuxiliaryVector([(1, 1), (3, 1)], 20),      #4
                AuxiliaryVector([(1, 1), (5, 1)], 20),      #5
                AuxiliaryVector([(2, 2)], 20),              #6
                AuxiliaryVector([(3, 2)], 20),              #7
                AuxiliaryVector([(5, 2)], 20),              #8
                AuxiliaryVector([(2, 3)], 20),              #9
                AuxiliaryVector([(2, 1), (5, 2)], 20),      #10
                AuxiliaryVector([(5, 3)], 20),]             #11
    hops_ad.basis.hierarchy.auxiliary_list = aux_list
    hops_ad.basis.system.state_list = [0, 1, 2]
    hops_ad.basis.mode.list_absindex_mode = [0, 1, 2, 3, 4, 5, 6]
    filter_longedge = hops_ad.basis.flux_filters.construct_filter_longedge_up()

    known_filter_longedge = np.array([
        #0  1  2  3  4  5  6  7  8  9  10  11
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0,  0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0,  0],
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0,  0],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,  1],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,  0],
    ])

    assert np.all(filter_longedge == known_filter_longedge)

def test_mode_setter():
    """
    Tests that the setter for HopsModes.list_absindex_modes updates the relevant
    parameters correctly.
    """
    noise_param = {"SEED": None, "MODEL": "FFT_FILTER", "TLEN": 250.0,
        # Units: fs
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

    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath, "L_HIER": lop_list, "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath, }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM', 'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[5] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(sys_param, noise_param=noise_param,

        hierarchy_param={'MAXHIER': 4}, eom_param=eom_param,
        integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)

    aux_list = [hops_ad.basis.hierarchy.auxiliary_list[0], AuxiliaryVector([(1, 1)], 20),
                AuxiliaryVector([(2, 1)], 20), AuxiliaryVector([(1, 1), (2, 1)], 20),
                AuxiliaryVector([(2, 3)], 20), AuxiliaryVector([(1, 4)], 20),
                AuxiliaryVector([(3, 4)], 20), AuxiliaryVector([(5, 1)], 20)]

    hops_ad.basis.hierarchy.auxiliary_list = aux_list
    hops_ad.basis.system.state_list = [1, 2]

    # Test list_absindex_mode
    known_list_absindex_mode = [1, 2, 3, 4, 5]
    assert np.all(list(set(hops_ad.basis.hierarchy.list_absindex_hierarchy_modes) | set(
        hops_ad.basis.system.list_absindex_state_modes)) == known_list_absindex_mode)

    # Set mode list
    hops_ad.basis.mode.list_absindex_mode = known_list_absindex_mode
    # Test list_absindex_L2
    known_list_absindex_L2 = [0, 1, 2]
    assert np.all(hops_ad.basis.mode.list_absindex_L2 == known_list_absindex_L2)
    # Test n_hmodes
    known_n_hmodes = 5
    assert hops_ad.basis.n_hmodes == known_n_hmodes
    # Test g
    known_g = hops_ad.basis.system.param["G"][1:6]
    assert np.all(known_g == hops_ad.basis.list_g)
    # Test w
    known_w = hops_ad.basis.system.param["W"][1:6]
    assert np.all(known_w == hops_ad.basis.list_w)
    # Test n_l2
    known_n_l2 = 3
    assert hops_ad.basis.mode.n_l2 == known_n_l2
    # Test list_index_L2_by_hmode
    known_list_index_L2_by_hmode = [0, 1, 1, 2, 2]
    assert np.all(
        hops_ad.basis.mode.list_index_L2_by_hmode == known_list_index_L2_by_hmode)
    # Test list_L2_coo
    known_list_L2_coo = []
    L2_coo_0 = sp.sparse.coo_matrix(([], ([], [])), shape=(2, 2))
    L2_coo_1 = sp.sparse.coo_matrix(([1.], ([0.], [0.])), shape=(2, 2))
    L2_coo_2 = sp.sparse.coo_matrix(([1.], ([1.], [1.])), shape=(2, 2))
    known_list_L2_coo.append(L2_coo_0)
    known_list_L2_coo.append(L2_coo_1)
    known_list_L2_coo.append(L2_coo_2)
    assert np.all(
        hops_ad.basis.mode.list_L2_coo[i].toarray() == known_list_L2_coo[i].toarray()
        for i in range(3))

