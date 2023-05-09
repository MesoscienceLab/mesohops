import numpy as np
import scipy as sp
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxiliaryVector
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp


def test_mode_setter():
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

    aux_list = [hops_ad.basis.hierarchy.auxiliary_list[0], AuxiliaryVector([(1, 1)], 10),
                AuxiliaryVector([(2, 1)], 10), AuxiliaryVector([(1, 1), (2, 1)], 10),
                AuxiliaryVector([(2, 3)], 10), AuxiliaryVector([(1, 4)], 10),
                AuxiliaryVector([(3, 4)], 10), AuxiliaryVector([(5, 1)], 10)]

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
    assert np.all(known_g == hops_ad.basis.g)
    # Test w
    known_w = hops_ad.basis.system.param["W"][1:6]
    assert np.all(known_w == hops_ad.basis.w)
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

def test_empty_modelist():
    """
    Tests that an empty list of modes does not crash a HOPS calculation.
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

    # Ensure that the first 2 sites have no associated modes in the hierarchy
    sys_param = {"HAMILTONIAN": np.array(hs, dtype=np.complex128),
                 "GW_SYSBATH": gw_sysbath[2:], "L_HIER": lop_list[2:], "L_NOISE1":
                                                                    lop_list[2:],
                 "ALPHA_NOISE1": bcf_exp, "PARAM_NOISE1": gw_sysbath[2:], }

    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA",
                        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
                        'EARLY_INTEGRATOR_STEPS': 5,
                        'INCHWORM_CAP': 5, 'STATIC_BASIS': None}

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[0] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_ad = HOPS(sys_param, noise_param=noise_param,

                   hierarchy_param={'MAXHIER': 4}, eom_param=eom_param,
                   integration_param=integrator_param, )
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    assert len(hops_ad.basis.mode.list_absindex_mode) == 0
    hops_ad.propagate(4.0, 2.0)