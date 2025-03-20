import sys
import numpy as np
import scipy as sp
from io import StringIO
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxVec
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.hops_storage import HopsStorage
from mesohops.dynamics.hops_noise import HopsNoise
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp
from mesohops.util.exceptions import UnsupportedRequest
from scipy import sparse
from mesohops.util.physical_constants import precision  # constant

__title__ = "test of hops_trajectory "
__author__ = "D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"
__date__ = ""

noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 10.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
}

loperator = np.zeros([2, 2, 2], dtype=np.float64)
loperator[0, 0, 0] = 1.0
loperator[1, 1, 1] = 1.0
sys_param = {
    "HAMILTONIAN": np.array([[0, 10.0], [10.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
}

hier_param = {"MAXHIER": 4}

eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None,
        'EFFECTIVE_NOISE_INTEGRATION': False,
    }
integrator_param_empty = {}
integrator_param_partial = {
    'EARLY_INTEGRATOR_STEPS': 3
}
integrator_param_broken = {
    'PANCAKE_MIXER': 7
}
t_max = 10.0
t_step = 2.0
psi_0 = [1.0 + 0.0 * 1j, 0.0 + 0.0 * 1j]

# Helper Function
# ===============
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
            AuxVec([tuple([key, aux_values[key]]) for key in aux_key], 4)
        )
    return list_aux_vec


def test_initialize():
    """
    test for the hops trajectory initialize function
    """


    # Checks to make sure storage is TrajectoryStorage when calculation is non-adaptive
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param_empty,
    )
    hops.initialize(psi_0)
    storage = hops.storage
    TS = HopsStorage(True,{})
    assert type(storage) == type(TS)

    # Checks to make sure noise was properly initialized
    t_axis = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    noise = np.array([[1, 2, 2, 1, 3], [2, 3, 3, 2, 4]], dtype=np.float64)
    noise = hops.noise1
    sys_param["NSITE"] = len(sys_param["HAMILTONIAN"][0])
    sys_param["NMODES"] = len(sys_param["GW_SYSBATH"][0])
    sys_param["N_L2"] = 2
    sys_param["L_IND_BY_NMODE1"] = [0, 1]
    sys_param["LIND_DICT"] = {0: loperator[0, :, :], 1: loperator[1, :, :]}
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    noiseModel = HopsNoise(noise_param, noise_corr)
    assert type(noise) == type(noiseModel)

    noise_param_0 = noise_param.copy()
    noise_param_0["MODEL"] = "ZERO"
    ZN = HopsNoise(noise_param_0, hops.basis.system.param)
    a = ZN
    b = hops.noise2
    assert type(a) == type(b)

    # checks to make sure correct dimensions are stored
    N_dim = hops.storage.n_dim
    known = 2
    assert N_dim == known

    # checks to make sure calculation is locked after initializing
    lock = hops.__initialized__
    known_lock = True
    assert lock == known_lock

    # checks to make sure integration_param is initialized with defaults correctly
    assert hops.integration_param == integrator_param
    hops_partial = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param_partial,
    )
    partial_dict = {key: val for (key, val) in integrator_param.items()}
    partial_dict['EARLY_INTEGRATOR_STEPS'] = 3
    assert hops_partial.integration_param == partial_dict
    try:
        hops_broken = HOPS(
            sys_param,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param_broken,
        )
    except UnsupportedRequest as excinfo:
        assert ("The current code does not support PANCAKE_MIXER in the integration"
                in str(excinfo))


def test_make_adaptive_delta_h_true():
    """
    Test to check if make_adaptive, with values of >0, is called that the proper DELTA_H
    dictionary values are stored
    """
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.make_adaptive(delta_h=1e-4, delta_s=0)
    adap = hops.basis.eom.param["ADAPTIVE"]
    known_adap = True
    assert adap == known_adap

    adap_h = hops.basis.eom.param["ADAPTIVE_H"]
    known_adap_h = True
    assert adap_h == known_adap_h

    delta_h = hops.basis.eom.param["DELTA_H"]
    known_delta_h = 1e-4
    assert delta_h == known_delta_h

    adap_s = hops.basis.eom.param["ADAPTIVE_S"]
    known_adap_s = False
    assert adap_s == known_adap_s


def test_make_adaptive_both_false():
    """
    Test to check if make_adaptive, with values of 0, is called that the proper DELTA_H
    dictionary values are stored
    """
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.make_adaptive(delta_h=0, delta_s=0)
    adap = hops.basis.eom.param["ADAPTIVE"]
    known_adap = False
    assert adap == known_adap

    adap_h = hops.basis.eom.param["ADAPTIVE_H"]
    known_adap_h = False
    assert adap_h == known_adap_h

    adap_s = hops.basis.eom.param["ADAPTIVE_S"]
    known_adap_s = False
    assert adap_s == known_adap_s

    delta_h = hops.basis.eom.param["DELTA_H"]
    known_delta_h = 0
    assert delta_h == known_delta_h


def test_make_adaptive_delta_s_true():
    """
    Test to check if make_adaptive, with values of >0, is called that the proper DELTA_S
    dictionary values are stored
    """
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.make_adaptive(delta_h=0, delta_s=1e-4)
    adap = hops.basis.eom.param["ADAPTIVE"]
    known_adap = True
    assert adap == known_adap

    adap_s = hops.basis.eom.param["ADAPTIVE_S"]
    known_adap_s = True
    assert adap_s == known_adap_s

    delta_s = hops.basis.eom.param["DELTA_S"]
    known_delta_s = 1e-4
    assert delta_s == known_delta_s

    adap_h = hops.basis.eom.param["ADAPTIVE_H"]
    known_adap_h = False
    assert adap_h == known_adap_h


def test_make_adaptive_both_true():
    """
    Test to check if make_adaptive, with values of 0, is called that the proper DELTA_S
    dictionary values are stored
    """
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.make_adaptive(delta_h=1e-4, delta_s=1e-4)
    adap = hops.basis.eom.param["ADAPTIVE"]
    known_adap = True
    assert adap == known_adap

    adap_h = hops.basis.eom.param["ADAPTIVE_H"]
    known_adap_h = True
    assert adap_h == known_adap_h

    adap_s = hops.basis.eom.param["ADAPTIVE_S"]
    known_adap_s = True
    assert adap_s == known_adap_s

    delta_s = hops.basis.eom.param["DELTA_S"]
    known_delta_s = 1e-4
    assert delta_s == known_delta_s

    delta_h = hops.basis.eom.param["DELTA_H"]
    known_delta_h = 1e-4
    assert delta_h == known_delta_h


def test_check_tau_step():
    """
    Test to make sure tau is within precision, which is a constant
    Our current Tau is 2.0 so at a tau of 2 it passes but if you try to introduce
    a smaller tau (1) it will not pass
    """
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    tau_bool = hops._check_tau_step(2.0, precision)
    known_tau_bool = True
    assert tau_bool == known_tau_bool

    tau_bool = hops._check_tau_step(1.0, precision)
    known_tau_bool = False
    assert tau_bool == known_tau_bool


def test_normalize_if():
    """
    test to make sure psi is being returned normalized when hops.basis.eom.normalized
    is True
    """
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.initialize([2, 3])
    hops.basis.eom.normalized = True
    norm = hops.normalize([2, 3])
    known_norm = [0.5547002, 0.83205029]
    assert np.allclose(norm, known_norm)


def test_normalize_else():
    """
    test to make sure psi is being returned un-normalized when hops.basis.eom.normalized
    is False
    """
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.initialize([2, 3])
    hops.basis.eom.normalized = False
    norm = hops.normalize([2, 3])
    known_norm = [2, 3]
    assert np.allclose(norm, known_norm)


def test_inchworm_aux():
    """
    test for inchworm_integrate to make sure the aux are properly being added
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
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[1] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_inchworm = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 4},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_inchworm.make_adaptive(1e-15, 1e-15)
    hops_inchworm.initialize(psi_0)
    aux_list = hops_inchworm.auxiliary_list
    known_aux_list = map_to_auxvec([(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)])
    assert set(aux_list) == set(known_aux_list)
    z_step = hops_inchworm._prepare_zstep(hops_inchworm.z_mem)  #hops_inchworm.storage.z_mem
    (state_update, aux_update) = hops_inchworm.basis.define_basis(
        hops_inchworm.phi, 2.0, z_step
    )

    # First inchworm
    # ----------------------------------------------------------------------------------
    state_update, aux_update, phi = hops_inchworm.inchworm_integrate(
        state_update, aux_update, 2.0
    )
    aux_new = aux_update
    known = map_to_auxvec(
        [
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 0, 2),
            (0, 0, 1, 0),
            (0, 0, 1, 1),
            (0, 0, 2, 0),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 0, 1, 0),
        ]
    )
    assert set(aux_new) == set(known)

    # Second inchworm
    # ----------------------------------------------------------------------------------
    state_update, aux_update, phi = hops_inchworm.inchworm_integrate(
        state_update, aux_update, 2.0
    )
    aux_new = aux_update
    known = map_to_auxvec(
        [
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 0, 2),
            (0, 0, 0, 3),
            (0, 0, 1, 0),
            (0, 0, 1, 1),
            (0, 0, 1, 2),
            (0, 0, 2, 0),
            (0, 0, 2, 1),
            (0, 0, 3, 0),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 1, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 0, 1, 0),
            (0, 1, 0, 2),
            (0, 1, 1, 0),
            (0, 1, 1, 1),
            (0, 1, 2, 0),
            (0, 2, 0, 0),
            (0, 2, 0, 1),
            (0, 2, 1, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 0, 0, 2),
            (1, 0, 1, 0),
            (1, 0, 1, 1),
            (1, 0, 2, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
            (2, 0, 0, 0),
            (2, 0, 0, 1),
            (2, 0, 1, 0),
        ]
    )
    assert set(aux_new) == set(known)

    # Third inchworm
    # ----------------------------------------------------------------------------------
    state_update, aux_update, phi = hops_inchworm.inchworm_integrate(
        state_update, aux_update, 2.0
    )
    aux_new = aux_update
    known = map_to_auxvec(
        [
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 0, 2),
            (0, 0, 0, 3),
            (0, 0, 1, 0),
            (0, 0, 1, 1),
            (0, 0, 1, 2),
            (0, 0, 2, 0),
            (0, 0, 2, 1),
            (0, 0, 3, 0),
            (0, 1, 0, 0),
            (0, 1, 0, 1),
            (0, 1, 0, 2),
            (0, 1, 1, 0),
            (0, 1, 1, 1),
            (0, 1, 2, 0),
            (0, 2, 0, 0),
            (0, 2, 0, 1),
            (0, 2, 1, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 1),
            (1, 0, 0, 2),
            (1, 0, 1, 0),
            (1, 0, 1, 1),
            (1, 0, 2, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
            (2, 0, 0, 0),
            (2, 0, 0, 1),
            (2, 0, 1, 0),
            (0, 0, 0, 4),
            (0, 0, 1, 3),
            (0, 0, 2, 2),
            (0, 0, 3, 1),
            (0, 0, 4, 0),
            (0, 1, 0, 3),
            (0, 1, 1, 2),
            (0, 1, 2, 1),
            (0, 1, 3, 0),
            (0, 2, 0, 2),
            (0, 2, 1, 1),
            (0, 2, 2, 0),
            (0, 3, 0, 0),
            (0, 3, 0, 1),
            (0, 3, 1, 0),
            (1, 0, 0, 3),
            (1, 0, 1, 2),
            (1, 0, 2, 1),
            (1, 0, 3, 0),
            (1, 1, 0, 2),
            (1, 1, 1, 1),
            (1, 1, 2, 0),
            (1, 2, 0, 0),
            (1, 2, 0, 1),
            (1, 2, 1, 0),
            (2, 0, 0, 2),
            (2, 0, 1, 1),
            (2, 1, 0, 0),
            (2, 1, 0, 1),
            (2, 1, 1, 0),
            (3, 0, 0, 0),
            (3, 0, 1, 0),
            (3, 0, 0, 1),
            (2, 0, 2, 0),
        ]
    )
    assert set(aux_new) == set(known)

def test_inchworm_state():
    """
    test to check inchworm is contributing to the state list
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
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

    psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
    psi_0[2] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)

    hops_inchworm = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_inchworm.make_adaptive(1e-15, 1e-15)
    hops_inchworm.initialize(psi_0)

    state_list = hops_inchworm.state_list
    known_state_list = [1, 2, 3]
    assert tuple(state_list) == tuple(known_state_list)

    z_step = hops_inchworm._prepare_zstep(hops_inchworm.z_mem)  #hops_inchworm.storage.z_mem
    (state_update, aux_update) = hops_inchworm.basis.define_basis(
        hops_inchworm.phi, 2.0, z_step
    )

    # First inchworm step
    # ----------------------------------------------------------------------------------
    state_update, aux_update, phi = hops_inchworm.inchworm_integrate(
        state_update, aux_update, 2.0
    )
    state_new = state_update
    known = [0, 1, 2, 3, 4]
    assert np.array_equal(state_new, known)

    # Second inchworm step
    # ----------------------------------------------------------------------------------
    state_update, aux_update, phi = hops_inchworm.inchworm_integrate(
        state_update, aux_update, 2.0
    )
    state_new = state_update
    known = [0, 1, 2, 3, 4, 5]
    assert np.array_equal(state_new, known)

    # Third inchworm step
    # ----------------------------------------------------------------------------------
    state_update, aux_update, phi = hops_inchworm.inchworm_integrate(
        state_update, aux_update, 2.0
    )
    state_new = state_update
    known = [0, 1, 2, 3, 4, 5, 6]
    assert np.array_equal(state_new, known)



def test_prepare_zstep():
    """
    test for preparing the noise terms for the next time step
    """
    noise_param = {
        "SEED": np.array([[1,1,2,2,3,3,4,4,5,5],[6,6,7,7,8,8,9,9,10,10]])*np.sqrt(2),
        "MODEL": "FFT_FILTER",
        "TLEN": 5.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }

    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.initialize(psi_0)
    z_mem_init = np.array([1,1,1,1])
    tau = 2
    hops.noise1._noise = np.array([[1, 2, 3, 4, 5, 6],
                                   [7, 8, 9, 10, 11, 12]])
    zran1, zrand2,z_mem = hops._prepare_zstep(z_mem_init)
    known_zran1 = np.array([1, 7])
    assert np.allclose(zran1,known_zran1)
    assert np.allclose(zrand2,np.zeros(2))
    assert np.array_equal(z_mem_init,z_mem)

def test_early_time_integrator():
    """
    Tests that the early time integrator is called only when expected, and that
    unsupported early time integrators throw an exception.
    """
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 50.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }

    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.make_adaptive(0.001, 0.001)
    hops.initialize(psi_0)
    assert hops.use_early_integrator
    hops.propagate(6.0, 2.0)
    assert hops.use_early_integrator
    hops.propagate(4.0, 2.0)
    assert not hops.use_early_integrator
    hops.reset_early_time_integrator()
    assert hops.use_early_integrator
    hops.propagate(4.0, 2.0)
    assert hops.use_early_integrator
    hops.propagate(8.0, 4.0)
    assert hops.use_early_integrator
    hops.propagate(2.0, 2.0)
    assert not hops.use_early_integrator

    integrator_param_broken = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'CHOOSE_BASIS_RANDOMLY',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param_broken,
    )
    hops.make_adaptive(0.001, 0.001)
    hops.initialize(psi_0)
    try:
        hops.propagate(2.0, 2.0)
    except UnsupportedRequest as excinfo:
        if "does not support CHOOSE_BASIS_RANDOMLY in the early time integrator " \
           "clause" not in str(excinfo):
            pytest.fail()

def test_static_state_inchworm_hierarchy():
    """
        Tests that choosing the early time integrator 'STATIC_STATE_INCHWORM_HIERARCHY'
        results in a static state basis but an evolving hierarchy basis.
        """
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 250.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }

    integrator_param_test = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'STATIC_STATE_INCHWORM_HIERARCHY',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': [[0], [AuxVec([],4)]]
    }

    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param_test,
    )
    hops.make_adaptive(0.00001, 0.001)
    hops.initialize(psi_0)
    state_0_fs = hops.state_list
    hier_0_fs = hops.auxiliary_list
    hops.propagate(10.0,2.0)
    state_10_fs = hops.state_list
    hier_10_fs = hops.auxiliary_list
    assert np.allclose(state_0_fs, np.array([0]))
    assert np.allclose(state_0_fs, state_10_fs)
    assert not hier_10_fs == hier_0_fs
    hops.propagate(2.0, 2.0)
    assert not np.allclose(state_0_fs, hops.state_list)

    # Control HOPS propagation to show that the state basis should in fact evolve in
    # time
    integrator_param_control = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': [[0], [AuxVec([], 4)]]
    }

    hops_control = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param_control,
    )
    hops_control.make_adaptive(0.00001, 0.001)
    hops_control.initialize(psi_0)
    hops_control.propagate(10.0, 2.0)
    assert not np.allclose(hops_control.state_list, state_10_fs)

def test_noise_hier_mismatch():
    """
    Tests that the noise is handled properly when the noise L-operators and hierarchy
    L-operators are not the same full set.
    """
    sys_param_noise_shorter = {
        "HAMILTONIAN": np.array([[0, 10.0], [10.0, 0]], dtype=np.float64),
        "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
        "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
        "L_NOISE1": [loperator[0], loperator[0]],
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0]],
    }
    hops = HOPS(
        sys_param_noise_shorter,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops.initialize(psi_0)
    assert np.shape(hops.noise1._noise) == (1,11)

    new_lop = np.array([[1,0],[0,1]])
    sys_param_noise_longer = {
        "HAMILTONIAN": np.array([[0, 10.0], [10.0, 0]], dtype=np.float64),
        "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
        "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
        "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1], new_lop,
                     new_lop],
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                         [10.0, 10.0], [5.0, 5.0]],
    }
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    hops = HOPS(
        sys_param_noise_longer,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    sys.stdout = old_stdout
    result_string = result.getvalue()
    hops.initialize(psi_0)
    assert np.shape(hops.noise1._noise) == (3, 11)
    assert "WARNING: the list of noise 1 L-operators contains an L" in result_string

def test_operator():
    """
    Tests management of operation in both non-adaptive and adaptive frameworks.
    """
    # Setting Parameters
    noise_param = {
        "SEED": 10,
        "MODEL": "FFT_FILTER",
        "TLEN": 50.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    nsite = 5
    lop_list = []
    for i in range(nsite):
        lop = np.zeros((nsite , nsite))
        lop[i, i] = 1.0
        lop_list.append(lop)
    V = 1
    H_sys = (np.diag([0] * nsite)
            + np.diag([V] * (nsite - 1), k=-1)
            + np.diag([V] * (nsite - 1), k=1))
    sys_param = {
        "HAMILTONIAN": H_sys,
        "GW_SYSBATH": [[10.0, 10.0]] * 5,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": [[10.0, 10.0]] * 5,
    }
    sys_param_adap = {
        "HAMILTONIAN": H_sys,
        "GW_SYSBATH": [[10.0, 10.0]] * 5,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": [[10.0, 10.0]] * 5,
    }
    hier_param = {"MAXHIER": 5}
    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}
    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }
    hops = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_adap = HOPS(
        sys_param_adap,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        storage_param={'phi_traj':True}
    )
    Op = np.zeros([nsite, nsite])
    Op_2 = np.zeros([nsite, nsite])
    Op[1:, 0] = 1
    Op_2[0, :] = 1
    t_max = 10.0
    t_step = 2.0
    psi_0 = np.zeros(nsite, dtype=np.complex64)
    psi_0[0] = 1

    # Non Adaptive
    # -------------
    hops.initialize(psi_0)
    hops._operator(Op)
    psi_ref=Op@psi_0
    np.testing.assert_allclose(hops.psi, psi_ref)

    # After Propagation
    # -----------------
    hops.propagate(t_max,t_step)
    phi_mat = np.reshape(hops.phi, [hops.n_state, hops.n_hier], order="F")
    phi_ref = np.reshape(Op_2 @ phi_mat, len(hops.phi), order="F")
    hops._operator(Op_2)
    np.testing.assert_allclose(hops.phi, phi_ref)

    # Adaptive
    # ---------
    hops_adap.make_adaptive(0.001, 0.001)
    hops_adap.initialize(psi_0)
    hops_adap._operator(Op)
    psi_ref = Op @ psi_0
    np.testing.assert_allclose(hops_adap.psi, psi_ref)

    # After Propagation
    # -----------------
    hops_adap.propagate(t_max,t_step)
    orig_state_list=hops_adap.state_list
    orig_traj=hops_adap.storage['phi_traj'][-1]
    phi_mat = np.reshape(orig_traj, [hops_adap.n_state, hops_adap.n_hier], order="F")
    hops_adap._operator(Op_2)
    new_state_list =list(set( list(orig_state_list) +
                              list(np.nonzero(Op_2[:, orig_state_list])[0])))
    phi_mat_new = np.zeros((len(new_state_list), hops_adap.n_hier),dtype=complex)

    state_index_map = {state: phi_mat[idx] for idx, state in enumerate(orig_state_list)}

    for key, row_array in state_index_map.items():
        phi_mat_new[key, :] = row_array
    phi_ref = np.reshape(Op_2 @ phi_mat_new, len(new_state_list)*hops_adap.n_hier,
                         order="F")
    np.testing.assert_allclose(hops_adap.phi, phi_ref)

    assert hops_adap._early_step_counter == 0
