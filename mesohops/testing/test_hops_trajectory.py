import numpy as np
import scipy as sp
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxVec
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.hops_storage import HopsStorage
from mesohops.dynamics.noise_fft import FFTFilterNoise as FFTFN
from mesohops.dynamics.noise_zero import ZeroNoise
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp
from mesohops.util.physical_constants import precision  # constant

__title__ = "test of hops_trajectory "
__author__ = "D. I. G. Bennett"
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
        'EARLY_ADAPTIVE_INTEGRATOR':'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS':None
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
        integration_param=integrator_param,
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
    noiseModel = FFTFN(noise_param, noise_corr)
    assert type(noise) == type(noiseModel)

    ZN = ZeroNoise(noise_param, hops.basis.system.param)
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

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
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

    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
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
