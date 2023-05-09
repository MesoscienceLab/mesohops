import os
import numpy as np
import scipy as sp
from scipy import sparse
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.eom_hops_ksuper import _permute_aux_by_matrix
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp

__title__ = "Test of eom_integrator_rk_nonlin_norm"
__author__ = "D. I. G. Bennett"
__version__ = "0.1"
__date__ = ""

# Run Test on Dimer of Dimers
# ===========================
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 25000.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
    "STORE_RAW_NOISE" : True,
    "RAND_MODEL" : "BOX_MULLER"
}

nsite = 4
e_lambda = 20.0
gamma = 50.0
temp = 140.0
(g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)

loperator = np.zeros([4, 4, 4], dtype=np.float64)
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

sys_param = {
    "HAMILTONIAN": np.array(hs, dtype=np.complex128),
    "GW_SYSBATH": gw_sysbath,
    "L_HIER": lop_list,
    "L_NOISE1": lop_list,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": gw_sysbath,
}

eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

integrator_param = {"INTEGRATOR": "RUNGE_KUTTA"}

psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
psi_0[2] = 1.0
psi_0 = psi_0 / np.linalg.norm(psi_0)

t_max = 200.0
t_step = 4.0
hops = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param={"MAXHIER": 2},
    eom_param=eom_param,
)
hops.initialize(psi_0)

# Saved Data
# ----------
path_data = os.path.realpath(__file__)
path_data = path_data[: -len("test_dimer_of_dimers.py")] + "/dimer_of_dimers"


def construct_permute_matrix(stable_aux, stable_state, hops):
    """
    Construct a matrix that rotates the adaptive basis
    of auxiliaries and states onto the original basis.

    PARAMETERS
    ----------
    1. stable_aux : list
                    list of stable auxiliaries in the basis
    2. stable_state : list
                      list of stable states in the basis
    3. hops : HopsTrajectory object
              object describing the stochastic dynamics.

    RETURNS
    -------
    1. P2_permute :  array
                     matrix that maps the retained auxiliaries and
                     states onto the original basis.
    """
    n_reduced = len(stable_aux) * len(stable_state)
    n_reduced2 = len(stable_aux)
    n_full = hops.n_hier * hops.n_state
    n_full2 = hops.n_hier
    permute_aux_row = []
    permute_aux_col = []
    for aux in stable_aux:
        for state in stable_state:
            permute_aux_row.append(
                list(stable_aux).index(aux) * len(stable_state)
                + list(stable_state).index(state)
            )
            permute_aux_col.append(
                hops.basis.hierarchy.auxiliary_list.index(aux)
                * hops.basis.system.param["NSTATES"]
                + list(hops.basis.system.state_list).index(state)
            )
    permute_aux_row2 = []
    permute_aux_col2 = []
    for aux in stable_aux:
        permute_aux_row2.append(
            list(stable_aux).index(aux)
        )
        permute_aux_col2.append(
            hops.basis.hierarchy.auxiliary_list.index(aux)
        )
    P2_permute = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row, permute_aux_col)),
        shape=(n_reduced, n_full),
        dtype=np.complex128,
    ).tocsc()
    P2_permute2 = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row2)), (permute_aux_row2, permute_aux_col2)),
        shape=(n_reduced2, n_full2),
        dtype=np.complex128,
    ).tocsc()
    return P2_permute, P2_permute2


def test_kmat_dimer_of_dimer():
    """
    A test for the super operators constructed for the dimer of dimers
    """
    # Test K2_k
    K2_k = np.load(path_data + "/K2_k.npy")
    assert np.all(K2_k == hops.basis.eom.K2_k.todense())
    
    # Test K2_kp1
    K2_kp1 = np.load(path_data + "/K2_kp1.npy")
    assert np.all(K2_kp1 == hops.basis.eom.K2_kp1.todense())

    # Test K2_km1
    K2_km1 = np.load(path_data + "/K2_km1.npy")
    assert np.all(K2_km1 == hops.basis.eom.K2_km1.todense())

    
    # Test Z2_kp1
    Z2_kp1 = np.load(path_data + "/Z2_kp1.npy")
    assert np.all(
        [
            np.all(Z2_lap == Z2_desk)
            for (Z2_lap, Z2_desk) in zip(hops.basis.eom.Z2_kp1, Z2_kp1)
        ]
    )
    


def test_integration_variables():
    """
    This is a test of the integration variables for the first time point of
    the time evolution for the dimer-of-dimers simulation.
    """

    var_list_desk = {}
    # You must select an initial time and time step that will match the t-axis of the
    # hopstrajectory object.
    var_list_lap = hops.integration_var([1, 1, 1, 1], 2873, 0, hops.noise1,
                                          hops.noise2, 4.0, {})  #this storage input should be something else probably
    var_list_desk["phi"] = [1,1,1,1]
    var_list_desk["z_mem"] = 2873
    var_list_desk["z_rnd"] = hops.noise1.get_noise([0, 2, 4])
    var_list_desk["z_rnd2"] = hops.noise2.get_noise([0, 2, 4])
    var_list_desk["tau"] = 4.0

    flag_pass = True
    for key in var_list_desk.keys():
        try:
            np.testing.assert_allclose(var_list_desk[key], var_list_lap[key])
            print(key)
        except:
            print(key, " <-- ERROR!")
            flag_pass = False
    assert flag_pass == True


def test_alpha():
    """
    This is a test of the correlation functions calculated for the dimer of dimers.
    """
    # Test alpha_t comparison
    alpha_desk = np.load(path_data + "/alpha.npy")
    alpha_lap = hops.noise1._corr_func_by_lop_taxis(hops.noise1.param["T_AXIS"])
    np.testing.assert_allclose(alpha_lap, alpha_desk)


def test_eta():
    """
    This is a test of the noise trajectory calculated for the dimer of dimers
    """
    eta_desk = np.load(path_data + "/eta.npy")
    alpha_lap = hops.noise1._corr_func_by_lop_taxis(hops.noise1.param["T_AXIS"])
    z_correlated = hops.noise1._construct_correlated_noise(alpha_lap,
                                            hops.noise1.param["Z_UNCORRELATED"])[0, :]
    np.testing.assert_allclose(z_correlated, eta_desk, rtol=1E-3)


def test_hops_dynamics():
    """
    This is a test of one time point in the HOPS trajectory simulated for the dimer
    of dimers
    """
    hops.propagate(t_max, t_step)
    path = os.path.realpath(__file__)
    path = (
        path[: -len("test_dimer_of_dimers.py")]
        + "dimer_of_dimers/traj_dimer_of_dimers.npy"
    )
    traj_dod = np.load(path)
    np.testing.assert_allclose(hops.storage.data['psi_traj'][25], traj_dod[25], rtol=1E-5)


def test_hops_adaptive_dynamics_full():
    """
    This is a test of the adaptive dynamics algorithm when threshold is small enough
    that all states and hierarchy are included.
    """
    
    
    hops_ah = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
    )
    hops_ah.make_adaptive(1e-15, 0)
    hops_ah.initialize(psi_0)
    hops_ah.propagate(t_max, t_step)
    
    np.testing.assert_allclose(hops.storage.data['psi_traj'][33], hops_ah.storage.data['psi_traj'][33])
    
    
    hops_as = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
    )
    hops_as.make_adaptive(0, 1e-100)
    hops_as.initialize(psi_0)
    hops_as.propagate(t_max, t_step)
    
    np.testing.assert_allclose(hops.storage.data['psi_traj'][33], hops_as.storage.data['psi_traj'][33])
    
    
    hops_a = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
    )
    hops_a.make_adaptive(1e-15, 1e-100)
    hops_a.initialize(psi_0)
    hops_a.propagate(t_max, t_step)

    np.testing.assert_allclose(hops.storage.data['psi_traj'][33], hops_a.storage.data['psi_traj'][33])
    

def test_hops_adaptive_dynamics_partial():
    """
    This is a test of the adaptive dynamics algorithm when threshold is large
    enough that some states and auxiliaries are missing.
    """
    # Test adaptive hierarchy
    # =======================
    hops_ah = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
    )
    hops_ah.make_adaptive(1e-3, 0)
    hops_ah.initialize(psi_0)
    hops_ah.propagate(t_max, t_step)

    # Construct permutation for full-->reduced basis
    # ----------------------------------------------
    P2_permute, P2_permute2 = construct_permute_matrix(
        hops_ah.auxiliary_list, hops_ah.state_list, hops
    )

    # Permute super operators from hops into reduced basis
    # ----------------------------------------------------
    K0 = _permute_aux_by_matrix(hops.basis.eom.K2_k, P2_permute2)
    Kp1 = _permute_aux_by_matrix(hops.basis.eom.K2_kp1, P2_permute)
    Km1 = _permute_aux_by_matrix(hops.basis.eom.K2_km1, P2_permute)
    Zp1 = [
        _permute_aux_by_matrix(hops.basis.eom.Z2_kp1[index_l2], P2_permute2)
        for index_l2 in hops_ah.basis.mode.list_absindex_L2
    ]

    # Compare reduced hops to adhops super operators
    # ----------------------------------------------
    assert (Kp1.todense() == hops_ah.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops_ah.basis.eom.K2_km1.todense()).all()
    assert (K0.todense() == hops_ah.basis.eom.K2_k.todense()).all()
    assert np.all(
        [
            np.all(Z2_kp1_hops.todense() == Z2_kp1_adhops.todense())
            for (Z2_kp1_hops, Z2_kp1_adhops) in zip(Zp1, hops_ah.basis.eom.Z2_kp1)
        ]
    )

    # Test adaptive system
    # ====================
    hops_ah = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
    )
    hops_ah.make_adaptive(0, 1e-3)
    hops_ah.initialize(psi_0)
    hops_ah.propagate(t_max, t_step)

    # Construct permutation for full-->reduced basis
    # ----------------------------------------------
    P2_permute, P2_permute2 = construct_permute_matrix(
        hops_ah.auxiliary_list, hops_ah.state_list, hops
    )

    # Permute super operators from hops into reduced basis
    # ----------------------------------------------------
    K0 = _permute_aux_by_matrix(hops.basis.eom.K2_k, P2_permute2)
    Kp1 = _permute_aux_by_matrix(hops.basis.eom.K2_kp1, P2_permute)
    Km1 = _permute_aux_by_matrix(hops.basis.eom.K2_km1, P2_permute)
    Zp1 = [
        _permute_aux_by_matrix(hops.basis.eom.Z2_kp1[index_l2], P2_permute2)
        for index_l2 in hops_ah.basis.mode.list_absindex_L2
    ]

    # Compare reduced hops to adhops super operators
    # ----------------------------------------------
    assert (Kp1.todense() == hops_ah.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops_ah.basis.eom.K2_km1.todense()).all()
    assert (K0.todense() == hops_ah.basis.eom.K2_k.todense()).all()
    assert np.all(
        [
            np.all(Z2_kp1_hops.todense() == Z2_kp1_adhops.todense())
            for (Z2_kp1_hops, Z2_kp1_adhops) in zip(Zp1, hops_ah.basis.eom.Z2_kp1)
        ]
    )

    # Test adaptive system and adaptive hierarchy
    # ===========================================
    hops_ah = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": 2},
        eom_param=eom_param,
    )
    hops_ah.make_adaptive(1e-3, 1e-3)
    hops_ah.initialize(psi_0)
    hops_ah.propagate(t_max, t_step)

    # Construct permutation for full-->reduced basis
    # ----------------------------------------------
    P2_permute, P2_permute2 = construct_permute_matrix(
        hops_ah.auxiliary_list, hops_ah.state_list, hops
    )

    # Permute super operators from hops into reduced basis
    # ----------------------------------------------------
    K0 = _permute_aux_by_matrix(hops.basis.eom.K2_k, P2_permute2)
    Kp1 = _permute_aux_by_matrix(hops.basis.eom.K2_kp1, P2_permute)
    Km1 = _permute_aux_by_matrix(hops.basis.eom.K2_km1, P2_permute)
    Zp1 = [
        _permute_aux_by_matrix(hops.basis.eom.Z2_kp1[index_l2], P2_permute2)
        for index_l2 in hops_ah.basis.mode.list_absindex_L2
    ]

    # Compare reduced hops to adhops super operators
    # ----------------------------------------------
    assert (Kp1.todense() == hops_ah.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops_ah.basis.eom.K2_km1.todense()).all()
    assert (K0.todense() == hops_ah.basis.eom.K2_k.todense()).all()
    assert np.all(
        [
            np.all(Z2_kp1_hops.todense() == Z2_kp1_adhops.todense())
            for (Z2_kp1_hops, Z2_kp1_adhops) in zip(Zp1, hops_ah.basis.eom.Z2_kp1)
        ]
    )


