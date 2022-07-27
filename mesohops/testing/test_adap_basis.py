import numpy as np
from scipy import sparse
from pyhops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from pyhops.util.physical_constants import hbar, kB
from pyhops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp


def const_gw_sysbath(nsite, e_lambda, gamma, temp, gamma_mark):
    """
    Generating the thermal objects for system-bath coupling.

    PARAMETERS
    ----------
    1. nsite : int
               number of states in the system.
    2. e_lambda : float
                  system's reorganization-energy
    3. gamma : float
               reorganization timescale
    4. temp : float
              temperature of the environment.
    5. gamma_mark : float
                    reorganization timescale of the Markovian mode.

    RETURNS
    -------
    1. gw_sysbath : list
                    The list that describes the system-bath coupling, defined
                    by a prefactor describing coupling-strength and
                    exponential scale-factor describing memory-effects
    2. lop_list : list
                  The diagonal Lindblad operator--a site-projection operator.
    """
    # define exponential parameters
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)
    # define parameter lists
    gw_sysbath = []
    lop_list = []
    for i in range(nsite):
        loperator = sparse.coo_matrix(([1], ([i], [i])), shape=(nsite, nsite))
        gw_sysbath.append([g_0, w_0])
        lop_list.append(loperator)
        gw_sysbath.append([-1j * np.imag(g_0), gamma_mark])
        lop_list.append(loperator)
    return gw_sysbath, lop_list


def linear_chain(nsite, sb_params, V, maxhier=3, seed=None):
    """
    Build a model system for HOPS dynamics.

    PARAMETERS
    ----------
    1. nsite : int
               number of states in the system.
    2. sb_params : list
                   parameters that will be used to construct the
                   bath-correlation functions, the temperature, and the
                   Markovian mode's rate of decay.
    3. V : float
           electronic coupling
    4. max_hier : int
                   depth of the hierarchy.
    5. seed : int
              seed value for random number generator

    RETURNS
    -------
    1. HOPS :  HopsTrajectory object
               object describing the dynamical simulation.
    """
    # Noise Dictionary
    noise_param = {
        "SEED": seed,
        "MODEL": "FFT_FILTER",
        "TLEN": 2100.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    # System Dictionary
    e_lambda = sb_params[0][0]
    gamma = sb_params[0][1]
    temp = sb_params[1]
    gamma_mark = sb_params[2]
    gw_sysbath, lop_list = const_gw_sysbath(nsite, e_lambda, gamma, temp, gamma_mark)
    hs = (np.eye(nsite, k=1) + np.eye(nsite, k=-1)) * V
    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),
        "GW_SYSBATH": gw_sysbath,
        "L_HIER": lop_list,
        "L_NOISE1": lop_list,
        "ALPHA_NOISE1": bcf_exp,
        "PARAM_NOISE1": gw_sysbath,
    }
    # eom dictionary
    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}
    # hierarchy dictionary
    hierarchy_param = {
        "MAXHIER": maxhier,
        "STATIC_FILTERS": [["Markovian", [False, True] * nsite]],
    }
    return HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hierarchy_param,
        eom_param=eom_param,
    )


def test_adap_hier():
    """
    Test to check adaptive hierarchy scheme is within accepted error
    """
    #     ====================================
    #     =   TESTING HIERARCHY DERIVATIVE   =
    #     ====================================
    e_lambda = 40.0
    gamma = 50.0
    temp = 140.0
    gamma_mark = 500
    V = 40
    maxhier = 4
    nsite = 3
    i_init = int(np.round(nsite / 2))  # initial occupation site
    seed = 100100
    sb_params = [(e_lambda, gamma), temp, gamma_mark]
    t_max = 200.0
    t_step = 4.0
    delta = 0.0005
    ## Phi(0)
    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[i_init] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    hops = linear_chain(nsite, sb_params, V, maxhier, int(seed))
    hops.initialize(psi_0)
    hops2 = linear_chain(nsite, sb_params, V, maxhier, int(seed))
    hops2.make_adaptive(delta, 0)
    hops2.initialize(psi_0)
    error_dnorm_comp = []

    for t in np.arange(0, t_max, t_step):
        hops2.propagate(t_step, t_step)
        # Match Aux Indices
        # -----------------
        list_aux_index = [
            hops.basis.hierarchy.auxiliary_list.index(aux)
            for aux in hops2.basis.hierarchy.auxiliary_list
        ]
        # Determine Derivative
        # --------------------
        D1_adap = (
            hops2.dsystem_dt(
                hops2.phi,
                hops2.z_mem,  #hops2.storage.z_mem,
                hops2.noise1.get_noise([t])[:, 0],
                hops2.noise2.get_noise([t])[:, 0],
            )[0]
            / hbar
        )
        phi_adap_comp = np.zeros(hops.n_state * hops.n_hier, dtype=np.complex128)
        P2_adap_comp = phi_adap_comp.view().reshape(
            [hops.n_state, hops.n_hier], order="F"
        )
        P2_adap_comp[:, list_aux_index] = hops2.phi.view().reshape(
            [hops2.n_state, hops2.n_hier], order="F"
        )
        D1_comp = (
            hops.dsystem_dt(
                phi_adap_comp,
                hops2.z_mem,  # hops2.storage.z_mem,
                hops2.noise1.get_noise([t])[:, 0],
                hops2.noise2.get_noise([t])[:, 0],
            )[0]
            / hbar
        )
        # Map Derivatives to the same space
        # ---------------------------------
        D1_full_adap = np.zeros(hops.n_state * hops.n_hier, dtype=np.complex128)
        D2_full_adap = D1_full_adap.view().reshape(
            [hops.n_state, hops.n_hier], order="F"
        )
        D2_adap = D1_adap.view().reshape([hops.n_state, hops2.n_hier], order="F")
        D2_full_adap[:, list_aux_index] = D2_adap

        # Calculate Error
        # ---------------
        error_dnorm_comp.append(np.linalg.norm(D1_comp - D1_full_adap))

        # Note: Use this error when delta is multiplied by norm of phi
        # error_dnorm_comp.append((np.linalg.norm(D1_comp - D1_full_adap)/ np.linalg.norm(hops2.phi)))

    for error in error_dnorm_comp:
        assert error <= delta


def test_adap_state():
    """
    Test to check adaptive state scheme is within accepted error
    """
    #     ================================
    #     =   TESTING STATE DERIVATIVE   =
    #     ================================
    e_lambda = 60.0
    gamma = 50.0
    temp = 140.0
    gamma_mark = 500
    V = 20
    maxhier = 2
    nsite = 10
    i_init = int(np.round(nsite / 2))  # initial occupation site
    seed = 100100
    sb_params = [(e_lambda, gamma), temp, gamma_mark]
    t_max = 100.0
    t_step = 4.0
    delta = 0.0005
    ## Phi(0)
    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[i_init] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    hops = linear_chain(nsite, sb_params, V, maxhier, int(seed))
    hops.initialize(psi_0)
    hops2 = linear_chain(nsite, sb_params, V, maxhier, int(seed))
    hops2.make_adaptive(0, delta)
    hops2.initialize(psi_0)
    error_dnorm_comp = []

    for t in np.arange(0, t_max, t_step):
        hops2.propagate(t_step, t_step)
        # Match Aux Indices
        # -----------------
        list_state_index = hops2.basis.system.state_list
        # Determine Derivative
        # --------------------
        D1_adap = (
            hops2.dsystem_dt(
                hops2.phi,
                hops2.z_mem, #hops2.storage.z_mem,
                hops2.noise1.get_noise([t])[:, 0],
                hops2.noise2.get_noise([t])[:, 0],
            )[0]
            / hbar
        )
        phi_adap_comp = np.zeros(hops.n_state * hops.n_hier, dtype=np.complex128)
        P2_adap_comp = phi_adap_comp.view().reshape(
            [hops.n_state, hops.n_hier], order="F"
        )
        P2_adap_comp[list_state_index, :] = hops2.phi.view().reshape(
            [hops2.n_state, hops2.n_hier], order="F"
        )
        D1_comp = (
            hops.dsystem_dt(
                phi_adap_comp,
                hops2.z_mem, #hops2.storage.z_mem,
                hops2.noise1.get_noise([t])[:, 0],
                hops2.noise2.get_noise([t])[:, 0],
            )[0]
            / hbar
        )
        # Map Derivatives to the same space
        # ---------------------------------
        D1_full_adap = np.zeros(hops.n_state * hops.n_hier, dtype=np.complex128)
        D2_full_adap = D1_full_adap.view().reshape(
            [hops.n_state, hops.n_hier], order="F"
        )
        D2_adap = D1_adap.view().reshape([hops2.n_state, hops2.n_hier], order="F")
        D2_full_adap[list_state_index, :] = D2_adap
        # Map Psi to the same space
        # -------------------------
        # Calculate Error
        # ---------------
        error_dnorm_comp.append(np.linalg.norm(D1_comp - D1_full_adap))

        # Note: Use this error when delta is multiplied by norm of phi
        # error_dnorm_comp.append((np.linalg.norm(D1_comp - D1_full_adap)/ np.linalg.norm(hops2.phi)))

    for error in error_dnorm_comp:
        assert error <= delta


def test_adap_hier_state():
    """
    Test to check adaptive basis (Hierarchy + State) scheme is within accepted error
     """
    #     ===========================================
    #     =   TESTING HIERARCHY +STATE DERIVATIVE   =
    #     ===========================================
    e_lambda = 50.0
    gamma = 50.0
    temp = 140.0
    gamma_mark = 500
    V = 10
    maxhier = 3
    nsite = 6
    i_init = int(np.round(nsite / 2))  # initial occupation site
    seed = 100100
    sb_params = [(e_lambda, gamma), temp, gamma_mark]
    t_max = 80.0
    t_step = 4.0
    delta = 0.0005
    ## Phi(0)
    psi_0 = np.array([0.0] * nsite, dtype=np.complex)
    psi_0[i_init] = 1.0
    psi_0 = psi_0 / np.linalg.norm(psi_0)
    hops = linear_chain(nsite, sb_params, V, maxhier, int(seed))
    hops.initialize(psi_0)
    hops2 = linear_chain(nsite, sb_params, V, maxhier, int(seed))
    hops2.make_adaptive(delta, delta)
    hops2.initialize(psi_0)
    error_dnorm_comp = []

    for t in np.arange(0, t_max, t_step):
        hops2.propagate(t_step, t_step)
        # Match Aux Indices
        # -----------------
        list_state_index = hops2.basis.system.state_list
        list_aux_index = [
            hops.basis.hierarchy.auxiliary_list.index(aux)
            for aux in hops2.basis.hierarchy.auxiliary_list
        ]
        # Determine Derivative
        # --------------------
        D1_adap = (
            hops2.dsystem_dt(
                hops2.phi,
                hops2.z_mem, #hops2.storage.z_mem,
                hops2.noise1.get_noise([t])[:, 0],
                hops2.noise2.get_noise([t])[:, 0],
            )[0]
            / hbar
        )
        phi_adap_comp = np.zeros(hops.n_state * hops.n_hier, dtype=np.complex128)
        P2_adap_comp = phi_adap_comp.view().reshape(
            [hops.n_state, hops.n_hier], order="F"
        )
        P2_adap_comp[
            np.ix_(list_state_index, list_aux_index)
        ] = hops2.phi.view().reshape([hops2.n_state, hops2.n_hier], order="F")[
            np.ix_(range(hops2.n_state), range(hops2.n_hier))
        ]
        D1_comp = (
            hops.dsystem_dt(
                phi_adap_comp,
                hops2.z_mem, #hops2.storage.z_mem,
                hops2.noise1.get_noise([t])[:, 0],
                hops2.noise2.get_noise([t])[:, 0],
            )[0]
            / hbar
        )
        # Map Derivatives to the same space
        # ---------------------------------
        D1_full_adap = np.zeros(hops.n_state * hops.n_hier, dtype=np.complex128)
        D2_full_adap = D1_full_adap.view().reshape(
            [hops.n_state, hops.n_hier], order="F"
        )
        D2_adap = D1_adap.view().reshape([hops2.n_state, hops2.n_hier], order="F")
        D2_full_adap[np.ix_(list_state_index, list_aux_index)] = D2_adap[
            np.ix_(range(hops2.n_state), range(hops2.n_hier))
        ]
        # Map Psi to the same space
        # -------------------------
        # Calculate Error
        # ---------------
        error_dnorm_comp.append(np.linalg.norm(D1_comp - D1_full_adap))

        # Note: Use this error when delta is multiplied by norm of phi
        # error_dnorm_comp.append((np.linalg.norm(D1_comp - D1_full_adap)/ np.linalg.norm(hops2.phi)))

    for error in error_dnorm_comp:
        assert error <= (delta + delta)
