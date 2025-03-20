import numpy as np
import scipy as sp
from mesohops.dynamics.eom_functions import (
    calc_LT_corr,
    calc_LT_corr_to_norm_corr,
    calc_LT_corr_linear
)
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp
from mesohops.util.physical_constants import hbar

__title__ = "Test of hops_eom.py"
__author__ = "J. K. Lynd"
__version__ = "1.3"
__date__ = "3/2/2023"

# Manual HOPS EoM helper functions (the "by-hand" solution)

def dsystem_dt_linear_manual(phi_t, list_state, list_aux, H2_ham, noise_t_pre,
                             list_modes, list_l_op, self_interaction = True,
                             downward_connection = True, upward_connection = True):
    """
    Helper function that takes in and returns dense array versions of all HOPS
    objects and the ensuing derivative for the linear equation of motion.

    Parameters
    ----------
    1. phi_t : array(complex)
               The full wavefunction, length n_aux*n_state. The first n_state entries
               must represent the physical wavefunction.
    2. list_state : array(int)
                    The state list, length n_state
    3. list_aux : array(array(int))
                  The list of auxiliary vectors (length n_mode), length n_aux
    4. H2_ham : array(complex)
                The system Hamiltonian, size (n_state, n_state)
    5. noise_t_pre : array(complex)
                     The noise at the current time for each L-operator prior to
                     conjugation, length n_mode
    6. list_modes : list(list(complex), list(complex), list(int))
                    The list of (g, w, absolute index) pairs for each mode in the
                    basis, made up of 3 lists each with length n_mode
    7. list_l_op : list(array(complex))
                   The list of system-bath coupling L-operators (size (n_state,
                   n_state)), length n_mode
    8. self_interaction : bool
                          Whether to calculate the self-interaction term of the HOPS
                          EoM
    9. downward_connection : bool
                             Whether to calculate the portion of the HOPS EoM
                             stemming from downward connections.
    10. upward_connection : bool
                            Whether to calculate the portion of the HOPS EoM
                            stemming from upward connections.

    Returns
    -------
    """
    noise_t = np.conj(noise_t_pre)
    n_state = len(list_state)
    n_aux = len(list_aux)
    P2_reshape = phi_t.reshape([n_aux, n_state])
    D2_deriv = np.zeros_like(P2_reshape)
    list_g = list_modes[0]
    list_w = list_modes[1]
    I2 = np.eye(n_state)
    noise_matrix = np.sum([noise_t[n]*list_l_op[n] for n in range(len(list_l_op))],
                          axis=0)
    for aux_index in range(n_aux):
        psi_k = P2_reshape[aux_index]
        k_vec = list_aux[aux_index]

        # Get the self-interaction term
        if self_interaction:
            D2_deriv[aux_index] += (-1j*H2_ham - k_vec@list_w*I2 + noise_matrix)@psi_k

        # For each mode,
        for m in range(len(list_g)):
            e_m = np.zeros_like(k_vec)
            e_m[m] = 1
            k_vec_down = k_vec - e_m

            if downward_connection:
                # Check for downward connections...
                connections_down_m = np.where([np.allclose(k_vec_down, aux_k) for
                                                aux_k in list_aux])[0]
                if len(connections_down_m):
                    # And add the downward connection term.
                    d_ind = connections_down_m[0]
                    D2_deriv[aux_index] += k_vec[m]*list_w[m]*list_l_op[m]@P2_reshape[
                        d_ind]

            # Check for upward connections...
            if upward_connection:
                k_vec_up = k_vec+e_m
                connections_up_m = np.where([np.allclose(k_vec_up, aux_k) for
                                             aux_k in list_aux])[0]
                if len(connections_up_m):
                    # And add the upward connection term.
                    u_ind = connections_up_m[0]
                    D2_deriv[aux_index] -= (list_g[m]/list_w[m])*list_l_op[
                        m]@P2_reshape[u_ind]

    return D2_deriv/hbar

def dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux, H2_ham, noise_t_pre,
                                list_modes, list_l_op, list_noise_memory,
                                self_interaction = True, downward_connection = True,
                                upward_connection = True, type =
                                "normalized_nonlinear"):
    """
    Helper function that takes in and returns dense array versions of all HOPS
    objects and the ensuing derivative for the normalized nonlinear equation of motion.

    Parameters
    ----------
    1. phi_t : array(complex)
               The full wavefunction, length n_aux*n_state. The first n_state entries
               must represent the physical wavefunction.
    2. list_state : array(int)
                    The state list, length n_state
    3. list_aux : array(array(int))
                  The list of auxiliary vectors (length n_mode), length n_aux - the
                  physical wavefunction (k_vec = 0) must be the first entry
    4. H2_ham : array(complex)
                The system Hamiltonian, size (n_state, n_state)
    5. noise_t_pre : array(complex)
                     The noise at the current time for each L-operator prior to
                     conjugation, length n_mode
    6. list_modes : list(list(complex), list(complex), list(int))
                    The list of (g, w, absolute index) pairs for each mode in the
                    basis, made up of 3 lists each with length n_mode
    7. list_l_op : list(array(complex))
                   The list of system-bath coupling L-operators (size (n_state,
                   n_state)), length n_mode
    8. list_noise_memory : list(complex)
                           The noise memory drift term associated with each mode at
                           the current time, length n_mode
    8. self_interaction : bool
                          Whether to calculate the self-interaction term of the HOPS
                          EoM
    9. downward_connection : bool
                             Whether to calculate the portion of the HOPS EoM
                             stemming from downward connections.
    10. upward_connection : bool
                            Whether to calculate the portion of the HOPS EoM
                            stemming from upward connections.
    11. type : string
               The EoM type: "normalized_nonlinear", "nonlinear",
               and "nonlinear_absorption" are the current options.

    Returns
    -------
    """
    if type not in ["normalized_nonlinear", "nonlinear", "nonlinear_absorption"]:
        print("Type of nonlinear equation of motion not recognized!")
        assert False

    noise_t = np.conj(noise_t_pre)
    n_state = len(list_state)
    n_aux = len(list_aux)
    P2_reshape = phi_t.reshape([n_aux, n_state])
    D2_deriv = np.zeros_like(P2_reshape)
    list_g = list_modes[0]
    list_w = list_modes[1]
    list_abs_modes = list_modes[2]
    list_rel_modes = range(len(list_abs_modes))
    I2 = np.eye(n_state)
    noise_matrix = np.sum([(noise_t[n]+list_noise_memory[n])*list_l_op[n] for n in
                           range(len(list_l_op))],axis=0)
    psi_0 = P2_reshape[0]
    # Build a list of l-operator expectation values
    if type == "nonlinear_absorption":
        list_l_exp = [(np.conj(psi_0) @ L2 @ psi_0) / (1 + np.conj(psi_0) @ psi_0) for
                      L2 in list_l_op]
    else:
        list_l_exp = [(np.conj(psi_0)@L2@psi_0)/(np.conj(psi_0)@psi_0) for L2 in
                       list_l_op]

    # Build normalization correction factor:
    if type == "normalized_nonlinear":
        norm_corr = np.sum(np.array(list_l_exp)*(np.real(np.array(noise_t)+np.array(
            list_noise_memory))))
        for m in list_rel_modes:
            e_m = np.zeros_like(list_aux[0])
            e_m[list_abs_modes[m]] = 1
            markovian_aux = np.where([np.allclose(e_m, aux_k) for
                                           aux_k in list_aux])[0]
            if len(markovian_aux):
                mode_ratio_m = list_g[m]/list_w[m]
                psi_m = P2_reshape[markovian_aux[0]]
                norm_corr -= np.real(mode_ratio_m*np.conj(psi_0)@list_l_op[m]@psi_m)
                norm_corr += list_l_exp[m]*np.real(mode_ratio_m*np.conj(psi_0)@psi_m)
    else:
        norm_corr = 0

    for aux_index in range(n_aux):
        psi_k = P2_reshape[aux_index]
        k_vec = list_aux[aux_index]

        # Get the self-interaction term
        if self_interaction:
            k_vec_trunc = k_vec[np.array(list_abs_modes)]
            D2_deriv[aux_index] += (-1j*H2_ham - (k_vec_trunc@list_w + norm_corr)*I2 +
                                    noise_matrix)@psi_k

        # For each mode,
        for m in list_rel_modes:
            e_m = np.zeros_like(k_vec)
            e_m[list_abs_modes[m]] = 1
            k_vec_down = k_vec - e_m

            if downward_connection:
                # Check for downward connections...
                connections_down_m = np.where([np.allclose(k_vec_down, aux_k) for
                                                aux_k in list_aux])[0]
                if len(connections_down_m):
                    # And add the downward connection term.
                    d_ind = connections_down_m[0]
                    D2_deriv[aux_index] += k_vec[list_abs_modes[m]]*list_w[m]*list_l_op[
                        m]@P2_reshape[d_ind]

            # Check for upward connections...
            if upward_connection:
                k_vec_up = k_vec+e_m
                connections_up_m = np.where([np.allclose(k_vec_up, aux_k) for
                                             aux_k in list_aux])[0]
                if len(connections_up_m):
                    # And add the upward connection term.
                    u_ind = connections_up_m[0]
                    D2_deriv[aux_index] -= (list_g[m]/list_w[m])*(list_l_op[m] -
                                            list_l_exp[m]*I2)@P2_reshape[u_ind]

    return D2_deriv/hbar

# Helper function to get an aux vector from a HopsAux object:
def build_aux_vector(hops_aux):
    """
    Converts AuxVec objects into an indexing vector in the space of all correlation
    function modes.
    """
    aux_vec = np.zeros(len(hops_aux))
    hops_aux_arr = hops_aux.toarray()
    for i in range(len(hops_aux_arr)):
        aux_vec[hops_aux_arr[i,0]] = hops_aux_arr[i,1]
    return aux_vec

# Helper function to order noise properly:
def prepare_noise(list_l_by_mode, noise_at_t, active_modes):
    """
    Organizes the noise given a list of L-operators by mode and the associated list
    of modes currently in the basis.
    """
    noise_t_prepared = []
    for mode in active_modes:
        mode_index = list_l_by_mode[mode]
        if not noise_at_t[mode_index] in noise_t_prepared:
            noise_t_prepared.append(noise_at_t[mode_index])
        else:
            noise_t_prepared.append(0)
    return noise_t_prepared

# TEST PARAMETERS
# ===============
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 10.0,  # Units: fs
    "TAU": 0.5,  # Units: fs
}

loperator = np.zeros([3, 3, 3], dtype=np.float64)
loperator[0, 0, 0] = 1.0
loperator[0, 1, 1] = -1.0
loperator[1, 1, 1] = 1.0
loperator[1, 2, 2] = -1.0
loperator[2, 2, 2] = 1.0
loperator[2, 0, 0] = -1.0

base_ham = np.array([[1.0, 14.0, 8.0], [14.0, 0, 10.0], [8.0, 10.0, -1.0]],
                    dtype=np.float64)

list_l_by_mode_6mode = [0, 0, 1, 1, 2, 2]

sys_param_diagonal = {
    "HAMILTONIAN": base_ham,
    "GW_SYSBATH": [[12.0, 18.0], [7.0, 3.0], [11.0, 9.0], [6.0, 4.0], [10.0, 10.0],
                   [5.0, 5.0]],
    "L_HIER": [loperator[i] for i in list_l_by_mode_6mode],
    "L_NOISE1": [loperator[i] for i in list_l_by_mode_6mode],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[12.0, 18.0], [7.0, 3.0], [11.0, 9.0], [6.0, 4.0], [10.0, 10.0],
                     [5.0, 5.0]],
}

hier_param = {"MAXHIER": 3}

integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

psi_0 = [1.0 + 0.0 * 1j, 0.0 + 0.0 * 1j, 0.0 + 0.0 * 1j]

# Generalized L-operators: the first is an off-diagonal coupling with imaginary
# entries, the second is an acoustic mode that couples all nearest-neighbors,
# and the last is a Holstein-type coupling that couples the same environment to all
# site energies with different magnitudes.
loperator_general = np.zeros([3, 3, 3], dtype=np.complex128)
loperator_general[0,0,1] = 1j
loperator_general[0,1,0] = -1j
loperator_general[1,1,2] = 1
loperator_general[1,2,1] = 1
loperator_general[2,0,0] = 0.5
loperator_general[2,1,1] = 1
loperator_general[2,2,2] = -1

# everything is the same except the L-operators here.
sys_param_general = {
    "HAMILTONIAN": base_ham,
    "GW_SYSBATH": [[12.0, 18.0], [7.0, 3.0], [11.0, 9.0], [6.0, 4.0], [10.0, 10.0],
                   [5.0, 5.0]],
    "L_HIER": [loperator_general[i] for i in list_l_by_mode_6mode],
    "L_NOISE1": [loperator_general[i] for i in list_l_by_mode_6mode],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[12.0, 18.0], [7.0, 3.0], [11.0, 9.0], [6.0, 4.0], [10.0, 10.0],
                     [5.0, 5.0]],
}

def test_linear_eom():
    """
    Tests the linear HOPS EoM.
    """
    # Non-adaptive case: initialize the HOPS object
    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "LINEAR"}
    for sys_param in [sys_param_diagonal, sys_param_general]:
        hops = HOPS(
            sys_param,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
        hops.initialize(psi_0)

        # Test just after initialization
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in hops.basis.mode.list_absindex_mode]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]
        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_linear_manual(phi_t, list_state, list_aux, H2_ham,
                                                  noise_t_prepared, list_modes, list_l_op,
                                                  self_interaction=True,
                                                  downward_connection=True,
                                                  upward_connection=True).flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0]/hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

        # Time-evolve the trajectory
        hops.propagate(5.0, 1.0)
        # Test after time evolution
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in
                     hops.basis.mode.list_absindex_mode]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]
        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_linear_manual(phi_t, list_state, list_aux, H2_ham,
                                                  noise_t_prepared, list_modes, list_l_op,
                                                  self_interaction=True,
                                                  downward_connection=True,
                                                  upward_connection=True).flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0] / hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

def test_normalized_nonlinear_nonadaptive_eom():
    """
    Tests the normalized nonlinear HOPS EoM in the non-adaptive case.
    """
    # Non-adaptive case: initialize the HOPS object
    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}
    for sys_param in [sys_param_diagonal, sys_param_general]:
        hops = HOPS(
            sys_param,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
        hops.initialize(psi_0)

        # Test just after initialization
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in hops.basis.mode.list_absindex_mode]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]
        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state,
                                                                list_aux, H2_ham,
                                                                noise_t_prepared,
                                 list_modes, list_l_op, list_noise_memory,
                                 type = "normalized_nonlinear").flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0]/hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

        # Time-evolve the trajectory
        hops.propagate(5.0, 1.0)
        # Test after time evolution
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in
                     hops.basis.mode.list_absindex_mode]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]
        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux, H2_ham,
                                                     noise_t_prepared, list_modes,
                                                     list_l_op, list_noise_memory,
                                                     type = "normalized_nonlinear").flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0] / hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

def test_nonlinear_eom():
    """
    Tests the non-normalized nonlinear HOPS EoM in the non-adaptive case.
    """
    # Non-adaptive case: initialize the HOPS object
    for sys_param in [sys_param_diagonal, sys_param_general]:
        eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NONLINEAR"}
        hops = HOPS(
            sys_param,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
        hops.initialize(psi_0)

        # Test just after initialization
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in hops.basis.mode.list_absindex_mode]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]
        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux, H2_ham,
                                                     noise_t_prepared, list_modes,
                                                     list_l_op, list_noise_memory,
                                                     type = "nonlinear").flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0]/hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

        # Time-evolve the trajectory
        hops.propagate(5.0, 1.0)
        # Test after time evolution
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in
                     hops.basis.mode.list_absindex_mode]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]
        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux, H2_ham,
                                                     noise_t_prepared, list_modes,
                                                     list_l_op, list_noise_memory,
                                                     type = "nonlinear").flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0] / hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

def test_nonlinear_absorption_eom():
    """
    Tests the nonlinear absorption HOPS EoM in the non-adaptive case.
    """
    # Non-adaptive case: initialize the HOPS object
    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NONLINEAR ABSORPTION"}
    for sys_param in [sys_param_diagonal, sys_param_general]:
        hops = HOPS(
            sys_param,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
        hops.initialize(psi_0)

        # Test just after initialization
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in hops.basis.mode.list_absindex_mode]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]
        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux, H2_ham,
                                                     noise_t_prepared, list_modes,
                                                     list_l_op, list_noise_memory,
                                                     type = "nonlinear_absorption").flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0]/hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

        # Time-evolve the trajectory
        hops.propagate(5.0, 1.0)
        # Test after time evolution
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in
                     hops.basis.mode.list_absindex_mode]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]
        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux, H2_ham,
                                                     noise_t_prepared, list_modes,
                                                     list_l_op, list_noise_memory,
                                                     type = "nonlinear_absorption").flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0] / hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

# Create a linear chain model
linear_chain_test_ham = np.eye(101, k=1)*50 + np.eye(101, k=-1)*50
linear_chain_loperator = np.zeros([101, 101, 101], dtype=np.float64)

for i in range(101):
    linear_chain_loperator[i, i, i] = 1.0

linear_chain_list_lop = [linear_chain_loperator[i] for i in range(101)]

factor = 10
linear_chain_list_modes = [[20850.9+i*factor-2000j, 40+0j] for i in range (101)]

linear_chain_noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 1000.0,  # Units: fs
    "TAU": 0.5,  # Units: fs
}

linear_chain_sys_param = {
    "HAMILTONIAN": linear_chain_test_ham,
    "GW_SYSBATH": linear_chain_list_modes,
    "L_HIER": linear_chain_list_lop,
    "L_NOISE1": linear_chain_list_lop,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": linear_chain_list_modes,
}

linear_chain_psi_0 = np.zeros(101)
linear_chain_psi_0[50] = 1.0

def linear_chain_prepare_noise(noise_at_t,active_modes):
    """
    Ensures that the noise is in the right order so long as the L-operators of the
    associated system are site-projection operators. Note: this is not universal and
    depends on the assumption that a single system dictionary is instantiated for use
    by all of the HOPS trajectory objects in this testing file.
    """
    noise_t_prepared = []
    for mode in active_modes:
        mode_index = np.where(linear_chain_sys_param["L_HIER"][mode])[0][0]
        if not noise_at_t[mode_index] in noise_t_prepared:
            noise_t_prepared.append(noise_at_t[mode_index])
        else:
            noise_t_prepared.append(0)
    return noise_t_prepared

def test_eom_adaptive():
    """
    Tests the normalized nonlinear HOPS EoM in the adaptive case.
    """
    # Adaptive case: initialize the HOPS object
    for sys_param in [sys_param_diagonal]:
        eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}
        hops = HOPS(
            sys_param,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
        hops.make_adaptive(0.00001, 0.0025)
        hops.initialize(psi_0)

        # Test just after initialization
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        H2_ham_trunc = H2_ham[np.ix_(list_state, list_state)]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in
                     hops.basis.mode.list_absindex_mode]
        list_l_op_trunc = [l_op[np.ix_(list_state, list_state)] for l_op in list_l_op]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]

        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux,
                                                     H2_ham_trunc, noise_t_prepared,
                                                     list_modes, list_l_op_trunc,
                                                     list_noise_memory, type =
                                                     "normalized_nonlinear").flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0] / hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

        # Time-evolve the trajectory
        hops.propagate(5.0, 1.0)
        # Test after time evolution
        phi_t = hops.phi
        list_state = hops.state_list
        list_aux = [build_aux_vector(aux) for aux in hops.auxiliary_list]
        H2_ham = sys_param["HAMILTONIAN"]
        H2_ham_trunc = H2_ham[np.ix_(list_state, list_state)]
        noise_t = hops.noise1.get_noise([hops.t])
        noise_t = np.array([noise_t[0][0], noise_t[1][0], noise_t[2][0]])
        list_modes = [hops.basis.mode.g, hops.basis.mode.w, hops.basis.mode.list_absindex_mode]
        list_l_op = [sys_param["L_HIER"][m] for m in
                     hops.basis.mode.list_absindex_mode]
        list_l_op_trunc = [l_op[np.ix_(list_state, list_state)] for l_op in list_l_op]
        list_noise_memory = hops.z_mem[hops.basis.mode.list_absindex_mode]
        noise_t_prepared = prepare_noise(list_l_by_mode_6mode, noise_t,
                                         hops.basis.mode.list_absindex_mode)
        dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux,
                                                     H2_ham_trunc, noise_t_prepared,
                                                     list_modes, list_l_op_trunc,
                                                     list_noise_memory, type =
                                                     "normalized_nonlinear").flatten()
        dsystem_dt_test = hops.dsystem_dt(phi_t, list_noise_memory, noise_t,
                                          np.zeros_like(noise_t))[0] / hbar
        assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

    # Linear chain case
    linear_chain_hops = HOPS(
        linear_chain_sys_param,
        noise_param=linear_chain_noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    linear_chain_hops.make_adaptive(0.00001, 0.0025)
    linear_chain_hops.initialize(linear_chain_psi_0)

    # Test just after initialization
    phi_t = linear_chain_hops.phi
    list_state = linear_chain_hops.state_list
    list_aux = [build_aux_vector(aux) for aux in linear_chain_hops.auxiliary_list]
    H2_ham = linear_chain_sys_param["HAMILTONIAN"]
    H2_ham_trunc = H2_ham[np.ix_(list_state, list_state)]
    noise_t_base = linear_chain_hops.noise1.get_noise([linear_chain_hops.t])
    noise_t = np.array([noise_t_base[i][0] for i in range(101)])
    list_modes = [linear_chain_hops.basis.mode.g, linear_chain_hops.basis.mode.w,
                  linear_chain_hops.basis.mode.list_absindex_mode]
    list_l_op = [linear_chain_sys_param["L_HIER"][m] for m in
                 linear_chain_hops.basis.mode.list_absindex_mode]
    list_l_op_trunc = [l_op[np.ix_(list_state, list_state)] for l_op in list_l_op]
    list_noise_memory = linear_chain_hops.z_mem[linear_chain_hops.basis.mode.list_absindex_mode]
    noise_t_prepared = linear_chain_prepare_noise(noise_t, linear_chain_hops.basis.mode.list_absindex_mode)
    dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux,
                                                 H2_ham_trunc, noise_t_prepared,
                                                 list_modes, list_l_op_trunc,
                                                 list_noise_memory, type=
                                                 "normalized_nonlinear").flatten()
    dsystem_dt_test = linear_chain_hops.dsystem_dt(phi_t, linear_chain_hops.z_mem, noise_t,
                                      np.zeros_like(noise_t))[0] / hbar
    assert np.allclose(dsystem_dt_test, dsystem_dt_ref)

    # Time-evolve the trajectory
    linear_chain_hops.propagate(25.0, 1.0)
    # Test after time evolution
    phi_t = linear_chain_hops.phi
    list_state = linear_chain_hops.state_list
    list_aux = [build_aux_vector(aux) for aux in linear_chain_hops.auxiliary_list]
    H2_ham = linear_chain_sys_param["HAMILTONIAN"]
    H2_ham_trunc = H2_ham[np.ix_(list_state, list_state)]
    noise_t_base = linear_chain_hops.noise1.get_noise([linear_chain_hops.t])
    noise_t = np.array([noise_t_base[i][0] for i in range(101)])
    list_modes = [linear_chain_hops.basis.mode.g, linear_chain_hops.basis.mode.w,
                  linear_chain_hops.basis.mode.list_absindex_mode]
    list_l_op = [linear_chain_sys_param["L_HIER"][m] for m in
                 linear_chain_hops.basis.mode.list_absindex_mode]
    list_l_op_trunc = [l_op[np.ix_(list_state, list_state)] for l_op in list_l_op]
    list_noise_memory = linear_chain_hops.z_mem[
        linear_chain_hops.basis.mode.list_absindex_mode]
    noise_t_prepared = linear_chain_prepare_noise(noise_t,
                                                  linear_chain_hops.basis.mode.list_absindex_mode)
    dsystem_dt_ref = dsystem_dt_nonlinear_manual(phi_t, list_state, list_aux,
                                                 H2_ham_trunc, noise_t_prepared,
                                                 list_modes, list_l_op_trunc,
                                                 list_noise_memory, type=
                                                 "normalized_nonlinear").flatten()
    dsystem_dt_test = \
    linear_chain_hops.dsystem_dt(phi_t, linear_chain_hops.z_mem, noise_t,
                                 np.zeros_like(noise_t))[0] / hbar
    assert np.allclose(dsystem_dt_test, dsystem_dt_ref)