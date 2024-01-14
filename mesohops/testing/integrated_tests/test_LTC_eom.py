import numpy as np
import scipy as sp
from mesohops.dynamics.eom_functions import (
    calc_LT_corr,
    calc_LT_corr_to_norm_corr,
    calc_LT_corr_linear
)
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp
from mesohops.dynamics.hops_aux import AuxiliaryVector
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS

__title__ = "Test of low-temperature correction"
__author__ = "J. K. Lynd"
__version__ = "1.4"

noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 50.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
}

loperator = np.zeros([3, 3, 3], dtype=np.float64)
loperator[0, 0, 0] = 1.0
loperator[1, 1, 1] = 1.0
loperator[2, 2, 2] = 1.0


# Set the Hamiltonian to have no coupling so that we can ensure that the adaptive
# basis will contain only one state.
sys_param_ltc_modes_match_l_op = {
    "HAMILTONIAN": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [10.0, 10.0],
                   [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1], loperator[2],
               loperator[2]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1], loperator[2],
               loperator[2], loperator[0], loperator[1], loperator[2]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [10.0, 10.0],
                    [5.0, 5.0], [250.0, 1000.0], [250.0, 2000.0], [250.0, 3000.0]],
    "PARAM_LT_CORR": [250.0/1000.0, 250.0/2000.0, 250.0/3000.0],
    "L_LT_CORR": [loperator[0], loperator[1], loperator[2]]
}

sys_param_no_ltc_modes = {
    "HAMILTONIAN": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [10.0, 10.0],
                   [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1], loperator[2],
               loperator[2]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1], loperator[2],
               loperator[2], loperator[0], loperator[1], loperator[2]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [10.0, 10.0],
                    [5.0, 5.0], [250.0, 1000.0], [250.0, 2000.0], [250.0, 3000.0]],
}

hier_param = {"MAXHIER": 1}

integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR':'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS':None
    }

t_max = 20.0
t_step = 2.0

def test_LTC_linear_eom():
    """
    Tests that the low-temperature correction to the linear equation of motion
    reproduces the correct effect on the full hierarchy and state bases.
    """
    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "LINEAR"}
    phi_0 = np.array([0.7071, 0.5773, 0.4082, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5])
    psi_0 = phi_0[:3]
    hops_ltc_modes_match_l_op = HOPS(
            sys_param_ltc_modes_match_l_op,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
    hops_ltc_modes_match_l_op.initialize(psi_0)
    hops_ltc_modes_match_l_op.phi = phi_0
    integration_var_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.integration_var(hops_ltc_modes_match_l_op.phi,
                                      hops_ltc_modes_match_l_op.z_mem,
                                      hops_ltc_modes_match_l_op.t,
                                      hops_ltc_modes_match_l_op.noise1,
                                      hops_ltc_modes_match_l_op.noise2,
                                      2.0, hops_ltc_modes_match_l_op.storage)
    dsystem_dt_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.dsystem_dt(
                integration_var_ltc_modes_match_l_op['phi'],
                integration_var_ltc_modes_match_l_op['z_mem'],
                integration_var_ltc_modes_match_l_op['z_rnd'][:,0],
                integration_var_ltc_modes_match_l_op['z_rnd2'][:,0],
        )[0]

    hops_no_ltc_modes = HOPS(
            sys_param_no_ltc_modes,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
    hops_no_ltc_modes.initialize(psi_0)
    hops_no_ltc_modes.phi = phi_0
    integration_var_no_ltc_modes = hops_no_ltc_modes.integration_var(hops_no_ltc_modes.phi,
                                      hops_no_ltc_modes.z_mem,
                                      hops_no_ltc_modes.t,
                                      hops_no_ltc_modes.noise1,
                                      hops_no_ltc_modes.noise2,
                                      2.0, hops_no_ltc_modes.storage)
    dsystem_dt_no_ltc_modes = hops_no_ltc_modes.dsystem_dt(
                integration_var_no_ltc_modes['phi'],
                integration_var_no_ltc_modes['z_mem'],
                integration_var_no_ltc_modes['z_rnd'][:,0],
                integration_var_no_ltc_modes['z_rnd2'][:,0],
        )[0]

    list_lt_corr_coeff = [250.0/1000.0, 250.0/2000.0, 250.0/3000.0]
    list_l_op = [loperator[0], loperator[1], loperator[2]]
    list_l_op_sq = np.array([l_op@l_op for l_op in list_l_op])

    C2_LT_corr_total = np.zeros([len(phi_0), len(phi_0)])
    C2_LT_corr_total[:len(psi_0), :len(psi_0)] = calc_LT_corr_linear(
        list_lt_corr_coeff, list_l_op_sq)
    C1_LT_corr_to_deriv = C2_LT_corr_total@phi_0

    assert(np.allclose(C1_LT_corr_to_deriv,
                       dsystem_dt_ltc_modes_match_l_op-dsystem_dt_no_ltc_modes))


def test_LTC_non_adaptive_nonlinear_norm_eom():
    """
    Tests that the low-temperature correction to the normalized nonlinear equation of
    motion reproduces the correct effect on the full hierarchy and state bases.
    """
    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}
    phi_0 = np.array([0.7071, 0.5773, 0.4082, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5])
    psi_0 = phi_0[:3]
    hops_ltc_modes_match_l_op = HOPS(
            sys_param_ltc_modes_match_l_op,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
    hops_ltc_modes_match_l_op.initialize(psi_0)
    hops_ltc_modes_match_l_op.phi = phi_0
    integration_var_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.integration_var(hops_ltc_modes_match_l_op.phi,
                                      hops_ltc_modes_match_l_op.z_mem,
                                      hops_ltc_modes_match_l_op.t,
                                      hops_ltc_modes_match_l_op.noise1,
                                      hops_ltc_modes_match_l_op.noise2,
                                      2.0, hops_ltc_modes_match_l_op.storage)
    dsystem_dt_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.dsystem_dt(
                integration_var_ltc_modes_match_l_op['phi'],
                integration_var_ltc_modes_match_l_op['z_mem'],
                integration_var_ltc_modes_match_l_op['z_rnd'][:,0],
                integration_var_ltc_modes_match_l_op['z_rnd2'][:,0],
        )[0]

    hops_no_ltc_modes = HOPS(
            sys_param_no_ltc_modes,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
    hops_no_ltc_modes.initialize(psi_0)
    hops_no_ltc_modes.phi = phi_0
    integration_var_no_ltc_modes = hops_no_ltc_modes.integration_var(hops_no_ltc_modes.phi,
                                      hops_no_ltc_modes.z_mem,
                                      hops_no_ltc_modes.t,
                                      hops_no_ltc_modes.noise1,
                                      hops_no_ltc_modes.noise2,
                                      2.0, hops_no_ltc_modes.storage)
    dsystem_dt_no_ltc_modes = hops_no_ltc_modes.dsystem_dt(
                integration_var_no_ltc_modes['phi'],
                integration_var_no_ltc_modes['z_mem'],
                integration_var_no_ltc_modes['z_rnd'][:,0],
                integration_var_no_ltc_modes['z_rnd2'][:,0],
        )[0]

    list_lt_corr_coeff = [250.0/1000.0, 250.0/2000.0, 250.0/3000.0]
    list_l_op = [loperator[0], loperator[1], loperator[2]]
    list_l_op_exp = [np.conj(psi_0)@l_op@psi_0/(np.conj(psi_0)@psi_0) for l_op in
                     list_l_op]
    list_l_op = [sp.sparse.csr_matrix(l_op) for l_op in list_l_op]
    list_l_op_sq = [l_op @ l_op for l_op in list_l_op]
    list_l_op_exp_sq = [np.conj(psi_0)@l_op@l_op@psi_0/(np.conj(psi_0)@psi_0) for
                        l_op in list_l_op]

    C2_LT_corr_physical, C2_LT_corr_hier = calc_LT_corr(list_lt_corr_coeff, list_l_op, list_l_op_exp,
                                       list_l_op_sq)
    G2_norm_corr_correction = calc_LT_corr_to_norm_corr(list_lt_corr_coeff,
                                                        list_l_op_exp, list_l_op_exp_sq)

    C2_LT_corr_total = np.eye(len(phi_0))*(-1*G2_norm_corr_correction)
    C2_LT_corr_total[:len(psi_0), :len(psi_0)] += C2_LT_corr_physical
    for i in range(len(phi_0)//len(psi_0)):
        i_up = len(psi_0)*(i+1)
        i_down = len(psi_0)*i
        C2_LT_corr_total[i_down:i_up, i_down:i_up] += C2_LT_corr_hier

    C1_LT_corr_to_deriv = C2_LT_corr_total@phi_0
    assert(np.allclose(C1_LT_corr_to_deriv,
                       dsystem_dt_ltc_modes_match_l_op-dsystem_dt_no_ltc_modes))


def test_LTC_nonlinear_eom():
    """
    Tests that the low-temperature correction to the nonlinear equation of
    motion reproduces the correct effect on the full hierarchy and state bases.
    """
    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NONLINEAR"}
    phi_0 = np.array([0.7071, 0.5773, 0.4082, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5])
    psi_0 = phi_0[:3]
    hops_ltc_modes_match_l_op = HOPS(
            sys_param_ltc_modes_match_l_op,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
    hops_ltc_modes_match_l_op.initialize(psi_0)
    hops_ltc_modes_match_l_op.phi = phi_0
    integration_var_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.integration_var(hops_ltc_modes_match_l_op.phi,
                                      hops_ltc_modes_match_l_op.z_mem,
                                      hops_ltc_modes_match_l_op.t,
                                      hops_ltc_modes_match_l_op.noise1,
                                      hops_ltc_modes_match_l_op.noise2,
                                      2.0, hops_ltc_modes_match_l_op.storage)
    dsystem_dt_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.dsystem_dt(
                integration_var_ltc_modes_match_l_op['phi'],
                integration_var_ltc_modes_match_l_op['z_mem'],
                integration_var_ltc_modes_match_l_op['z_rnd'][:,0],
                integration_var_ltc_modes_match_l_op['z_rnd2'][:,0],
        )[0]

    hops_no_ltc_modes = HOPS(
            sys_param_no_ltc_modes,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
    hops_no_ltc_modes.initialize(psi_0)
    hops_no_ltc_modes.phi = phi_0
    integration_var_no_ltc_modes = hops_no_ltc_modes.integration_var(hops_no_ltc_modes.phi,
                                      hops_no_ltc_modes.z_mem,
                                      hops_no_ltc_modes.t,
                                      hops_no_ltc_modes.noise1,
                                      hops_no_ltc_modes.noise2,
                                      2.0, hops_no_ltc_modes.storage)
    dsystem_dt_no_ltc_modes = hops_no_ltc_modes.dsystem_dt(
                integration_var_no_ltc_modes['phi'],
                integration_var_no_ltc_modes['z_mem'],
                integration_var_no_ltc_modes['z_rnd'][:,0],
                integration_var_no_ltc_modes['z_rnd2'][:,0],
        )[0]

    list_lt_corr_coeff = [250.0/1000.0, 250.0/2000.0, 250.0/3000.0]
    list_l_op = [loperator[0], loperator[1], loperator[2]]
    list_l_op_exp = [np.conj(psi_0)@l_op@psi_0/(np.conj(psi_0)@psi_0) for l_op in
                     list_l_op]
    list_l_op = [sp.sparse.csr_matrix(l_op) for l_op in list_l_op]
    list_l_op_sq = [l_op@l_op for l_op in list_l_op]
    list_l_op_exp_sq = [np.conj(psi_0)@l_op@l_op@psi_0/(np.conj(psi_0)@psi_0) for
                        l_op in list_l_op]

    C2_LT_corr_physical, C2_LT_corr_hier = calc_LT_corr(list_lt_corr_coeff, list_l_op, list_l_op_exp,
                                       list_l_op_sq)
    G2_norm_corr_correction = 0

    C2_LT_corr_total = np.eye(len(phi_0))*(-1*G2_norm_corr_correction)
    C2_LT_corr_total[:len(psi_0), :len(psi_0)] += C2_LT_corr_physical
    for i in range(len(phi_0)//len(psi_0)):
        i_up = len(psi_0)*(i+1)
        i_down = len(psi_0)*i
        C2_LT_corr_total[i_down:i_up, i_down:i_up] += C2_LT_corr_hier

    C1_LT_corr_to_deriv = C2_LT_corr_total@phi_0
    assert(np.allclose(C1_LT_corr_to_deriv,
                       dsystem_dt_ltc_modes_match_l_op-dsystem_dt_no_ltc_modes))

def test_LTC_nonlinear_absorption_eom():
    """
    Tests that the low-temperature correction to the nonlinear absorption equation of
    motion reproduces the correct effect on the full hierarchy and state bases.
    """
    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NONLINEAR ABSORPTION"}
    phi_0 = np.array([0.7071, 0.5773, 0.4082, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5])
    psi_0 = phi_0[:3]
    hops_ltc_modes_match_l_op = HOPS(
            sys_param_ltc_modes_match_l_op,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
    hops_ltc_modes_match_l_op.initialize(psi_0)
    hops_ltc_modes_match_l_op.phi = phi_0
    integration_var_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.integration_var(hops_ltc_modes_match_l_op.phi,
                                      hops_ltc_modes_match_l_op.z_mem,
                                      hops_ltc_modes_match_l_op.t,
                                      hops_ltc_modes_match_l_op.noise1,
                                      hops_ltc_modes_match_l_op.noise2,
                                      2.0, hops_ltc_modes_match_l_op.storage)
    dsystem_dt_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.dsystem_dt(
                integration_var_ltc_modes_match_l_op['phi'],
                integration_var_ltc_modes_match_l_op['z_mem'],
                integration_var_ltc_modes_match_l_op['z_rnd'][:,0],
                integration_var_ltc_modes_match_l_op['z_rnd2'][:,0],
        )[0]

    hops_no_ltc_modes = HOPS(
            sys_param_no_ltc_modes,
            noise_param=noise_param,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,
        )
    hops_no_ltc_modes.initialize(psi_0)
    hops_no_ltc_modes.phi = phi_0
    integration_var_no_ltc_modes = hops_no_ltc_modes.integration_var(hops_no_ltc_modes.phi,
                                      hops_no_ltc_modes.z_mem,
                                      hops_no_ltc_modes.t,
                                      hops_no_ltc_modes.noise1,
                                      hops_no_ltc_modes.noise2,
                                      2.0, hops_no_ltc_modes.storage)
    dsystem_dt_no_ltc_modes = hops_no_ltc_modes.dsystem_dt(
                integration_var_no_ltc_modes['phi'],
                integration_var_no_ltc_modes['z_mem'],
                integration_var_no_ltc_modes['z_rnd'][:,0],
                integration_var_no_ltc_modes['z_rnd2'][:,0],
        )[0]

    list_lt_corr_coeff = [250.0/1000.0, 250.0/2000.0, 250.0/3000.0]
    list_l_op = [loperator[0], loperator[1], loperator[2]]
    list_l_op_exp = [np.conj(psi_0)@l_op@psi_0/(1 + (np.conj(psi_0)@psi_0)) for l_op in
                     list_l_op]
    list_l_op = [sp.sparse.csr_matrix(l_op) for l_op in list_l_op]
    list_l_op_sq = [l_op@l_op for l_op in list_l_op]
    list_l_op_exp_sq = [np.conj(psi_0)@l_op@l_op@psi_0/(1 + (np.conj(psi_0)@psi_0)) for
                        l_op in list_l_op]

    C2_LT_corr_physical, C2_LT_corr_hier = calc_LT_corr(list_lt_corr_coeff, list_l_op, list_l_op_exp,
                                       list_l_op_sq)
    G2_norm_corr_correction = 0

    C2_LT_corr_total = np.eye(len(phi_0))*(-1*G2_norm_corr_correction)
    C2_LT_corr_total[:len(psi_0), :len(psi_0)] += C2_LT_corr_physical
    for i in range(len(phi_0)//len(psi_0)):
        i_up = len(psi_0)*(i+1)
        i_down = len(psi_0)*i
        C2_LT_corr_total[i_down:i_up, i_down:i_up] += C2_LT_corr_hier

    C1_LT_corr_to_deriv = C2_LT_corr_total@phi_0
    assert(np.allclose(C1_LT_corr_to_deriv,
                       dsystem_dt_ltc_modes_match_l_op-dsystem_dt_no_ltc_modes))


def test_LTC_adaptive_nonlinear_norm_eom():
    """
    Tests that the low-temperature correction to the normalized nonlinear equation of
    motion reproduces the correct effect in truncated hierarchy and state bases.
    """
    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}
    adap_state_list = np.array([0, 2])
    phi_0 = np.array(
        [0.7071, 0.5773, 0.4082, 0.5, 0, 0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5,
         0.5, 0, 0, 0.5])
    phi_0_reshape = phi_0.reshape(7, 3)
    phi_0_reshape[:, [1]] = 0
    phi_0_reshape[[3, 4], :] = 0
    phi_0_subset = ((phi_0_reshape[[0, 1, 2, 5, 6]])[:,[0, 1, 2]]).flatten()
    psi_0 = phi_0[:3]

    hops_ltc_modes_match_l_op = HOPS(
        sys_param_ltc_modes_match_l_op,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_match_l_op.make_adaptive(1e-5, 1e-5)
    hops_ltc_modes_match_l_op.initialize(psi_0)
    adap_auxiliary_list = [hops_ltc_modes_match_l_op.auxiliary_list[0],
                           hops_ltc_modes_match_l_op.auxiliary_list[1],
                           hops_ltc_modes_match_l_op.auxiliary_list[3],
                           hops_ltc_modes_match_l_op.auxiliary_list[4]]
    hops_ltc_modes_match_l_op.phi = phi_0_subset
    hops_ltc_modes_match_l_op.phi, hops_ltc_modes_match_l_op.dsystem_dt = \
        hops_ltc_modes_match_l_op.basis.update_basis(
        phi_0_subset, adap_state_list, adap_auxiliary_list)
    integration_var_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.integration_var(
        hops_ltc_modes_match_l_op.phi,
        hops_ltc_modes_match_l_op.z_mem,
        hops_ltc_modes_match_l_op.t,
        hops_ltc_modes_match_l_op.noise1,
        hops_ltc_modes_match_l_op.noise2,
        2.0, hops_ltc_modes_match_l_op.storage)
    dsystem_dt_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.dsystem_dt(
        integration_var_ltc_modes_match_l_op['phi'],
        integration_var_ltc_modes_match_l_op['z_mem'],
        integration_var_ltc_modes_match_l_op['z_rnd'][:, 0],
        integration_var_ltc_modes_match_l_op['z_rnd2'][:, 0],
    )[0]

    hops_no_ltc_modes = HOPS(
        sys_param_no_ltc_modes,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_no_ltc_modes.make_adaptive(1e-5, 1e-5)
    hops_no_ltc_modes.initialize(psi_0)
    adap_auxiliary_list = [hops_no_ltc_modes.auxiliary_list[0],
                           hops_no_ltc_modes.auxiliary_list[1],
                           hops_no_ltc_modes.auxiliary_list[3],
                           hops_no_ltc_modes.auxiliary_list[4]]
    hops_no_ltc_modes.phi = phi_0_subset
    hops_no_ltc_modes.phi, hops_no_ltc_modes.dsystem_dt = \
        hops_no_ltc_modes.basis.update_basis(
            phi_0_subset, adap_state_list, adap_auxiliary_list)
    integration_var_no_ltc_modes = hops_no_ltc_modes.integration_var(
        hops_no_ltc_modes.phi,
        hops_no_ltc_modes.z_mem,
        hops_no_ltc_modes.t,
        hops_no_ltc_modes.noise1,
        hops_no_ltc_modes.noise2,
        2.0, hops_no_ltc_modes.storage)
    dsystem_dt_no_ltc_modes = hops_no_ltc_modes.dsystem_dt(
        integration_var_no_ltc_modes['phi'],
        integration_var_no_ltc_modes['z_mem'],
        integration_var_no_ltc_modes['z_rnd'][:, 0],
        integration_var_no_ltc_modes['z_rnd2'][:, 0],
    )[0]

    S2_slicer = np.ix_(adap_state_list, adap_state_list)
    list_lt_corr_coeff = [250.0 / 1000.0, 250.0 / 2000.0, 250.0 / 3000.0]
    list_l_op_reduced = [loperator[0][S2_slicer], loperator[1][S2_slicer],
                         loperator[2][S2_slicer]]
    psi_0_red = psi_0[adap_state_list]
    list_l_op_exp = [np.conj(psi_0_red) @ l_op @ psi_0_red / (np.conj(psi_0_red) @
                                    psi_0_red) for l_op in list_l_op_reduced]
    list_l_op_reduced = [sp.sparse.csr_matrix(l_op) for l_op in list_l_op_reduced]
    list_l_op_sq = [l_op@l_op for l_op in list_l_op_reduced]
    list_l_op_exp_sq = [np.conj(psi_0_red) @ l_op @ l_op @ psi_0_red /
                        (np.conj(psi_0_red) @ psi_0_red) for l_op in list_l_op_reduced]

    phi_0_adap = hops_no_ltc_modes.phi
    list_lt_corr_coeff_adap = np.array(list_lt_corr_coeff)[adap_state_list]
    list_l_op_reduced_adap = np.array(list_l_op_reduced)[adap_state_list]
    list_l_op_exp_adap = np.array(list_l_op_exp)[adap_state_list]
    list_l_op_reduced_adap = [sp.sparse.csr_matrix(l_op) for l_op in list_l_op_reduced_adap]
    list_l_op_reduced_adap_sq = [l_op@l_op for l_op in list_l_op_reduced_adap]
    list_l_op_exp_sq_adap = np.array(list_l_op_exp_sq)[adap_state_list]

    C2_LT_corr_physical_adap, C2_LT_corr_hier_adap = calc_LT_corr(list_lt_corr_coeff_adap,
                                             list_l_op_reduced_adap,
                                       list_l_op_exp_adap, list_l_op_reduced_adap_sq)
    G2_norm_corr_correction_adap = calc_LT_corr_to_norm_corr(list_lt_corr_coeff_adap,
                                                        list_l_op_exp_adap, list_l_op_exp_sq_adap)

    C2_LT_corr_total_adap = np.eye(len(phi_0_adap)) * (-1 *
                                                       G2_norm_corr_correction_adap)
    C2_LT_corr_total_adap[:len(psi_0_red), :len(psi_0_red)] += C2_LT_corr_physical_adap
    for i in range(len(phi_0_adap) // len(psi_0_red)):
        i_up = len(psi_0_red) * (i + 1)
        i_down = len(psi_0_red) * i
        C2_LT_corr_total_adap[i_down:i_up, i_down:i_up] += C2_LT_corr_hier_adap

    C1_LT_corr_to_deriv_adap = C2_LT_corr_total_adap @ phi_0_adap

    assert (np.allclose(C1_LT_corr_to_deriv_adap,
                        dsystem_dt_ltc_modes_match_l_op - dsystem_dt_no_ltc_modes))

loperator_4_site = np.zeros([4, 4, 4], dtype=np.float64)
loperator_4_site[0, 0, 0] = 1.0
loperator_4_site[0, 1, 1] = 1.0
# Note: because this site will end up fully-occupied, we actually need to use 
# imaginary components of the correction to avoid degeneracies!
loperator_4_site[1, 0, 0] = 1.0
loperator_4_site[1, 2, 2] = 1.0
loperator_4_site[2, 2, 2] = 1.0
loperator_4_site[2, 3, 3] = 1.0
loperator_4_site[3, 1, 1] = 1.0
loperator_4_site[3, 3, 3] = 1.0

sys_param_ltc_modes_match_l_op_4_site = {
    "HAMILTONIAN": np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                            dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [10.0, 10.0],
                   [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator_4_site[0], loperator_4_site[0], loperator_4_site[1], loperator_4_site[1], loperator_4_site[2],
               loperator_4_site[2], loperator_4_site[3], loperator_4_site[3]],
    "L_NOISE1": [loperator_4_site[0], loperator_4_site[0], loperator_4_site[1], loperator_4_site[1], loperator_4_site[2],
               loperator_4_site[2], loperator_4_site[3], loperator_4_site[3], loperator_4_site[0], loperator_4_site[1], 
                 loperator_4_site[2], loperator_4_site[3]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [10.0, 10.0],
                    [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [(250.0+10j), 1000.0],
                    [(250.0+10j), 2000.0], [(250.0+10j), 3000.0], [(250.0+10j), 4000.0]],
    "PARAM_LT_CORR": [(250.0+10j)/1000.0, (250.0+10j)/2000.0, (250.0+10j)/3000.0, (250.0+10j)/4000.0],
    "L_LT_CORR": [loperator_4_site[0], loperator_4_site[1], loperator_4_site[2], loperator_4_site[3]]
}

sys_param_no_ltc_modes_4_site = {
    "HAMILTONIAN": np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                            dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [10.0, 10.0],
                   [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator_4_site[0], loperator_4_site[0], loperator_4_site[1], loperator_4_site[1], loperator_4_site[2],
               loperator_4_site[2], loperator_4_site[3], loperator_4_site[3]],
    "L_NOISE1": [loperator_4_site[0], loperator_4_site[0], loperator_4_site[1], loperator_4_site[1], loperator_4_site[2],
               loperator_4_site[2], loperator_4_site[3], loperator_4_site[3], loperator_4_site[0], loperator_4_site[1],
                 loperator_4_site[2], loperator_4_site[3]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [10.0, 10.0],
                    [5.0, 5.0], [10.0, 10.0], [5.0, 5.0], [(250.0+10j), 1000.0],
                     [(250.0+10j), 2000.0], [(250.0+10j), 3000.0], [(250.0+10j), 4000.0]],
}


def test_LTC_adaptive_nonlinear_norm_eom_multiparticle():
    """
    Tests that the low-temperature correction to the normalized nonlinear equation of
    motion reproduces the correct effect in truncated hierarchy and state bases in
    the general case of complex LTC parameters matched to L-operators that project on
    multiple Hamiltonian states.
    """
    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}
    adap_state_list = np.array([0, 2])
    phi_0 = np.zeros([8, 4])
    phi_0[0, :] = [0.7071, 0.5773, 0.4082, 0.1872]
    for i in range(7):
        if i%2 == 0:
            phi_0[i+1, 0] = 0.5
        if i > 3:
            phi_0[i+1, 2] = 0.5

    phi_0_subset = phi_0.flatten()
    psi_0 = phi_0[0]
    hops_ltc_modes_match_l_op = HOPS(
        sys_param_ltc_modes_match_l_op_4_site,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_match_l_op.make_adaptive(1e-5, 1e-5)
    hops_ltc_modes_match_l_op.initialize(psi_0)
    adap_auxiliary_list = [hops_ltc_modes_match_l_op.auxiliary_list[0],
                           hops_ltc_modes_match_l_op.auxiliary_list[1],
                           hops_ltc_modes_match_l_op.auxiliary_list[3],
                           hops_ltc_modes_match_l_op.auxiliary_list[4],
                           hops_ltc_modes_match_l_op.auxiliary_list[5],
                           hops_ltc_modes_match_l_op.auxiliary_list[6],
                           ]
    hops_ltc_modes_match_l_op.phi = phi_0_subset
    hops_ltc_modes_match_l_op.phi, hops_ltc_modes_match_l_op.dsystem_dt = \
        hops_ltc_modes_match_l_op.basis.update_basis(
        phi_0_subset, adap_state_list, adap_auxiliary_list)
    integration_var_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.integration_var(
        hops_ltc_modes_match_l_op.phi,
        hops_ltc_modes_match_l_op.z_mem,
        hops_ltc_modes_match_l_op.t,
        hops_ltc_modes_match_l_op.noise1,
        hops_ltc_modes_match_l_op.noise2,
        2.0, hops_ltc_modes_match_l_op.storage)
    dsystem_dt_ltc_modes_match_l_op = hops_ltc_modes_match_l_op.dsystem_dt(
        integration_var_ltc_modes_match_l_op['phi'],
        integration_var_ltc_modes_match_l_op['z_mem'],
        integration_var_ltc_modes_match_l_op['z_rnd'][:, 0],
        integration_var_ltc_modes_match_l_op['z_rnd2'][:, 0],
    )[0]

    hops_no_ltc_modes = HOPS(
        sys_param_no_ltc_modes_4_site,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_no_ltc_modes.make_adaptive(1e-5, 1e-5)
    hops_no_ltc_modes.initialize(psi_0)
    adap_auxiliary_list = [hops_no_ltc_modes.auxiliary_list[0],
                           hops_no_ltc_modes.auxiliary_list[1],
                           hops_no_ltc_modes.auxiliary_list[3],
                           hops_no_ltc_modes.auxiliary_list[4],
                           hops_no_ltc_modes.auxiliary_list[5],
                           hops_no_ltc_modes.auxiliary_list[6],
                           ]
    hops_no_ltc_modes.phi = phi_0_subset
    hops_no_ltc_modes.phi, hops_no_ltc_modes.dsystem_dt = \
        hops_no_ltc_modes.basis.update_basis(
            phi_0_subset, adap_state_list, adap_auxiliary_list)
    integration_var_no_ltc_modes = hops_no_ltc_modes.integration_var(
        hops_no_ltc_modes.phi,
        hops_no_ltc_modes.z_mem,
        hops_no_ltc_modes.t,
        hops_no_ltc_modes.noise1,
        hops_no_ltc_modes.noise2,
        2.0, hops_no_ltc_modes.storage)
    dsystem_dt_no_ltc_modes = hops_no_ltc_modes.dsystem_dt(
        integration_var_no_ltc_modes['phi'],
        integration_var_no_ltc_modes['z_mem'],
        integration_var_no_ltc_modes['z_rnd'][:, 0],
        integration_var_no_ltc_modes['z_rnd2'][:, 0],
    )[0]

    S2_slicer = np.ix_(adap_state_list, adap_state_list)
    list_lt_corr_coeff = [(250.0+10j) / 1000.0, (250.0+10j) / 2000.0, (250.0+10j) / 3000.0, (250.0+10j) /
                          4000.0]
    list_l_op_reduced = [loperator_4_site[i][S2_slicer] for i in range(4)]
    psi_0_red = psi_0[adap_state_list]
    list_l_op_exp = [np.conj(psi_0_red) @ l_op @ psi_0_red / (np.conj(psi_0_red) @
                                    psi_0_red) for l_op in list_l_op_reduced]
    list_l_op_sq = [l_op@l_op for l_op in list_l_op_reduced]
    list_l_op_exp_sq = [np.conj(psi_0_red) @ l_op @ l_op @ psi_0_red /
                        (np.conj(psi_0_red) @ psi_0_red) for l_op in list_l_op_reduced]

    active_l_list = list(hops_ltc_modes_match_l_op.basis.system.list_absindex_L2_active)
    phi_0_adap = hops_no_ltc_modes.phi
    list_lt_corr_coeff_adap = np.array(list_lt_corr_coeff)[active_l_list]
    list_l_op_reduced_adap = np.array(list_l_op_reduced)[active_l_list]
    list_l_op_exp_adap = np.array(list_l_op_exp)[active_l_list]
    list_l_op_reduced_adap = [sp.sparse.csr_matrix(l_op) for l_op in list_l_op_reduced_adap]
    list_l_op_reduced_adap_sq = np.array(
        [l_op @ l_op for l_op in list_l_op_reduced_adap])
    list_l_op_exp_sq_adap = np.array(list_l_op_exp_sq)[active_l_list]

    C2_LT_corr_physical_adap, C2_LT_corr_hier_adap = calc_LT_corr(list_lt_corr_coeff_adap,
                                             list_l_op_reduced_adap,
                                       list_l_op_exp_adap, list_l_op_reduced_adap_sq)
    G2_norm_corr_correction_adap = calc_LT_corr_to_norm_corr(list_lt_corr_coeff_adap,
                                                        list_l_op_exp_adap, list_l_op_exp_sq_adap)

    C2_LT_corr_total_adap = np.eye(len(phi_0_adap), dtype=np.complex128) * (-1 *
                                                       G2_norm_corr_correction_adap)
    C2_LT_corr_total_adap[:len(psi_0_red), :len(psi_0_red)] += C2_LT_corr_physical_adap
    for i in range(len(phi_0_adap) // len(psi_0_red)):
        i_up = len(psi_0_red) * (i + 1)
        i_down = len(psi_0_red) * i
        C2_LT_corr_total_adap[i_down:i_up, i_down:i_up] += C2_LT_corr_hier_adap

    C1_LT_corr_to_deriv_adap = C2_LT_corr_total_adap @ phi_0_adap

    assert (np.allclose(C1_LT_corr_to_deriv_adap,
                        dsystem_dt_ltc_modes_match_l_op - dsystem_dt_no_ltc_modes))