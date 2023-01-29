import numpy as np
import scipy as sp
from mesohops.dynamics.eom_functions import (
    calc_LT_corr,
    calc_LT_corr_to_norm_corr,
    calc_LT_corr_linear
)
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp

__title__ = "test of low-temperature correction "
__author__ = "J. K. Lynd"
__version__ = "1.2"

noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 50.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
}

loperator = np.zeros([2, 2, 2], dtype=np.float64)
loperator[0, 0, 0] = 1.0
loperator[1, 1, 1] = 1.0

loperator_secondary = np.array(
    [[0, 1],
     [1, 0]]
)

# Set the Hamiltonian to have no coupling so that we can ensure that the adaptive
# basis will contain only one state.
sys_param_ltc_modes_match_l_op = {
    "HAMILTONIAN": np.array([[0, 0.0], [0.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1],
                 loperator[0], loperator[1]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [250.0, 1000.0], [250.0, 2000.0]],
    "PARAM_LT_CORR": [250.0/1000.0, 250.0/2000.0],
    "L_LT_CORR": [loperator[0], loperator[1]]
}

sys_param_ltc_modes_empty = {
    "HAMILTONIAN": np.array([[0, 0.0], [0.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1],
                 loperator[0], loperator[1]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [250.0, 1000.0], [250.0, 2000.0]],
    "PARAM_LT_CORR": [],
    "L_LT_CORR": []
}

sys_param_ltc_modes_swap_l_op = {
    "HAMILTONIAN": np.array([[0, 0.0], [0.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1],
                 loperator[1], loperator[0]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [250.0, 1000.0], [250.0, 2000.0]],
    "PARAM_LT_CORR": [250.0/1000.0, 250.0/2000.0],
    "L_LT_CORR": [loperator[1], loperator[0]]
}

sys_param_ltc_modes_less_than_l_op = {
    "HAMILTONIAN": np.array([[0, 0.0], [0.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1],
                 loperator[0]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [250.0, 1000.0]],
    "PARAM_LT_CORR": [250.0/1000.0],
    "L_LT_CORR": [loperator[0]]
}

sys_param_ltc_modes_more_than_l_op = {
    "HAMILTONIAN": np.array([[0, 0.0], [0.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1],
                 loperator[0], loperator[1], loperator_secondary],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [250.0, 1000.0], [250.0, 2000.0], [250.0, 3000.0]],
    "PARAM_LT_CORR": [250.0/1000.0, 250.0/2000.0, 250.0/3000.0],
    "L_LT_CORR": [loperator[0], loperator[1], loperator_secondary]
}

sys_param_ltc_modes_wrong_l_op = {
    "HAMILTONIAN": np.array([[0, 0.0], [0.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1],
                 loperator[0], loperator[1]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [250.0, 1000.0], [250.0, 2000.0]],
    "PARAM_LT_CORR": [250.0/1000.0, 250.0/2000.0],
    "L_LT_CORR": [loperator_secondary, 1j*loperator_secondary]
}

sparse_l0 = sp.sparse.coo_matrix(loperator[0])
sparse_l1 = sp.sparse.coo_matrix(loperator[1])
sparse_lsec = sp.sparse.coo_matrix(loperator_secondary)


hier_param = {"MAXHIER": 4}

eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR':'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS':None
    }
t_max = 20.0
t_step = 2.0
psi_0 = [1.0 + 0.0 * 1j, 0.0 + 0.0 * 1j]

def test_lt_corr_list():
    """
    Tests that the list of low-temperature modes is in the order of associated
    hierarchy L-operators, and that only low-temperature modes associated with an
    L-operator that exists in the hierarchy are put into place.
    """
    # Case where the L-operators of the low-temperature correction match the
    # L-operators of the hierarchy
    hops_ltc_modes_match_l_op = HOPS(
        sys_param_ltc_modes_match_l_op,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_match_l_op.initialize(psi_0)

    sys_ltc_modes_match = hops_ltc_modes_match_l_op.basis.mode
    assert list(sys_ltc_modes_match.lt_corr_param) == \
           [250.0 / 1000.0, 250.0 / 2000.0]
    assert np.allclose(sys_ltc_modes_match.list_L2_coo[0].todense(),sparse_l0.todense())
    assert np.allclose(sys_ltc_modes_match.list_L2_coo[1].todense(),sparse_l1.todense())

    # Case where the low-temperature correction term list fed into HOPSSystem is empty.
    hops_ltc_modes_empty = HOPS(
        sys_param_ltc_modes_empty,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_empty.initialize(psi_0)

    sys_ltc_modes_empty = hops_ltc_modes_empty.basis.mode
    assert list(sys_ltc_modes_empty.lt_corr_param) == \
           [0, 0]

    # Case where the L-operators of the low-temperature correction match the
    # L-operators of the hierarchy, but are swapped in order to test that the
    # coefficients are matched to the right hierarchy L-operators
    hops_ltc_modes_swap_l_op = HOPS(
        sys_param_ltc_modes_swap_l_op,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_swap_l_op.initialize(psi_0)

    sys_ltc_modes_swap = hops_ltc_modes_swap_l_op.basis.mode
    assert list(sys_ltc_modes_swap.lt_corr_param) == \
           [250.0 / 2000.0, 250.0 / 1000.0]
    assert np.allclose(sys_ltc_modes_swap.list_L2_coo[0].todense(),
                       sparse_l0.todense())
    assert np.allclose(sys_ltc_modes_swap.list_L2_coo[1].todense(),
                       sparse_l1.todense())

    # Case where there is an extra L-operator not found in the hierarchy L-operator
    # list (currently, the correction associated with that L-operator is ignored)
    hops_ltc_modes_more_than_l_op = HOPS(
        sys_param_ltc_modes_more_than_l_op,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_more_than_l_op.initialize(psi_0)

    sys_ltc_modes_more_than = hops_ltc_modes_more_than_l_op.basis.mode
    assert list(sys_ltc_modes_more_than.lt_corr_param) == \
           [250.0 / 1000.0, 250.0 / 2000.0]
    assert np.allclose(sys_ltc_modes_more_than.list_L2_coo[0].todense(),
                       sparse_l0.todense())
    assert np.allclose(sys_ltc_modes_more_than.list_L2_coo[1].todense(),
                       sparse_l1.todense())

    # Case where the low-temperature correction is only acted on one of the
    # L-operators found in the hierarchy
    hops_ltc_modes_less_than_l_op = HOPS(
        sys_param_ltc_modes_less_than_l_op,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_less_than_l_op.initialize(psi_0)

    hopsmodes_ltc_modes_less_than = hops_ltc_modes_less_than_l_op.basis.mode
    assert list(hopsmodes_ltc_modes_less_than.lt_corr_param) == \
           [250.0 / 1000.0, 0]
    assert np.allclose(hopsmodes_ltc_modes_less_than.list_L2_coo[0].todense(),
                       sparse_l0.todense())

    # Case where none of the L-operators of the low-temperature correction match the
    # L-operators of the hierarchy
    hops_ltc_modes_wrong_l_op = HOPS(
        sys_param_ltc_modes_wrong_l_op,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_wrong_l_op.initialize(psi_0)

    hopsmodes_ltc_modes_wrong = hops_ltc_modes_wrong_l_op.basis.mode
    assert list(hopsmodes_ltc_modes_wrong.lt_corr_param) == \
           [0, 0]

    # Case where the L-operators of the low-temperature correction match the
    # L-operators of the hierarchy, run with aggressive adaptivity to reduce the basis
    # to a single state
    hops_ltc_modes_match_l_op_adaptive = HOPS(
        sys_param_ltc_modes_match_l_op,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_match_l_op_adaptive.make_adaptive(1e-1, 1e-1)
    hops_ltc_modes_match_l_op_adaptive.initialize(psi_0)

    hopsmodes_ltc_modes_adap = hops_ltc_modes_match_l_op_adaptive.basis.mode
    hopsmodes_ltc_modes_adap.system.state_list = [0]
    assert list(hopsmodes_ltc_modes_adap.lt_corr_param) == \
           [250.0 / 1000.0]
    assert np.allclose(hopsmodes_ltc_modes_adap.list_L2_coo[0].todense(),
                       np.array([[1]]))

    hopsmodes_ltc_modes_adap.system.state_list = [1]
    assert list(hopsmodes_ltc_modes_adap.lt_corr_param) == \
           [250.0 / 2000.0]
    assert np.allclose(hopsmodes_ltc_modes_adap.list_L2_coo[0].todense(),
                       np.array([[1]]))

    hopsmodes_ltc_modes_adap.system.state_list = [0, 1]
    hopsmodes_ltc_modes_adap.list_absindex_mode = [0, 1, 2, 3]
    assert list(hopsmodes_ltc_modes_adap.lt_corr_param) == \
           [250.0 / 1000.0, 250.0 / 2000.0]
    assert np.allclose(hopsmodes_ltc_modes_adap.list_L2_coo[0].todense(),
                       sparse_l0.todense())
    assert np.allclose(hopsmodes_ltc_modes_adap.list_L2_coo[1].todense(),
                       sparse_l1.todense())


def test_calc_lt_corr():
    """
    Tests that, in a trivial case, the low-temperature correction to the nonlinear
    equation of motion returns the expected term.

    Recall that the formula for the correction is:
    \sum_n (2<L_n>Re[c_n] - L_nc_n)L_n
    Where c_n is the nth low-temperature correction factor, and L_n is the nth
    L-operator associated with that factor.
    """
    # Physical wavefunction case with the noise memory drift term approximation added in
    test_ltc_full = np.array(
        [[-5 - 10j, 0],
         [0, 5 + 10j]]
    )
    # Non-physical auxiliary case with only the noise memory drift term approximation
    test_ltc_not_physical = np.array(
        [[2.5 - 2.5j, 0],
         [0, -2.5 + 2.5j]]
    )
    test_ltc_flux_from_above_only = test_ltc_full - test_ltc_not_physical
    assert np.allclose(
        calc_LT_corr([10 + 10j, -10 -10j], [loperator[0], loperator[1]], [0.25,
                                                                          0.25], True),
        test_ltc_flux_from_above_only
    )
    assert np.allclose(
        calc_LT_corr([0,0], [loperator[0], loperator[1]], [0.25, 0.25], True),
        np.zeros([2,2])
    )
    assert np.allclose(
        calc_LT_corr([10, -10], [loperator[0], loperator[1]], [0.5, 0.5], True),
        -1*calc_LT_corr([10, -10], [loperator[0], loperator[1]],
                        [0.5, 0.5], False)
    )
    assert np.allclose(
        calc_LT_corr([10+10j, -10 - 10j], [np.zeros([2, 2]), np.zeros([2, 2])],
        [0.25, 0.25], True), np.zeros([2, 2])
    )
    assert np.allclose(
        calc_LT_corr([], [], [], True),
        np.zeros([])
    )
    assert np.allclose(
        calc_LT_corr([10 + 10j, -10 - 10j], [loperator[0], loperator[1]], [0.25,
                                                                           0.25], False),
        test_ltc_not_physical
    )
    assert np.allclose(
        calc_LT_corr([0, 0], [loperator[0], loperator[1]], [0.25, 0.25], False),
        np.zeros([2, 2])
    )
    assert np.allclose(
        calc_LT_corr([10, -10], [loperator[0], loperator[1]], [0, 0], False),
        np.zeros([2, 2])
    )
    assert np.allclose(
        calc_LT_corr([10 + 10j, -10 - 10j], [np.zeros([2, 2]), np.zeros([2, 2])],
                     [0.25, 0.25], False), np.zeros([2, 2])
    )
    assert np.allclose(
        calc_LT_corr([], [], [], False),
        np.zeros([])
    )


def test_calc_lt_corr_to_norm_corr():
    """
    Tests that, in a trivial case, the low-temperature correction to the normalization
    correction of the normalized nonlinear equation of motion returns the correct
    normalization constant.

    Recall that the formula for the correction is:
    \sum_n Re[c_n](2<L_n>^2 - <L_n^2>)
    Where c_n is the nth low-temperature correction factor, and L_n is the nth
    L-operator associated with that factor.
    """
    test_ltc_to_norm_corr = 9*0.25
    assert np.allclose(
        calc_LT_corr_to_norm_corr([1 + 10j, -10 -1j], [0.5, 0.5], [0.75, 0.75]),
        test_ltc_to_norm_corr
    )
    assert np.allclose(
        calc_LT_corr_to_norm_corr([0, 0], [0.5, 0.5], [0.75, 0.75]),
        0
    )
    assert np.allclose(
        calc_LT_corr_to_norm_corr([1 + 10j, -10 - 1j], [0.5, 0.5], [0.5, 0.5]),
        0
    )
    assert np.allclose(
        calc_LT_corr_to_norm_corr([], [], []),
        0
    )

def test_calc_LT_corr_linear():
    """
    Tests that, in a trivial case, the low-temperature correction to the linear
    equation of motion returns the expected term.

    Recall that the formula for the correction is:
    -\sum_n c_nL_n^2
    Where c_n is the nth low-temperature correction factor, and L_n is the nth
    L-operator associated with that factor.
    """
    test_ltc_linear = np.array([[1, 0],[0, 1j]])
    assert np.allclose(
        calc_LT_corr_linear([-1, -1j], [loperator[0], loperator[1]]),
        test_ltc_linear
    )
    assert np.allclose(
        calc_LT_corr_linear([0, 0], [loperator[0], loperator[1]]),
        0
    )
    assert np.allclose(
        calc_LT_corr_linear([-1, -1j], [np.zeros([2,2]), np.zeros([2,2])]),
        0
    )
    assert np.allclose(
        calc_LT_corr_linear([], []),
        0
    )