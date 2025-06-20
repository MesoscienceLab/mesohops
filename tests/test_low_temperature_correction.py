import numpy as np
import scipy as sp
from mesohops.eom.eom_functions import (
    calc_LT_corr,
    calc_LT_corr_to_norm_corr,
    calc_LT_corr_linear
)
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS


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
    [[0, 2j],
     [-2j, 0]]
)
# WE NEED A TEST CASE WITH AN IMAGINARY BUT STILL HERMITIAN OFF-DIAGONAL L-OPERATOR.

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

    sys_ltc_modes_match = hops_ltc_modes_match_l_op.basis
    assert list(sys_ltc_modes_match.system.list_lt_corr_param) == \
           [250.0 / 1000.0, 250.0 / 2000.0]
    assert np.allclose(sys_ltc_modes_match.mode.list_L2_coo[0].todense(),sparse_l0.todense())
    assert np.allclose(sys_ltc_modes_match.mode.list_L2_coo[1].todense(),sparse_l1.todense())

    # Case where the low-temperature correction term list fed into HOPSSystem is empty.
    hops_ltc_modes_empty = HOPS(
        sys_param_ltc_modes_empty,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops_ltc_modes_empty.initialize(psi_0)

    sys_ltc_modes_empty = hops_ltc_modes_empty.basis
    assert list(sys_ltc_modes_empty.system.list_lt_corr_param) == \
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

    sys_ltc_modes_swap = hops_ltc_modes_swap_l_op.basis
    assert list(sys_ltc_modes_swap.system.list_lt_corr_param) == \
           [250.0 / 2000.0, 250.0 / 1000.0]
    assert np.allclose(sys_ltc_modes_swap.mode.list_L2_coo[0].todense(),
                       sparse_l0.todense())
    assert np.allclose(sys_ltc_modes_swap.mode.list_L2_coo[1].todense(),
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

    sys_ltc_modes_more_than = hops_ltc_modes_more_than_l_op.basis
    assert list(sys_ltc_modes_more_than.system.list_lt_corr_param) == \
           [250.0 / 1000.0, 250.0 / 2000.0]
    assert np.allclose(sys_ltc_modes_more_than.mode.list_L2_coo[0].todense(),
                       sparse_l0.todense())
    assert np.allclose(sys_ltc_modes_more_than.mode.list_L2_coo[1].todense(),
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

    hopsmodes_ltc_modes_less_than = hops_ltc_modes_less_than_l_op.basis
    assert list(hopsmodes_ltc_modes_less_than.system.list_lt_corr_param) == \
           [250.0 / 1000.0, 0]
    assert np.allclose(hopsmodes_ltc_modes_less_than.mode.list_L2_coo[0].todense(),
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

    hopsmodes_ltc_modes_wrong = hops_ltc_modes_wrong_l_op.basis
    assert list(hopsmodes_ltc_modes_wrong.system.list_lt_corr_param) == \
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

    hopsmodes_ltc_modes_adap = hops_ltc_modes_match_l_op_adaptive.basis
    hopsmodes_ltc_modes_adap.system.state_list = [0]
    assert list(hopsmodes_ltc_modes_adap.system.list_lt_corr_param) == \
           [250.0 / 1000.0]
    assert np.allclose(hopsmodes_ltc_modes_adap.mode.list_L2_coo[0].todense(),
                       np.array([[1]]))

    hopsmodes_ltc_modes_adap.system.state_list = [1]
    assert list(hopsmodes_ltc_modes_adap.system.list_lt_corr_param) == \
           [250.0 / 2000.0]
    assert np.allclose(hopsmodes_ltc_modes_adap.mode.list_L2_coo[0].todense(),
                       np.array([[1]]))

    hopsmodes_ltc_modes_adap.system.state_list = [0, 1]
    hopsmodes_ltc_modes_adap.mode.list_absindex_mode = [0, 1, 2, 3]
    assert list(hopsmodes_ltc_modes_adap.system.list_lt_corr_param) == \
           [250.0 / 1000.0, 250.0 / 2000.0]
    assert np.allclose(hopsmodes_ltc_modes_adap.mode.list_L2_coo[0].todense(),
                       sparse_l0.todense())
    assert np.allclose(hopsmodes_ltc_modes_adap.mode.list_L2_coo[1].todense(),
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
    # We consider LTC terms with G = 10 + 10j, -10 -10j, and an expectation value of
    # 0.25 for each L-operator as the "base" case here. Recall that
    # T_hier = \sum_n G_n*<L_n>L_n and
    # T_phys = \sum_n 2Re[G_n]<L_n>L_n - G_n(L_n^2)
    # For the secondary (Peierls) L-operators, we work with G = 10 + 10j

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
    # For the real Peierls L-operator
    test_ltc_full_sec = np.array(
        [[-40 - 40j, 10j],
         [-10j, -40 - 40j]]
    )
    test_ltc_not_physical_sec = np.array(
        [[0, 5 + 5j],
         [-5 - 5j, 0]]
    )

    test_ltc_flux_from_above_only = test_ltc_full - test_ltc_not_physical
    assert np.allclose(
        calc_LT_corr([10 + 10j, -10 -10j], [sparse_l0, sparse_l1],
        [0.25, 0.25], [sparse_l0@sparse_l0, sparse_l1@sparse_l1])[0].todense(),
        test_ltc_flux_from_above_only
    )
    assert np.allclose(
        calc_LT_corr([10 + 10j, -10 - 10j], [sparse_l0, sparse_l1],
        [0.25, 0.25], [sparse_l0 @ sparse_l0, sparse_l1 @ sparse_l1])[1].todense(),
        test_ltc_not_physical
    )
    test_ltc_flux_from_above_only_sec = test_ltc_full_sec - test_ltc_not_physical_sec
    assert np.allclose(
        calc_LT_corr([10 + 10j], [sparse_lsec], [0.25], [sparse_lsec @ sparse_lsec])[
            0].todense(), test_ltc_flux_from_above_only_sec
    )
    assert np.allclose(
        calc_LT_corr([10 + 10j], [sparse_lsec], [0.25], [sparse_lsec @ sparse_lsec])[
            1].todense(), test_ltc_not_physical_sec
    )
    assert np.allclose(
        calc_LT_corr([0,0], [sparse_l0, sparse_l1],
                     [0.25, 0.25], [sparse_l0 @ sparse_l0, sparse_l1 @ sparse_l1])[0].todense(),
        np.zeros([2,2])
    )
    assert np.allclose(
        calc_LT_corr([10, -10], [sparse_l0, sparse_l1],
                     [0.5, 0.5], [sparse_l0 @ sparse_l0, sparse_l1 @ sparse_l1])[0].todense(),
        -1*calc_LT_corr([10, -10], [sparse_l0, sparse_l1],
                     [0.5, 0.5], [sparse_l0 @ sparse_l0, sparse_l1 @ sparse_l1])[1].todense()
    )
    assert np.allclose(
        calc_LT_corr([10+10j, -10 - 10j], [np.zeros([2, 2]), np.zeros([2, 2])],
        [0.25, 0.25], [np.zeros([2, 2]), np.zeros([2, 2])])[0],
        np.zeros([2, 2])
    )
    assert np.allclose(
        calc_LT_corr([], [], [], [])[0],
        np.zeros([])
    )
    assert np.allclose(
        calc_LT_corr([0, 0], [sparse_l0, sparse_l1],
                     [0.25, 0.25], [sparse_l0 @ sparse_l0, sparse_l1 @ sparse_l1])[1].todense(),
        np.zeros([2, 2])
    )
    assert np.allclose(
        calc_LT_corr([10, -10], [sparse_l0, sparse_l1],
                     [0, 0], [sparse_l0 @ sparse_l0, sparse_l1 @ sparse_l1])[1].todense(),
        np.zeros([2, 2])
    )
    assert np.allclose(
        calc_LT_corr([10 + 10j, -10 - 10j], [np.zeros([2, 2]), np.zeros([2, 2])],
                     [0.25, 0.25], [np.zeros([2, 2]), np.zeros([2, 2])])[1],
        np.zeros([2, 2])
    )
    assert np.allclose(
        calc_LT_corr([], [], [], [])[1],
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
    # We consider LTC terms with G = =1, -j.
    # T_phys = -\sum_n G_n*(L_n@L_n)
    # Because the squared L-operators are a direct input, no operator algebra is done,
    # and we don't need to  explicitly test a Peierls case.
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
