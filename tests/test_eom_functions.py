import numpy as np
from mesohops.eom.eom_functions import (
    operator_expectation,
    calc_delta_zmem,
    compress_zmem,
)
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS

__title__ = "Test of eom_functions"
__author__ = "D. I. G. Bennett"
__version__ = "1.2"
__date__ = ""

# TEST PARAMETERS
# ===============
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 10.0,  # Units: fs
    "TAU": 0.5,  # Units: fs
}

loperator = np.zeros([4, 2, 2], dtype=np.complex128)
loperator[0, 0, 0] = 1.0
loperator[1, 1, 1] = 1.0
loperator[2, 0, 0] = -1.0
loperator[2, 1, 1] = 1.0
loperator[3, 0, 1] = 1.0j
loperator[3, 1, 0] = -1.0j

sys_param = {
    "HAMILTONIAN": np.array([[0, 10.0], [10.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                   [10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1],
               loperator[2], loperator[2], loperator[3], loperator[3]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1],
                 loperator[2], loperator[2], loperator[3], loperator[3]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0],
                     [10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
}

hier_param = {"MAXHIER": 4}

eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

psi_0 = [1.0 + 0.0 * 1j, 0.0 + 0.0 * 1j]

hops = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
)
hops.initialize(psi_0)


# Test basic functions
# ----------------------
def test_operator_expectation():
    """
    Tests that operator expectation is correctly calculated and normalized in both
    the ground state-corrected and non-ground state-corrected cases.
    """
    psi = np.array([np.sqrt(2)*np.exp(-1j), np.sqrt(2)*np.exp(1j)])
    I2_identity = np.array([[1, 0],
                            [0, 1]])
    I2 = I2_identity
    C2_cancellation = np.array([[1, 0],
                                [0, -1]])
    C2 = C2_cancellation
    P2_off_diagonal = np.array([[0, 1],
                                [1, 0]])
    P2 = P2_off_diagonal
    O2_off_diagonal_imag = np.array([[0, -1j],
                                [1j, 0]])
    O2 = O2_off_diagonal_imag
    exact_answers = [1.0, 0, np.cos(2), np.sin(2)]
    calculated_answers = [operator_expectation(I2, psi),
                          operator_expectation(C2, psi),
                          operator_expectation(P2, psi),
                          operator_expectation(O2, psi)]
    assert np.allclose(exact_answers, calculated_answers)


def test_l_avg_calculation():
    lind_dict = hops.basis.system.param["LIST_L2_COO"]
    lop_list = lind_dict
    lavg_list = [operator_expectation(L2, hops.psi) for L2 in lop_list]
    assert lavg_list[0] == 1.0
    assert lavg_list[1] == 0.0
    assert lavg_list[2] == -1.0
    assert lavg_list[3] == 0.0


def test_calc_delta_zmem():
    """
    This is a test to ensure that memory-effects
    in the noise are properly taken into account
    during a HOPS simulation.
    """
    lind_dict = hops.basis.system.param["LIST_L2_COO"]
    lop_list = lind_dict
    lavg_list = [operator_expectation(L2, hops.psi) for L2 in lop_list]
    g_list = hops.basis.system.param["G"]
    w_list = hops.basis.system.param["W"]

    # Tests calc_delta_zmem when all noise memory terms are zero
    z_mem = np.array([0.0 for g in g_list])
    d_zmem = calc_delta_zmem(
        z_mem,
        lavg_list,
        g_list,
        w_list,
        hops.basis.system.param["LIST_INDEX_L2_BY_NMODE1"],
        np.array(range(len(g_list))),
        list(np.arange(len(lavg_list)))
    )
    assert len(d_zmem) == len(g_list)
    assert d_zmem[0] == 10.0
    assert d_zmem[1] == 5.0
    assert d_zmem[2] == 0
    assert d_zmem[3] == 0
    assert d_zmem[4] == -10.0
    assert d_zmem[5] == -5.0
    assert d_zmem[6] == 0
    assert d_zmem[7] == 0
    assert type(d_zmem) == type(np.array([]))

    # Tests calc_delta_zmem when nonzero noise memory terms are present
    z_mem = np.array([5.0, 0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 0.0])
    d_zmem = calc_delta_zmem(
        z_mem,
        [1, 1, -1, -1],
        g_list,
        w_list,
        hops.basis.system.param["LIST_INDEX_L2_BY_NMODE1"],
        np.array([0, 1, 6]),
        list(np.arange(len(lavg_list)))
    )
    assert len(d_zmem) == len(g_list)
    assert d_zmem[0] == 10.0 - (5.0*10.0)
    assert d_zmem[1] == 5.0
    assert d_zmem[2] == 0.0
    assert d_zmem[3] == -3.0*5.0
    assert d_zmem[4] == 0.0
    assert d_zmem[5] == -1.0*5.0
    assert d_zmem[6] == -1*10.0 - (1.0*10.0)
    assert d_zmem[7] == 0

    assert type(d_zmem) == type(np.array([]))

def test_compress_zmem():
    """
    This is a test to ensure that memory-compression,
    or the implicit accumulation of Matsubara modes,
    is properly taken into account during a HOPS
    simulation.
    """
    lind_dict = hops.basis.system.param["LIST_L2_COO"]
    lop_list = lind_dict
    lavg_list = [operator_expectation(L2, hops.psi) for L2 in lop_list]
    g_list = hops.basis.system.param["G"]
    w_list = hops.basis.system.param["W"]
    z_mem = np.array([0.0 for g in g_list])
    z_mem = calc_delta_zmem(
        z_mem,
        lavg_list,
        g_list,
        w_list,
        hops.basis.system.param["LIST_INDEX_L2_BY_NMODE1"],
        range(len(g_list)),
        list(np.arange(len(lavg_list)))
    )
    z_compress = compress_zmem(
        z_mem,
        hops.basis.system.param["LIST_INDEX_L2_BY_NMODE1"],
        hops.basis.list_absindex_mode,
    )
    assert len(z_compress) == 4
    assert z_compress[0] == 15.0
    assert z_compress[1] == 0.0
    assert z_compress[2] == -15.0
    assert z_compress[3] == 0.0
