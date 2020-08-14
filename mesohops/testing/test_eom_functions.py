import numpy as np
from mesohops.dynamics.bath_corr_functions import bcf_exp
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.eom_functions import (
    operator_expectation,
    calc_delta_zmem,
    compress_zmem,
)

__title__ = "Test of eom_functions"
__author__ = "D. I. G. Bennett"
__version__ = "0.1"
__date__ = ""

# TEST PARAMETERS
# ===============
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 10.0,  # Units: fs
    "TAU": 0.5,  # Units: fs
    "DIAGONAL": True,
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

integrator_param = {"INTEGRATOR": "RUNGE_KUTTA", "INCHWORM": True, "INCHWORM_MIN": 5}

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
def test_l_avg_calculation():
    lind_dict = hops.basis.system.param["LIST_L2_COO"]
    lop_list = lind_dict
    lavg_list = [operator_expectation(L2, hops.psi) for L2 in lop_list]
    assert lavg_list[0] == 1.0
    assert lavg_list[1] == 0.0


def test_calc_deltz_zmem():
    lind_dict = hops.basis.system.param["LIST_L2_COO"]
    lop_list = lind_dict
    lavg_list = [operator_expectation(L2, hops.psi) for L2 in lop_list]
    g_list = hops.basis.system.param["G"]
    w_list = hops.basis.system.param["W"]
    z_mem = np.array([0.0 for g in g_list])
    d_zmem = calc_delta_zmem(
        z_mem,
        lavg_list,
        g_list,
        w_list,
        hops.basis.system.param["LIST_INDEX_L2_BY_NMODE1"],
        np.array(range(len(g_list))),
    )
    assert len(d_zmem) == len(g_list)
    assert d_zmem[0] == 10.0
    assert d_zmem[1] == 5.0
    assert d_zmem[2] == 0
    assert d_zmem[3] == 0
    assert type(d_zmem) == type(np.array([]))


def test_compress_zmem():
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
    )
    z_compress = compress_zmem(
        z_mem,
        hops.basis.system.param["LIST_INDEX_L2_BY_NMODE1"],
        hops.basis.system.list_absindex_mode,
    )
    assert len(z_compress) == 2
    assert z_compress[0] == 15.0
    assert z_compress[1] == 0.0
