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
    lop_list = hops.basis.system.param["LIST_L2_COO"]
    lavg_list = [operator_expectation(L2, hops.psi) for L2 in lop_list]
    g_list = hops.basis.noise_memory.list_zmemg_abs
    w_list = hops.basis.noise_memory.list_zmemw_abs
    list_index_L2_by_mode = hops.basis.mode.list_index_L2_by_hmode
    list_modeidx_abs = hops.basis.mode.list_modeidx_abs
    list_zmemmodeidx_abs = hops.basis.noise_memory.list_zmemmodeidx_abs
    list_l2idx_abs = hops.basis.mode.list_l2idx_abs
    list_activel2idx_abs = list_l2idx_abs
    # Tests calc_delta_zmem when all noise memory terms are zero
    z_mem = np.array([0.0 for g in g_list])

    # l_avg = [1,    0, -1,   0]
    # g = w = [10,5,10,5,10,5,10,5]
    # z_mem = [0,0,0,0,0,0,0,0]

    d_zmem = calc_delta_zmem(
        z_mem,
        lavg_list,
        g_list,
        w_list,
        list_index_L2_by_mode,
        list_modeidx_abs,
        list_zmemmodeidx_abs,
        list_l2idx_abs,
        list_activel2idx_abs
    )

    # d_zmem[i] = l_avg * np.conj(g) - np.conj(w) * z_mem[i]
    assert len(d_zmem) == len(z_mem)
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
    lavg_list = [1, -1, -1]

    hops.basis.system.state_list = [0]
    hops.basis.mode.list_modeidx_abs = [0, 1, 4, 5, 6, 7]

    g_list = hops.basis.noise_memory.list_zmemg_abs
    w_list = hops.basis.noise_memory.list_zmemw_abs
    list_index_L2_by_mode = hops.basis.mode.list_index_L2_by_hmode
    list_modeidx_abs = hops.basis.mode.list_modeidx_abs
    list_zmemmodeidx_abs = hops.basis.noise_memory.list_zmemmodeidx_abs
    list_l2idx_abs = hops.basis.mode.list_l2idx_abs
    list_activel2idx_abs = list_l2idx_abs

    # l_avg = [1,        -1,  -1]
    # g = w = [10,5,10,5,10,5,10,5]
    # z_mem = [5, 0, 0,3, 0,1, 1,0]
    d_zmem = calc_delta_zmem(
        z_mem,
        lavg_list,
        g_list,
        w_list,
        list_index_L2_by_mode,
        list_modeidx_abs,
        hops.basis.noise_memory.list_zmemmodeidx_abs,
        list_l2idx_abs,
        list_activel2idx_abs,
    )
    # d_zmem[i] = l_avg * np.conj(g) - np.conj(w) * z_mem[i]
    assert len(d_zmem) == len(z_mem)
    assert d_zmem[0] == 10.0 - (5.0*10.0)
    assert d_zmem[1] == 5.0
    assert d_zmem[2] == 0.0
    assert d_zmem[3] == -3.0*5.0
    assert d_zmem[4] == -1.0*10.0
    assert d_zmem[5] == -1.0*5.0 - (1.0*5.0)
    assert d_zmem[6] == -1.0*10.0 - (1.0*10.0)
    assert d_zmem[7] == -1.0*5.0
    assert type(d_zmem) == type(np.array([]))


    # Tests that it still works when not all L2 are active
    z_mem = [1, 2, 3, 4, 5, 6, 7, 8]
    lavg_list = [1,1,-1] #Note:  lavg_list must have same length as list_activel2idx_abs!
    g_list = w_list = [10,5,10,5,10,5,10,5]
    list_index_L2_by_mode = [0,0,1,1,2,2,3,3]
    list_modeidx_abs = [0,1,2,3,4,5,6,7]
    list_zmemmodeidx_abs = [0,1,2,3,4,5,6,7]
    list_l2idx_abs = [0,1,2,3]
    list_activel2idx_abs = [0,2,3]
    d_zmem = calc_delta_zmem(
        z_mem,
        lavg_list,
        g_list,
        w_list,
        list_index_L2_by_mode,
        list_modeidx_abs,
        hops.basis.noise_memory.list_zmemmodeidx_abs,
        list_l2idx_abs,
        list_activel2idx_abs,
    )
    # d_zmem[i] = l_avg * np.conj(g) - np.conj(w) * z_mem[i]
    assert len(d_zmem) == len(z_mem)
    assert d_zmem[0] == (1.0*10.0) - (10.0*1.0)
    assert d_zmem[1] == (1.0*5.0) - (5.0*2.0)
    assert d_zmem[2] == (0.0*10.0) - (10.0*3.0)
    assert d_zmem[3] == (0.0*5.0) - (5.0*4.0)
    assert d_zmem[4] == (1.0*10.0) - (10.0*5.0)
    assert d_zmem[5] == (1.0*5.0) - (5.0*6.0)
    assert d_zmem[6] == (-1.0*10.0) - (10.0*7.0)
    assert d_zmem[7] == (-1.0*5.0) - (5.0*8.0)

    # Tests that it still works when z_mem contains extra modes
    z_mem = [1, 2, 3, 4, 5, 6, 7, 8]
    lavg_list = [1,-1,1,-1] #Note:  lavg_list must have same length as list_activel2idx_abs!
    g_list = w_list = [10,5,10,5,10,5,10,5]
    list_index_L2_by_mode = [0,0,1,2,3,3]
    list_modeidx_abs = [0,1,3,4,6,7]
    list_zmemmodeidx_abs = [0,1,2,3,4,5,6,7]
    list_l2idx_abs = [0,1,2,3]
    list_activel2idx_abs = [0,1,2,3]
    d_zmem = calc_delta_zmem(
        z_mem,
        lavg_list,
        g_list,
        w_list,
        list_index_L2_by_mode,
        list_modeidx_abs,
        hops.basis.noise_memory.list_zmemmodeidx_abs,
        list_l2idx_abs,
        list_activel2idx_abs,
    )
    # d_zmem[i] = l_avg * np.conj(g) - np.conj(w) * z_mem[i]
    assert len(d_zmem) == len(z_mem)
    assert d_zmem[0] == (1.0*10.0) - (10.0*1.0)
    assert d_zmem[1] == (1.0*5.0) - (5.0*2.0)
    assert d_zmem[2] == (0.0*10.0) - (10.0*3.0)
    assert d_zmem[3] == (-1.0*5.0) - (5.0*4.0)
    assert d_zmem[4] == (1.0*10.0) - (10.0*5.0)
    assert d_zmem[5] == (0.0*5.0) - (5.0*6.0)
    assert d_zmem[6] == (-1.0*10.0) - (10.0*7.0)
    assert d_zmem[7] == (-1.0*5.0) - (5.0*8.0)


def test_compress_zmem():
    """
    This is a test to ensure that all modes corresponding to
    each L-operator is summed correctly
    """
    z_mem = [10, 5, 0, 0, -10, -5, 0, 0]
    list_zmemactivemodeidx_rel = [0,1,2,3,4,5,6,7]
    list_index_L2_by_hmode = [0,0,1,1,2,2,3,3]

    z_compress = compress_zmem(
        z_mem,
        list_index_L2_by_hmode,
        list_zmemactivemodeidx_rel,
    )
    assert len(z_compress) == 4
    assert z_compress[0] == 15.0
    assert z_compress[1] == 0.0
    assert z_compress[2] == -15.0
    assert z_compress[3] == 0.0

    # Now we test to ensure that the method can handle partial bases.

    # The z_mem array can be larger than the relindex_mode_active list, but it must
    # contain the indices therein.

    # We start with a two mode per site system, but leave some modes out.
    z_mem = [1,2,3,4,5,6,7,8]
    list_index_L2_by_hmode = [0,0,1,2]
    list_zmemactivemodeidx_rel = [0,1,5,7]

    z_compress = compress_zmem(
        z_mem,
        list_index_L2_by_hmode,
        list_zmemactivemodeidx_rel
    )
    # The length of z_compress is the number of unique L2-indices in "list_index_L2_by_hmode"
    assert len(z_compress) == 3
    assert z_compress[0] == 1 + 2
    assert z_compress[1] ==  6
    assert z_compress[2] == 8

    # Tests that the compression still works when list_index_L2_by_hmode is not trivial
    z_mem = [1,2,3,4,5,6,7,8]
    list_index_L2_by_hmode = [0,0,0,0,1,2,3,3]
    list_zmemactivemodeidx_rel = [0,1,2,3,4,5,6,7]
    z_compress = compress_zmem(
        z_mem,
        list_index_L2_by_hmode,
        list_zmemactivemodeidx_rel
    )
    assert len(z_compress) == 4
    assert z_compress[0] == 1 + 2 + 3 + 4
    assert z_compress[1] ==  5
    assert z_compress[2] == 6
    assert z_compress[3] == 7 + 8
