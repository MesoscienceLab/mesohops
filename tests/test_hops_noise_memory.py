import numpy as np
import scipy as sp
import pytest
from types import SimpleNamespace
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.noise.hops_noise import HopsNoise
from mesohops.basis.hops_aux import AuxiliaryVector as AuxiliaryVector
from mesohops.basis.hops_hierarchy import HopsHierarchy as HHier
from mesohops.basis.hops_noise_memory import HopsNoiseMemory
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.physical_constants import hbar

__title__ = "Test of HOPS Zmem"
__author__ = "B. Z. Citty"
__version__ = "1.6"




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
(g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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

psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
psi_0[5] = 1.0
psi_0 = psi_0 / np.linalg.norm(psi_0)

# Adaptive Hops
hops_ad = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param={"MAXHIER": 2},
    eom_param=eom_param,
    integration_param=integrator_param,
)



def test_zmem_indexing():
    """
    This test performs a sequence of zmem basis updates
    to test accuracy
    """
    hops_ad.make_adaptive(1e-3, 1e-3)
    hops_ad.initialize(psi_0)
    # Initialize the test.

    # We set state_list to [5] to minimize the number of modes which have to be in the basis.
    # Each mode associated with a state in the basis must be in the mode basis, so in this case,
    # Modes 10,11 cannot be removed.  The other modes can be removed.
    hops_ad.basis.system.state_list = [5]
    hops_ad.basis.mode.list_modeidx_abs = [10,11,12,13]
    hops_ad.basis.noise_memory.update_zmem_indexing(hops_ad.z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [10,11,12,13]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0,1,2,3]

    # Remove mode 13 from HOPS.mode.  Z_mem mode 13 persists.
    z_mem = [1.0, 1.0, 1.0, 1.0]
    hops_ad.basis.mode.list_modeidx_abs = [10,11,12]
    tuple_index_mapping = hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [10,11,12,13]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0,1,2]
    # The tuple_index_mapping tests that the mapping between the old z_mem to new
    # z_mem is done correctly.  In this case, z_mem is unchanged, so the transformation is
    # trivial.
    assert tuple_index_mapping == ([0,1,2,3],[0,1,2,3])

    # Add mode 9 to the mode basis.
    z_mem = [1.0, 1.0, 1.0 , 1.0]
    hops_ad.basis.mode.list_modeidx_abs = [9, 10, 11, 12]
    tuple_index_mapping = hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [9, 10, 11, 12, 13]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0, 1, 2, 3]
    # In this case, there is a new mode "9" at index 0.  Therefore the old z_mem values
    # at entries [0,1,2,3] are shifted over to [1,2,3,4].
    assert tuple_index_mapping == ([0,1,2,3],[1,2,3,4])

    # Now make the non-mode basis zmem entry decay
    z_mem = [1.0, 1.0, 1.0, 1.0, 1e-10]
    hops_ad.basis.mode.list_modeidx_abs = [9, 10, 11, 12]
    tuple_index_mapping = hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [9, 10, 11, 12]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0, 1, 2, 3]
    assert tuple_index_mapping == ([0,1,2,3],[0,1,2,3])

    # Add modes 7,8,13 and remove mode 9,12
    # Simulaneously decay mode 9
    z_mem = [1e-10, 1.0, 1.0, 1.0]
    hops_ad.basis.mode.list_modeidx_abs = [7,8,10,11,13]
    tuple_index_mapping = hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [7,8,10,11,12,13]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0,1,2,3,5]
    assert tuple_index_mapping == ([1,2,3],[2,3,4])

    # Now make mode 12 decay
    z_mem = [1.0, 1.0, 1.0, 1.0, 1e-10, 1.0]
    hops_ad.basis.mode.list_modeidx_abs = [7,8,10,11,13]
    tuple_index_mapping = hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [7,8,10,11,13]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0,1,2,3,4]
    assert tuple_index_mapping == ([0,1,2,3,5],[0,1,2,3,4])

    # Add modes 9,15.
    z_mem = [1.0, 1.0, 1.0, 1.0, 1.0]
    hops_ad.basis.mode.list_modeidx_abs = [7,8,9,10,11,13,15]
    tuple_index_mapping = hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [7,8,9,10,11,13,15]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0,1,2,3,4,5,6]
    assert tuple_index_mapping == ([0,1,2,3,4],[0,1,3,4,5])

    # Remove modes 8,13 from Mode basis
    z_mem = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    hops_ad.basis.mode.list_modeidx_abs = [7,9,10,11,15]
    tuple_index_mapping = hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [7,8,9,10,11,13,15]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0,2,3,4,6]
    assert tuple_index_mapping == ([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])

    # Decay mode 8, but add it to the basis at the same time (edge case)
    # The z_mem indexing arrays should stay the same.
    z_mem = [1.0, 1e-10, 1.0, 1.0, 1.0, 1.0, 1.0]
    hops_ad.basis.mode.list_modeidx_abs = [7,8,9,10,11,15]
    tuple_index_mapping = hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [7,8,9,10,11,13,15]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0,1,2,3,4,6]
    assert tuple_index_mapping == ([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])


def test_set_zmem_indexing():
    """
    Test that set_zmem_indexing correctly sets list_zmemmodeidx_abs and all
    derived indexing arrays, including when zmem contains modes not in the
    active mode basis.
    """
    # hops_ad is already initialized by test_zmem_indexing (module-level object).
    # Set a known active mode basis and re-initialize noise memory.
    hops_ad.basis.system.state_list = [5]
    hops_ad.basis.mode.list_modeidx_abs = [10, 11, 12]
    hops_ad.basis.noise_memory.initialize()

    g_global = hops_ad.basis.system.param["G"]
    w_global = hops_ad.basis.system.param["W"]

    # Call set_zmem_indexing with a list that includes mode 13, which is NOT
    # in the active mode basis [10, 11, 12].
    hops_ad.basis.noise_memory.set_zmem_indexing([10, 11, 12, 13])

    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [10, 11, 12, 13]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0, 1, 2]
    np.testing.assert_allclose(
        hops_ad.basis.noise_memory.list_zmemg_abs,
        np.array([g_global[m] for m in [10, 11, 12, 13]]),
    )
    np.testing.assert_allclose(
        hops_ad.basis.noise_memory.list_zmemw_abs,
        np.array([w_global[m] for m in [10, 11, 12, 13]]),
    )

    # Change the active mode basis and call again to verify recomputation.
    hops_ad.basis.mode.list_modeidx_abs = [10, 11]
    hops_ad.basis.noise_memory.set_zmem_indexing([9, 10, 11])

    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [9, 10, 11]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [1, 2]
    np.testing.assert_allclose(
        hops_ad.basis.noise_memory.list_zmemg_abs,
        np.array([g_global[m] for m in [9, 10, 11]]),
    )
    np.testing.assert_allclose(
        hops_ad.basis.noise_memory.list_zmemw_abs,
        np.array([w_global[m] for m in [9, 10, 11]]),
    )
# ------------------------------------------------------------
# TEST: set_zmem_indexing raises ValueError for missing active modes
# ------------------------------------------------------------
def test_set_zmem_indexing_missing_active_mode():
    """
    Test that set_zmem_indexing raises a ValueError when the provided
    list_zmemmodeidx_abs does not contain all active modes from
    mode.list_modeidx_abs.
    """
    hops_ad.basis.mode.list_modeidx_abs = [10, 11, 12]
    hops_ad.basis.noise_memory.initialize()

    # Mode 12 is active but absent from the zmem list.
    with pytest.raises(ValueError, match='active modes'):
        hops_ad.basis.noise_memory.set_zmem_indexing([10, 11, 13])


# ------------------------------------------------------------
# TEST: initialize sets all properties correctly
# ------------------------------------------------------------
def test_initialize():
    """
    Test that initialize correctly sets list_zmemmodeidx_abs,
    list_zmemactivemodeidx_rel, list_zmemg_abs, and list_zmemw_abs
    from the current mode basis.
    """
    hops_ad.basis.system.state_list = [5]
    hops_ad.basis.mode.list_modeidx_abs = [10, 11, 12, 13]
    hops_ad.basis.noise_memory.initialize()

    g_global = hops_ad.basis.system.param['G']
    w_global = hops_ad.basis.system.param['W']

    # This case tests that zmem modes match the active mode basis
    assert list(hops_ad.basis.noise_memory.list_zmemmodeidx_abs) == [10, 11, 12, 13]

    # This case tests that relative indices are sequential [0..n-1]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0, 1, 2, 3]

    # This case tests that g values match global G for each mode
    np.testing.assert_allclose(
        hops_ad.basis.noise_memory.list_zmemg_abs,
        np.array([g_global[m] for m in [10, 11, 12, 13]]),
    )

    # This case tests that w values match global W for each mode
    np.testing.assert_allclose(
        hops_ad.basis.noise_memory.list_zmemw_abs,
        np.array([w_global[m] for m in [10, 11, 12, 13]]),
    )

    # This case tests the single-mode initialization path on a minimal object
    single_system = SimpleNamespace(
        param={'G': np.array([3.0 + 0.0j]), 'W': np.array([7.0 + 0.0j])}
    )
    single_mode = SimpleNamespace(
        list_modeidx_abs=[0],
        list_g=np.array([single_system.param['G'][0]]),
        list_w=np.array([single_system.param['W'][0]]),
    )
    single_noise_mem = HopsNoiseMemory(single_system, single_mode)
    single_noise_mem.initialize()
    assert list(single_noise_mem.list_zmemmodeidx_abs) == [0]
    assert single_noise_mem.list_zmemactivemodeidx_rel == [0]
    np.testing.assert_allclose(
        single_noise_mem.list_zmemg_abs,
        np.array([single_system.param['G'][0]]),
    )
    np.testing.assert_allclose(
        single_noise_mem.list_zmemw_abs,
        np.array([single_system.param['W'][0]]),
    )


# ------------------------------------------------------------
# TEST: update_zmem_indexing raises ValueError on length mismatch
# ------------------------------------------------------------
def test_update_zmem_length_mismatch():
    """
    Test that update_zmem_indexing raises a ValueError when
    len(z_mem) does not match len(list_zmemmodeidx_abs).
    """
    hops_ad.basis.system.state_list = [5]
    hops_ad.basis.mode.list_modeidx_abs = [10, 11, 12]
    hops_ad.basis.noise_memory.initialize()

    # This case tests z_mem too short
    with pytest.raises(ValueError, match='update_zmem_indexing'):
        hops_ad.basis.noise_memory.update_zmem_indexing([1.0, 1.0])

    # This case tests z_mem too long
    with pytest.raises(ValueError, match='update_zmem_indexing'):
        hops_ad.basis.noise_memory.update_zmem_indexing(
            [1.0, 1.0, 1.0, 1.0]
        )


# ------------------------------------------------------------
# TEST: update_zmem_indexing with unchanged basis is a no-op
# ------------------------------------------------------------
def test_update_zmem_no_change():
    """
    Test that calling update_zmem_indexing when the mode basis
    has not changed produces identity-like index mapping and
    leaves all indexing arrays unchanged.
    """
    hops_ad.basis.system.state_list = [5]
    hops_ad.basis.mode.list_modeidx_abs = [10, 11, 12]
    hops_ad.basis.noise_memory.initialize()
    list_zmemg_abs_before = hops_ad.basis.noise_memory.list_zmemg_abs.copy()
    list_zmemw_abs_before = hops_ad.basis.noise_memory.list_zmemw_abs.copy()

    z_mem = [1.0, 1.0, 1.0]
    tuple_index_mapping = (
        hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    )

    # This case tests that zmem modes are unchanged
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [10, 11, 12]

    # This case tests that relative indices are unchanged
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0, 1, 2]

    # This case tests that the mapping is identity
    assert tuple_index_mapping == ([0, 1, 2], [0, 1, 2])

    # This case tests that g/w arrays are unchanged
    np.testing.assert_allclose(
        hops_ad.basis.noise_memory.list_zmemg_abs, list_zmemg_abs_before
    )
    np.testing.assert_allclose(
        hops_ad.basis.noise_memory.list_zmemw_abs, list_zmemw_abs_before
    )


# ------------------------------------------------------------
# TEST: all ghost modes decay below precision simultaneously
# ------------------------------------------------------------
def test_update_zmem_all_ghosts_decay():
    """
    Test that when multiple ghost modes all decay below precision
    at the same time, they are all truncated in a single call.
    """
    hops_ad.basis.system.state_list = [5]
    hops_ad.basis.mode.list_modeidx_abs = [10, 11, 12, 13]
    hops_ad.basis.noise_memory.initialize()

    # This case creates ghost modes 12 and 13 by removing them
    # from the active basis while z_mem values remain large
    z_mem = [1.0, 1.0, 1.0, 1.0]
    hops_ad.basis.mode.list_modeidx_abs = [10, 11]
    hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [10, 11, 12, 13]

    # This case tests that both ghosts are truncated when they
    # decay below precision simultaneously
    z_mem = [1.0, 1.0, 1e-10, 1e-10]
    hops_ad.basis.mode.list_modeidx_abs = [10, 11]
    tuple_index_mapping = (
        hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    )
    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [10, 11]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0, 1]
    assert tuple_index_mapping == ([0, 1], [0, 1])


def test_update_zmem_all_ghosts_decay_with_mapping_change():
    """
    Test simultaneous ghost decay with a changed tuple_index_mapping caused by
    introducing a new lower-index active mode.
    """
    hops_ad.basis.system.state_list = [5]
    hops_ad.basis.mode.list_modeidx_abs = [10, 11, 12, 13]
    hops_ad.basis.noise_memory.initialize()

    # Modes 12 and 13 are ghosts that decay below precision. At the same time,
    # new active mode 9 is introduced, shifting surviving modes to higher
    # relative indices in the new zmem basis.
    z_mem = [1.0, 1.0, 1e-10, 1e-10]
    hops_ad.basis.mode.list_modeidx_abs = [9, 10, 11]
    tuple_index_mapping = (
        hops_ad.basis.noise_memory.update_zmem_indexing(z_mem)
    )

    assert hops_ad.basis.noise_memory.list_zmemmodeidx_abs == [9, 10, 11]
    assert hops_ad.basis.noise_memory.list_zmemactivemodeidx_rel == [0, 1, 2]
    assert tuple_index_mapping == ([0, 1], [1, 2])
