# tests/test_hops_tensor_eom.py
import numpy as np
import pytest
import scipy as sp

from mesohops.basis.hops_modes import HopsModes
from mesohops.basis.hops_noise_memory import HopsNoiseMemory
from mesohops.basis.hops_system import HopsSystem
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.hops_tensor_basis import HopsTensorBasis
from mesohops.eom.eom_functions import calc_delta_zmem, operator_expectation
from mesohops.tensor.hops_tensor_eom import HopsTensorEOM
from mesohops.tensor.tensor_eom_functions import tensor_matvec_prod
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.tensor_operations import (
    _statenumber_offsets,
    extract_psi,
    tensor_add,
)

__title__ = 'Unit Tests for HopsTensorEOM'
__author__ = 'N. Covalsen'
__maintainer__ = 'N. Covalsen'

# ---- shared fixtures (dimer-of-dimers, same as test_hops_tensor_unit.py) ----
nsite = 4
e_lambda = 20.0
gamma = 50.0
temp = 140.0
(g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

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
    'HAMILTONIAN': np.array(hs, dtype=np.complex128),
    'GW_SYSBATH': gw_sysbath,
    'L_HIER': lop_list,
    'L_NOISE1': lop_list,
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': gw_sysbath,
}
psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
psi_0[2] = 1.0
k_max = 2
state_list = np.arange(nsite)
delta_s = 0
eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}


def _make_tb(sp=sys_param, ds=delta_s, psi=psi_0, sl=state_list):
    """Creates an initialized HopsTensorBasis from sys_param dict."""
    system = HopsSystem(sp)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    tb = HopsTensorBasis(system, mode, noise_memory)
    system.initialize(ds > 0, psi)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = sl
    tb.initialize(ds)
    return tb


def _make_eom(method='fullstate', flag_mpo_optimize=True):
    """Creates an initialized HopsTensorEOM for the dimer-of-dimers."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
        'FLAG_MPO_OPTIMIZE': flag_mpo_optimize,
    }
    tb = _make_tb()
    ht = HopsTensorWavefunction(
        k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param,
    )
    ht.initialize(psi_0, tb.system)
    return HopsTensorEOM(ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_param)


def _make_noise(eom):
    """Returns non-trivial z_mem, z_rnd, z_rnd2 arrays sized for eom.

    Each array uses a different complex prefactor times np.arange to break
    all symmetries and ensure z_rnd2 is never zero.
    """
    n_mem = len(eom.noise_memory.list_zmemmodeidx_abs)
    n_l2 = len(eom.mode.list_l2idx_abs)
    z_mem = (0.01 + 0.02j) * np.arange(n_mem, dtype=np.complex128)
    z_rnd = (0.03 - 0.01j) * np.arange(n_l2, dtype=np.complex128)
    z_rnd2 = (-0.02 + 0.04j) * np.arange(n_l2, dtype=np.complex128)
    return z_mem, z_rnd, z_rnd2


def _make_noise_from_basis(tb):
    """Returns non-trivial z_mem, z_rnd, z_rnd2 from a HopsTensorBasis.

    Same prefactors as _make_noise, for use in tests that construct EOM
    instances manually from a shared basis.
    """
    n_mem = len(tb.noise_memory.list_zmemmodeidx_abs)
    n_l2 = len(tb.mode.list_l2idx_abs)
    z_mem = (0.01 + 0.02j) * np.arange(n_mem, dtype=np.complex128)
    z_rnd = (0.03 - 0.01j) * np.arange(n_l2, dtype=np.complex128)
    z_rnd2 = (-0.02 + 0.04j) * np.arange(n_l2, dtype=np.complex128)
    return z_mem, z_rnd, z_rnd2


# ============================================================
# TEST SUITE: build_generator()
# ============================================================


# ------------------------------------------------------------
# TEST: Return value has correct shape and type
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_returns_correct_shape():
    # This case tests that dz_dt has the same shape as z_mem and is complex.
    eom = _make_eom()
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    dz_dt = eom.build_generator(z_mem, z_rnd, z_rnd2)
    assert dz_dt.shape == z_mem.shape
    assert np.iscomplexobj(dz_dt)
    # Cross-validation: fullstate and statenumber dz_dt should agree
    eom_sn = _make_eom('number')
    z_mem_sn, z_rnd_sn, z_rnd2_sn = _make_noise(eom_sn)
    dz_dt_sn = eom_sn.build_generator(z_mem_sn, z_rnd_sn, z_rnd2_sn)
    np.testing.assert_allclose(
        dz_dt, dz_dt_sn, atol=1e-8,
        err_msg='dz_dt should agree between fullstate and statenumber',
    )


# ------------------------------------------------------------
# TEST: MPO cores populated for fullstate representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_populates_mpo_cores_fullstate():
    # This case tests that mpo_cores is empty before and non-empty after.
    # The fullstate MPO is a single chain: one system core followed by one
    # core per bath mode, so build_generator should produce n_lop*modes_per_state + 1
    # 4-D cores.
    eom = _make_eom('fullstate')
    assert eom.mpo_cores == []
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    eom.build_generator(z_mem, z_rnd, z_rnd2)
    # Analytical: fullstate MPO has 1 state core + n_total_modes mode cores
    n_modes = sum(eom.wavefunction.M1_modes_per_state)
    expected_n_cores = 1 + n_modes
    assert len(eom.mpo_cores) == expected_n_cores, (
        f'Expected {expected_n_cores} MPO cores, got {len(eom.mpo_cores)}'
    )
    # Each core is a 4-index MPO tensor (bL, d_out, d_in, bR)
    assert eom.mpo_cores[0].ndim == 4


# ------------------------------------------------------------
# TEST: MPO cores populated for statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_populates_mpo_cores_statenumber():
    # This case tests that mpo_cores is populated for statenumber method.
    # The statenumber MPO interleaves state and mode cores, so build_generator
    # produces n_state * (modes_per_state + 1) 4-D cores after adding the
    # hierarchy and Hamiltonian MPOs together.
    eom = _make_eom('number')
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    eom.build_generator(z_mem, z_rnd, z_rnd2)
    # Analytical: statenumber MPO has n_state + sum(modes_per_state) cores
    n_state = len(eom.wavefunction.M1_modes_per_site)
    n_modes = sum(eom.wavefunction.M1_modes_per_state)
    expected_n_cores = n_state + n_modes
    assert len(eom.mpo_cores) == expected_n_cores, (
        f'Expected {expected_n_cores} MPO cores, got {len(eom.mpo_cores)}'
    )
    # Each core is a 4-index MPO tensor (bL, d_out, d_in, bR)
    assert eom.mpo_cores[0].ndim == 4


# ------------------------------------------------------------
# TEST: Deterministic output for fullstate representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_matches_across_instances_fullstate():
    # This case tests that two EOM instances on identical state produce
    # identical dz_dt, mpo_cores, and derivative outputs.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    tb = _make_tb()

    def make():
        ht = HopsTensorWavefunction(
            k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param,
        )
        ht.initialize(psi_0, tb.system)
        return HopsTensorEOM(ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_param)

    z_mem, z_rnd, z_rnd2 = _make_noise_from_basis(tb)

    eom1 = make()
    eom2 = make()
    dz1 = eom1.build_generator(z_mem, z_rnd, z_rnd2)
    dz2 = eom2.build_generator(z_mem, z_rnd, z_rnd2)

    assert np.allclose(dz1, dz2)
    for c1, c2 in zip(eom1.mpo_cores, eom2.mpo_cores):
        assert np.allclose(c1, c2)
    cores1 = eom1.derivative()
    cores2 = eom2.derivative()
    assert np.allclose(
        np.array([c.sum() for c in cores1]),
        np.array([c.sum() for c in cores2]),
    )


# ------------------------------------------------------------
# TEST: Deterministic output for statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_matches_across_instances_statenumber():
    # This case tests that two EOM instances on identical state produce
    # identical dz_dt and mpo_cores for statenumber representation.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }
    tb = _make_tb()

    def make():
        ht = HopsTensorWavefunction(
            k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param,
        )
        ht.initialize(psi_0, tb.system)
        return HopsTensorEOM(ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_param)

    z_mem, z_rnd, z_rnd2 = _make_noise_from_basis(tb)

    eom1 = make()
    eom2 = make()
    dz1 = eom1.build_generator(z_mem, z_rnd, z_rnd2)
    dz2 = eom2.build_generator(z_mem, z_rnd, z_rnd2)

    assert np.allclose(dz1, dz2)
    for c1, c2 in zip(eom1.mpo_cores, eom2.mpo_cores):
        assert np.allclose(c1, c2)


# ------------------------------------------------------------
# TEST: Linear EOM returns dz_dt = 0 and builds valid MPO
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_linear_eom():
    # CASE: With EQUATION_OF_MOTION = LINEAR, build_generator should
    # return dz_dt = 0 (no memory feedback), build a valid MPO, and
    # omit z_mem from the noise field (z_hat uses only z_rnd, z_rnd2).
    eom_linear_param = {'EQUATION_OF_MOTION': 'LINEAR'}
    for method in ['fullstate', 'number']:
        tensor_param = {
            'MPS_EPSILON': 1e-10,
            'METHOD': method,
            'BOND_DIM_MAX': 20,
        }
        tb = _make_tb()
        ht = HopsTensorWavefunction(
            k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'},
            eom_linear_param,
        )
        ht.initialize(psi_0, tb.system)
        eom = HopsTensorEOM(
            ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive,
            eom_linear_param,
        )
        assert eom.flag_linear is True
        z_mem, z_rnd, z_rnd2 = _make_noise(eom)
        dz_dt = eom.build_generator(z_mem, z_rnd, z_rnd2)
        # dz_dt must be exactly zero for linear EOM
        np.testing.assert_array_equal(
            dz_dt, np.zeros_like(z_mem),
            err_msg=f'dz_dt should be zero for LINEAR EOM ({method})',
        )
        # MPO should still be populated
        assert len(eom.mpo_cores) > 0, (
            f'mpo_cores should be populated for LINEAR EOM ({method})'
        )


# ------------------------------------------------------------
# TEST: dz_dt matches by-hand formula
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_dz_dt_values():
    # CASE: Verify dz_dt values against the analytical formula
    # dz/dt[i] = <L> * conj(g) - conj(w) * z_mem[i]
    # for a non-adaptive, non-linear EOM.
    eom = _make_eom('fullstate')
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    dz_dt = eom.build_generator(z_mem, z_rnd, z_rnd2)
    # Compute expected dz_dt by hand using the same formula
    psi = eom.wavefunction.psi
    list_L2_coo = eom.mode.list_L2_coo
    list_expect_L2 = [
        operator_expectation(list_L2_coo[idx], psi)
        for idx in range(len(list_L2_coo))
    ]
    dz_expected = calc_delta_zmem(
        z_mem,
        list_expect_L2,
        eom.noise_memory.list_zmemg_abs,
        eom.noise_memory.list_zmemw_abs,
        eom.mode.list_index_L2_by_hmode,
        eom.mode.list_modeidx_abs,
        eom.noise_memory.list_zmemmodeidx_abs,
        eom.mode.list_l2idx_abs,
        eom.system.list_activel2idx_abs,
    )
    np.testing.assert_allclose(
        dz_dt, dz_expected, atol=1e-12,
        err_msg='dz_dt does not match by-hand calc_delta_zmem formula',
    )


# ============================================================
# TEST SUITE: derivative()
# ============================================================


# ------------------------------------------------------------
# TEST: Returns list of cores with correct length
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_derivative_returns_cores_list():
    # This case tests that derivative() returns a list of MPS cores
    # with the same length as list_cores_phi.
    eom = _make_eom()
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    eom.build_generator(z_mem, z_rnd, z_rnd2)
    cores = eom.derivative()
    assert isinstance(cores, list)
    assert len(cores) == len(eom.wavefunction.list_cores_phi)
    # Cross-validation: derivative from fullstate and statenumber
    # should produce the same physical state (psi component)
    eom_sn = _make_eom('number')
    z_mem_sn, z_rnd_sn, z_rnd2_sn = _make_noise(eom_sn)
    eom_sn.build_generator(z_mem_sn, z_rnd_sn, z_rnd2_sn)
    cores_sn = eom_sn.derivative()
    psi_deriv = extract_psi(
        cores, eom.wavefunction.method, eom.wavefunction.M1_modes_per_state,
    )
    psi_deriv_sn = extract_psi(
        cores_sn,
        eom_sn.wavefunction.method,
        eom_sn.wavefunction.M1_modes_per_state,
    )
    np.testing.assert_allclose(
        psi_deriv, psi_deriv_sn, atol=1e-8,
        err_msg='Derivative psi should agree between representations',
    )


# ------------------------------------------------------------
# TEST: No side effects on list_cores_phi
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_derivative_no_side_effects_on_list_cores_phi():
    # This case tests that derivative() does not mutate wavefunction.list_cores_phi.
    # Note: only covers the non-TDVP path. In the TDVP path, build_generator
    # itself recenters list_cores_phi (documented side effect), but derivative()
    # remains pure regardless.
    eom = _make_eom()
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    eom.build_generator(z_mem, z_rnd, z_rnd2)
    snapshot = [c.copy() for c in eom.wavefunction.list_cores_phi]
    eom.derivative()
    for before, after in zip(snapshot, eom.wavefunction.list_cores_phi):
        assert np.allclose(before, after)


# ============================================================
# TEST SUITE: _construct_MPO() — statenumber round-trip
# ============================================================


# ------------------------------------------------------------
# TEST: _construct_MPO fuse-add-unfuse round-trip produces rank-4 cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_construct_mpo_statenumber_roundtrip():
    # This case tests that _construct_MPO for statenumber representation
    # produces rank-4 MPO cores whose physical dimensions are consistent
    # with the original unfused cores (d_out == d_in == sqrt(fused_dim)).
    eom = _make_eom('number')
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    eom.build_generator(z_mem, z_rnd, z_rnd2)
    for i, core in enumerate(eom.mpo_cores):
        assert core.ndim == 4, f'Core {i} should be rank-4 after unfuse'
        w_l, d_out, d_in, w_r = core.shape
        assert d_out == d_in, (
            f'Core {i} physical dims mismatch: d_out={d_out}, d_in={d_in}'
        )
    # Also verify _list_cores_op and _list_cores_ham were unfused back to rank-4
    for i, core in enumerate(eom._list_cores_op):
        assert core.ndim == 4, f'_list_cores_op[{i}] should be rank-4 after unfuse'
    for i, core in enumerate(eom._list_cores_ham):
        assert core.ndim == 4, f'_list_cores_ham[{i}] should be rank-4 after unfuse'


# ============================================================
# TEST SUITE: _build_mpo_parts() — error paths
# ============================================================


# ------------------------------------------------------------
# TEST: Invalid method raises UnsupportedRequest
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_invalid_method_raises():
    # This case tests that _build_mpo_parts raises UnsupportedRequest when
    # wavefunction.method is not a recognized representation.
    eom = _make_eom('fullstate')
    # Monkey-patch the method to an invalid value after setup,
    # then call _build_mpo_parts directly (build_generator fails earlier
    # at phi_0 extraction before reaching the method dispatch).
    eom.wavefunction.method = 'bogus'
    with pytest.raises(UnsupportedRequest):
        eom._build_mpo_parts(
            np.zeros(eom.system.size, dtype=np.complex128),
            np.zeros(eom.system.size, dtype=np.complex128),
            0.0,
        )


# ------------------------------------------------------------
# TEST: Invalid normalization raises UnsupportedRequest
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_invalid_normalization_raises():
    # This case tests that _build_mpo_parts raises UnsupportedRequest
    # when an unsupported normalization is passed.
    eom = _make_eom('fullstate')
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    # Set invalid normalization on the instance and verify _build_mpo_parts raises
    eom.normalization = 'unknown'
    with pytest.raises(UnsupportedRequest):
        eom._build_mpo_parts(
            np.zeros(eom.system.size, dtype=np.complex128),
            np.zeros(eom.system.size, dtype=np.complex128),
            0.0,
        )


# ------------------------------------------------------------
# TEST: nearest_neighbors_ham branch uses nn MPO builder
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_nn_ham_branch():
    # This case tests that _build_mpo_parts takes the
    # nearest_neighbors_ham=True branch for the dimer-of-dimers
    # Hamiltonian (tridiagonal). The nn MPO has bond dimension 4
    # for the Hamiltonian cores, while the general MPO has
    # bond dimension 4 + 2*(n_state - 2). We verify that the
    # nn branch produces a smaller Hamiltonian bond dimension.
    # The separate Hamiltonian MPO under test is built only without the
    # topology optimization, which otherwise absorbs it into one generator.
    eom_nn = _make_eom('number', flag_mpo_optimize=False)
    assert eom_nn.system.flag_nearest_neighbor_ham is True
    z_conj = (0.01 + 0.02j) * np.arange(
        eom_nn.mode.n_l2, dtype=np.complex128,
    )
    L_conj_avg = np.zeros(eom_nn.mode.n_l2, dtype=np.complex128)
    eom_nn._build_mpo_parts(z_conj, L_conj_avg, 0.0)
    nn_ham_bond = max(c.shape[-1] for c in eom_nn._list_cores_ham)
    # Force general path by patching nearest_neighbors_ham
    eom_gen = _make_eom('number', flag_mpo_optimize=False)
    eom_gen.system.flag_nearest_neighbor_ham = False
    eom_gen.mpo_builder.flag_nearest_neighbor_ham = False
    eom_gen._build_mpo_parts(z_conj, L_conj_avg, 0.0)
    gen_ham_bond = max(c.shape[-1] for c in eom_gen._list_cores_ham)
    # nn bond dim (4) should be strictly less than general (4 + 2*(n-2))
    assert nn_ham_bond < gen_ham_bond, (
        f'nn ham bond {nn_ham_bond} should be < general {gen_ham_bond}'
    )


# ============================================================
# TEST SUITE: build_generator() — flag_norm=False
# ============================================================


def _make_eom_unnormalized(method='fullstate'):
    """Create a HopsTensorEOM with unnormalized (Linear) EOM."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    eom_param_linear = {'EQUATION_OF_MOTION': 'LINEAR'}
    tb = _make_tb()
    ht = HopsTensorWavefunction(
        k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param_linear,
    )
    ht.initialize(psi_0, tb.system)
    return HopsTensorEOM(ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_param_linear)


# ------------------------------------------------------------
# TEST: flag_norm=False sets norm_corr=0 and produces valid MPO
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_flag_norm_false_runs():
    # This case tests that build_generator completes without error
    # when flag_norm is False (Linear EOM), producing valid MPO cores.
    eom = _make_eom_unnormalized('fullstate')
    assert not eom.wavefunction.flag_norm, 'flag_norm should be False'
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    dz_dt = eom.build_generator(z_mem, z_rnd, z_rnd2)
    # MPO cores should be populated
    assert len(eom.mpo_cores) > 0
    # dz_dt should have the right shape
    assert dz_dt.shape == z_mem.shape
    # Differential: unnormalized MPO cores should differ from normalized.
    # Note: dz_dt (the z_mem derivative) is identical between representations
    # because norm_corr only enters the MPO, not the memory term.
    eom_norm = _make_eom('fullstate')
    z_mem_norm, z_rnd_norm, z_rnd2_norm = _make_noise(eom_norm)
    eom_norm.build_generator(z_mem_norm, z_rnd_norm, z_rnd2_norm)
    any_core_differs = any(
        not np.allclose(c1, c2)
        for c1, c2 in zip(eom.mpo_cores, eom_norm.mpo_cores)
    )
    assert any_core_differs, (
        'Unnormalized and normalized MPO cores should differ when norm_corr != 0'
    )


# ------------------------------------------------------------
# TEST: Unnormalized MPO differs from normalized MPO
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_norm_vs_unnorm_differ():
    # This case tests that the norm correction term actually changes
    # the MPO. With identical inputs, normalized and unnormalized
    # EOMs should produce different MPO cores.
    eom_norm = _make_eom('fullstate')
    eom_unnorm = _make_eom_unnormalized('fullstate')
    z_mem_n, z_rnd_n, z_rnd2_n = _make_noise(eom_norm)
    z_mem_u, z_rnd_u, z_rnd2_u = _make_noise(eom_unnorm)
    eom_norm.build_generator(z_mem_n, z_rnd_n, z_rnd2_n)
    eom_unnorm.build_generator(z_mem_u, z_rnd_u, z_rnd2_u)
    # At least one core should differ (norm_corr != 0 changes the MPO)
    any_differ = any(
        not np.allclose(c1, c2)
        for c1, c2 in zip(eom_norm.mpo_cores, eom_unnorm.mpo_cores)
    )
    assert any_differ, (
        'Normalized and unnormalized MPOs should differ when norm correction is nonzero'
    )


# ============================================================
# TEST SUITE: _build_mpo_parts() — Hamiltonian MPO structure
# ============================================================


# ------------------------------------------------------------
# TEST: Hamiltonian MPO has correct number of cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_ham_structure():
    # This case tests that the Hamiltonian MPO has the correct number
    # of cores: n_state site cores + n_state * modes_per_state mode cores.
    # A separate Hamiltonian MPO exists only without the optimization.
    eom = _make_eom('number', flag_mpo_optimize=False)
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._build_mpo_parts(z_conj_t, L_conj_avg, 0.0)
    V1_mps = eom.wavefunction.M1_modes_per_state
    expected = nsite + int(np.sum(V1_mps))
    assert len(eom._list_cores_ham) == expected


# ------------------------------------------------------------
# TEST: Hamiltonian MPO site cores have correct physical dimensions
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_ham_site_dims():
    # This case tests that site cores in the Hamiltonian MPO have
    # physical dimension 2x2 (binary occupation per site).
    # A separate Hamiltonian MPO exists only without the optimization.
    eom = _make_eom('number', flag_mpo_optimize=False)
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._build_mpo_parts(z_conj_t, L_conj_avg, 0.0)
    offsets = _statenumber_offsets(eom.wavefunction.M1_modes_per_state)
    for site in range(nsite):
        idx = offsets[site]
        core = eom._list_cores_ham[idx]
        assert core.shape[1] == 2 and core.shape[2] == 2, (
            f'Site core {site} has shape {core.shape}, expected (*, 2, 2, *)'
        )


# ------------------------------------------------------------
# TEST: Fullstate _list_cores_ham is empty after build
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_fullstate_list_cores_ham_empty():
    # CASE: For fullstate representation, the Hamiltonian is embedded
    # directly into _list_cores_op by build_fullstate_mpo. _list_cores_ham should
    # remain an empty list.
    eom = _make_eom('fullstate')
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._build_mpo_parts(z_conj_t, L_conj_avg, 0.0)
    assert eom._list_cores_ham == [], (
        f'_list_cores_ham should be empty for fullstate, got {len(eom._list_cores_ham)} cores'
    )
    assert len(eom._list_cores_op) > 0, '_list_cores_op should be populated'


# ------------------------------------------------------------
# TEST: Hamiltonian MPO mode cores are identity
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_ham_mode_cores_identity():
    # This case tests that mode cores in the Hamiltonian MPO act as
    # identity operators (the Hamiltonian doesn't couple to hierarchy modes).
    # A separate Hamiltonian MPO exists only without the optimization.
    eom = _make_eom('number', flag_mpo_optimize=False)
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._build_mpo_parts(z_conj_t, L_conj_avg, 0.0)
    V1_mps = eom.wavefunction.M1_modes_per_state
    offsets = _statenumber_offsets(V1_mps)
    for site in range(nsite):
        idx_base = offsets[site]
        for m in range(V1_mps[site]):
            idx = idx_base + 1 + m
            core = eom._list_cores_ham[idx]
            # For each bond index pair (i, i), the mode core should be identity
            for bond_idx in range(core.shape[0]):
                np.testing.assert_allclose(
                    core[bond_idx, :, :, bond_idx],
                    np.eye(k_max + 1),
                    atol=1e-12,
                    err_msg=f'Mode core {idx} not identity at bond index {bond_idx}',
                )


# ============================================================
# TEST SUITE: __init__()
# ============================================================


# ------------------------------------------------------------
# TEST: Constructor stores references and builds MpoBuilder
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_stores_references():
    eom = _make_eom()
    # This case tests that constructor stores all required references
    # with correct types
    assert isinstance(eom.wavefunction, HopsTensorWavefunction)
    assert isinstance(eom.system, HopsSystem)
    assert isinstance(eom.mode, HopsModes)
    assert isinstance(eom.noise_memory, HopsNoiseMemory)
    assert eom.adaptive is not None
    # This case tests that normalization defaults to 'homps'
    assert eom.normalization == 'homps'
    # This case tests that MpoBuilder is constructed
    assert eom.mpo_builder is not None
    # This case tests that MPO storage starts empty
    assert eom.mpo_cores == []
    assert eom._list_cores_op == []
    assert eom._list_cores_ham == []
    # This case tests that flag_linear is derived from eom_param
    assert eom.flag_linear is False


# ------------------------------------------------------------
# TEST: flag_linear is True for LINEAR EOM
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_flag_linear_true():
    # CASE: When EQUATION_OF_MOTION is LINEAR, flag_linear should be True.
    tb = _make_tb()
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    eom_linear = {'EQUATION_OF_MOTION': 'LINEAR'}
    ht = HopsTensorWavefunction(
        k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_linear,
    )
    ht.initialize(psi_0, tb.system)
    eom = HopsTensorEOM(
        ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_linear,
    )
    assert eom.flag_linear is True


# ============================================================
# TEST SUITE: refresh_builder()
# ============================================================


# ------------------------------------------------------------
# TEST: refresh_builder updates builder state count
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_refresh_builder_updates_state_count():
    eom = _make_eom()
    original_n = eom.mpo_builder.n_state
    eom.system.state_list = [0, 2]
    eom.refresh_builder()
    # Analytical: n_state should match the new state list length
    assert eom.mpo_builder.n_state == 2
    assert eom.mpo_builder.n_state != original_n


# ============================================================
# TEST SUITE: _build_mpo_parts() — hierarchy MPO structure
# ============================================================


# ------------------------------------------------------------
# TEST: Fullstate hierarchy MPO has correct number of cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_fullstate_n_cores():
    # This case tests that the fullstate hierarchy MPO has 1 system core
    # + n_modes mode cores.
    eom = _make_eom('fullstate')
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._build_mpo_parts(z_conj_t, L_conj_avg, 0.0)
    n_total_modes = int(np.sum(eom.wavefunction.M1_modes_per_state))
    assert len(eom._list_cores_op) == 1 + n_total_modes


# ------------------------------------------------------------
# TEST: Statenumber hierarchy MPO has correct number of cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_statenumber_n_cores():
    # This case tests that the statenumber hierarchy MPO has
    # n_state * (1 + modes_per_state) cores.
    eom = _make_eom('number')
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._build_mpo_parts(z_conj_t, L_conj_avg, 0.0)
    expected = nsite + int(np.sum(eom.wavefunction.M1_modes_per_state))
    assert len(eom._list_cores_op) == expected


# ------------------------------------------------------------
# TEST: Fullstate system core has correct bond dimension
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_fullstate_bondsize():
    # This case tests that the fullstate system core has the expected
    # MPO bond dimension of n_lop_full + 2.
    eom = _make_eom('fullstate')
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._build_mpo_parts(z_conj_t, L_conj_avg, 0.0)
    expected_bond = eom.mode.n_l2 + 2
    assert eom._list_cores_op[0].shape[3] == expected_bond, (
        f'Fullstate system core bond dim = {eom._list_cores_op[0].shape[3]}, '
        f'expected {expected_bond}'
    )


# ------------------------------------------------------------
# TEST: Statenumber site cores have bond dimension 5
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_statenumber_site_bond():
    # This case tests that statenumber site cores have the fixed
    # MPO bond dimension of 5 (except first core left bond = 1). This is the
    # hierarchy-only MPO, so the optimization is off.
    eom = _make_eom('number', flag_mpo_optimize=False)
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._build_mpo_parts(z_conj_t, L_conj_avg, 0.0)
    offsets = _statenumber_offsets(eom.wavefunction.M1_modes_per_state)
    for site in range(nsite):
        idx = offsets[site]
        core = eom._list_cores_op[idx]
        if site == 0:
            assert core.shape[0] == 1, (
                f'First site core left bond should be 1, got {core.shape[0]}'
            )
        else:
            assert core.shape[0] == 5
        if site == nsite - 1:
            # Last site group: final core (last mode core) should have Dr=1
            last_idx = offsets[site] + eom.wavefunction.M1_modes_per_state[site]
            last_core = eom._list_cores_op[last_idx]
            assert last_core.shape[3] == 1, (
                f'Last core right bond should be 1, got {last_core.shape[3]}'
            )
        else:
            assert core.shape[3] == 5


# ------------------------------------------------------------
# TEST: Statenumber hierarchy site cores have correct physical dims
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_mpo_parts_statenumber_site_phys_dims():
    # This case tests that statenumber site cores have physical
    # dimension 2 (binary occupied/unoccupied) and mode cores have
    # physical dimension k_max + 1.
    eom = _make_eom('number')
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._build_mpo_parts(z_conj_t, L_conj_avg, 0.0)
    offsets = _statenumber_offsets(eom.wavefunction.M1_modes_per_state)
    d_mode = k_max + 1
    for site in range(nsite):
        idx_site = offsets[site]
        core_site = eom._list_cores_op[idx_site]
        assert core_site.shape[1] == 2, (
            f'Site {site} core d_out should be 2, got {core_site.shape[1]}'
        )
        assert core_site.shape[2] == 2, (
            f'Site {site} core d_in should be 2, got {core_site.shape[2]}'
        )
        for m in range(eom.wavefunction.M1_modes_per_state[site]):
            idx_mode = idx_site + 1 + m
            core_mode = eom._list_cores_op[idx_mode]
            assert core_mode.shape[1] == d_mode, (
                f'Mode core {idx_mode} d_out should be {d_mode}, '
                f'got {core_mode.shape[1]}'
            )
            assert core_mode.shape[2] == d_mode, (
                f'Mode core {idx_mode} d_in should be {d_mode}, '
                f'got {core_mode.shape[2]}'
            )


# ============================================================
# TEST SUITE: _construct_MPO()
# ============================================================


# ------------------------------------------------------------
# TEST: Fullstate _construct_MPO assigns cores_op directly
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_construct_mpo_fullstate_direct():
    # This case tests that for fullstate, _construct_MPO sets
    # mpo_cores = _list_cores_op (no tensor addition or compression).
    eom = _make_eom('fullstate')
    z_conj_t = np.zeros(nsite, dtype=np.complex128)
    L_conj_avg = np.zeros(nsite, dtype=np.complex128)
    eom._construct_MPO(z_conj_t, L_conj_avg, 0.0)
    assert eom.mpo_cores is eom._list_cores_op


# ------------------------------------------------------------
# TEST: MPO bond dim is not capped by tight MPS bond_dim_max
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_construct_mpo_statenumber_not_capped_by_mps_bond_dim():
    # This case tests that _construct_MPO uses its own bond_dim_max
    # (sum of hierarchy + Hamiltonian bond dims) rather than
    # wavefunction.bond_dim_max. With a tight MPS cap of 5, the old code
    # would silently truncate the 9-dim combined MPO; the new code should
    # preserve the full MPO regardless.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 5,   # tight MPS cap — must NOT affect MPO compression
        'EOM': 'Normalized_Nonlinear',
        # The two MPOs whose sum is compressed here are built only without
        # the topology optimization.
        'FLAG_MPO_OPTIMIZE': False,
    }
    tb = _make_tb()
    ht = HopsTensorWavefunction(
        k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param,
    )
    ht.initialize(psi_0, tb.system)
    eom = HopsTensorEOM(ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_param)
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    eom.build_generator(z_mem, z_rnd, z_rnd2)
    # Combined MPO has dim 9 (hierarchy 5 + Ham NN 4); MPS cap of 5
    # must not truncate it below this.
    max_bond = max(c.shape[-1] for c in eom.mpo_cores[:-1])
    assert max_bond > 5, (
        f'MPO bond dim = {max_bond} was capped by MPS bond_dim_max=5'
    )


# ------------------------------------------------------------
# TEST: Compressed MPO is numerically equivalent to uncompressed sum
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_construct_mpo_statenumber_exact_compression_preserves_operator():
    # This case tests that exact MPO compression (epsilon=0) produces the
    # same wavefunction derivative as the uncompressed block-concatenated
    # sum of hierarchy + Hamiltonian MPOs. The compressed and uncompressed
    # operators represent the same linear map. Both MPOs exist only without
    # the topology optimization.
    eom = _make_eom('number', flag_mpo_optimize=False)
    z_mem, z_rnd, z_rnd2 = _make_noise(eom)
    eom.build_generator(z_mem, z_rnd, z_rnd2)
    deriv_compressed, _ = tensor_matvec_prod(
        eom.wavefunction.flat_cores,
        eom.mpo_cores,
        eom.wavefunction.mps_epsilon,
        eom.wavefunction.bond_dim_max,
    )

    # Build the uncompressed block-concatenated sum (no SVD truncation)
    # by calling _build_mpo_parts and fusing/adding without compression.
    eom2 = _make_eom('number', flag_mpo_optimize=False)
    eom2.build_generator(z_mem, z_rnd, z_rnd2)
    # Fuse physical dims to rank-3, form uncompressed block sum, unfuse
    cores_op = [c.reshape(c.shape[0], c.shape[1]*c.shape[2], c.shape[3])
                for c in eom2._list_cores_op]
    cores_ham = [c.reshape(c.shape[0], c.shape[1]*c.shape[2], c.shape[3])
                 for c in eom2._list_cores_ham]
    bond_max = (max(c.shape[-1] for c in cores_op)
                + max(c.shape[-1] for c in cores_ham))
    cores_uncompressed = tensor_add(cores_op, cores_ham, 0, bond_max)
    cores_uncompressed_r4 = []
    for core in cores_uncompressed:
        phys = int(round(np.sqrt(core.shape[1])))
        cores_uncompressed_r4.append(
            core.reshape(core.shape[0], phys, phys, core.shape[2])
        )
    deriv_uncompressed, _ = tensor_matvec_prod(
        eom2.wavefunction.flat_cores,
        cores_uncompressed_r4,
        eom2.wavefunction.mps_epsilon,
        eom2.wavefunction.bond_dim_max,
    )

    for i, (dc, du) in enumerate(zip(deriv_compressed, deriv_uncompressed)):
        np.testing.assert_allclose(
            dc, du, atol=1e-10,
            err_msg=f'Derivative core {i}: compressed vs uncompressed mismatch',
        )


# ============================================================
# TEST SUITE: Linear EOM (EQUATION_OF_MOTION = 'LINEAR')
# ============================================================

eom_param_linear = {'EQUATION_OF_MOTION': 'LINEAR'}


def _make_eom_linear(method='fullstate'):
    """Creates an initialized HopsTensorEOM with the linear EOM."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    tb = _make_tb()
    ht = HopsTensorWavefunction(
        k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param_linear,
    )
    ht.initialize(psi_0, tb.system)
    return HopsTensorEOM(ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive,
                         eom_param_linear)


# ------------------------------------------------------------
# TEST: MPO is independent of z_mem for the linear EOM
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_linear_mpo_independent_of_zmem():
    # This case tests that the linear EOM ignores z_mem when building the MPO,
    # consistent with the scalar EOM where z_hat = conj(z_rnd) only.
    for method in ['fullstate', 'number']:
        eom = _make_eom_linear(method)
        z_mem, z_rnd, z_rnd2 = _make_noise(eom)
        eom.build_generator(np.zeros_like(z_mem), z_rnd, z_rnd2)
        mpo_zero_zmem = [c.copy() for c in eom.mpo_cores]
        eom.build_generator(z_mem, z_rnd, z_rnd2)
        for i, (c_zero, c_nonzero) in enumerate(zip(mpo_zero_zmem, eom.mpo_cores)):
            np.testing.assert_allclose(
                c_zero, c_nonzero, atol=1e-14,
                err_msg=(
                    f'Linear EOM: MPO core {i} changed with z_mem '
                    f'for {method} — z_mem must be suppressed'
                ),
            )


# ============================================================
# LTC fixture
# ============================================================

# sys_param with L_LT_CORR / PARAM_LT_CORR populated. The LTC
# L-operators reuse the site-diagonal projectors from L_HIER so
# that system_functions maps them to the same unique L2 indices.
sys_param_ltc = {
    'HAMILTONIAN': np.array(hs, dtype=np.complex128),
    'GW_SYSBATH': gw_sysbath,
    'L_HIER': lop_list,
    'L_NOISE1': lop_list,
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': gw_sysbath,
    'L_LT_CORR': [loperator[i] for i in range(nsite)],
    'PARAM_LT_CORR': [250.0 / 1000.0] * nsite,
}


def _make_eom_ltc(method='fullstate',
                  eom_p=eom_param):
    """Creates a HopsTensorEOM with LTC parameters populated."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    tb = _make_tb(sp=sys_param_ltc)
    ht = HopsTensorWavefunction(
        k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_p,
    )
    ht.initialize(psi_0, tb.system)
    return HopsTensorEOM(
        ht, tb.system, tb.mode, tb.noise_memory, tb.adaptive, eom_p,
    )


# ============================================================
# TEST SUITE: build_generator() — LTC
# ============================================================


# ------------------------------------------------------------
# TEST: build_generator sets _has_lt_corr = True (nonlinear)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_ltc_sets_flag_nonlinear():
    # This case tests that _has_lt_corr is True after build_generator
    # when the system has nonzero LTC parameters (nonlinear EOM).
    for method in ['fullstate', 'number']:
        eom = _make_eom_ltc(method)
        z_mem, z_rnd, z_rnd2 = _make_noise(eom)
        eom.build_generator(z_mem, z_rnd, z_rnd2)
        assert eom._has_lt_corr is True, (
            f'{method}: _has_lt_corr should be True with LTC params'
        )


# ------------------------------------------------------------
# TEST: build_generator sets _has_lt_corr = True (linear)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_ltc_sets_flag_linear():
    # This case tests that _has_lt_corr is True after build_generator
    # when the system has nonzero LTC parameters (linear EOM).
    linear_eom_param = {'EQUATION_OF_MOTION': 'LINEAR'}
    for method in ['fullstate', 'number']:
        eom = _make_eom_ltc(method, eom_p=linear_eom_param)
        z_mem, z_rnd, z_rnd2 = _make_noise(eom)
        eom.build_generator(z_mem, z_rnd, z_rnd2)
        assert eom._has_lt_corr is True, (
            f'{method}: _has_lt_corr should be True with LTC params (linear)'
        )


# ------------------------------------------------------------
# TEST: no-LTC fixture has _has_lt_corr = False
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_no_ltc_flag_false():
    # This case tests that _has_lt_corr remains False when the
    # system has no LTC parameters.
    for method in ['fullstate', 'number']:
        eom = _make_eom(method)
        z_mem, z_rnd, z_rnd2 = _make_noise(eom)
        eom.build_generator(z_mem, z_rnd, z_rnd2)
        assert eom._has_lt_corr is False, (
            f'{method}: _has_lt_corr should be False without LTC params'
        )


# ------------------------------------------------------------
# TEST: LTC produces different mpo_cores than no-LTC (nonlinear)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_generator_ltc_changes_mpo_nonlinear():
    # This case tests that the MPO cores differ between LTC and
    # no-LTC fixtures. The LTC norm correction alters the state
    # core of the MPO via the Hamiltonian channel.
    for method in ['fullstate', 'number']:
        eom_no_ltc = _make_eom(method)
        z_mem, z_rnd, z_rnd2 = _make_noise(eom_no_ltc)
        eom_no_ltc.build_generator(z_mem, z_rnd, z_rnd2)
        cores_no_ltc = [c.copy() for c in eom_no_ltc.mpo_cores]

        eom_ltc = _make_eom_ltc(method)
        z_mem_ltc, z_rnd_ltc, z_rnd2_ltc = _make_noise(eom_ltc)
        eom_ltc.build_generator(z_mem_ltc, z_rnd_ltc, z_rnd2_ltc)
        cores_ltc = eom_ltc.mpo_cores

        # At least one core must differ (shape or values)
        any_diff = (
            len(cores_no_ltc) != len(cores_ltc)
            or any(
                c1.shape != c2.shape or not np.allclose(c1, c2, atol=1e-14)
                for c1, c2 in zip(cores_no_ltc, cores_ltc)
            )
        )
        assert any_diff, (
            f'{method}: MPO cores should differ between LTC and no-LTC'
        )
