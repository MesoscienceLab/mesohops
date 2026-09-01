import numpy as np
import pytest
import scipy as sp

from mesohops.basis.hops_modes import HopsModes
from mesohops.basis.hops_noise_memory import HopsNoiseMemory
from mesohops.basis.hops_system import HopsSystem
from mesohops.eom.eom_functions import (
    calc_norm_corr,
    compress_zmem,
    operator_expectation,
)
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.hops_tensor_basis import HopsTensorBasis
from mesohops.tensor.mpo_constructors import build_statenumber_operator_mpo
from mesohops.tensor.tdvp import recenter_to_zero
from mesohops.tensor.tensor_eom_functions import (
    apply_system_operator,
    calc_norm_corr_tensor,
    tensor_matvec_prod,
)
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.tensor_operations import (
    calc_mps_complexity,
    extract_psi,
    phi_aux,
    unflatten_cores,
)

__title__ = 'Unit Tests for Tensor EOM Functions'
__author__ = 'N. Covalsen'
__maintainer__ = 'N. Covalsen'


def _build_mode_l2_map(system, mode):
    """Build the (mode_idx, l2_idx) mapping for calc_norm_corr_tensor tests."""
    dict_state_to_idx = {
        s: i for i, s in enumerate(sorted(system.state_list))
    }
    list_mode_l2_map = []
    for mode_idx in range(len(mode.list_modeidx_abs)):
        abs_hmode_idx = mode.list_modeidx_abs[mode_idx]
        sys_state = system.param['LIST_STATE_INDICES_BY_HMODE'][abs_hmode_idx]
        sys_state_key = int(np.asarray(sys_state).ravel()[0])
        if sys_state_key not in dict_state_to_idx:
            continue
        ordered_state_idx = dict_state_to_idx[sys_state_key]
        list_l2_for_state = system.param[
            'LIST_INDEX_L2_BY_STATE_INDICES'
        ][ordered_state_idx]
        list_mode_l2_map.append((mode_idx, list_l2_for_state[0]))
    return list_mode_l2_map


# ============================================================
# Shared Setup
# ============================================================

nsite = 4
e_lambda = 20.0
gamma = 50.0
temp = 140.0
(g_0, w_0) = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

T3_loperator = np.zeros([4, 4, 4], dtype=np.float64)
list_gw_sysbath = []
list_lop = []
for i in range(nsite):
    T3_loperator[i, i, i] = 1.0
    list_gw_sysbath.append([g_0, w_0])
    list_lop.append(sp.sparse.coo_matrix(T3_loperator[i]))
    list_gw_sysbath.append([-1j * np.imag(g_0), 500.0])
    list_lop.append(T3_loperator[i])

H2_hamiltonian = np.zeros([nsite, nsite])
H2_hamiltonian[0, 1] = 40
H2_hamiltonian[1, 0] = 40
H2_hamiltonian[1, 2] = 10
H2_hamiltonian[2, 1] = 10
H2_hamiltonian[2, 3] = 40
H2_hamiltonian[3, 2] = 40

sys_param = {
    'HAMILTONIAN': np.array(H2_hamiltonian, dtype=np.complex128),
    'GW_SYSBATH': list_gw_sysbath,
    'L_HIER': list_lop,
    'L_NOISE1': list_lop,
    'ALPHA_NOISE1': bcf_exp,
    'PARAM_NOISE1': list_gw_sysbath,
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


def _make_tensor(method):
    """Creates and initializes a HopsTensorWavefunction for the dimer-of-dimers system.
    """
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    tb = _make_tb()
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    return ht


def _make_tensor_with_psi(method, psi):
    """Same as _make_tensor but initializes with a caller-provided psi.

    Lets tests that need a non-default initial state reuse the standard
    fixture without copy-pasting the HopsTensorWavefunction setup.
    """
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    tb = _make_tb(psi=psi)
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi, tb.system)
    return ht


def _make_simple_mps(n_cores, phys_dims, bond_dims):
    """Build a simple MPS with specified structure.

    Parameters
    ----------
    1. n_cores: int
                Number of MPS cores (sites).
    2. phys_dims: list(int)
                  Physical dimension of each core.
    3. bond_dims: list(int)
                  Bond dimensions, length n_cores + 1. bond_dims[0] and
                  bond_dims[-1] should be 1 for open boundary conditions.

    Returns
    -------
    1. cores: list(np.ndarray)
              MPS cores, each shaped (bond_left, phys_dim, bond_right).
    """
    cores = []
    for i in range(n_cores):
        core = np.zeros(
            (bond_dims[i], phys_dims[i], bond_dims[i + 1]),
            dtype=np.complex128,
        )
        core += 0.01 * (
            np.random.randn(*core.shape) + 1j * np.random.randn(*core.shape)
        )
        cores.append(core)
    return cores


# ============================================================
# TEST SUITE: recenter_to_zero()
# ============================================================


# ------------------------------------------------------------
# TEST: Sites 1..L-1 become left-canonical
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_left_canonical():
    # This case tests that after recentering, all sites n >= 1 satisfy
    # M_n^dagger M_n = I (left-canonical form).
    np.random.seed(42)
    cores = _make_simple_mps(4, [2, 3, 3, 2], [1, 3, 4, 3, 1])
    result = recenter_to_zero(cores)
    for n in range(1, len(result)):
        Dl, d, Dr = result[n].shape
        M = result[n].reshape(Dl * d, Dr)
        eye_check = M.conj().T @ M
        np.testing.assert_allclose(
            eye_check,
            np.eye(Dr, dtype=np.complex128),
            atol=1e-12,
            err_msg=f'Site {n} is not left-canonical',
        )


# ------------------------------------------------------------
# TEST: Recentering preserves the represented state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_preserves_state():
    # This case tests that the full contraction of the MPS gives the same
    # state vector before and after recentering.
    np.random.seed(123)
    cores = _make_simple_mps(3, [2, 2, 2], [1, 2, 2, 1])

    def _contract(cs):
        result = cs[0]
        for c in cs[1:]:
            result = np.tensordot(result, c, axes=([-1], [0]))
        return result.squeeze()

    state_before = _contract([c.copy() for c in cores])
    result = recenter_to_zero(cores)
    state_after = _contract(result)
    np.testing.assert_allclose(state_after, state_before, atol=1e-12)


# ------------------------------------------------------------
# TEST: ValueError when last core Dr != 1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_bad_boundary():
    # This case tests that a ValueError is raised when the last core
    # has Dr != 1 (violating open boundary conditions).
    bad_core = np.zeros((1, 2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match='open boundaries'):
        recenter_to_zero([bad_core])


# ------------------------------------------------------------
# TEST: recenter_to_zero raises on internal bond mismatch
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_bond_mismatch():
    # This case tests that a ValueError is raised when adjacent cores
    # have incompatible bond dimensions. The upfront validation catches
    # mismatches for any number of cores, including 2-core MPS.

    # 2-core case: right bond of core_0 (3) != left bond of core_1 (2)
    core_0 = np.ones((1, 2, 3), dtype=np.complex128)
    core_1 = np.ones((2, 2, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match='Bond mismatch'):
        recenter_to_zero([core_0, core_1])

    # 3-core case: right bond of core_1 (4) != left bond of core_2 (2)
    core_0 = np.ones((1, 2, 3), dtype=np.complex128)
    core_1 = np.ones((3, 2, 4), dtype=np.complex128)
    core_2 = np.ones((2, 2, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match='Bond mismatch'):
        recenter_to_zero([core_0, core_1, core_2])


# ------------------------------------------------------------
# TEST: normalize=True makes ||A[0]||_F = 1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_normalize():
    # This case tests that normalize=True rescales the MPS so that
    # the Frobenius norm of A[0] equals 1 and the state direction is preserved.
    np.random.seed(7)
    cores = _make_simple_mps(3, [2, 3, 2], [1, 3, 3, 1])

    def contract(c_list):
        s = c_list[0]
        for c in c_list[1:]:
            s = np.tensordot(s, c, axes=([-1], [0]))
        return s.squeeze().ravel()

    state_orig = contract(cores)
    result = recenter_to_zero(cores, normalize=True)
    norm_A0 = np.linalg.norm(result[0].ravel())
    np.testing.assert_allclose(norm_A0, 1.0, atol=1e-12)
    # The contracted state should be proportional to the original
    state_norm = contract(result)
    ratio = state_orig / state_norm
    np.testing.assert_allclose(
        ratio, ratio[0] * np.ones_like(ratio), atol=1e-12,
        err_msg='Normalized state is not proportional to original',
    )


# ------------------------------------------------------------
# TEST: recenter_to_zero copy=False reuses the input list
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_no_copy():
    # This case tests that copy=False returns the same list object and
    # that the cores are actually left-canonical after the call.
    ht = _make_tensor('fullstate')
    ht.inflate_bonds_to(4, eps=0.01)
    cores = ht.list_cores_phi
    result = recenter_to_zero(cores, copy=False)
    assert result is cores
    # Verify cores 1..L-1 are left-canonical (Q^dQ = I)
    for n in range(1, len(result)):
        Dl, d, Dr = result[n].shape
        Q = result[n].reshape(Dl * d, Dr)
        np.testing.assert_allclose(
            Q.conj().T @ Q, np.eye(Dr), atol=1e-12,
            err_msg=f'Core {n} is not left-canonical after copy=False',
        )


# ------------------------------------------------------------
# TEST: recenter_to_zero copy=True returns independent cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_copy_independence():
    # This case tests that copy=True returns cores that are genuinely
    # independent from the input: mutating the result must not affect
    # the original. A shallow-copy bug (new list object but shared
    # ndarray references) would pass the list-identity check yet leak
    # mutations through; this test catches that.
    ht = _make_tensor('fullstate')
    ht.inflate_bonds_to(4, eps=0.01)
    cores = ht.list_cores_phi
    # Snapshot every input core so we can detect any leaked mutation
    cores_snapshot = [c.copy() for c in cores]
    result = recenter_to_zero(cores, copy=True)
    # Outer list must be a new object
    assert result is not cores
    # Mutate every core of the result; the originals must be unaffected
    for core in result:
        core[...] = 0.0
    for i, (before, after) in enumerate(zip(cores_snapshot, cores)):
        np.testing.assert_array_equal(
            after, before,
            err_msg=(
                f'copy=True leaked: mutating result core {i} also '
                f'modified original cores[{i}]'
            ),
        )


# ------------------------------------------------------------
# TEST: recenter_to_zero with single core preserves state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_single_core():
    # Limiting case: single core MPS, sweep is empty, state preserved
    core = np.array([[[1.0 + 0j], [0.5 + 0.1j]]])  # shape (1, 2, 1)
    result = recenter_to_zero([core])
    np.testing.assert_allclose(result[0], core, atol=1e-14)


# ------------------------------------------------------------
# TEST: recenter_to_zero with two cores (R goes directly to scalar)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_two_cores():
    # This case tests the two-core MPS where the sweep runs exactly once
    # and R folds directly into site 0 (no absorb-into-next-core branch).
    np.random.seed(42)
    cores = _make_simple_mps(2, [3, 2], [1, 4, 1])

    def contract(c_list):
        s = c_list[0]
        for c in c_list[1:]:
            s = np.tensordot(s, c, axes=([-1], [0]))
        return s.squeeze().ravel()

    state_before = contract(cores)
    result = recenter_to_zero(cores)
    state_after = contract(result)
    np.testing.assert_allclose(state_after, state_before, atol=1e-12)


# ------------------------------------------------------------
# TEST: recenter_to_zero on already-canonical MPS is near-no-op
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_already_canonical():
    # This case tests that passing an already-canonical MPS is a near-no-op:
    # both the contracted physical state and the individual cores are
    # preserved (within numerical precision).
    np.random.seed(99)
    cores = _make_simple_mps(3, [2, 3, 2], [1, 3, 3, 1])
    canonical = recenter_to_zero(cores)

    def contract(c_list):
        s = c_list[0]
        for c in c_list[1:]:
            s = np.tensordot(s, c, axes=([-1], [0]))
        return s.squeeze().ravel()

    state_first = contract(canonical)
    result = recenter_to_zero(canonical)
    state_second = contract(result)
    # State-level idempotency: the contracted quantum state is preserved
    np.testing.assert_allclose(state_second, state_first, atol=1e-12)
    # Core-level idempotency: each core is preserved element-wise. Catches
    # per-core drift (e.g. sign/phase shifts) that would cancel at contraction.
    for n in range(len(canonical)):
        np.testing.assert_allclose(
            result[n], canonical[n], atol=1e-12,
            err_msg=f'Core {n} was modified by re-canonicalization',
        )


# ------------------------------------------------------------
# TEST: recenter_to_zero with all bond dims = 1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_trivial_bonds():
    # This case tests the degenerate geometry where every bond is 1,
    # so QR reduces to scalar extraction at every site.
    np.random.seed(11)
    cores = _make_simple_mps(4, [2, 3, 2, 2], [1, 1, 1, 1, 1])

    def contract(c_list):
        s = c_list[0]
        for c in c_list[1:]:
            s = np.tensordot(s, c, axes=([-1], [0]))
        return s.squeeze().ravel()

    state_before = contract(cores)
    result = recenter_to_zero(cores)
    state_after = contract(result)
    np.testing.assert_allclose(state_after, state_before, atol=1e-12)


# ------------------------------------------------------------
# TEST: recenter_to_zero normalize with zero-norm MPS
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_zero_norm():
    # This case tests the norm_phi > 0 guard: a zero MPS should be
    # returned without division by zero. The QR sweep may introduce
    # non-zero Q factors, but A[0] should remain zero (R[0,0] = 0
    # is folded into it), so the overall state is still zero.
    cores = [
        np.zeros((1, 2, 3), dtype=np.complex128),
        np.zeros((3, 3, 1), dtype=np.complex128),
    ]
    result = recenter_to_zero(cores, normalize=True)
    # The orthogonality center (site 0) should be zero
    np.testing.assert_allclose(result[0], 0.0, atol=1e-15)


# ============================================================
# TEST SUITE: tensor_matvec_prod()
# ============================================================


# ------------------------------------------------------------
# TEST: Mismatched core counts raise ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_length_mismatch():
    # This case tests that a ValueError is raised when the MPS and MPO
    # have different numbers of cores.
    ht = _make_tensor('fullstate')
    list_cores_short_mpo = [np.zeros((1, 2, 2, 1), dtype=np.complex128)]
    with pytest.raises(ValueError, match='same number of cores'):
        tensor_matvec_prod(
            ht.list_cores_phi,
            list_cores_short_mpo,
            ht.mps_epsilon,
            ht.bond_dim_max,
        )


# ------------------------------------------------------------
# TEST: Non-trivial MPO (diagonal scaling) produces correct result
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_diagonal_scaling():
    # This case tests that a diagonal scaling MPO (multiply every physical
    # index by 2) correctly doubles the physical wavefunction. This exercises
    # the einsum contraction with non-identity operator entries.
    ht = _make_tensor('fullstate')
    V1_phi_before = ht.psi.copy()
    # Build a 2*I MPO: each core applies 2*identity on the physical index
    list_cores_scale = []
    for core in ht.list_cores_phi:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        mpo_core[0, :, :, 0] = 2.0 * np.eye(phys_dim)
        list_cores_scale.append(mpo_core)
    # Apply the scaling MPO
    list_cores_result, _ = tensor_matvec_prod(
        ht.list_cores_phi,
        list_cores_scale,
        ht.mps_epsilon,
        ht.bond_dim_max,
    )
    V1_phi_after = extract_psi(
        list_cores_result,
        'fullstate',
        ht.M1_modes_per_state,
    )
    # The system core carries the physical dimension; mode cores get identity
    # scaled by 2 each, so phi_0 is multiplied by 2^(n_cores).
    n_cores = len(ht.list_cores_phi)
    expected = V1_phi_before * (2.0**n_cores)
    np.testing.assert_allclose(
        V1_phi_after,
        expected,
        atol=1e-10,
        err_msg='Diagonal scaling MPO did not produce correct result',
    )


# ------------------------------------------------------------
# TEST: Compression respects bond_dim_max
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_compression():
    # This case tests that the SVD compression enforces bond_dim_max.
    # We inflate the MPS bonds first so the contracted result would
    # exceed bond_dim_max=2 without compression.
    ht = _make_tensor('fullstate')
    ht.inflate_bonds_to(4, eps=0.01)
    # Build identity MPO with bond dim 2 to further inflate contracted bonds
    list_cores_mpo = []
    for core in ht.list_cores_phi:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((2, phys_dim, phys_dim, 2), dtype=np.complex128)
        mpo_core[0, :, :, 0] = np.eye(phys_dim)
        mpo_core[1, :, :, 1] = np.eye(phys_dim) * 0.001
        list_cores_mpo.append(mpo_core)
    # Fix boundary bonds to 1
    list_cores_mpo[0] = list_cores_mpo[0][:1, :, :, :]
    list_cores_mpo[-1] = list_cores_mpo[-1][:, :, :, :1]
    # Apply with tight bond_dim_max
    small_bond = 3
    list_cores_result, _ = tensor_matvec_prod(
        ht.list_cores_phi,
        list_cores_mpo,
        1e-10,
        small_bond,
    )
    # Verify all internal bonds respect bond_dim_max
    for i, core in enumerate(list_cores_result):
        assert core.shape[0] <= small_bond, (
            f'Core {i} left bond {core.shape[0]} exceeds bond_dim_max={small_bond}'
        )
        assert core.shape[2] <= small_bond, (
            f'Core {i} right bond {core.shape[2]} exceeds bond_dim_max={small_bond}'
        )
    # Invariant: compressed result should approximate the uncompressed contraction
    list_cores_uncompressed, _ = tensor_matvec_prod(
        ht.list_cores_phi,
        list_cores_mpo,
        1e-14,  # very tight epsilon
        999,    # no bond cap
    )
    psi_compressed = extract_psi(
        list_cores_result, ht.method, ht.M1_modes_per_state,
    )
    psi_uncompressed = extract_psi(
        list_cores_uncompressed, ht.method, ht.M1_modes_per_state,
    )
    np.testing.assert_allclose(
        psi_compressed, psi_uncompressed, atol=1e-4,
        err_msg='Compressed matvec should approximate uncompressed',
    )


# ------------------------------------------------------------
# TEST: Single-site MPS edge case
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_single_site():
    # This case tests that tensor_matvec_prod works correctly when the
    # MPS has exactly one core (the loop body runs once).
    phys_dim = 3
    # Single MPS core: (1, 3, 1)
    list_cores_vec = [np.array([[[1.0 + 0j], [0.5 + 0.3j], [0.0 + 0.2j]]])]
    # Single MPO core: 2*I on phys_dim=3
    mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
    mpo_core[0, :, :, 0] = 2.0 * np.eye(phys_dim)
    list_cores_mpo = [mpo_core]
    list_cores_result, _ = tensor_matvec_prod(
        list_cores_vec,
        list_cores_mpo,
        1e-10,
        20,
    )
    # Should be a single core with the same shape
    assert len(list_cores_result) == 1
    assert list_cores_result[0].shape == (1, phys_dim, 1)
    # Values should be doubled
    np.testing.assert_allclose(
        list_cores_result[0],
        2.0 * list_cores_vec[0],
        atol=1e-14,
    )


# ------------------------------------------------------------
# TEST: Rectangular MPO (dim_out != dim_in) reshapes physical dimension
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_rectangular_mpo():
    # This case tests that tensor_matvec_prod correctly handles an MPO
    # whose physical output dimension differs from its physical input
    # dimension (rectangular operator). The einsum 'LoiR,lir->LloRr'
    # contracts only the i index, so dim_out and dim_in need not match;
    # the output core's physical dim equals the MPO's dim_out.
    dim_in = 2
    dim_out = 3
    # Single MPS core with phys_dim = dim_in
    V1_vec = np.array([1.0 + 0j, 0.5 + 0.2j], dtype=np.complex128)
    list_cores_vec = [V1_vec.reshape(1, dim_in, 1)]
    # Non-trivial rectangular MPO core: embeds the 2-dim vector into 3
    # dimensions. The last row sums the two input components so an
    # index swap or scalar factor bug would fail the value check.
    M2_embed = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.complex128,
    )
    mpo_core = M2_embed.reshape(1, dim_out, dim_in, 1)
    list_cores_mpo = [mpo_core]
    list_cores_result, _ = tensor_matvec_prod(
        list_cores_vec, list_cores_mpo, 1e-10, 20,
    )
    # Output shape carries dim_out as the physical dimension
    assert len(list_cores_result) == 1
    assert list_cores_result[0].shape == (1, dim_out, 1)
    # Output values match the direct rectangular matrix product M @ v
    V1_expected = M2_embed @ V1_vec
    np.testing.assert_allclose(
        list_cores_result[0].reshape(dim_out), V1_expected, atol=1e-12,
        err_msg='Rectangular MPO did not match direct matrix-vector product',
    )


# ------------------------------------------------------------
# TEST: Input cores are not mutated by tensor_matvec_prod
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_input_not_mutated():
    # This case tests that tensor_matvec_prod produces a new result list
    # without modifying the input MPS cores. Element-wise equality against
    # a snapshot of the inputs catches any in-place mutation (e.g. a
    # downstream helper reshaping input cores instead of copies).
    ht = _make_tensor('fullstate')
    # Snapshot copies of every input core so any in-place mutation by
    # tensor_matvec_prod would diverge the originals from the snapshot.
    list_cores_vec_snapshot = [c.copy() for c in ht.list_cores_phi]
    # Build a non-trivial MPO so the contraction path is fully exercised
    list_cores_mpo = []
    for core in ht.list_cores_phi:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        mpo_core[0, :, :, 0] = 2.0 * np.eye(phys_dim)
        list_cores_mpo.append(mpo_core)
    _, _ = tensor_matvec_prod(
        ht.list_cores_phi, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )
    # Verify every input core is unchanged from its pre-call snapshot
    for i, (before, after) in enumerate(
        zip(list_cores_vec_snapshot, ht.list_cores_phi)
    ):
        np.testing.assert_array_equal(
            after, before,
            err_msg=f'Input core {i} was mutated by tensor_matvec_prod',
        )


# ------------------------------------------------------------
# TEST: Nested input cores are not mutated by tensor_matvec_prod
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_input_not_mutated_nested():
    # This case tests that the nested (number) path
    # also preserves input cores. flatten_cores returns a new list but
    # reuses ndarray references, so any in-place mutation downstream
    # would be visible through the original nested structure.
    ht = _make_tensor('number')
    # Snapshot every core inside every group before the call
    list_cores_phi_snapshot = [
        [c.copy() for c in group] for group in ht.list_cores_phi
    ]
    # Build a non-trivial MPO matching the flat core count
    list_cores_mpo = []
    for core in ht.flat_cores:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        mpo_core[0, :, :, 0] = 2.0 * np.eye(phys_dim)
        list_cores_mpo.append(mpo_core)
    _, _ = tensor_matvec_prod(
        ht.list_cores_phi, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )
    # Verify every nested input core is unchanged from its pre-call snapshot
    for s, (group_before, group_after) in enumerate(
        zip(list_cores_phi_snapshot, ht.list_cores_phi)
    ):
        assert len(group_before) == len(group_after), (
            f'Group {s} size changed from {len(group_before)} to {len(group_after)}'
        )
        for i, (before, after) in enumerate(zip(group_before, group_after)):
            np.testing.assert_array_equal(
                after, before,
                err_msg=(
                    f'Nested input core at group {s}, position {i} '
                    f'was mutated by tensor_matvec_prod'
                ),
            )


# ------------------------------------------------------------
# TEST: Empty input raises a clear ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_empty_list():
    # This case tests that tensor_matvec_prod guards against empty input
    # with a clear ValueError rather than leaking an IndexError from a
    # downstream helper (tensor_compress previously indexed into the
    # empty list).
    with pytest.raises(ValueError, match='at least one core'):
        tensor_matvec_prod([], [], 1e-10, 20)


# ------------------------------------------------------------
# TEST: Off-diagonal MPO mixes physical states correctly
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_off_diagonal_mpo():
    # This case tests that tensor_matvec_prod correctly handles an
    # off-diagonal MPO that mixes physical states. Uses a Pauli-X
    # (NOT gate) on the state core of a fullstate MPS:
    # X = [[0,1],[1,0]] swaps |0⟩ ↔ |1⟩.
    # For psi = [a, b], X|psi⟩ = [b, a].
    #
    # We build a 2-state MPS, apply an MPO with X on the state core
    # and identity on mode cores, then verify the physical wavefunction
    # components are swapped.
    nsite_local = 2
    sys_param_local = {
        'HAMILTONIAN': np.zeros([nsite_local, nsite_local], dtype=np.complex128),
        'GW_SYSBATH': list_gw_sysbath[:2],
        'L_HIER': list_lop[:1],
        'L_NOISE1': list_lop[:1],
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': list_gw_sysbath[:2],
    }
    tensor_param_local = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    psi_local = np.array([0.3 + 0.1j, 0.7 - 0.2j], dtype=np.complex128)
    tb = _make_tb(sp=sys_param_local, ds=0, psi=psi_local, sl=np.arange(nsite_local))
    ht = HopsTensorWavefunction(
        k_max, tensor_param_local, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param,
    )
    ht.initialize(psi_local, tb.system)
    # Extract phi before applying MPO
    V1_phi_before = ht.psi.copy()
    # Build an MPO with Pauli-X on state core, identity on mode cores
    list_cores_mpo = []
    for core in ht.list_cores_phi:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        if phys_dim == nsite_local:
            # State core: apply Pauli-X (swap)
            pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            mpo_core[0, :, :, 0] = pauli_x
        else:
            # Mode core: identity
            mpo_core[0, :, :, 0] = np.eye(phys_dim)
        list_cores_mpo.append(mpo_core)
    # Apply the MPO
    list_cores_result, _ = tensor_matvec_prod(
        ht.list_cores_phi,
        list_cores_mpo,
        ht.mps_epsilon,
        ht.bond_dim_max,
    )
    V1_phi_after = extract_psi(
        list_cores_result,
        'fullstate',
        ht.M1_modes_per_state,
    )
    # Pauli-X swaps components: [a, b] → [b, a]
    V1_expected = np.array(
        [V1_phi_before[1], V1_phi_before[0]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(
        V1_phi_after,
        V1_expected,
        atol=1e-10,
        err_msg='Off-diagonal MPO (Pauli-X) did not swap physical states',
    )


# ------------------------------------------------------------
# TEST: Nested statenumber MPS path (flatten/contract/unflatten)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_nested_statenumber():
    # This case tests the is_nested branch: passing a list-of-lists
    # statenumber MPS and verifying the result is correctly unflattened.
    ht = _make_tensor('number')
    V1_phi_before = ht.psi.copy()
    # Build identity MPO matching the flattened core count
    from mesohops.util.tensor_operations import flatten_cores
    flat = flatten_cores(ht.list_cores_phi)
    list_cores_mpo = []
    for core in flat:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        mpo_core[0, :, :, 0] = np.eye(phys_dim)
        list_cores_mpo.append(mpo_core)
    # Pass the nested (list-of-lists) MPS directly
    result, _ = tensor_matvec_prod(
        ht.list_cores_phi, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )
    # Result should be nested (list-of-lists) with same structure
    assert isinstance(result[0], list), (
        'Nested input should produce nested output'
    )
    assert len(result) == len(ht.list_cores_phi)
    for g_res, g_orig in zip(result, ht.list_cores_phi):
        assert len(g_res) == len(g_orig), (
            'Group size should be preserved after flatten/unflatten'
        )
    # Identity MPO should preserve the wavefunction
    V1_phi_after = extract_psi(result, 'number',
                               ht.M1_modes_per_state)
    np.testing.assert_allclose(V1_phi_after, V1_phi_before, atol=1e-10)


# ------------------------------------------------------------
# TEST: Nested input with MPO length mismatch raises ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_length_mismatch_nested():
    # This case tests that the length-mismatch guard fires on the nested
    # trajectory too: a statenumber MPS is flattened first, so an MPO
    # whose length matches neither the outer-group count nor the flat
    # core count must still trip the "same number of cores" check.
    ht = _make_tensor('number')
    list_cores_short_mpo = [np.zeros((1, 2, 2, 1), dtype=np.complex128)]
    with pytest.raises(ValueError, match='same number of cores'):
        tensor_matvec_prod(
            ht.list_cores_phi,
            list_cores_short_mpo,
            ht.mps_epsilon,
            ht.bond_dim_max,
        )


# ------------------------------------------------------------
# TEST: Tuple-of-tuples nested input is routed through the nested path
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_tuple_nested_input():
    # This case tests that a tuple-of-tuples MPS (not list-of-lists) is
    # detected as nested and produces the same result as the equivalent
    # list-of-lists input. Catches regressions of the isinstance check
    # that previously accepted only `list`, silently misclassifying
    # tuple-wrapped nested MPSs as flat and failing in the einsum.
    ht = _make_tensor('number')
    # Build identity MPO matching the flat core count
    list_cores_mpo = []
    for core in ht.flat_cores:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        mpo_core[0, :, :, 0] = np.eye(phys_dim)
        list_cores_mpo.append(mpo_core)
    # Apply with the list-of-lists input (baseline)
    list_result, _ = tensor_matvec_prod(
        ht.list_cores_phi, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )
    # Apply with the tuple-of-tuples copy of the same data
    tuple_input = tuple(tuple(group) for group in ht.list_cores_phi)
    tuple_result, _ = tensor_matvec_prod(
        tuple_input, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )
    # Both results should be nested and structurally identical
    assert isinstance(list_result[0], list), 'list input should stay nested'
    assert isinstance(tuple_result[0], list), 'tuple input should unflatten to lists'
    assert len(list_result) == len(tuple_result)
    for group_list, group_tuple in zip(list_result, tuple_result):
        assert len(group_list) == len(group_tuple)
        for core_list, core_tuple in zip(group_list, group_tuple):
            np.testing.assert_allclose(core_list, core_tuple, atol=1e-14)


# ------------------------------------------------------------
# TEST: Per-site different operators catches mispairing
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_per_site_scaling():
    # This case tests that different operators on different sites are
    # applied to the correct cores. A uniform scaling (2*I on every site)
    # would mask a site-pairing bug; here we use 2*I on the state core
    # and 3*I on mode cores to verify each factor is applied correctly.
    ht = _make_tensor('fullstate')
    V1_phi_before = ht.psi.copy()
    list_cores_mpo = []
    for i, core in enumerate(ht.list_cores_phi):
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        if i == 0:
            mpo_core[0, :, :, 0] = 2.0 * np.eye(phys_dim)
        else:
            mpo_core[0, :, :, 0] = 3.0 * np.eye(phys_dim)
        list_cores_mpo.append(mpo_core)
    result, _ = tensor_matvec_prod(
        ht.list_cores_phi, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )
    V1_phi_after = extract_psi(
        result, 'fullstate', ht.M1_modes_per_state,
    )
    # State core scaled by 2, each of n_mode mode cores scaled by 3
    n_modes = len(ht.list_cores_phi) - 1
    expected = V1_phi_before * 2.0 * (3.0 ** n_modes)
    np.testing.assert_allclose(V1_phi_after, expected, atol=1e-10,
        err_msg='Per-site scaling factors not applied to correct cores')


# ------------------------------------------------------------
# TEST: Non-trivial Hamiltonian-style MPO vs dense H2_op @ psi
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_general_operator():
    # This case tests tensor_matvec_prod against the actual statenumber
    # operator MPO from mpo_constructors. Unlike the identity / scaling /
    # Pauli-X tests, this MPO has:
    #   - Non-trivial bond dimension (> 1) carrying state across sites
    #   - Off-diagonal entries that mix physical indices across sites
    #     via daisy-chained transfer matrices
    # The physical wavefunction after contraction is cross-validated
    # against the direct dense H2_op @ psi product, providing a reference
    # independent of the MPS contraction logic itself.
    ht = _make_tensor('number')
    V1_phi_before = extract_psi(
        ht.list_cores_phi, ht.method, ht.M1_modes_per_state,
    )
    # Non-trivial dense system operator: every entry is nonzero so every
    # bond channel of the MPO (diagonal, left/right transfer, daisy-chain)
    # is exercised.
    H2_op = np.array([
        [1.0 + 0.1j, 0.3 + 0.1j, 0.2 - 0.1j, 0.1 + 0.05j],
        [0.3 - 0.1j, 0.8 + 0.0j, 0.4 + 0.2j, 0.2 + 0.0j],
        [0.2 + 0.1j, 0.4 - 0.2j, 0.5 + 0.0j, 0.3 + 0.1j],
        [0.1 - 0.05j, 0.2 + 0.0j, 0.3 - 0.1j, 0.9 + 0.0j],
    ], dtype=np.complex128)
    list_cores_mpo = build_statenumber_operator_mpo(
        H2_op, nsite, k_max, ht.M1_modes_per_state,
    )
    # Confirm the MPO is genuinely non-trivial (bond dim > 1 somewhere)
    max_bond = max(
        max(c.shape[0] for c in list_cores_mpo),
        max(c.shape[3] for c in list_cores_mpo),
    )
    assert max_bond > 1, f'Expected non-trivial MPO bond, got {max_bond}'
    # Apply the MPO via flat-core path (nested-path coverage is in a
    # dedicated test upstream).
    list_cores_result_flat, _ = tensor_matvec_prod(
        ht.flat_cores, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )
    list_cores_result = unflatten_cores(
        list_cores_result_flat, ht.M1_modes_per_state,
    )
    V1_phi_after = extract_psi(
        list_cores_result, ht.method, ht.M1_modes_per_state,
    )
    # Dense cross-validation: the MPO applies H2_op to the system space
    # and identity to mode cores, so the resulting physical wavefunction
    # equals H2_op @ V1_phi_before.
    V1_expected = H2_op @ V1_phi_before
    np.testing.assert_allclose(
        V1_phi_after, V1_expected, atol=1e-10,
        err_msg='MPS contraction does not match dense H2_op @ psi reference',
    )


# ------------------------------------------------------------
# TEST: Returned complexity scalar matches calc_mps_complexity
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_complexity_scalar_matches_formula():
    # This case tests that the complexity scalar returned by
    # tensor_matvec_prod equals calc_mps_complexity(list_cores_compressed)
    # — i.e. the per-core sum of D_left * D_right * max(D_left, D_right) *
    # d_phys evaluated on the contracted-but-uncompressed cores.
    # Manually compute the post-contraction core shapes so the assertion
    # is independent of the implementation's internal variable name.
    ht = _make_tensor('fullstate')
    # Build a non-trivial 2*I MPO (bond dim 1) so each contracted core
    # has shape (mpo_bond * mps_bond, dim_phys, mpo_bond * mps_bond);
    # for bond-1 inputs that's identical to the input bond shape.
    list_cores_mpo = []
    for core in ht.list_cores_phi:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        mpo_core[0, :, :, 0] = 2.0 * np.eye(phys_dim)
        list_cores_mpo.append(mpo_core)
    # Reproduce the contracted-but-uncompressed core shapes the
    # implementation builds internally (see tensor_eom_functions L92-95):
    # shape (mpo_dl * vec_dl, dim_out, mpo_dr * vec_dr).
    list_uncompressed = []
    for core_mpo, core_vec in zip(list_cores_mpo, ht.list_cores_phi):
        mpo_dl, dim_out, _, mpo_dr = core_mpo.shape
        vec_dl, _, vec_dr = core_vec.shape
        list_uncompressed.append(
            np.zeros(
                (mpo_dl * vec_dl, dim_out, mpo_dr * vec_dr),
                dtype=np.complex128,
            )
        )
    expected_complexity = calc_mps_complexity(list_uncompressed)
    _, complexity = tensor_matvec_prod(
        ht.list_cores_phi, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )
    assert complexity == expected_complexity, (
        f'tensor_matvec_prod complexity {complexity} does not match '
        f'calc_mps_complexity of pre-compress cores {expected_complexity}'
    )


# ------------------------------------------------------------
# TEST: Output cores are 3-D with compatible bonds between neighbors
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_output_bond_compatibility():
    # This case tests that the output MPS has a valid bond structure:
    # every core is 3-dimensional, and the right bond of each core
    # matches the left bond of its neighbor. Catches bugs where the
    # contraction or compression produces mismatched adjacent cores,
    # which would only surface later when the result is used in any
    # downstream tensor operation (contract, extract_psi, etc.).
    ht = _make_tensor('fullstate')
    # Build a non-trivial 2*I MPO so the full contraction + compression
    # pipeline runs (not a degenerate bond-1 pass-through).
    list_cores_mpo = []
    for core in ht.list_cores_phi:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        mpo_core[0, :, :, 0] = 2.0 * np.eye(phys_dim)
        list_cores_mpo.append(mpo_core)
    list_cores_result, _ = tensor_matvec_prod(
        ht.list_cores_phi, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )
    # Every output core must be 3-D (dim_left, dim_phys, dim_right)
    for i, core in enumerate(list_cores_result):
        assert core.ndim == 3, (
            f'Output core {i} is not 3-D (got shape {core.shape})'
        )
    # Bond compatibility: right bond of core i matches left bond of core i+1
    for i in range(len(list_cores_result) - 1):
        dim_right_i = list_cores_result[i].shape[2]
        dim_left_next = list_cores_result[i + 1].shape[0]
        assert dim_right_i == dim_left_next, (
            f'Bond mismatch between cores {i} and {i + 1}: '
            f'right bond of core {i} is {dim_right_i} but left bond '
            f'of core {i + 1} is {dim_left_next}'
        )
    # Left boundary of the first core and right boundary of the last
    # core should both be size 1 (open-boundary MPS)
    assert list_cores_result[0].shape[0] == 1, (
        f'First core left bond is not 1 (got {list_cores_result[0].shape[0]})'
    )
    assert list_cores_result[-1].shape[2] == 1, (
        f'Last core right bond is not 1 (got {list_cores_result[-1].shape[2]})'
    )


# ------------------------------------------------------------
# TEST: Identity MPO preserves auxiliary-core slices, not just phi_0
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_matvec_prod_identity_preserves_aux_slices():
    # This case tests that an identity MPO preserves the full quantum
    # state across the entire system x modes product basis, not just the
    # phi_0 projection that extract_psi (which slices each mode core at
    # index 0) returns. A contraction bug that corrupted auxiliary
    # slices (mode-core indices > 0) while leaving the [0] slice intact
    # would be invisible to extract_psi-based tests; comparing the full
    # tensor contraction catches it.
    # Individual cores aren't compared directly because SVD compression
    # has gauge freedom (sign/phase flips on adjacent bonds that cancel
    # in the contracted state), so element-wise core equality is too
    # strict; the full contraction is the gauge-invariant observable.
    ht = _make_tensor('fullstate')
    # Inflate bonds so mode cores carry non-trivial auxiliary amplitudes
    # (not just [0] = 1). Without this the auxiliary slices are all
    # zero and corruption would be undetectable.
    ht.inflate_bonds_to(4, eps=0.05)
    # Build an identity MPO: one core per MPS core, bond dim 1
    list_cores_mpo = []
    for core in ht.list_cores_phi:
        phys_dim = core.shape[1]
        mpo_core = np.zeros((1, phys_dim, phys_dim, 1), dtype=np.complex128)
        mpo_core[0, :, :, 0] = np.eye(phys_dim)
        list_cores_mpo.append(mpo_core)
    list_cores_result, _ = tensor_matvec_prod(
        ht.list_cores_phi, list_cores_mpo, ht.mps_epsilon, ht.bond_dim_max,
    )

    def _contract_cores(list_cores):
        # Sequentially tensordot adjacent bonds to build the full state
        # across every physical index (system + all modes). Output shape
        # collapses to the product of physical dimensions.
        state = list_cores[0]
        for core in list_cores[1:]:
            state = np.tensordot(state, core, axes=([-1], [0]))
        return state.squeeze()

    state_before = _contract_cores(ht.list_cores_phi)
    state_after = _contract_cores(list_cores_result)
    np.testing.assert_allclose(
        state_after, state_before, atol=1e-10,
        err_msg=(
            'Full state contraction changed under identity MPO; '
            'auxiliary-slice contributions may be corrupted'
        ),
    )


# ============================================================
# TEST SUITE: calc_norm_corr_tensor()
# ============================================================


def _make_norm_corr_args(
    method='fullstate', z_scale=0.0, psi=None,
    z_rnd_func=None, aux_populate=None,
):
    """Helper: builds all arguments for calc_norm_corr_tensor.

    Parameters
    ----------
    1. method: str
    2. z_scale: float
                Scale factor for uniform noise. Ignored if z_rnd_func given.
    3. psi: np.ndarray or None
            Initial state. Defaults to module-level psi_0.
    4. z_rnd_func: callable or None
                   f(n_l2) -> np.ndarray of noise values. Overrides z_scale.
    5. aux_populate: list of (mode_idx, value) or None
                     Manually populate first-order auxiliaries in the MPS.

    Returns
    -------
    1. dict with keys: 'wavefunction', 'z_hat', 'list_avg_L2', 'mode',
       'list_mode_l2_map', and 'system' (for cross-validation).
    """
    if psi is None:
        psi = psi_0

    # Build the tensor wavefunction that calc_norm_corr_tensor consumes
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    tb = _make_tb(psi=psi)
    ht = HopsTensorWavefunction(
        k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param,
    )
    ht.initialize(psi, tb.system)

    # Optionally inject amplitude into first-order mode cores. Use
    # flat_cores + update_phi_from_flat so the same (mode_idx, value)
    # pair targets the correct core in both representations even
    # though the flat layouts differ (fullstate interleaves no state
    # cores among modes; statenumber interleaves one state core per
    # site group).
    if aux_populate is not None:
        flat = [c.copy() for c in ht.flat_cores]
        for (m_idx, val) in aux_populate:
            if method == 'fullstate':
                core_idx = 1 + m_idx
            else:
                # Walk through state groups until m_idx lands in one
                modes_per_state = np.asarray(
                    ht.M1_modes_per_state, dtype=int,
                )
                state_idx = 0
                remaining = int(m_idx)
                while remaining >= modes_per_state[state_idx]:
                    remaining -= modes_per_state[state_idx]
                    state_idx += 1
                offset = sum(
                    1 + int(modes_per_state[s]) for s in range(state_idx)
                )
                core_idx = offset + 1 + remaining
            flat[core_idx][0, 1, 0] = val
        ht.update_phi_from_flat(flat)

    # Pull mode/system indexers needed for downstream construction
    mode = tb.mode
    system = tb.system
    list_l2idx_abs = mode.list_l2idx_abs
    list_index_L2_by_hmode = mode.list_index_L2_by_hmode

    # Compute <L_m> expectation values — second input to calc_norm_corr_tensor
    V1_psi = ht.psi
    list_avg_L2 = [
        operator_expectation(mode.list_L2_coo[i], V1_psi)
        for i in range(len(mode.list_L2_coo))
    ]

    # Stochastic noise vector — uniform z_scale unless a generator is passed
    if z_rnd_func is not None:
        z_rnd = z_rnd_func(len(list_l2idx_abs))
    else:
        z_rnd = np.ones(len(list_l2idx_abs), dtype=np.complex128) * z_scale

    # z_hat: conjugate noise plus compressed memory — the drive for norm correction
    z_mem = np.zeros(
        len(tb.noise_memory.list_zmemmodeidx_abs), dtype=np.complex128,
    )
    z_hat = np.conj(z_rnd[list_l2idx_abs]) + compress_zmem(
        z_mem, list_index_L2_by_hmode,
        tb.noise_memory.list_zmemactivemodeidx_rel,
    )

    result = dict(
        wavefunction=ht,
        psi=V1_psi,
        z_hat=z_hat,
        list_avg_L2=list_avg_L2,
        mode=mode,
        list_index_L2_by_mode=mode.list_index_L2_by_hmode,
    )
    # Store system for cross-validation but not as a calc_norm_corr_tensor arg
    result['_system'] = system
    return result


def _flat_norm_corr(kwargs):
    """Cross-validate calc_norm_corr_tensor against flat calc_norm_corr."""
    ht = kwargs['wavefunction']
    mode = kwargs['mode']
    system = kwargs['_system']
    z_hat = kwargs['z_hat']
    list_avg_L2 = kwargs['list_avg_L2']
    V1_psi = ht.psi
    n_state = len(V1_psi)
    n_modes_total = len(mode.list_modeidx_abs)
    phi_flat = np.zeros(n_state * (1 + n_modes_total), dtype=np.complex128)
    phi_flat[:n_state] = V1_psi
    dict_mode_to_l2 = dict(_build_mode_l2_map(system, mode))
    list_index_phi_L2_mode_flat = []
    for m_idx in range(n_modes_total):
        list_aux_idx_m = [0] * n_modes_total
        list_aux_idx_m[m_idx] = 1
        phi_flat[n_state * (1 + m_idx):n_state * (2 + m_idx)] = phi_aux(
            ht.list_cores_phi, ht.method, ht.M1_modes_per_state,
            list_aux_idx_m,
        )
        if m_idx in dict_mode_to_l2:
            list_index_phi_L2_mode_flat.append(
                (1 + m_idx, dict_mode_to_l2[m_idx], m_idx)
            )
    return calc_norm_corr(
        phi_flat, z_hat, list_avg_L2, mode.list_L2_coo,
        n_state, list_index_phi_L2_mode_flat, mode.list_g, mode.list_w,
    )


# ------------------------------------------------------------
# TEST: Nonzero noise gives nonzero correction
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_norm_corr_tensor_nonzero_noise():
    # Scope: z-component-only regime. With the initial localized state
    # (only zeroth auxiliary populated) phi_aux returns zeros, so the
    # per-mode correction loop contributes nothing and the result
    # collapses to Re(z_hat . list_avg_L2). This test pins the
    # z-component behavior; per-mode loop coverage is in
    # test_calc_norm_corr_tensor_populated_hierarchy and
    # test_calc_norm_corr_tensor_all_modes_active.
    kwargs = _make_norm_corr_args(z_scale=1.0)
    tensor_kwargs = {k: v for k, v in kwargs.items() if k != '_system'}
    result = calc_norm_corr_tensor(**tensor_kwargs)
    # Compute expected leading-order term
    expected = np.real(np.dot(kwargs['z_hat'], kwargs['list_avg_L2']))
    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-12,
        err_msg=(
            'Norm correction should equal Re(z_hat . list_avg_L2) for initial state'
        ),
    )


# ------------------------------------------------------------
# TEST: Statenumber representation gives same result as fullstate
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_norm_corr_tensor_statenumber():
    # This case tests that number produces the same
    # norm correction as fullstate for identical inputs.
    kwargs_full = _make_norm_corr_args(
        method='fullstate',
        z_scale=0.5,
    )
    kwargs_sn = _make_norm_corr_args(
        method='number',
        z_scale=0.5,
    )
    tensor_full = {k: v for k, v in kwargs_full.items() if k != '_system'}
    tensor_sn = {k: v for k, v in kwargs_sn.items() if k != '_system'}
    result_full = calc_norm_corr_tensor(**tensor_full)
    result_sn = calc_norm_corr_tensor(**tensor_sn)
    np.testing.assert_allclose(
        result_sn,
        result_full,
        atol=1e-10,
        err_msg='statenumber and fullstate gave different norm corrections',
    )


# ------------------------------------------------------------
# TEST: Multi-site initial state (superposition)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_norm_corr_tensor_superposition():
    # This case tests that a superposition initial state (amplitude on
    # multiple sites) exercises the L2 operator application more
    # meaningfully than a single-site localized state.
    psi_super = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
    psi_super = psi_super / np.linalg.norm(psi_super)
    kwargs = _make_norm_corr_args(
        psi=psi_super,
        z_rnd_func=lambda n: (0.03 - 0.01j) * np.arange(n, dtype=np.complex128),
    )
    tensor_kwargs = {k: v for k, v in kwargs.items() if k != '_system'}
    result = calc_norm_corr_tensor(**tensor_kwargs)
    assert np.isreal(result)
    expected = _flat_norm_corr(kwargs)
    np.testing.assert_allclose(
        result, expected, atol=1e-12,
        err_msg='Superposition: tensor norm corr does not match flat calc_norm_corr',
    )


# ------------------------------------------------------------
# TEST: Populated hierarchy exercises per-mode loop
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_norm_corr_tensor_populated_hierarchy():
    # This case tests the per-mode correction loop by manually populating
    # the first-order auxiliary in the MPS. With zeroth-order only, phi_aux
    # returns zeros and the loop contributes nothing. Here we set a nonzero
    # first-order auxiliary so that -<phi_0|L|phi_1> + <phi_0|phi_1>*<L>
    # is actually computed.
    # Mode 4 corresponds to site 2 (modes_per_state=2, site 2 → mode 2*2=4).
    kwargs = _make_norm_corr_args(
        z_rnd_func=lambda n: (0.05 + 0.02j) * np.arange(n, dtype=np.complex128),
        aux_populate=[(4, 0.3 + 0.1j)],
    )
    tensor_kwargs = {k: v for k, v in kwargs.items() if k != '_system'}
    result = calc_norm_corr_tensor(**tensor_kwargs)
    assert np.isreal(result)

    # Verify that the per-mode loop actually contributed by comparing against
    # the z-component alone (delta = z_hat . list_avg_L2)
    z_component_only = np.real(np.dot(kwargs['z_hat'], kwargs['list_avg_L2']))
    assert result != z_component_only, (
        'Per-mode loop contributed nothing despite populated auxiliary'
    )

    # Cross-validate against the flat-vector calc_norm_corr
    expected = _flat_norm_corr(kwargs)
    np.testing.assert_allclose(
        result, expected, atol=1e-12,
        err_msg='calc_norm_corr_tensor does not match flat calc_norm_corr',
    )


# ------------------------------------------------------------
# TEST: Statenumber representation with populated hierarchy
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_norm_corr_tensor_statenumber_populated():
    # This case tests that the per-mode correction loop works correctly
    # in statenumber representation by populating a first-order auxiliary
    # and comparing against the fullstate result. Both representations
    # are built from _make_norm_corr_args with identical inputs, which
    # produces the same z_hat deterministically so the cross-check is
    # meaningful without rebuilding the fixture by hand.
    z_rnd_func = lambda n: (0.05 + 0.02j) * np.arange(n, dtype=np.complex128)
    aux_populate = [(4, 0.3 + 0.1j)]

    kwargs_full = _make_norm_corr_args(
        method='fullstate',
        z_rnd_func=z_rnd_func,
        aux_populate=aux_populate,
    )
    kwargs_sn = _make_norm_corr_args(
        method='number',
        z_rnd_func=z_rnd_func,
        aux_populate=aux_populate,
    )
    tensor_full = {k: v for k, v in kwargs_full.items() if k != '_system'}
    tensor_sn = {k: v for k, v in kwargs_sn.items() if k != '_system'}
    result_full = calc_norm_corr_tensor(**tensor_full)
    result_sn = calc_norm_corr_tensor(**tensor_sn)

    # Per-mode loop must have contributed (otherwise the comparison
    # degenerates to the z-component alone and covers nothing new)
    z_only = np.real(
        np.dot(kwargs_full['z_hat'], kwargs_full['list_avg_L2'])
    )
    assert result_full != z_only, (
        'Fullstate per-mode loop contributed nothing'
    )
    np.testing.assert_allclose(
        result_full, result_sn, atol=1e-10,
        err_msg='Statenumber norm correction diverges from fullstate with '
                'populated auxiliary',
    )


# ------------------------------------------------------------
# TEST: calc_norm_corr_tensor with all modes active is nonzero
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_norm_corr_tensor_all_modes_active():
    # This case tests the full norm correction with every state-mode
    # exercised: one first-order auxiliary is populated per system state
    # so the per-mode correction loop contributes across all states
    # (not just a single mode), alongside nonzero noise that drives the
    # z-component term. The result is cross-validated against the
    # flat-vector calc_norm_corr reference at atol=1e-12 — strictly
    # stronger than the prior `abs(result) > 1e-6` check, which only
    # verified the output wasn't trivially zero.
    # Populate one first-order mode auxiliary per system state: with
    # modes_per_state = 2 and nsite = 4, the first mode of each state
    # sits at flat mode indices 0, 2, 4, 6.
    kwargs = _make_norm_corr_args(
        z_rnd_func=lambda n: (0.04 + 0.02j) * np.arange(n, dtype=np.complex128),
        aux_populate=[
            (0, 0.10 + 0.05j),
            (2, 0.15 - 0.03j),
            (4, 0.20 + 0.08j),
            (6, 0.12 - 0.06j),
        ],
    )
    tensor_kwargs = {k: v for k, v in kwargs.items() if k != '_system'}
    result = calc_norm_corr_tensor(**tensor_kwargs)
    assert np.isreal(result)
    # Per-mode loop must actually contribute; otherwise the populated
    # auxiliaries didn't feed through and the test would be z-only.
    z_component_only = np.real(
        np.dot(kwargs['z_hat'], kwargs['list_avg_L2'])
    )
    assert result != z_component_only, (
        'Per-mode loop contributed nothing despite populated auxiliaries '
        'across all states'
    )
    # Cross-validate against the flat-vector reference implementation
    expected = _flat_norm_corr(kwargs)
    np.testing.assert_allclose(
        result, expected, atol=1e-12,
        err_msg=(
            'all_modes_active: calc_norm_corr_tensor does not match '
            'flat calc_norm_corr with all state auxiliaries populated'
        ),
    )


# ------------------------------------------------------------
# TEST: Sign of correction can be negative
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_norm_corr_tensor_sign():
    # Scope: z-component sign-flip property. The leading-order term
    # Re(z_hat . <L>) is linear in z_scale, so flipping z_scale's sign
    # flips the correction sign. This test uses the initial state
    # (per-mode loop inactive) because sign-flipping is a property of
    # the z-component alone; populated-auxiliary behavior is covered by
    # test_calc_norm_corr_tensor_populated_hierarchy and
    # test_calc_norm_corr_tensor_all_modes_active.
    kwargs_pos = _make_norm_corr_args(z_scale=1.0)
    kwargs_neg = _make_norm_corr_args(z_scale=-1.0)
    tensor_pos = {k: v for k, v in kwargs_pos.items() if k != '_system'}
    tensor_neg = {k: v for k, v in kwargs_neg.items() if k != '_system'}
    result_pos = calc_norm_corr_tensor(**tensor_pos)
    result_neg = calc_norm_corr_tensor(**tensor_neg)
    # With opposite z_scales, the leading z_hat . list_avg_L2 term flips sign
    assert result_pos * result_neg < 0, (
        f'Expected opposite signs: z_scale=1.0 gave {result_pos}, '
        f'z_scale=-1.0 gave {result_neg}'
    )
    # Analytical: the leading term Re(z_hat . <L>) flips sign with z_scale,
    # so magnitudes should be approximately equal
    np.testing.assert_allclose(
        abs(result_pos), abs(result_neg), atol=1e-10,
        err_msg='Magnitude should be equal for opposite z_scales',
    )


# ------------------------------------------------------------
# TEST: Single-mode system (n_modes = 1)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_norm_corr_tensor_single_mode():
    # Scope: minimal-system edge case (1 site, 1 bath mode) in the
    # z-component-only regime. Per-mode loop is not exercised here
    # because the goal is to confirm the function runs on a degenerate
    # single-mode fixture without indexing errors; multi-mode per-mode
    # loop coverage lives in test_calc_norm_corr_tensor_populated_hierarchy
    # and test_calc_norm_corr_tensor_all_modes_active.
    H2_hamiltonian_1 = np.array([[0.0]], dtype=np.complex128)
    lop_1 = [sp.sparse.coo_matrix(np.array([[1.0]]))]
    sys_param_1 = {
        'HAMILTONIAN': H2_hamiltonian_1,
        'GW_SYSBATH': [[g_0, w_0]],
        'L_HIER': lop_1,
        'L_NOISE1': lop_1,
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': [[g_0, w_0]],
    }
    psi_1 = np.array([1.0], dtype=np.complex128)
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    tb = _make_tb(sp=sys_param_1, ds=0, psi=psi_1, sl=np.array([0]))
    ht = HopsTensorWavefunction(
        k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param,
    )
    ht.initialize(psi_1, tb.system)

    mode = tb.mode
    system = tb.system
    list_L2_coo = mode.list_L2_coo

    V1_psi = ht.psi
    list_avg_L2 = [
        operator_expectation(list_L2_coo[i], V1_psi) for i in range(len(list_L2_coo))
    ]
    z_hat = np.array([0.5 + 0.3j], dtype=np.complex128)

    result = calc_norm_corr_tensor(
        ht,
        V1_psi,
        z_hat,
        list_avg_L2,
        mode,
        mode.list_index_L2_by_hmode,
    )
    assert np.isreal(result)
    # With 1 mode and initial MPS (zeroth-order only), the per-mode
    # loop contributes nothing, so result should equal Re(z_hat . avg_L2)
    expected = np.real(np.dot(z_hat, list_avg_L2))
    np.testing.assert_allclose(result, expected, atol=1e-12)


# ============================================================
# TEST SUITE: apply_system_operator()
# ============================================================

# Shared swap matrices
# Swaps sites 0<->2 and 1<->3
O2_SWAP_02_13 = np.array(
    [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
    dtype=np.complex128,
)
# Swaps sites 0<->1 and 2<->3
O2_SWAP_01_23 = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)


# ------------------------------------------------------------
# TEST: Projection operator zeros out components
# ------------------------------------------------------------
@pytest.mark.level(1)
@pytest.mark.parametrize('method', ['fullstate', 'number'])
def test_apply_system_operator_projection(method):
    # This case tests that projecting onto site 0 zeros out the
    # entire wavefunction, because psi_0 lives entirely on site 2.
    ht = _make_tensor(method)
    # |0><0| projector: keeps only site 0 amplitude
    O2_proj = np.zeros((nsite, nsite), dtype=np.complex128)
    O2_proj[0, 0] = 1.0
    ht.list_cores_phi = apply_system_operator(
        ht.list_cores_phi, O2_proj, ht.method, ht.k_max,
        ht.M1_modes_per_state, ht.mps_epsilon, ht.bond_dim_max,
    )
    # psi_0 had zero weight on site 0, so projection gives zero everywhere
    np.testing.assert_allclose(ht.psi, 0.0, atol=1e-10)


# ------------------------------------------------------------
# TEST: Sparse operator is handled correctly
# ------------------------------------------------------------
@pytest.mark.level(1)
@pytest.mark.parametrize('method', ['fullstate', 'number'])
def test_apply_system_operator_sparse(method):
    # This case tests that passing a sparse operator directly gives
    # the same result as a dense one. apply_system_operator converts
    # sparse to dense internally via .toarray().
    ht_dense = _make_tensor(method)
    ht_sparse = _make_tensor(method)
    O2_swap_sparse = sp.sparse.coo_matrix(O2_SWAP_01_23)
    ht_dense.list_cores_phi = apply_system_operator(
        ht_dense.list_cores_phi, O2_SWAP_01_23, ht_dense.method, ht_dense.k_max,
        ht_dense.M1_modes_per_state, ht_dense.mps_epsilon, ht_dense.bond_dim_max,
    )
    # Pass sparse directly — apply_system_operator handles conversion
    ht_sparse.list_cores_phi = apply_system_operator(
        ht_sparse.list_cores_phi, O2_swap_sparse,
        ht_sparse.method, ht_sparse.k_max,
        ht_sparse.M1_modes_per_state, ht_sparse.mps_epsilon, ht_sparse.bond_dim_max,
    )
    np.testing.assert_allclose(ht_dense.psi, ht_sparse.psi, atol=1e-12)


# ------------------------------------------------------------
# TEST: Off-diagonal operator swaps populations
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_system_operator_offdiagonal():
    # This case tests that an off-diagonal swap operator correctly
    # permutes the state core entries. psi_0 = [0,0,1,0], so after
    # swapping states 0<->2 and 1<->3, we expect [1,0,0,0].
    ht = _make_tensor('fullstate')
    ht.list_cores_phi = apply_system_operator(
        ht.list_cores_phi, O2_SWAP_02_13, ht.method, ht.k_max,
        ht.M1_modes_per_state, ht.mps_epsilon, ht.bond_dim_max,
    )
    expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    np.testing.assert_allclose(ht.psi, expected, atol=1e-12)


# ------------------------------------------------------------
# TEST: Statenumber off-diagonal swap matches fullstate
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_system_operator_cross_representation():
    # This case tests that an off-diagonal swap operator gives the same
    # result in statenumber and fullstate representations.
    ht_full = _make_tensor('fullstate')
    ht_snum = _make_tensor('number')
    ht_full.list_cores_phi = apply_system_operator(
        ht_full.list_cores_phi, O2_SWAP_02_13, ht_full.method, ht_full.k_max,
        ht_full.M1_modes_per_state, ht_full.mps_epsilon, ht_full.bond_dim_max,
    )
    ht_snum.list_cores_phi = apply_system_operator(
        ht_snum.list_cores_phi, O2_SWAP_02_13, ht_snum.method, ht_snum.k_max,
        ht_snum.M1_modes_per_state, ht_snum.mps_epsilon, ht_snum.bond_dim_max,
    )
    np.testing.assert_allclose(ht_snum.psi, ht_full.psi, atol=1e-10)


# ------------------------------------------------------------
# TEST: Raise operator moves population (both representations)
# ------------------------------------------------------------
@pytest.mark.level(1)
@pytest.mark.parametrize(
    'method', ['fullstate', 'number'],
)
def test_apply_system_operator_raise(method):
    # This case tests a fluorescence-style raise operator |1><0|
    # applied to the initial wavefunction.
    # Custom psi_0 on site 0 (rather than the module-level default at
    # site 2) is needed because |1><0| only acts non-trivially when
    # the input has amplitude on site 0 — starting at site 2 would
    # give a zero result and not distinguish a working operator from
    # a broken one.
    # |1><0| is the simplest off-diagonal operator that moves
    # population between non-adjacent MPS entries. In statenumber
    # representation this forces the operator MPO to route amplitude
    # through the daisy-chained transfer-matrix channels between
    # state cores 0 and 1, exercising cross-site coupling that
    # diagonal operators cannot.
    psi_site0 = np.zeros(nsite, dtype=np.complex128)
    psi_site0[0] = 1.0
    ht = _make_tensor_with_psi(method, psi_site0)
    # Raise operator: |1><0|
    O2_raise = np.zeros((nsite, nsite), dtype=np.complex128)
    O2_raise[1, 0] = 1.0
    ht.list_cores_phi = apply_system_operator(
        ht.list_cores_phi, O2_raise, ht.method, ht.k_max,
        ht.M1_modes_per_state, ht.mps_epsilon, ht.bond_dim_max,
    )
    # Expected: site 0 population moves to site 1, all other sites
    # stay at zero. This specific check distinguishes a working
    # raise from silent no-ops (e.g., identity applied in place of
    # the actual operator) and from destination-mislocation bugs.
    V1_expected = np.zeros(nsite, dtype=np.complex128)
    V1_expected[1] = 1.0
    np.testing.assert_allclose(ht.psi, V1_expected, atol=1e-10)



# ------------------------------------------------------------
# TEST: Operator preserves norm (both representations)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_system_operator_preserves_norm():
    # This case tests that a unitary operator preserves the norm
    # of the wavefunction in both representations.
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        norm_before = np.linalg.norm(ht.psi)
        # Permutation matrix (unitary): swap sites 0<->2 and 1<->3
        O2_perm = np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.complex128,
        )
        ht.list_cores_phi = apply_system_operator(
            ht.list_cores_phi, O2_perm, ht.method, ht.k_max,
            ht.M1_modes_per_state, ht.mps_epsilon, ht.bond_dim_max,
        )
        norm_after = np.linalg.norm(ht.psi)
        np.testing.assert_allclose(
            norm_after,
            norm_before,
            rtol=1e-8,
            err_msg=f'Norm not preserved for {method}',
        )


# ------------------------------------------------------------
# TEST: Superposition initial state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_system_operator_superposition():
    # This case tests apply_system_operator on a delocalized
    # superposition state, not just a single-site basis state.
    # psi = (|0> + |2>) / sqrt(2)
    tb = _make_tb()
    psi_super = np.array([1, 0, 1, 0], dtype=np.complex128) / np.sqrt(2)
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    ht = HopsTensorWavefunction(
        k_max, tensor_param, {'INTEGRATOR': 'RUNGE_KUTTA'}, eom_param,
    )
    ht.initialize(psi_super, tb.system)
    # Apply |2><2| projector: should keep only the |2> component
    O2_proj_2 = np.zeros((nsite, nsite), dtype=np.complex128)
    O2_proj_2[2, 2] = 1.0
    ht.list_cores_phi = apply_system_operator(
        ht.list_cores_phi, O2_proj_2, ht.method, ht.k_max,
        ht.M1_modes_per_state, ht.mps_epsilon, ht.bond_dim_max,
    )
    expected = np.array([0, 0, 1, 0], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(
        ht.psi,
        expected,
        atol=1e-12,
        err_msg='Projection on superposition state failed',
    )


# ------------------------------------------------------------
# TEST: Unknown method raises NotImplementedError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_system_operator_invalid_method():
    # This case tests the NotImplementedError guard for unrecognized
    # method strings. Without it, an invalid method would fall through
    # the function with undefined behavior rather than raising a clear
    # error at the API boundary.
    ht = _make_tensor('fullstate')
    O2_identity = np.eye(nsite, dtype=np.complex128)
    with pytest.raises(NotImplementedError, match='not implemented'):
        apply_system_operator(
            ht.list_cores_phi, O2_identity, 'invalid_method',
            ht.k_max, ht.M1_modes_per_state,
            ht.mps_epsilon, ht.bond_dim_max,
        )


# ------------------------------------------------------------
# TEST: Complex-valued operator is applied with correct phase
# ------------------------------------------------------------
@pytest.mark.level(1)
@pytest.mark.parametrize(
    'method', ['fullstate', 'number'],
)
def test_apply_system_operator_complex_operator(method):
    # This case tests that a complex-valued operator (1j * permutation)
    # is applied with the correct phase. psi_0 = [0,0,1,0] on site 2;
    # swapping states 0<->2 and 1<->3 with a 1j prefactor should give
    # [1j, 0, 0, 0]. Catches any silent complex -> real casting bug.
    ht = _make_tensor(method)
    O2_complex_swap = 1.0j * O2_SWAP_02_13
    ht.list_cores_phi = apply_system_operator(
        ht.list_cores_phi, O2_complex_swap, ht.method, ht.k_max,
        ht.M1_modes_per_state, ht.mps_epsilon, ht.bond_dim_max,
    )
    V1_expected = np.array([1.0j, 0.0, 0.0, 0.0], dtype=np.complex128)
    np.testing.assert_allclose(
        ht.psi, V1_expected, atol=1e-10,
        err_msg=(
            f'Complex operator (1j * swap) produced wrong result in '
            f'{method}; phase handling may be broken'
        ),
    )


# ------------------------------------------------------------
# TEST: General dense complex operator — fullstate vs statenumber
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_system_operator_general_dense_cross_representation():
    # This case tests that fullstate and statenumber representations
    # produce the same result for a general dense complex operator
    # (every entry nonzero, not a permutation). This is the strongest
    # cross-check for the statenumber MPO path, which must reconstruct
    # the full dense action through daisy-chained transfer matrices.
    rng = np.random.default_rng(0)
    O2_op = (
        rng.standard_normal((nsite, nsite))
        + 1.0j * rng.standard_normal((nsite, nsite))
    ).astype(np.complex128)

    ht_full = _make_tensor('fullstate')
    ht_snum = _make_tensor('number')
    ht_full.list_cores_phi = apply_system_operator(
        ht_full.list_cores_phi, O2_op, ht_full.method, ht_full.k_max,
        ht_full.M1_modes_per_state,
        ht_full.mps_epsilon, ht_full.bond_dim_max,
    )
    ht_snum.list_cores_phi = apply_system_operator(
        ht_snum.list_cores_phi, O2_op, ht_snum.method, ht_snum.k_max,
        ht_snum.M1_modes_per_state,
        ht_snum.mps_epsilon, ht_snum.bond_dim_max,
    )
    np.testing.assert_allclose(
        ht_snum.psi, ht_full.psi, atol=1e-10,
        err_msg=(
            'Statenumber and fullstate diverged on a general dense '
            'complex operator; MPO cross-site channels may be wrong'
        ),
    )


