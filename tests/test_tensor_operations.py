import numpy as np
import pytest
import scipy as sp

from mesohops.basis.hops_modes import HopsModes
from mesohops.basis.hops_noise_memory import HopsNoiseMemory
from mesohops.basis.hops_system import HopsSystem
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.hops_tensor_basis import HopsTensorBasis
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.basis.basis_functions import determine_error_thresh
from mesohops.storage.storage_functions import save_psi_g_traj_tensor
from mesohops.util.tensor_operations import (
    calc_mps_complexity,
    contract_down,
    contract_down_exact,
    extract_gs_amp,
    extract_psi,
    flatten_cores,
    scale_mps,
    tensor_add,
    tensor_compress,
    tensor_to_array,
    unflatten_cores,
)
from mesohops.util.tensor_operations import (
    phi_aux as extract_phi_aux,
)

__title__ = 'Unit Tests for Tensor Operations'
__author__ = 'N. Covalsen'
__maintainer__ = 'N. Covalsen'

# ============================================================
# Shared Setup
# ============================================================

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


def _make_tensor(method):
    """Creates and initializes a HopsTensorWavefunction for the dimer-of-dimers system."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    system = HopsSystem(sys_param)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    tb = HopsTensorBasis(system, mode, noise_memory)
    system.initialize(delta_s > 0, psi_0)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = state_list
    tb.initialize(delta_s)
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    return ht


def _make_tb():
    """Creates and returns a HopsTensorBasis for the dimer-of-dimers system."""
    system = HopsSystem(sys_param)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    tb = HopsTensorBasis(system, mode, noise_memory)
    system.initialize(delta_s > 0, psi_0)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = state_list
    tb.initialize(delta_s)
    return tb


def _make_simple_mps(n_cores, phys_dims, bond_dims):
    """Build a simple MPS with deterministic complex values."""
    cores = []
    offset = 0
    for i in range(n_cores):
        size = bond_dims[i] * phys_dims[i] * bond_dims[i + 1]
        real_part = np.arange(offset, offset + size, dtype=np.float64) * 0.01
        imag_part = np.arange(offset + size, offset + 2 * size, dtype=np.float64) * 0.01
        core = (real_part + 1j * imag_part).reshape(
            bond_dims[i], phys_dims[i], bond_dims[i + 1],
        )
        cores.append(core)
        offset += 2 * size
    return cores


# ============================================================
# TEST SUITE: extract_psi()
# ============================================================

# ------------------------------------------------------------
# TEST: extract_psi recovers initial wavefunction
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_extract_psi_recovers_initial_wavefunction():
    # This case tests that extract_psi on the initialized MPS returns
    # the original psi_0 = [0, 0, 1, 0] for both representations.
    for method in ['fullstate', 'number']:
        # Setup: create an initialized HopsTensorWavefunction
        ht = _make_tensor(method)
        # Action: extract phi_0 from the MPS
        result = extract_psi(
            ht.list_cores_phi, method, ht.M1_modes_per_state,
        )
        # Assertion: result matches psi_0
        np.testing.assert_allclose(
            result, psi_0, atol=1e-10,
            err_msg=f'phi_0 does not recover psi_0 for {method}',
        )


# ------------------------------------------------------------
# TEST: statenumber extract_psi on a bond-dim-2 superposition
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_extract_psi_statenumber_superposition_bond_dim_2():
    # This case tests that extract_psi correctly recovers a dense
    # superposition in statenumber representation, which requires bond
    # dim > 1. Also exercises the per-target-state one-hot selection
    # with non-trivial coefficients (not zero/one).
    a = 0.6 + 0.2j
    b = -0.3 + 0.5j
    # Bond dim 2 encodes two branches:
    #   k=0: state 0 occupied, state 1 unoccupied  -> amplitude a for |1,0>
    #   k=1: state 0 unoccupied, state 1 occupied  -> amplitude b for |0,1>
    state_0 = np.zeros((1, 2, 2), dtype=np.complex128)
    state_0[0, 1, 0] = a   # branch 0 carries a into phys=1 (occupied)
    state_0[0, 0, 1] = b   # branch 1 carries b into phys=0 (unoccupied)
    state_1 = np.zeros((2, 2, 1), dtype=np.complex128)
    state_1[0, 0, 0] = 1.0  # branch 0: state 1 unoccupied
    state_1[1, 1, 0] = 1.0  # branch 1: state 1 occupied
    list_cores_phi = [[state_0], [state_1]]
    M1_modes = np.array([0, 0], dtype=int)
    # Extract the physical wavefunction; expect [a, b]
    result = extract_psi(list_cores_phi, 'number', M1_modes)
    np.testing.assert_allclose(
        result, np.array([a, b], dtype=np.complex128), atol=1e-12,
        err_msg='Bond-dim-2 superposition amplitudes not recovered',
    )


# ------------------------------------------------------------
# TEST: statenumber extract_psi with non-uniform modes per state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_extract_psi_statenumber_nonuniform_modes():
    # This case tests that extract_psi correctly traverses varying group
    # sizes in the statenumber branch. Builds a 3-site MPS with
    # M1_modes_per_state = [1, 3, 2] (non-uniform) and verifies the
    # extracted wavefunction on a product state [0, 1, 0].
    # Use k_max_local = 2 so the state core (dim 2) and mode core
    # (dim k_max_local + 1 = 3) are structurally distinct.
    k_max_local = 2
    n_state = 3
    # Build an unoccupied state core (amp in phys=0) and an occupied
    # state core (amp in phys=1), both with trivial bond dim 1.
    def _state_core(occupied):
        core = np.zeros((1, 2, 1), dtype=np.complex128)
        core[0, 1 if occupied else 0, 0] = 1.0
        return core
    # Build a mode core in the ground-state slot (phys=0).
    def _mode_core():
        core = np.zeros((1, k_max_local + 1, 1), dtype=np.complex128)
        core[0, 0, 0] = 1.0
        return core
    # Only state 1 is occupied so expected phi = [0, 1, 0]
    list_cores_phi = [
        [_state_core(False)] + [_mode_core() for _ in range(1)],
        [_state_core(True)]  + [_mode_core() for _ in range(3)],
        [_state_core(False)] + [_mode_core() for _ in range(2)],
    ]
    # Sanity: non-uniform group sizes as flagged
    assert [len(g) - 1 for g in list_cores_phi] == [1, 3, 2]
    M1_modes = np.array([1, 3, 2], dtype=int)
    result = extract_psi(list_cores_phi, 'number', M1_modes)
    V1_expected = np.zeros(n_state, dtype=np.complex128)
    V1_expected[1] = 1.0
    np.testing.assert_allclose(
        result, V1_expected, atol=1e-12,
        err_msg='Non-uniform modes per state not handled correctly',
    )


# ------------------------------------------------------------
# TEST: statenumber extract_psi on a 1-state system
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_extract_psi_statenumber_single_state():
    # This case tests the degenerate 1-state edge case, where the
    # one-hot selection collapses to "always pick occupied for the only
    # state". 1 state + 1 mode.
    k_max_local = 2
    state_core = np.zeros((1, 2, 1), dtype=np.complex128)
    state_core[0, 1, 0] = 1.0  # only state occupied
    mode_core = np.zeros((1, k_max_local + 1, 1), dtype=np.complex128)
    mode_core[0, 0, 0] = 1.0   # mode in ground state
    list_cores_phi = [[state_core, mode_core]]
    M1_modes = np.array([1], dtype=int)
    result = extract_psi(list_cores_phi, 'number', M1_modes)
    np.testing.assert_allclose(
        result, np.array([1.0], dtype=np.complex128), atol=1e-12,
        err_msg='1-state extract_psi did not return unit amplitude',
    )


# ------------------------------------------------------------
# TEST: extract_psi raises on invalid method
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_extract_psi_invalid_method_raises():
    # This case tests that extract_psi raises ValueError for an unrecognized
    # method string, ensuring the dispatcher does not silently return wrong data.
    ht = _make_tensor('fullstate')
    with pytest.raises(ValueError):
        extract_psi(ht.list_cores_phi, 'bogus', ht.M1_modes_per_state)


# ============================================================
# TEST SUITE: extract_gs_amp()
# ============================================================

def _number_mps_from_dense(dense, list_modes_per_site, epsilon=1e-12):
    """Build a nested number MPS from a dense config tensor via exact TT-SVD.

    The dense axes are the MPS physical legs in order (site_0, mode_00, ...,
    site_1, ...); the TT round-trip is lossless, so the MPS represents the
    dense state exactly.
    """
    size = list(dense.shape)
    flat_cores = []
    C = dense
    rank = 1
    for i in range(len(size) - 1):
        C = np.reshape(C, (int(rank * size[i]), int(C.size / (rank * size[i]))))
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        normalized_S = S / np.linalg.norm(S)
        thr = determine_error_thresh(np.flip(normalized_S), epsilon * epsilon)
        S[normalized_S <= thr] = 0.0
        prev_rank, rank = rank, int(np.count_nonzero(S))
        flat_cores.append(
            U[:, :rank].astype(np.complex128).reshape(prev_rank, size[i], rank)
        )
        C = np.diag(S[:rank]).astype(np.complex128) @ Vt[:rank, :]
    flat_cores.append(C.reshape(C.shape[0], C.shape[1], 1))
    return unflatten_cores(flat_cores, list_modes_per_site)


def _fullstate_mps_from_dense(dense, epsilon=1e-12):
    """Build a flat fullstate MPS from a dense config tensor via exact TT-SVD."""
    size = list(dense.shape)
    flat_cores = []
    C = dense
    rank = 1
    for i in range(len(size) - 1):
        C = np.reshape(C, (int(rank * size[i]), int(C.size / (rank * size[i]))))
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        normalized_S = S / np.linalg.norm(S)
        thr = determine_error_thresh(np.flip(normalized_S), epsilon * epsilon)
        S[normalized_S <= thr] = 0.0
        prev_rank, rank = rank, int(np.count_nonzero(S))
        flat_cores.append(
            U[:, :rank].astype(np.complex128).reshape(prev_rank, size[i], rank)
        )
        C = np.diag(S[:rank]).astype(np.complex128) @ Vt[:rank, :]
    flat_cores.append(C.reshape(C.shape[0], C.shape[1], 1))
    return flat_cores


# ------------------------------------------------------------
# TEST: extract_gs_amp recovers the all-zeros (vacuum) amplitude
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_extract_gs_amp_recovers_vacuum_amplitude():
    # A random number MPS is built from a dense hierarchy state via TT; the
    # vacuum amplitude extract_gs_amp returns must equal the dense state's
    # all-zeros entry. k_max_local = 2 gives mode cores dimension 3 vs state
    # cores dimension 2 (size [2, 3, 2, 3]), so state and mode cores are
    # distinguishable and a core-ordering bug would surface.
    rng = np.random.default_rng(0)
    k_max_local = 2
    list_modes_per_site = [1, 1]
    size = []
    for n_modes in list_modes_per_site:
        size += [2] + [k_max_local + 1] * n_modes
    dense = rng.standard_normal(size) + 1j * rng.standard_normal(size)

    mps_number = _number_mps_from_dense(dense, list_modes_per_site)
    amp_number = extract_gs_amp(mps_number, 'number')
    np.testing.assert_allclose(amp_number, dense[(0,) * len(size)], atol=1e-10)

    mps_fullstate = _fullstate_mps_from_dense(dense)
    amp_fullstate = extract_gs_amp(mps_fullstate, 'fullstate')
    np.testing.assert_allclose(amp_fullstate, dense[(0,) * len(size)], atol=1e-10)


# ------------------------------------------------------------
# TEST: save_psi_g_traj_tensor returns the vacuum amplitude
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_save_psi_g_traj_tensor_returns_vacuum_amplitude():
    # The storage hook delegates to extract_gs_amp; on a TT-built number MPS
    # it must return the all-zeros (vacuum) amplitude.
    rng = np.random.default_rng(2)
    list_modes_per_site = [1, 1]
    size = []
    for n_modes in list_modes_per_site:
        size += [2] + [2] * n_modes
    dense = rng.standard_normal(size) + 1j * rng.standard_normal(size)

    ht = _make_tensor('number')
    ht.list_cores_phi = _number_mps_from_dense(dense, list_modes_per_site)
    amp = save_psi_g_traj_tensor(ht)
    np.testing.assert_allclose(amp, dense[(0,) * len(size)], atol=1e-10)


# ============================================================
# TEST SUITE: contract_down_exact()
# ============================================================

# ------------------------------------------------------------
# TEST: Per-state norms match |psi_0[i]|^2 for ground-state MPS
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_exact_per_state():
    # This case tests that for the initialized MPS (all hierarchy in ground
    # state), contract_down_exact returns |psi_0[i]|^2 for each state.
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        n_state = nsite
        result = contract_down_exact(
            ht.list_cores_phi, method, n_state,
        )
        expected = np.abs(psi_0) ** 2
        np.testing.assert_allclose(
            result.real, expected, atol=1e-12,
            err_msg=f'Per-state norm mismatch for {method}',
        )
        # Imag part must be negligible — a conjugation error in the
        # double-layer contraction could match real but drift imag.
        np.testing.assert_allclose(
            result.imag, 0.0, atol=1e-12,
            err_msg=f'Per-state norm imag part nonzero for {method}',
        )


# ------------------------------------------------------------
# TEST: Statenumber bond-dim-2 superposition — per-state norms match |a|^2, |b|^2
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_exact_superposition_bond_dim_2():
    # This case tests contract_down_exact on a dense 2-site superposition
    # a|s0> + b|s1> encoded with bond dim 2 in statenumber representation.
    # Exercises both the einsum double-layer contraction with non-trivial
    # matrices and the per-target-state one-hot selection at nonzero
    # amplitudes (not just zero/one), closing the bond-dim-1 limitation of
    # test_contract_down_exact_per_state.
    a = 0.6 + 0.2j
    b = -0.3 + 0.5j
    state_0 = np.zeros((1, 2, 2), dtype=np.complex128)
    state_0[0, 1, 0] = a
    state_0[0, 0, 1] = b
    state_1 = np.zeros((2, 2, 1), dtype=np.complex128)
    state_1[0, 0, 0] = 1.0
    state_1[1, 1, 0] = 1.0
    list_cores_phi = [[state_0], [state_1]]
    result = contract_down_exact(
        list_cores_phi, 'number', 2,
    )
    V1_expected = np.array(
        [np.abs(a) ** 2, np.abs(b) ** 2], dtype=np.complex128,
    )
    np.testing.assert_allclose(
        result.real, V1_expected.real, atol=1e-12,
        err_msg='Superposition per-state norms not recovered',
    )
    np.testing.assert_allclose(
        result.imag, 0.0, atol=1e-12,
        err_msg='Superposition per-state norm imag part nonzero',
    )


# ------------------------------------------------------------
# TEST: Fullstate with bond dim > 1 still returns |psi[i]|^2
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_exact_fullstate_bond_dim_4():
    # This case tests that contract_down_exact still returns |psi[i]|^2
    # when the MPS is inflated to a non-trivial bond dimension. Using
    # eps=0 makes the inflation exactly lossless (no noise added to pad
    # out new bond channels), so the contracted per-state norms must
    # match the ground-state values to machine precision. Exercises the
    # einsum double-layer contraction on fullstate cores with chi > 1.
    ht = _make_tensor('fullstate')
    ht.inflate_bonds_to(4, eps=0.0)
    result = contract_down_exact(
        ht.list_cores_phi, 'fullstate', nsite,
    )
    expected = np.abs(psi_0) ** 2
    np.testing.assert_allclose(
        result.real, expected, atol=1e-12,
        err_msg='Inflated fullstate did not recover |psi[i]|^2',
    )
    np.testing.assert_allclose(
        result.imag, 0.0, atol=1e-12,
        err_msg='Inflated fullstate imag part nonzero',
    )


# ------------------------------------------------------------
# TEST: Statenumber raises ValueError on too-few cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_exact_too_few_cores():
    # This case tests that contract_down_exact raises a ValueError when
    # given fewer groups than n_state for number.
    # 1 group (list-of-lists) for n_state=2 is too few.
    short_cores = [[np.zeros((1, 2, 1), dtype=np.complex128),
                    np.zeros((1, 3, 1), dtype=np.complex128),
                    np.zeros((1, 3, 1), dtype=np.complex128)]]
    with pytest.raises(ValueError, match='expects at least'):
        contract_down_exact(
            short_cores, 'number', 2,
        )


# ------------------------------------------------------------
# TEST: Invalid method raises ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_exact_invalid_method():
    # This case tests that contract_down_exact raises a ValueError when
    # given an unrecognized method string.
    ht = _make_tensor('fullstate')
    with pytest.raises(ValueError, match='Unknown method'):
        contract_down_exact(
            ht.list_cores_phi, 'badmethod', nsite,
        )


# ------------------------------------------------------------
# TEST: Statenumber raises ValueError on bond mismatch
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_exact_bond_mismatch():
    # This case tests that contract_down_exact raises a ValueError when
    # adjacent group bond dimensions are incompatible.
    # Group 0 has right bond 3; group 1 has left bond 1 — mismatch.
    bad_cores = [
        [np.zeros((1, 2, 3), dtype=np.complex128),
         np.zeros((3, 3, 3), dtype=np.complex128),
         np.zeros((3, 3, 3), dtype=np.complex128)],
        [np.zeros((1, 2, 1), dtype=np.complex128),
         np.zeros((1, 3, 1), dtype=np.complex128),
         np.zeros((1, 3, 1), dtype=np.complex128)],
    ]
    with pytest.raises(ValueError, match='Environment/bond mismatch'):
        contract_down_exact(bad_cores, 'number', 2)


# ============================================================
# TEST SUITE: contract_down()
# ============================================================

# ------------------------------------------------------------
# TEST: Approximate contraction agrees with exact for bond-dim-1 MPS
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_agrees_with_exact_bonddim1():
    # This case tests that for a bond-dim-1 MPS (no entanglement between
    # cores), the approximate contraction matches the exact contraction.
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        n_state = nsite
        approx = contract_down(
            ht.list_cores_phi, method, n_state,
        )
        exact = contract_down_exact(
            ht.list_cores_phi, method, n_state,
        )
        np.testing.assert_allclose(
            approx.real, exact.real, atol=1e-10,
            err_msg=f'Approximate vs exact mismatch for {method}',
        )

    # This case tests with bond dim > 1 (entangled MPS) where the
    # approximation may differ but invariants still hold
    ht_inflated = _make_tensor('fullstate')
    ht_inflated.inflate_bonds_to(4, eps=0.1)
    approx_infl = contract_down(
        ht_inflated.list_cores_phi, 'fullstate',
        nsite,
    )
    # Invariant: per-state norms should be non-negative
    assert np.all(approx_infl.real >= -1e-10), (
        'Per-state norms should be non-negative'
    )
    # Invariant: sum should approximate total norm squared
    psi_infl = extract_psi(
        ht_inflated.list_cores_phi, 'fullstate',
        ht_inflated.M1_modes_per_state,
    )
    expected_norm_sq = np.sum(np.abs(psi_infl) ** 2)
    np.testing.assert_allclose(
        np.sum(approx_infl).real, expected_norm_sq.real, atol=1e-4,
        err_msg='Sum of contract_down should approximate norm squared',
    )


# ------------------------------------------------------------
# TEST: Both representations agree
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_both_representations_agree():
    # This case tests that contract_down() returns the same per-state
    # values for both MPS representations.
    # Setup: create tensors with both methods
    ht_full = _make_tensor('fullstate')
    ht_state = _make_tensor('number')
    # Action: compute per-state norms from each
    result_full = contract_down(
        ht_full.list_cores_phi, 'fullstate',
        nsite,
    )
    result_state = contract_down(
        ht_state.list_cores_phi, 'number',
        nsite,
    )
    # Assertion: both representations give the same result
    np.testing.assert_allclose(
        result_full, result_state, atol=1e-10,
        err_msg='contract_down differs between fullstate and statenumber',
    )
    # Imag part must be negligible in both representations
    np.testing.assert_allclose(
        result_full.imag, 0.0, atol=1e-12,
        err_msg='Fullstate contract_down imag part nonzero',
    )
    np.testing.assert_allclose(
        result_state.imag, 0.0, atol=1e-12,
        err_msg='Statenumber contract_down imag part nonzero',
    )

    # This case tests invariant for inflated (entangled) MPS
    ht_infl = _make_tensor('fullstate')
    ht_infl.inflate_bonds_to(4, eps=0.1)
    result_infl = contract_down(
        ht_infl.list_cores_phi, 'fullstate',
        nsite,
    )
    # Invariant: sum of per-state norms equals norm squared
    psi_infl = extract_psi(
        ht_infl.list_cores_phi, 'fullstate',
        ht_infl.M1_modes_per_state,
    )
    np.testing.assert_allclose(
        np.sum(result_infl).real,
        np.sum(np.abs(psi_infl) ** 2).real,
        atol=1e-4,
    )


# ------------------------------------------------------------
# TEST: Sum equals norm squared
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_sum_equals_norm_squared():
    # This case tests that the sum of per-state values from contract_down()
    # equals the total MPS norm squared.
    for method in ['fullstate', 'number']:
        # Setup: create an initialized HopsTensorWavefunction
        ht = _make_tensor(method)
        # Action: compute per-state norms
        result = contract_down(
            ht.list_cores_phi, method, nsite,
        )
        # Compute expected norm squared from phi_0 (for bond-dim-1 MPS,
        # only the zeroth auxiliary contributes)
        phi = extract_psi(
            ht.list_cores_phi, method, ht.M1_modes_per_state,
        )
        expected_norm_sq = np.sum(np.abs(phi) ** 2)
        # Assertion: sum of per-state values equals norm squared
        np.testing.assert_allclose(
            np.sum(result).real, expected_norm_sq.real, atol=1e-10,
            err_msg=f'Sum of contract_down != norm squared for {method}',
        )
        # Imag part must be negligible — a conjugation error would match
        # the real sum but drift imag across per-state values.
        np.testing.assert_allclose(
            result.imag, 0.0, atol=1e-12,
            err_msg=f'Per-state imag part nonzero for {method}',
        )

    # This case tests the invariant for inflated (bond dim > 1) MPS
    ht_infl = _make_tensor('fullstate')
    ht_infl.inflate_bonds_to(4, eps=0.1)
    result_infl = contract_down(
        ht_infl.list_cores_phi, 'fullstate',
        nsite,
    )
    phi_infl = extract_psi(
        ht_infl.list_cores_phi, 'fullstate',
        ht_infl.M1_modes_per_state,
    )
    expected_infl = np.sum(np.abs(phi_infl) ** 2)
    np.testing.assert_allclose(
        np.sum(result_infl).real, expected_infl.real, atol=1e-4,
        err_msg='Sum of contract_down != norm squared for inflated MPS',
    )


# ------------------------------------------------------------
# TEST: Invalid method raises ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_invalid_method_raises():
    # This case tests that contract_down raises ValueError for an
    # unrecognized method string, matching the parallel test on
    # contract_down_exact.
    ht = _make_tensor('fullstate')
    with pytest.raises(ValueError, match='Unknown method'):
        contract_down(ht.list_cores_phi, 'bogus', nsite)


# ------------------------------------------------------------
# TEST: Statenumber bond-dim-2 superposition per-state norms
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_down_superposition_statenumber_bond_dim_2():
    # This case tests contract_down on a dense 2-site superposition
    # a|s0> + b|s1> encoded with bond dim 2 in statenumber representation.
    # Covers both "statenumber with bond dim > 1" and "superposition
    # state" in one test — a dense statenumber superposition naturally
    # requires bond > 1, so the inline traversal + per-target
    # contraction is exercised on non-trivial bond matrices.
    a = 0.6 + 0.2j
    b = -0.3 + 0.5j
    state_0 = np.zeros((1, 2, 2), dtype=np.complex128)
    state_0[0, 1, 0] = a
    state_0[0, 0, 1] = b
    state_1 = np.zeros((2, 2, 1), dtype=np.complex128)
    state_1[0, 0, 0] = 1.0
    state_1[1, 1, 0] = 1.0
    list_cores_phi = [[state_0], [state_1]]
    result = contract_down(
        list_cores_phi, 'number', 2,
    )
    V1_expected = np.array(
        [np.abs(a) ** 2, np.abs(b) ** 2], dtype=np.complex128,
    )
    np.testing.assert_allclose(
        result.real, V1_expected.real, atol=1e-12,
        err_msg='Superposition per-state norms not recovered by contract_down',
    )
    np.testing.assert_allclose(
        result.imag, 0.0, atol=1e-12,
        err_msg='Superposition contract_down imag part nonzero',
    )


# ============================================================
# TEST SUITE: phi_aux()
# ============================================================

# ------------------------------------------------------------
# TEST: First-order excitation is zero for initialized MPS
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_aux_first_order_zero_at_init():
    # This case tests that all first-order auxiliary wavefunctions
    # are zero for the initial MPS (all modes start in |0>).
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        n_modes = len(gw_sysbath)
        for i_mode in range(n_modes):
            indices = [0] * n_modes
            indices[i_mode] = 1
            phi_1 = extract_phi_aux(
                ht.list_cores_phi, method, ht.M1_modes_per_state, indices,
            )
            np.testing.assert_allclose(
                phi_1, 0.0, atol=1e-12,
                err_msg=f'Mode {i_mode} first-order not zero for {method}',
            )


# ------------------------------------------------------------
# TEST: phi_aux recovers nonzero auxiliary after manual injection
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_aux_nonzero_injection():
    # This case tests that phi_aux returns the exact analytically
    # computable auxiliary amplitude after a known injection. A
    # function that returned random nonzero garbage would pass the
    # prior "any nonzero" check, so we assert the exact value.
    # For a bond-dim-1 MPS with psi_0 = [0,0,1,0] and an injection of
    # z = 0.5+0.1j into mode 0's occupation-1 slice, phi_aux at
    # indices=[1,0,0,...] contracts to z * psi_0 = [0, 0, z, 0].
    z_inject = 0.5 + 0.1j
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        n_modes = int(np.sum(ht.M1_modes_per_state))
        # Inject z_inject into mode 0's occupation-1 slice.
        # For fullstate: list_cores_phi[1][:, 1, :] is the occ-1 slice.
        # For statenumber: list_cores_phi[0][1][:, 1, :] is the first mode.
        if method == 'fullstate':
            ht.list_cores_phi[1][:, 1, :] = z_inject
        else:
            ht.list_cores_phi[0][1][:, 1, :] = z_inject
        indices = [0] * n_modes
        indices[0] = 1
        V1_phi_1 = extract_phi_aux(
            ht.list_cores_phi, method, ht.M1_modes_per_state, indices,
        )
        V1_expected = z_inject * psi_0
        np.testing.assert_allclose(
            V1_phi_1, V1_expected, atol=1e-12,
            err_msg=f'phi_aux injection value wrong for {method}',
        )


# ------------------------------------------------------------
# TEST: phi_aux with all-zero indices equals extract_psi
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_aux_zero_indices_equals_extract_psi():
    # This case pins the docstring guarantee that phi_aux with
    # all-zero indices returns phi_0 — identical to extract_psi.
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        n_modes = int(np.sum(ht.M1_modes_per_state))
        V1_aux_zero = extract_phi_aux(
            ht.list_cores_phi, method, ht.M1_modes_per_state,
            [0] * n_modes,
        )
        V1_phi_0 = extract_psi(
            ht.list_cores_phi, method, ht.M1_modes_per_state,
        )
        np.testing.assert_allclose(
            V1_aux_zero, V1_phi_0, atol=1e-12,
            err_msg=f'phi_aux([0]*n_modes) != extract_psi for {method}',
        )


# ------------------------------------------------------------
# TEST: phi_aux with higher-order and multi-mode indices
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_aux_higher_order_and_multi_mode():
    # This case tests phi_aux on auxiliary indices beyond the common
    # [0,...,0,1,0,...] pattern: a single mode at occupation 2
    # (second-order) and two modes simultaneously at occupation 1
    # (multi-mode). For a bond-dim-1 MPS with psi_0 = [0,0,1,0], each
    # injection contributes multiplicatively, so the expected aux is
    # (product of injections) * psi_0.
    z_0 = 0.4 - 0.2j   # injected into mode 0 at occ 2 (second-order)
    z_1 = 0.3 + 0.1j   # injected into mode 1 at occ 1
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        n_modes = int(np.sum(ht.M1_modes_per_state))
        if method == 'fullstate':
            ht.list_cores_phi[1][:, 2, :] = z_0
            ht.list_cores_phi[2][:, 1, :] = z_1
        else:
            ht.list_cores_phi[0][1][:, 2, :] = z_0
            ht.list_cores_phi[0][2][:, 1, :] = z_1
        # Sub-case A: higher-order on a single mode
        indices_second_order = [0] * n_modes
        indices_second_order[0] = 2
        V1_aux_A = extract_phi_aux(
            ht.list_cores_phi, method, ht.M1_modes_per_state,
            indices_second_order,
        )
        np.testing.assert_allclose(
            V1_aux_A, z_0 * psi_0, atol=1e-12,
            err_msg=f'phi_aux second-order wrong for {method}',
        )
        # Sub-case B: two modes at occupation 1 simultaneously
        indices_multi = [0] * n_modes
        indices_multi[0] = 1  # mode 0 — but we injected at occ 2, not 1
        indices_multi[1] = 1
        # Mode 0's occ-1 slice is still zero (we only populated occ-2),
        # so the product includes a zero factor and the aux vanishes.
        V1_aux_B = extract_phi_aux(
            ht.list_cores_phi, method, ht.M1_modes_per_state,
            indices_multi,
        )
        np.testing.assert_allclose(
            V1_aux_B, 0.0, atol=1e-12,
            err_msg=(
                f'phi_aux multi-mode with one zero factor should be '
                f'zero for {method}'
            ),
        )


# ------------------------------------------------------------
# TEST: Statenumber phi_aux zero-pads short indices
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_aux_statenumber_padding():
    # This case tests the statenumber-path padding logic: when the
    # caller passes an `indices` list shorter than the total mode
    # count, the function zero-pads via the n_total_padded check
    # rather than crashing. Short all-zero indices must yield the
    # same result as [0]*n_total (which equals extract_psi by the
    # docstring guarantee).
    ht = _make_tensor('number')
    n_total = int(np.sum(ht.M1_modes_per_state))
    # Pass half the indices — forces the padding branch to fire.
    indices_short = [0] * (n_total // 2)
    V1_aux_short = extract_phi_aux(
        ht.list_cores_phi, 'number',
        ht.M1_modes_per_state, indices_short,
    )
    V1_phi_0 = extract_psi(
        ht.list_cores_phi, 'number',
        ht.M1_modes_per_state,
    )
    np.testing.assert_allclose(
        V1_aux_short, V1_phi_0, atol=1e-12,
        err_msg='Short indices should zero-pad to the full mode count',
    )


# ============================================================
# TEST SUITE: tensor_add()
# ============================================================

# ------------------------------------------------------------
# TEST: Mismatched core counts raise ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_length_mismatch():
    # This case tests that tensor_add raises ValueError for mismatched
    # core counts.
    cores_a = [np.zeros((1, 2, 1), dtype=np.complex128)]
    cores_b = [np.zeros((1, 2, 1), dtype=np.complex128)] * 2
    with pytest.raises(ValueError, match='core count mismatch'):
        tensor_add(cores_a, cores_b, 1e-10, 20)


# ------------------------------------------------------------
# TEST: Physical dimension mismatch raises ValueError (Jacob T6)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_phys_dim_mismatch():
    # This case tests that tensor_add raises ValueError when
    # corresponding cores have different physical dimensions.
    cores_a = [np.zeros((1, 2, 1), dtype=np.complex128)]
    cores_b = [np.zeros((1, 3, 1), dtype=np.complex128)]
    with pytest.raises(ValueError, match='Physical dimension mismatch'):
        tensor_add(cores_a, cores_b, 1e-10, 20)


# ------------------------------------------------------------
# TEST: Adding known MPS gives correct sum (Jacob T5)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_known_sum():
    # This case tests tensor_add with two initialized HopsTensorWavefunction MPS
    # and verifies that the physical wavefunction sums correctly.
    # Uses the shared dimer system: psi_0 = [0, 0, 1, 0].
    ht = _make_tensor('fullstate')
    V1_phi_before = ht.psi.copy()
    # Scale the second MPS by 0.5+0.3j to break symmetry
    scale = 0.5 + 0.3j
    list_scaled = [c.copy() for c in ht.list_cores_phi]
    list_scaled[0] = list_scaled[0] * scale
    list_cores_sum = tensor_add(
        ht.list_cores_phi, list_scaled, ht.mps_epsilon, ht.bond_dim_max,
    )
    # Extract phi_0 from the summed MPS
    V1_phi_sum = extract_psi(
        list_cores_sum, 'fullstate', ht.M1_modes_per_state,
    )
    expected = V1_phi_before * (1.0 + scale)
    np.testing.assert_allclose(
        V1_phi_sum, expected, atol=1e-10,
        err_msg='tensor_add known sum does not match expected',
    )


# ------------------------------------------------------------
# TEST: Statenumber (nested) tensor_add gives correct sum
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_statenumber_nested():
    # This case tests tensor_add with nested statenumber MPS (list-of-lists).
    # The nested code path flattens, adds, compresses, then unflattens.
    ht = _make_tensor('number')
    V1_phi_before = ht.psi.copy()
    scale = 0.7 - 0.2j
    list_scaled = [
        [c.copy() for c in group] for group in ht.list_cores_phi
    ]
    list_scaled[0][0] = list_scaled[0][0] * scale
    list_cores_sum = tensor_add(
        ht.list_cores_phi, list_scaled, ht.mps_epsilon, ht.bond_dim_max,
    )
    # Result should be nested (list-of-lists)
    assert isinstance(list_cores_sum[0], list), (
        'tensor_add should return nested structure for nested input'
    )
    V1_phi_sum = extract_psi(
        list_cores_sum, 'number', ht.M1_modes_per_state,
    )
    expected = V1_phi_before * (1.0 + scale)
    np.testing.assert_allclose(
        V1_phi_sum, expected, atol=1e-10,
        err_msg='tensor_add statenumber sum does not match expected',
    )


# ------------------------------------------------------------
# TEST: Large epsilon does not catastrophically lose information
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_large_epsilon():
    # This case tests that tensor_add with a deliberately large epsilon
    # still produces an approximate result rather than silently dropping
    # most information during SVD truncation.
    ht = _make_tensor('fullstate')
    V1_psi_before = ht.psi.copy()
    other_cores = [c.copy() for c in ht.list_cores_phi]
    epsilon = 0.5
    list_cores_sum = tensor_add(
        ht.list_cores_phi, other_cores, epsilon, ht.bond_dim_max,
    )
    V1_psi_sum = extract_psi(
        list_cores_sum, 'fullstate', ht.M1_modes_per_state,
    )
    expected = 2.0 * V1_psi_before
    np.testing.assert_allclose(
        V1_psi_sum, expected, atol=epsilon,
        err_msg='tensor_add with large epsilon lost too much information',
    )


# ============================================================
# TEST SUITE: tensor_compress()
# ============================================================

# ------------------------------------------------------------
# TEST: Compression respects bond_dim_max
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_compress_bond_dim_max():
    # This case tests that after compression, no bond dimension
    # exceeds bond_dim_max.
    cores = _make_simple_mps(4, [2, 3, 3, 2], [1, 8, 8, 8, 1])
    max_bond = 3
    compressed = tensor_compress(cores, 1e-10, max_bond)
    for i, core in enumerate(compressed):
        assert core.shape[0] <= max_bond, (
            f'Core {i} left bond {core.shape[0]} > {max_bond}'
        )
        assert core.shape[2] <= max_bond, (
            f'Core {i} right bond {core.shape[2]} > {max_bond}'
        )


# ------------------------------------------------------------
# TEST: Compression accuracy within epsilon
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_compress_accuracy():
    # This case tests that the compressed MPS approximates the original
    # by contracting both to full state vectors and comparing.
    cores = _make_simple_mps(3, [2, 2, 2], [1, 4, 4, 1])
    compressed = tensor_compress(cores, 1e-10, 20)
    # Contract original
    state_orig = cores[0]
    for c in cores[1:]:
        state_orig = np.tensordot(state_orig, c, axes=([-1], [0]))
    # Contract compressed
    state_comp = compressed[0]
    for c in compressed[1:]:
        state_comp = np.tensordot(state_comp, c, axes=([-1], [0]))
    np.testing.assert_allclose(
        state_comp, state_orig, atol=1e-8,
        err_msg='Compressed MPS does not approximate original',
    )


# ------------------------------------------------------------
# TEST: tensor_compress with large epsilon reduces bond dims
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_compress_epsilon_truncation():
    # This case tests that compressing with a large epsilon discards small
    # singular values and reduces bond dimensions below the inflated baseline.
    ht = _make_tensor('fullstate')
    ht.inflate_bonds_to(8, eps=0.01)
    cores_inflated = [c.copy() for c in ht.list_cores_phi]
    psi_before = extract_psi(
        cores_inflated, 'fullstate', ht.M1_modes_per_state,
    )
    # Compress with large epsilon but no hard bond cap
    cores_compressed = tensor_compress(cores_inflated, 0.5, 999)
    # Bound: bonds should be reduced
    max_bond_before = max(c.shape[2] for c in cores_inflated[:-1])
    max_bond_after = max(c.shape[2] for c in cores_compressed[:-1])
    assert max_bond_after <= max_bond_before
    # Invariant: state approximately preserved
    psi_after = extract_psi(
        cores_compressed, 'fullstate', ht.M1_modes_per_state,
    )
    np.testing.assert_allclose(psi_after, psi_before, atol=0.5)


# ============================================================
# TEST SUITE: tensor_to_array()
# ============================================================

# ------------------------------------------------------------
# TEST: tensor_to_array recovers full state vector with nonzero auxiliaries
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_to_array_recovers_vector():
    # Analytical: returns phi_0 followed by first-order auxiliaries.
    # Test both representations with zero auxiliaries (initial state)
    # and fullstate with nonzero auxiliaries after injection.
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        tb = _make_tb()
        result = tensor_to_array(
            ht.list_cores_phi, method, ht.M1_modes_per_state,
            tb.system, tb.mode,
        )
        np.testing.assert_allclose(result[:nsite], psi_0, atol=1e-12)
        np.testing.assert_allclose(result[nsite:], 0.0, atol=1e-12)

    # Inject nonzero occupation-1 slice into the first mode core
    # and verify both phi_0 and auxiliaries are correct.
    ht = _make_tensor('fullstate')
    tb = _make_tb()
    ht.list_cores_phi[1][:, 1, :] = 0.3 + 0.2j
    result = tensor_to_array(
        ht.list_cores_phi, 'fullstate', ht.M1_modes_per_state,
        tb.system, tb.mode,
    )
    # Physical wavefunction should still be correct after injection
    np.testing.assert_allclose(result[:nsite], psi_0, atol=1e-12)
    # First-order auxiliary for mode 0 should now be nonzero
    aux_block = result[nsite:2 * nsite]
    assert np.any(np.abs(aux_block) > 1e-10), (
        'First-order auxiliary should be nonzero after injection'
    )


# ============================================================
# TEST SUITE: flatten_cores() / unflatten_cores()
# ============================================================

# ------------------------------------------------------------
# TEST: flatten then unflatten is identity
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_flatten_unflatten_roundtrip():
    # Algebraic property: unflatten(flatten(x)) == x for any list of core groups.
    ht = _make_tensor('number')
    original = ht.list_cores_phi
    flat = flatten_cores(original)
    modes_per_site = [len(g) - 1 for g in original]
    restored = unflatten_cores(flat, modes_per_site)
    assert len(restored) == len(original)
    for s in range(len(original)):
        assert len(restored[s]) == len(original[s])
        for i in range(len(original[s])):
            np.testing.assert_array_equal(restored[s][i], original[s][i])

    # Second pass with non-uniform modes_per_site [1, 3, 2] to exercise
    # unflatten_cores's int(n_modes) indexing with varying group sizes.
    # The _make_tensor fixture uses uniform 2 modes per state, which
    # never exercises that path.
    original_nonuniform = [
        [
            np.zeros((1, 2, 1), dtype=np.complex128),
            np.zeros((1, 3, 1), dtype=np.complex128),
        ],
        [
            np.zeros((1, 2, 1), dtype=np.complex128),
            np.zeros((1, 3, 1), dtype=np.complex128),
            np.zeros((1, 3, 1), dtype=np.complex128),
            np.zeros((1, 3, 1), dtype=np.complex128),
        ],
        [
            np.zeros((1, 2, 1), dtype=np.complex128),
            np.zeros((1, 3, 1), dtype=np.complex128),
            np.zeros((1, 3, 1), dtype=np.complex128),
        ],
    ]
    # Mark each core with a unique pattern so a mis-indexed restore
    # would be detectable element-wise.
    marker = 1
    for group in original_nonuniform:
        for core in group:
            core[0, 0, 0] = marker
            marker += 1
    modes_nonuniform = [1, 3, 2]
    flat_nonuniform = flatten_cores(original_nonuniform)
    restored_nonuniform = unflatten_cores(flat_nonuniform, modes_nonuniform)
    assert len(restored_nonuniform) == len(original_nonuniform)
    for s in range(len(original_nonuniform)):
        assert len(restored_nonuniform[s]) == len(original_nonuniform[s])
        for i in range(len(original_nonuniform[s])):
            np.testing.assert_array_equal(
                restored_nonuniform[s][i], original_nonuniform[s][i],
            )


# ============================================================
# TEST SUITE: tensor_add() — OBC guards
# ============================================================

# ------------------------------------------------------------
# TEST: tensor_add raises on non-OBC boundaries
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_obc_guards():
    # This case tests that tensor_add raises ValueError with the
    # expected error message when either MPS violates open boundary
    # conditions (left bond != 1 or right bond != 1). Covers all four
    # combinations — bad left / right bond on the first / second MPS —
    # with match strings so a wrong-ValueError would not silently pass.
    cores_ok = [
        np.ones((1, 2, 3), dtype=np.complex128),
        np.ones((3, 2, 1), dtype=np.complex128),
    ]
    # Left bond != 1 on first MPS
    cores_bad_left = [
        np.ones((2, 2, 3), dtype=np.complex128),
        np.ones((3, 2, 1), dtype=np.complex128),
    ]
    with pytest.raises(ValueError, match='left bond of first core must be 1'):
        tensor_add(cores_bad_left, cores_ok, 1e-10, 10)
    # Right bond != 1 on first MPS
    cores_bad_right = [
        np.ones((1, 2, 3), dtype=np.complex128),
        np.ones((3, 2, 2), dtype=np.complex128),
    ]
    with pytest.raises(ValueError, match='right bond of last core must be 1'):
        tensor_add(cores_bad_right, cores_ok, 1e-10, 10)
    # Left bond != 1 on second MPS
    with pytest.raises(ValueError, match='left bond of first core must be 1'):
        tensor_add(cores_ok, cores_bad_left, 1e-10, 10)
    # Right bond != 1 on second MPS
    with pytest.raises(ValueError, match='right bond of last core must be 1'):
        tensor_add(cores_ok, cores_bad_right, 1e-10, 10)


# ------------------------------------------------------------
# TEST: tensor_add doubles phi_0
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_doubles_phi0():
    # This case tests that adding a tensor to itself doubles the physical
    # wavefunction phi_0, verifying tensor_add produces the correct sum.
    ht = _make_tensor('fullstate')
    phi_0_before = ht.psi.copy()
    other_cores = [c.copy() for c in ht.list_cores_phi]
    ht.list_cores_phi = tensor_add(
        ht.list_cores_phi, other_cores, ht.mps_epsilon, ht.bond_dim_max,
    )
    np.testing.assert_allclose(ht.psi, 2 * phi_0_before, atol=1e-12)


# ------------------------------------------------------------
# TEST: tensor_add bond dimensions are compatible
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_bond_dims_compatible():
    # This case tests that after tensor_add, the resulting MPS has
    # compatible bond dimensions between adjacent cores.
    ht = _make_tensor('fullstate')
    other_cores = [c.copy() for c in ht.list_cores_phi]
    ht.list_cores_phi = tensor_add(
        ht.list_cores_phi, other_cores, ht.mps_epsilon, ht.bond_dim_max,
    )
    assert ht.check_bondsize(), 'Bond dims incompatible after tensor_add'


# ------------------------------------------------------------
# TEST: tensor_add respects bond_dim_max after compression
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_respects_bond_dim_max():
    # This case tests that after tensor_add with compression,
    # no bond dimension exceeds bond_dim_max.
    ht = _make_tensor('fullstate')
    other_cores = [c.copy() for c in ht.list_cores_phi]
    ht.list_cores_phi = tensor_add(
        ht.list_cores_phi, other_cores, ht.mps_epsilon, ht.bond_dim_max,
    )
    for i, core in enumerate(ht.list_cores_phi):
        assert core.shape[0] <= ht.bond_dim_max, (
            f'Core {i} left bond {core.shape[0]} > bond_dim_max {ht.bond_dim_max}'
        )
        assert core.shape[2] <= ht.bond_dim_max, (
            f'Core {i} right bond {core.shape[2]} > bond_dim_max {ht.bond_dim_max}'
        )


# ------------------------------------------------------------
# TEST: tensor_add works in statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_statenumber():
    # This case tests that adding a tensor to itself doubles phi_0
    # in statenumber representation.
    ht = _make_tensor('number')
    V1_phi_before = ht.psi.copy()
    list_other = [c.copy() for c in ht.flat_cores]
    ht.update_phi_from_flat(
        tensor_add(ht.flat_cores, list_other, ht.mps_epsilon, ht.bond_dim_max),
    )
    np.testing.assert_allclose(
        ht.psi,
        2 * V1_phi_before,
        atol=1e-10,
        err_msg='tensor_add did not double phi_0 in statenumber',
    )


# ------------------------------------------------------------
# TEST: tensor_add result has no NaN or zero-dim bonds
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_tensor_add_no_pathological_output():
    # This case tests that the output of tensor_add has no NaN values
    # and all bond dimensions are at least 1 (no degenerate cores).
    ht = _make_tensor('fullstate')
    list_other = [c.copy() for c in ht.list_cores_phi]
    ht.list_cores_phi = tensor_add(
        ht.list_cores_phi, list_other, ht.mps_epsilon, ht.bond_dim_max,
    )
    for i, core in enumerate(ht.list_cores_phi):
        assert not np.any(np.isnan(core)), f'NaN found in core {i} after tensor_add'
        assert core.shape[0] >= 1, f'Core {i} left bond is 0'
        assert core.shape[2] >= 1, f'Core {i} right bond is 0'


# ============================================================
# TEST SUITE: scale_mps()
# ============================================================


# ------------------------------------------------------------
# TEST: scale_mps with nested statenumber MPS
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_scale_mps_statenumber():
    # This case tests that scale_mps correctly handles nested
    # list-of-lists statenumber MPS by scaling cores[0][0].
    ht = _make_tensor('number')
    V1_psi_before = ht.psi.copy()
    scale_mps(ht.list_cores_phi, 3.0)
    np.testing.assert_allclose(
        ht.psi,
        3.0 * V1_psi_before,
        atol=1e-10,
        err_msg='scale_mps did not scale statenumber MPS correctly',
    )


# ============================================================
# TEST SUITE: calc_mps_complexity()
# ============================================================
# Per-core formula: D_left * D_right * max(D_left, D_right) * d_phys
# Total: sum over all cores.

# ------------------------------------------------------------
# TEST: Empty list returns 0
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_mps_complexity_empty():
    # This case tests that an empty core list yields complexity 0.
    assert calc_mps_complexity([]) == 0


# ------------------------------------------------------------
# TEST: Single core matches per-core formula
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_mps_complexity_single_core_symmetric():
    # This case pins the symmetric (D_left == D_right) formula:
    # core shape (3, 4, 3) -> 3 * 3 * max(3,3) * 4 = 108.
    core = np.zeros((3, 4, 3), dtype=np.complex128)
    expected = 3 * 3 * 3 * 4
    assert calc_mps_complexity([core]) == expected


# ------------------------------------------------------------
# TEST: Single core with asymmetric bonds exercises the max() branch
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_mps_complexity_single_core_asymmetric():
    # This case pins the max(D_left, D_right) branch when bonds differ.
    # core shape (5, 2, 3) -> 5 * 3 * max(5,3) * 2 = 150
    # core shape (3, 2, 5) -> 3 * 5 * max(3,5) * 2 = 150 (transpose-equivalent)
    left_heavy = np.zeros((5, 2, 3), dtype=np.complex128)
    right_heavy = np.zeros((3, 2, 5), dtype=np.complex128)
    assert calc_mps_complexity([left_heavy]) == 5 * 3 * 5 * 2
    assert calc_mps_complexity([right_heavy]) == 3 * 5 * 5 * 2


# ------------------------------------------------------------
# TEST: Multi-core complexity sums per-core scores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_mps_complexity_multi_core_sum():
    # This case pins additivity: total = sum of per-core scores.
    # Three cores with mixed shapes: (1, 2, 3), (3, 2, 4), (4, 2, 1).
    list_cores = [
        np.zeros((1, 2, 3), dtype=np.complex128),
        np.zeros((3, 2, 4), dtype=np.complex128),
        np.zeros((4, 2, 1), dtype=np.complex128),
    ]
    expected = (
        1 * 3 * 3 * 2  # max(1,3) = 3
        + 3 * 4 * 4 * 2  # max(3,4) = 4
        + 4 * 1 * 4 * 2  # max(4,1) = 4
    )
    assert calc_mps_complexity(list_cores) == expected


# ------------------------------------------------------------
# TEST: Complexity is invariant under value changes (depends only on shape)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_calc_mps_complexity_shape_only():
    # This case tests that the formula depends purely on shapes; element
    # values do not affect the score.
    core_shape = (2, 3, 4)
    zeros_core = np.zeros(core_shape, dtype=np.complex128)
    random_core = np.random.default_rng(0).standard_normal(core_shape) \
        + 1j * np.random.default_rng(1).standard_normal(core_shape)
    assert calc_mps_complexity([zeros_core]) == calc_mps_complexity([random_core])
