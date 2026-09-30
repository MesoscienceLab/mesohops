import numpy as np
import pytest
import scipy.sparse as sparse

from mesohops.basis.hops_modes import HopsModes
from mesohops.basis.hops_noise_memory import HopsNoiseMemory
from mesohops.basis.hops_system import HopsSystem
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.hops_tensor_basis import HopsTensorBasis
from mesohops.basis.basis_functions import determine_error_thresh
from mesohops.tensor.mpo_constructors import (
    MpoBuilder,
    build_statenumber_dipole_lower_plus_ident_mpo,
    build_statenumber_dipole_mpo,
    build_statenumber_dipole_raise_plus_ground_ident_mpo,
    build_statenumber_operator_mpo,
)
from mesohops.tensor.tensor_eom_functions import tensor_matvec_prod
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.tensor_operations import extract_psi, unflatten_cores
from mesohops.util.tensor_operations import tensor_add

__title__ = 'Unit Tests for MPO Constructors'
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
    lop_list.append(sparse.coo_matrix(loperator[i]))
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


def _make_tensor_pair(method):
    """Returns (HopsTensorWavefunction, HopsTensorBasis), both initialized."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    tb = _make_tb()
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    return ht, tb


def _apply_ham_mpo_and_extract(ht, tb, list_cores_ham):
    """Apply a Hamiltonian MPO to the MPS and extract the physical wavefunction.

    Pads the Hamiltonian MPO with identity mode cores to match the
    full MPS length, applies via tensor_matvec_prod, then extracts
    phi_0 from the result.
    """
    # Build identity mode cores matching MPS mode core dimensions
    list_cores_full_mpo = []
    mps_cores_flat = ht.flat_cores
    ham_idx = 0
    for idx, mps_core in enumerate(mps_cores_flat):
        if ham_idx < len(list_cores_ham):
            ham_core = list_cores_ham[ham_idx]
            # Check if this ham core's physical dims match the MPS core
            if ham_core.shape[1] == mps_core.shape[1]:
                list_cores_full_mpo.append(ham_core)
                ham_idx += 1
                continue
        # Identity mode core: (1, d, d, 1) with I on the diagonal
        d = mps_core.shape[1]
        identity_core = np.zeros((1, d, d, 1), dtype=np.complex128)
        for k in range(d):
            identity_core[0, k, k, 0] = 1.0
        list_cores_full_mpo.append(identity_core)
    # Append remaining ham cores if any
    while ham_idx < len(list_cores_ham):
        list_cores_full_mpo.append(list_cores_ham[ham_idx])
        ham_idx += 1

    result_cores_flat, _ = tensor_matvec_prod(
        mps_cores_flat,
        list_cores_full_mpo,
        ht.mps_epsilon,
        ht.bond_dim_max,
    )
    # For statenumber, phi_0 expects list-of-lists; restore structure
    if ht.method == 'number':
        result_cores = unflatten_cores(result_cores_flat, ht.M1_modes_per_state)
    else:
        result_cores = result_cores_flat
    return extract_psi(result_cores, ht.method, ht.M1_modes_per_state)


# ============================================================
# TEST SUITE: MpoBuilder.__init__()
# ============================================================


# ------------------------------------------------------------
# TEST: homps normalization produces correct ladder operators
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_elementary_operators_homps_ladder():
    # This case tests that homps normalization produces the correct
    # ladder operators B2_lower, B2_raise, and occupation number N2_occ.
    k_max = 2

    # build a minimal mock mode
    class MockMode:
        list_g = np.array([1.0 + 0j, 2.0 + 0j])
        list_w = np.array([10.0, 20.0])
        n_l2 = 1
        n_hmodes = 2
        list_L2_coo = []
        list_L2_masks = [[[0], [0], None]]
        list_index_L2_by_hmode = [0, 0]

    mode = MockMode()
    elem = MpoBuilder(
        k_max,
        2,
        1,
        1,
        np.eye(2, dtype=np.complex128),
        np.array([0, 1]),
        mode,
        'homps',
    )
    # B2_lower[i][i+1] = sqrt(i+1) for i in range(k_max)
    assert elem.B2_lower[0, 1] == pytest.approx(1.0)
    assert elem.B2_lower[1, 2] == pytest.approx(np.sqrt(2))
    # B2_raise is transpose of B2_lower for homps
    assert elem.B2_raise[1, 0] == pytest.approx(1.0)
    assert elem.B2_raise[2, 1] == pytest.approx(np.sqrt(2))
    # Analytical: N2_occ is the number operator, N2_occ[n,n] = n
    assert elem.N2_occ[0, 0] == pytest.approx(0.0)
    assert elem.N2_occ[1, 1] == pytest.approx(1.0)
    assert elem.N2_occ[2, 2] == pytest.approx(2.0)
    # This case tests that off-diagonal elements are zero
    assert elem.N2_occ[0, 1] == pytest.approx(0.0)
    assert elem.N2_occ[1, 0] == pytest.approx(0.0)


# ------------------------------------------------------------
# TEST: adhops normalization produces correct ladder operators
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_elementary_operators_adhops_ladder():
    # This case tests that adhops normalization produces the correct
    # unit ladder operators B2_lower, B2_raise, diagonal N2_occ, and
    # coupling vectors C1_coupling_raise = list_w and C1_coupling_lower = list_g / list_w.
    k_max = 2

    class MockMode:
        list_g = np.array([1.0 + 0j, 2.0 + 0j])
        list_w = np.array([10.0, 20.0])
        n_l2 = 1
        n_hmodes = 2
        list_L2_coo = []
        list_L2_masks = [[[0], [0], None]]
        list_index_L2_by_hmode = [0, 0]

    mode = MockMode()
    elem = MpoBuilder(
        k_max,
        2,
        1,
        1,
        np.eye(2, dtype=np.complex128),
        np.array([0, 1]),
        mode,
        'adhops',
    )
    # adhops: B2_lower[i][i+1] = 1 (unit)
    assert elem.B2_lower[0, 1] == pytest.approx(1.0)
    assert elem.B2_lower[1, 2] == pytest.approx(1.0)
    # N2_occ[i+1][i+1] = i+1 (explicit diagonal)
    assert elem.N2_occ[1, 1] == pytest.approx(1.0)
    assert elem.N2_occ[2, 2] == pytest.approx(2.0)
    # C1_coupling_raise = list_w, C1_coupling_lower = list_g / list_w
    np.testing.assert_allclose(elem.C1_coupling_raise, mode.list_w)
    np.testing.assert_allclose(elem.C1_coupling_lower, mode.list_g / mode.list_w)


# ------------------------------------------------------------
# TEST: Invalid normalization raises ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_mpo_builder_invalid_normalization_raises():
    class MockMode:
        list_g = np.array([1.0 + 0j])
        list_w = np.array([10.0])
        n_l2 = 1
        n_hmodes = 1
        list_L2_coo = []
        list_L2_masks = [[[0], [0], None]]
        list_index_L2_by_hmode = [0]

    with pytest.raises(ValueError, match='Unknown normalization'):
        MpoBuilder(
            2, 2, 1, 1, np.eye(2, dtype=np.complex128),
            np.array([0, 1]), MockMode(), 'bogus',
        )


# ------------------------------------------------------------
# TEST: Scalar and derived attributes stored correctly
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_mpo_builder_stores_scalars():
    ht, tb = _make_tensor_pair('fullstate')
    builder = MpoBuilder(
        k_max=k_max,
        n_state=tb.system.size,
        modes_per_state=ht.M1_modes_per_state,
        n_lop_full=tb.mode.n_l2,
        ham=tb.system.param['HAMILTONIAN'],
        state_list=tb.system.state_list,
        mode=tb.mode,
        normalization='homps',
    )
    assert builder.k_max == k_max
    assert builder.n_state == tb.system.size
    assert builder.n_lop_full == tb.mode.n_l2
    np.testing.assert_array_equal(
        builder.M1_modes_per_state, ht.M1_modes_per_state,
    )
    # M1_mode_offset is cumulative sum with leading zero
    expected_offset = np.concatenate(
        [[0], np.cumsum(ht.M1_modes_per_state)]
    )
    np.testing.assert_array_equal(builder.M1_mode_offset, expected_offset)


# ------------------------------------------------------------
# TEST: State-dimension operators have correct shape and values
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_mpo_builder_state_operators():
    class MockMode:
        list_g = np.array([1.0 + 0j])
        list_w = np.array([10.0])
        n_l2 = 1
        n_hmodes = 1
        list_L2_coo = []
        list_L2_masks = [[[0], [0], None]]
        list_index_L2_by_hmode = [0]

    builder = MpoBuilder(
        2, 2, 1, 1, np.eye(2, dtype=np.complex128),
        np.array([0, 1]), MockMode(), 'homps',
    )
    # All state operators are (1, 2, 2, 1)
    for name in ['T4_plus', 'T4_minus', 'Q4_site', 'P4_site']:
        op = getattr(builder, name)
        assert op.shape == (1, 2, 2, 1), f'{name} shape should be (1,2,2,1)'
    # P = |1><1|, Q = |0><0|, T_left = |1><0|, T_right = |0><1|
    np.testing.assert_array_equal(
        builder.P4_site.reshape(2, 2), np.array([[0, 0], [0, 1]]),
    )
    np.testing.assert_array_equal(
        builder.Q4_site.reshape(2, 2), np.array([[1, 0], [0, 0]]),
    )
    np.testing.assert_array_equal(
        builder.T4_plus.reshape(2, 2), np.array([[0, 0], [1, 0]]),
    )
    np.testing.assert_array_equal(
        builder.T4_minus.reshape(2, 2), np.array([[0, 1], [0, 0]]),
    )
    # I2_mode is identity of size k_max + 1
    np.testing.assert_array_equal(
        builder.I2_mode, np.eye(3, dtype=np.complex128),
    )


# ============================================================
# TEST SUITE: _build_statenumber_ham_general_mpo()
# ============================================================


# ------------------------------------------------------------
# TEST: Ham general MPO has correct bond dimension
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_statenumber_ham_general_mpo_shape():
    # This case tests that the general Hamiltonian MPO bond dimension
    # equals 4 + 2*(n_state-2) for a non-nearest-neighbor Hamiltonian.
    # Build a 3-state non-NN system: state 0 couples to state 2 (skips 1).
    n = 3
    ham_gen = np.zeros((n, n), dtype=np.complex128)
    ham_gen[0, 2] = 5.0
    ham_gen[2, 0] = 5.0
    # Site-diagonal L-operators (projectors onto each site)
    lop_gen = np.zeros((n, n, n), dtype=np.float64)
    lop_list_gen = []
    for i in range(n):
        lop_gen[i, i, i] = 1.0
        lop_list_gen.append(sparse.coo_matrix(lop_gen[i]))
    gw_gen = [[g_0, w_0]] * n
    sys_param_gen = {
        'HAMILTONIAN': ham_gen,
        'GW_SYSBATH': gw_gen,
        'L_HIER': lop_list_gen,
        'L_NOISE1': lop_list_gen,
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': gw_gen,
    }
    tensor_param = {
        'METHOD': 'number',
        'MPS_EPSILON': 1e-10,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    psi_gen = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    state_list_gen = np.arange(n)
    tb = _make_tb(sp=sys_param_gen, ds=0, psi=psi_gen, sl=state_list_gen)
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_gen, tb.system)
    elem = MpoBuilder(
        ht.k_max,
        tb.system.size,
        ht.M1_modes_per_state,
        tb.mode.n_l2,
        tb.system.param['HAMILTONIAN'],
        tb.system.state_list,
        tb.mode,
        'homps',
        n_states_full=n,
        flag_nearest_neighbor_ham=False,
    )
    cores = elem._build_statenumber_ham_general_mpo()
    bonddim = int(4 + 2 * (n - 2))
    assert cores[0].shape[3] == bonddim
    # This case tests that the MPO can be contracted without producing NaN
    flat_cores = [
        c for group in ht.list_cores_phi
        for c in (group if isinstance(group, list) else [group])
    ]
    result_cores, _ = tensor_matvec_prod(flat_cores, cores, 1e-10, 50)
    for c in result_cores:
        assert not np.any(np.isnan(c)), 'MPO contraction produced NaN'


# ------------------------------------------------------------
# TEST: General Hamiltonian MPO gives H @ psi (statenumber)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_statenumber_ham_general_mpo_value():
    # This case tests the general (non-NN) Hamiltonian MPO builder.
    # Uses the same 4-site system — build_statenumber_ham_general_mpo
    # should produce the same result as the NN builder for NN Hamiltonians.
    ht, tb = _make_tensor_pair('number')
    elem = MpoBuilder(
        k_max,
        tb.system.size,
        ht.M1_modes_per_state,
        tb.mode.n_l2,
        tb.system.param['HAMILTONIAN'],
        tb.system.state_list,
        tb.mode,
        'homps',
        n_states_full=tb.system.param['NSTATES'],
        flag_nearest_neighbor_ham=False,
    )
    list_cores_ham = elem._build_statenumber_ham_general_mpo()
    psi_result = _apply_ham_mpo_and_extract(ht, tb, list_cores_ham)
    psi_input = extract_psi(
        ht.list_cores_phi,
        ht.method,
        ht.M1_modes_per_state,
    )
    psi_expected = hs @ psi_input
    np.testing.assert_allclose(
        psi_result,
        psi_expected,
        atol=1e-8,
        err_msg='General Hamiltonian MPO does not match H @ psi',
    )


# ------------------------------------------------------------
# TEST: General ham MPO handles n_state=1 without crashing
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_statenumber_ham_general_single_site():
    # This case tests that the general Hamiltonian MPO builder handles
    # a single-site system (n_state=1) without crashing. With one site
    # there are no off-diagonal terms — only the diagonal energy.
    E = 3.5
    ham_1 = np.array([[E]], dtype=np.complex128)
    k_max_1 = 2
    n_modes = 1

    class MockMode:
        list_g = np.array([1.0 + 0j])
        list_w = np.array([10.0])
        n_l2 = 1
        n_hmodes = 1
        list_L2_coo = [sparse.coo_matrix(np.array([[1.0]]))]
        list_L2_masks = [[[0], [0], None]]
        list_index_L2_by_hmode = [0]

    mode = MockMode()
    elem = MpoBuilder(
        k_max_1,
        1,
        np.array([n_modes]),
        1,
        ham_1,
        np.array([0]),
        mode,
        'homps',
        n_states_full=1,
        flag_nearest_neighbor_ham=False,
    )
    cores = elem.build_statenumber_ham_mpo()

    # Should produce 1 state core + n_modes identity mode cores = 2 cores
    assert len(cores) == 1 + n_modes

    # State core shape: (1, 2, 2, 1) — single bond on each side
    state_core = cores[0]
    assert state_core.shape[0] == 1
    assert state_core.shape[3] == 1

    # The occupied state should carry the diagonal energy E
    # state_core[0, 1, 1, 0] = E (from H[0,0] * P4_site[0,1,1,0])
    assert state_core[0, 1, 1, 0] == pytest.approx(E)
    # The unoccupied state should be identity pass-through
    # state_core[0, 0, 0, 0] = 1.0 (from Q4_site[0,0,0,0])
    assert state_core[0, 0, 0, 0] == pytest.approx(1.0)


# ------------------------------------------------------------
# TEST: General MPO with non-NN Ham reproduces H @ psi via contraction
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_statenumber_ham_general_mpo_nonnn_value():
    # Analytical: MPO applied to MPS should give H @ psi for a
    # Hamiltonian with long-range coupling H[0,2] = 5.0 (skips site 1).
    # The nearest-neighbor builder cannot represent this coupling, so this
    # test specifically exercises the general MPO builder.
    n = 3
    ham_gen = np.zeros((n, n), dtype=np.complex128)
    ham_gen[0, 1] = 2.0   # NN coupling (covered by NN builder too)
    ham_gen[1, 0] = 2.0
    ham_gen[0, 2] = 5.0   # Long-range coupling — requires general builder
    ham_gen[2, 0] = 5.0
    ham_gen[1, 1] = 1.0   # On-site energies
    ham_gen[2, 2] = 3.0

    # Site-diagonal L-operators (projectors onto each site)
    lop_gen = np.zeros((n, n, n), dtype=np.float64)
    lop_list_gen = []
    for i in range(n):
        lop_gen[i, i, i] = 1.0
        lop_list_gen.append(sparse.coo_matrix(lop_gen[i]))
    gw_gen = [[g_0, w_0]] * n
    sys_param_gen = {
        'HAMILTONIAN': ham_gen,
        'GW_SYSBATH': gw_gen,
        'L_HIER': lop_list_gen,
        'L_NOISE1': lop_list_gen,
        'ALPHA_NOISE1': bcf_exp,
        'PARAM_NOISE1': gw_gen,
    }
    tensor_param = {
        'METHOD': 'number',
        'MPS_EPSILON': 1e-10,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    # Delocalized initial state with complex phases so that all Hamiltonian
    # matrix elements contribute non-trivially to H @ psi.
    psi_gen = np.array([0.5 + 0.3j, -0.4 + 0.2j, 0.6 - 0.1j], dtype=np.complex128)
    psi_gen /= np.linalg.norm(psi_gen)
    state_list_gen = np.arange(n)

    tb = _make_tb(sp=sys_param_gen, ds=0, psi=psi_gen, sl=state_list_gen)
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_gen, tb.system)

    elem = MpoBuilder(
        ht.k_max,
        tb.system.size,
        ht.M1_modes_per_state,
        tb.mode.n_l2,
        tb.system.param['HAMILTONIAN'],
        tb.system.state_list,
        tb.mode,
        'homps',
        n_states_full=n,
        flag_nearest_neighbor_ham=False,
    )
    list_cores_ham = elem._build_statenumber_ham_general_mpo()

    psi_result = _apply_ham_mpo_and_extract(ht, tb, list_cores_ham)
    # Use psi_gen directly — avoids extract_psi dependency and matches
    # the known input state (no SVD truncation at bond dim 1).
    psi_expected = ham_gen @ psi_gen

    np.testing.assert_allclose(
        psi_result,
        psi_expected,
        atol=1e-8,
        err_msg='General MPO with non-NN coupling does not match H @ psi',
    )


# ============================================================
# TEST SUITE: MpoBuilder.build_elementary_ops()
# ============================================================


def _mock_mode(list_g=None, list_w=None):
    """Helper: build a minimal mock mode object."""

    class MockMode:
        pass

    m = MockMode()
    m.list_g = np.array(list_g or [1.0 + 0j, 2.0 + 0j])
    m.list_w = np.array(list_w or [10.0, 20.0])
    m.n_l2 = 1
    m.n_hmodes = len(m.list_w)
    m.list_L2_coo = []
    # Rows/cols/ix_ per L2 in the active basis; [0][0] is the L2's site.
    m.list_L2_masks = [[[i], [i], None] for i in range(m.n_l2)]
    m.list_index_L2_by_hmode = [0] * m.n_hmodes
    return m


def _make_elem(k_max_val=2, normalization='homps', mode=None):
    """Helper: build and return an MpoBuilder instance."""
    if mode is None:
        mode = _mock_mode()
    return MpoBuilder(
        k_max_val,
        2,
        1,
        1,
        np.eye(2, dtype=np.complex128),
        np.array([0, 1]),
        mode,
        normalization,
        n_states_full=2,
        flag_nearest_neighbor_ham=True,
    )


# ------------------------------------------------------------
# TEST: Invalid normalization raises ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_elementary_ops_invalid_normalization():
    # This case tests that an unknown normalization string raises ValueError.
    mode = _mock_mode()
    with pytest.raises(ValueError, match='Unknown normalization'):
        MpoBuilder(
            2,
            2,
            1,
            1,
            np.eye(2, dtype=np.complex128),
            np.array([0, 1]),
            mode,
            'invalid',
            n_states_full=2,
            flag_nearest_neighbor_ham=True,
        )


# ------------------------------------------------------------
# TEST: State-dimension operators have correct values
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_elementary_ops_state_dimension_operators():
    # This case tests P4_site, Q4_site, T4_plus, T4_minus, I2_state.
    elem = _make_elem()

    # P4_site = |1><1| projector in 4-D core shape
    expected_P4 = np.array([[0, 0], [0, 1]], dtype=np.complex128).reshape(1, 2, 2, 1)
    np.testing.assert_allclose(
        elem.P4_site, expected_P4, atol=1e-14, err_msg='P4_site should be |1><1|'
    )

    # Q4_site = |0><0| projector in 4-D core shape
    expected_Q4 = np.array([[1, 0], [0, 0]], dtype=np.complex128).reshape(1, 2, 2, 1)
    np.testing.assert_allclose(
        elem.Q4_site, expected_Q4, atol=1e-14, err_msg='Q4_site should be |0><0|',
    )

    # T4_plus = |1><0| in 4-D core shape
    expected_T4_plus = np.array([[0, 0], [1, 0]], dtype=np.complex128).reshape(1, 2, 2, 1)
    np.testing.assert_allclose(
        elem.T4_plus, expected_T4_plus, atol=1e-14, err_msg='T4_plus should have [1,0]=1',
    )

    # T4_minus = |0><1| in 4-D core shape
    expected_T4_minus = np.array([[0, 1], [0, 0]], dtype=np.complex128).reshape(1, 2, 2, 1)
    np.testing.assert_allclose(
        elem.T4_minus, expected_T4_minus, atol=1e-14, err_msg='T4_minus should have [0,1]=1',
    )

    # I2_state is identity of size n_state=2
    np.testing.assert_allclose(
        elem.I2_state,
        np.eye(2, dtype=np.complex128),
        atol=1e-14,
        err_msg='I2_state should be identity',
    )


# ------------------------------------------------------------
# TEST: k_max=0 produces valid degenerate operators
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_elementary_ops_k_max_zero():
    # This case tests that k_max=0 produces (1,1) mode operators
    # with no ladder steps.
    elem = _make_elem(k_max_val=0)

    # All mode operators should be (1, 1)
    assert elem.B2_raise.shape == (1, 1)
    assert elem.B2_lower.shape == (1, 1)
    assert elem.N2_occ.shape == (1, 1)
    assert elem.I2_mode.shape == (1, 1)

    # Ladder operators should be zero (no transitions possible)
    np.testing.assert_allclose(
        elem.B2_raise, [[0]], atol=1e-14, err_msg='B2_raise should be zero for k_max=0'
    )
    np.testing.assert_allclose(
        elem.B2_lower, [[0]], atol=1e-14, err_msg='B2_lower should be zero for k_max=0'
    )
    np.testing.assert_allclose(
        elem.N2_occ, [[0]], atol=1e-14, err_msg='N2_occ should be zero for k_max=0'
    )

    # Identity should still be [[1]]
    np.testing.assert_allclose(
        elem.I2_mode, [[1]], atol=1e-14, err_msg='I2_mode should be [[1]] for k_max=0'
    )


# ------------------------------------------------------------
# TEST: homps C1_coupling_raise and C1_coupling_lower have correct values
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_elementary_ops_homps_coupling_vectors():
    # Convention of Gao et al., Phys. Rev. A 105, L030202 (2022), Eq. (10):
    # C1_coupling_raise pairs with B2_raise (b†): V_m^+ = g/sqrt(|g|)
    # C1_coupling_lower pairs with B2_lower (b):  V_m^- = sqrt(|g|)
    # A complex g is required: for real positive g the two orderings coincide.
    list_g = [3.0 + 0j, 4.0 + 1j]
    mode = _mock_mode(list_g=list_g)
    elem = _make_elem(normalization='homps', mode=mode)

    for i, g in enumerate(list_g):
        expected_raise = g / np.sqrt(np.abs(g))
        expected_lower = np.sqrt(np.abs(g))
        np.testing.assert_allclose(
            elem.C1_coupling_raise[i],
            expected_raise,
            atol=1e-12,
            err_msg=f'C1_coupling_raise[{i}] should be g/sqrt(|g|) (V_m^+, pairs with b†)',
        )
        np.testing.assert_allclose(
            elem.C1_coupling_lower[i],
            expected_lower,
            atol=1e-12,
            err_msg=f'C1_coupling_lower[{i}] should be sqrt(|g|) (V_m^-, pairs with b)',
        )
        # The split must reproduce the physical coupling either way round.
        np.testing.assert_allclose(
            elem.C1_coupling_raise[i] * elem.C1_coupling_lower[i],
            g,
            atol=1e-12,
            err_msg=f'V_m^+ * V_m^- should equal g for mode {i}',
        )


# ------------------------------------------------------------
# TEST: Core entries match hand-computed values for minimal system
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_statenumber_hierarchy_mpo_values_minimal():
    # This case tests that the statenumber hierarchy MPO core entries
    # match hand-computed values for a 1-state, 1-mode, k_max=1 system
    # with zero noise, using adhops normalization with known g and w.
    # Setup: adhops with known g, w so entries are hand-computable
    g, w = 2.0 + 0j, 10.0
    mode = _mock_mode(list_g=[g], list_w=[w])
    mode.n_hmodes = 1
    elem = _make_elem(k_max_val=1, normalization='adhops', mode=mode)
    # Override n_state and modes_per_state for 1-site system
    elem.n_state = 1
    elem.M1_modes_per_state = np.array([1])

    z_conj_t = np.zeros(1, dtype=np.complex128)
    L_conj_avg = np.zeros(1, dtype=np.complex128)
    cores = elem.build_statenumber_hierarchy_mpo(
        z_conj_t,
        L_conj_avg,
        0.0,
    )
    assert len(cores) == 2

    # --- Core 0: state site (1, 2, 2, 5) ---
    site = cores[0]
    # This case tests the [0,:,:,0] block is Q4_site = |0><0|.
    np.testing.assert_allclose(
        site[0, :, :, 0],
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        atol=1e-14,
        err_msg='[0,:,:,0] should be |0><0|',
    )
    # This case tests [0,:,:,1] and [0,:,:,2] are 1j * |1><1|.
    expected_proj = 1j * np.array([[0, 0], [0, 1]], dtype=np.complex128)
    np.testing.assert_allclose(
        site[0, :, :, 1],
        expected_proj,
        atol=1e-14,
        err_msg='[0,:,:,1] should be 1j * |1><1|',
    )
    # Channel 2 (damp1) at site 0 is 1j * |1><1| in the default (non-vacuum)
    # convention; the site-0 widening to 1j * I_site only fires when
    # flag_gs_vacuum is set (see the gated-widening test below).
    np.testing.assert_allclose(
        site[0, :, :, 2],
        expected_proj,
        atol=1e-14,
        err_msg='[0,:,:,2] at site 0 should be 1j * |1><1| (non-vacuum)',
    )
    # This case tests remaining bond slots [0,:,:,3] and [0,:,:,4] are zero.
    np.testing.assert_allclose(
        site[0, :, :, 3],
        0.0,
        atol=1e-14,
        err_msg='[0,:,:,3] should be zero',
    )
    np.testing.assert_allclose(
        site[0, :, :, 4],
        0.0,
        atol=1e-14,
        err_msg='[0,:,:,4] should be zero',
    )

    # --- Core 1: mode site (5, 2, 2, 1), last mode ---
    mode_core = cores[1]
    # adhops k_max=1: B2_raise=[[0,0],[1,0]], B2_lower=[[0,1],[0,0]]
    # C1_coupling_raise=w=10, C1_coupling_lower=g/w=0.2
    # This case tests [1,:,:,0] = C1_coupling_raise*B2_raise - C1_coupling_lower*B2_lower.
    expected_coupling = np.array(
        [[0, -0.2], [10, 0]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(
        mode_core[1, :, :, 0],
        expected_coupling,
        atol=1e-14,
        err_msg='[1,:,:,0] coupling block incorrect',
    )
    # This case tests [4,:,:,0] = identity (pass-through).
    np.testing.assert_allclose(
        mode_core[4, :, :, 0],
        np.eye(2, dtype=np.complex128),
        atol=1e-14,
        err_msg='[4,:,:,0] should be identity',
    )
    # This case tests [2,:,:,0] = -w * N2_occ = -10 * diag(0,1).
    expected_damping = np.array(
        [[0, 0], [0, -10]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(
        mode_core[2, :, :, 0],
        expected_damping,
        atol=1e-14,
        err_msg='[2,:,:,0] damping block incorrect',
    )
    # This case tests remaining bond slots [0,:,:,0] and [3,:,:,0] are zero.
    np.testing.assert_allclose(
        mode_core[0, :, :, 0],
        0.0,
        atol=1e-14,
        err_msg='[0,:,:,0] should be zero for last mode',
    )
    np.testing.assert_allclose(
        mode_core[3, :, :, 0],
        0.0,
        atol=1e-14,
        err_msg='[3,:,:,0] should be zero for last mode',
    )


# ------------------------------------------------------------
# TEST: site-0 damp1 widening is a no-op on the single-excitation manifold
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_statenumber_hierarchy_mpo_site0_widening_noop_on_manifold():
    # The site-0 damp1 widening (gated by flag_gs_vacuum) opens the
    # vacuum-drift channel on the electronic-ground sector and must be a
    # bit-exact no-op on the single-excitation manifold. This builds a
    # 2-site / 2-mode hierarchy MPO (so the interior state core also fires),
    # applies it to a manifold wavefunction via the generic
    # vector -> MPS -> apply -> vector path, and checks the widened and
    # un-widened actions agree on the single-excitation inputs while the
    # ground-sector action differs.
    layout = _dipole_layout(2, np.array([1, 1]), 1)

    def _hierarchy_mpo(flag_gs_vacuum):
        mode = _mock_mode(list_g=[1.5 + 0j, 0.8 + 0j], list_w=[5.0, 15.0])
        mode.n_hmodes = 2
        mode.n_l2 = 2
        mode.list_L2_masks = [[[0], [0], None], [[1], [1], None]]
        elem = MpoBuilder(
            1, 2, np.array([1, 1]), 2, np.eye(2, dtype=np.complex128),
            np.array([0, 1]), mode, 'adhops',
        n_states_full=2,
            flag_nearest_neighbor_ham=True,
            flag_gs_vacuum=flag_gs_vacuum,
        )
        z_conj_t = np.array([0.2 + 0.1j, -0.1 + 0.3j], dtype=np.complex128)
        L_conj_avg = np.array([0.3 - 0.05j, 0.4 + 0.2j], dtype=np.complex128)
        return elem.build_statenumber_hierarchy_mpo(z_conj_t, L_conj_avg, 0.5)

    def _act(V2_phi, flag):
        flat_cores = _vector_to_mps_flat(V2_phi, layout)
        result_flat, _ = tensor_matvec_prod(
            flat_cores, _hierarchy_mpo(flag), 1e-12, 64,
        )
        return _mps_flat_to_vector(result_flat, layout)

    # Manifold vector form phi[state, aux]: rows = [ground, e_site0, e_site1],
    # cols = [k=vac, k=mode0, k=mode1].
    rng = np.random.default_rng(0)
    V2_phi = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))

    # Single-excitation manifold input: zero the electronic-ground row.
    V2_single = V2_phi.copy()
    V2_single[0, :] = 0.0
    np.testing.assert_allclose(
        _act(V2_single, True), _act(V2_single, False), atol=1e-12,
        err_msg='site-0 widening must be a no-op on the single-excitation '
                'manifold',
    )

    # The widening is live: with the ground sector populated (the
    # Phi[vac, k=e_n] drift configurations) the action differs on/off.
    assert not np.allclose(
        _act(V2_phi, True), _act(V2_phi, False), atol=1e-12
    ), (
        'site-0 widening should alter the ground-sector (vacuum-drift) action'
    )


# ============================================================
# TEST SUITE: build_statenumber_ham_mpo()
# ============================================================


# ------------------------------------------------------------
# TEST: NN Hamiltonian MPO gives H @ psi (statenumber)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_statenumber_ham_nn_mpo_value():
    # This case tests that the nearest-neighbor Hamiltonian MPO,
    # when applied to the MPS, produces H @ psi. Uses a delocalized
    # psi with complex phases so all four components contribute to H @ psi,
    # and a negative coupling (-10) on bond 1-2 to verify sign handling.
    #
    # The input psi is used directly for the expected value — we compare
    # psi_result against H_neg @ psi_test without going through extract_psi
    # on the input MPS, which would only recover the normalized vector.
    psi_test = np.array([0.3 + 0.1j, -0.5, 0.2 - 0.4j, 0.6 + 0.2j], dtype=np.complex128)
    psi_test = psi_test / np.linalg.norm(psi_test)

    # Build Hamiltonian with negative coupling on bond 1-2
    H2_ham_neg = np.zeros([nsite, nsite], dtype=np.complex128)
    H2_ham_neg[0, 1] = 40
    H2_ham_neg[1, 0] = 40
    H2_ham_neg[1, 2] = -10
    H2_ham_neg[2, 1] = -10
    H2_ham_neg[2, 3] = 40
    H2_ham_neg[3, 2] = 40

    sys_param_neg = dict(sys_param)
    sys_param_neg['HAMILTONIAN'] = H2_ham_neg

    tb = _make_tb(sp=sys_param_neg, psi=psi_test)
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_test, tb.system)

    elem = MpoBuilder(
        k_max,
        tb.system.size,
        ht.M1_modes_per_state,
        tb.mode.n_l2,
        tb.system.param['HAMILTONIAN'],
        tb.system.state_list,
        tb.mode,
        'homps',
        n_states_full=tb.system.param['NSTATES'],
        flag_nearest_neighbor_ham=True,
    )
    list_cores_ham = elem.build_statenumber_ham_mpo()

    # This case tests interior state core shapes.
    # For 4 sites with 2 modes each: state0(0), mode0a(1), mode0b(2),
    # state1(3), mode1a(4), mode1b(5), state2(6), ..., state3(9), ...
    # Interior state cores (sites 1, 2) should be (4, 2, 2, 4).
    modes_per_state = ht.M1_modes_per_state[0]  # 2 modes per state
    interior_state_core_indices = [
        1 + modes_per_state,                        # state1 core index
        1 + modes_per_state + (1 + modes_per_state),  # state2 core index
    ]
    for core_idx in interior_state_core_indices:
        assert list_cores_ham[core_idx].shape == (4, 2, 2, 4), (
            f'Interior state core at index {core_idx} should be (4, 2, 2, 4), '
            f'got {list_cores_ham[core_idx].shape}'
        )

    # Apply MPO to MPS and extract result
    psi_result = _apply_ham_mpo_and_extract(ht, tb, list_cores_ham)
    # Compare against H_neg @ psi_test directly (not via extract_psi on input MPS)
    psi_expected = H2_ham_neg @ psi_test
    np.testing.assert_allclose(
        psi_result,
        psi_expected,
        atol=1e-8,
        err_msg='NN Hamiltonian MPO does not match H @ psi',
    )


# ------------------------------------------------------------
# TEST: NN Ham MPO mode cores are block-diagonal identity
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_nn_ham_mpo_mode_cores_identity():
    # This case tests that mode cores in the NN Hamiltonian MPO are
    # block-diagonal identity: for each bond index j,
    # T4_core_mode[j, :, :, j] == eye(k_max+1) and all off-diagonal
    # bond blocks are zero. This reflects that the Hamiltonian acts only
    # on state cores and passes the hierarchy dimension through unchanged.
    ht, tb = _make_tensor_pair('number')
    elem = MpoBuilder(
        k_max,
        tb.system.size,
        ht.M1_modes_per_state,
        tb.mode.n_l2,
        tb.system.param['HAMILTONIAN'],
        tb.system.state_list,
        tb.mode,
        'homps',
        n_states_full=tb.system.param['NSTATES'],
        flag_nearest_neighbor_ham=True,
    )
    list_cores_ham = elem.build_statenumber_ham_mpo()

    # Core layout: state0, mode0a, mode0b, state1, mode1a, mode1b, ...
    # Identify mode core indices by skipping state cores (one per site).
    I2_mode = np.eye(k_max + 1, dtype=np.complex128)
    core_idx = 0
    for site in range(nsite):
        # Skip the state core for this site
        core_idx += 1
        n_modes = ht.M1_modes_per_state[site]
        for _ in range(n_modes):
            T4_core = list_cores_ham[core_idx]
            bond_dim = T4_core.shape[0]
            assert T4_core.shape == (bond_dim, k_max + 1, k_max + 1, bond_dim), (
                f'Mode core at index {core_idx} has unexpected shape {T4_core.shape}'
            )
            # This case tests that diagonal bond blocks are identity
            for j in range(bond_dim):
                np.testing.assert_allclose(
                    T4_core[j, :, :, j],
                    I2_mode,
                    atol=1e-14,
                    err_msg=(
                        f'Mode core {core_idx}, bond block [{j},:,:,{j}] '
                        f'should be identity'
                    ),
                )
            # This case tests that off-diagonal bond blocks are zero
            for j in range(bond_dim):
                for k in range(bond_dim):
                    if j != k:
                        np.testing.assert_allclose(
                            T4_core[j, :, :, k],
                            0.0,
                            atol=1e-14,
                            err_msg=(
                                f'Mode core {core_idx}, off-diagonal block '
                                f'[{j},:,:,{k}] should be zero'
                            ),
                        )
            core_idx += 1


# ------------------------------------------------------------
# TEST: NN Ham MPO works with adaptive sub-basis
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_nn_ham_mpo_adaptive_subbasis():
    # This case tests that the NN Hamiltonian MPO applied to an MPS
    # initialized with state_list = [1, 2] gives H_sub @ psi, where
    # H_sub = hs[np.ix_([1, 2], [1, 2])]. This verifies that the MPO
    # correctly uses relative site indices mapped through state_list
    # and does not accidentally reference out-of-basis states.
    #
    # psi_full is a 4-element vector nonzero only at sites 1 and 2.
    # ht.initialize slices to system.state_list=[1,2], producing the
    # 2-element active wavefunction used to build the MPS.
    sub_state_list = np.array([1, 2])
    psi_sub = np.array([0.6 + 0.3j, -0.4 - 0.5j], dtype=np.complex128)
    psi_sub = psi_sub / np.linalg.norm(psi_sub)
    # Embed into full 4-site space for initialize() which indexes by state_list
    psi_full = np.zeros(nsite, dtype=np.complex128)
    psi_full[1] = psi_sub[0]
    psi_full[2] = psi_sub[1]

    tb = _make_tb(psi=psi_full, sl=sub_state_list)
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_full, tb.system)

    elem = MpoBuilder(
        k_max,
        tb.system.size,
        ht.M1_modes_per_state,
        tb.mode.n_l2,
        tb.system.param['HAMILTONIAN'],
        tb.system.state_list,
        tb.mode,
        'homps',
        n_states_full=tb.system.param['NSTATES'],
        flag_nearest_neighbor_ham=True,
    )
    list_cores_ham = elem.build_statenumber_ham_mpo()

    # Apply MPO and extract using the 2-state modes_per_state (active only).
    # ht.M1_modes_per_state has shape (4,) covering all states, but the MPS
    # only has cores for the 2 active states. Slice to the active sub-basis.
    M1_modes_active = ht.M1_modes_per_state[sub_state_list]
    result_cores_flat, _ = tensor_matvec_prod(
        ht.flat_cores, list_cores_ham, ht.mps_epsilon, ht.bond_dim_max,
    )
    result_cores = unflatten_cores(result_cores_flat, M1_modes_active)
    psi_result = extract_psi(result_cores, ht.method, M1_modes_active)

    # Expected: H restricted to states [1, 2] applied to the active psi
    H2_sub = hs[np.ix_(sub_state_list, sub_state_list)]
    psi_expected = H2_sub @ psi_sub
    np.testing.assert_allclose(
        psi_result,
        psi_expected,
        atol=1e-8,
        err_msg='NN MPO with adaptive sub-basis does not match H_sub @ psi',
    )


# ============================================================
# TEST SUITE: build_fullstate_mpo()
# ============================================================


# ------------------------------------------------------------
# TEST: Fullstate MPO with zero noise gives H @ psi
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_fullstate_mpo_value_ham_only():
    # This case tests the fullstate MPO builder with zero noise inputs,
    # so only the Hamiltonian contribution survives. The MPO should
    # act as H on the state core and identity on mode cores.
    ht, tb = _make_tensor_pair('fullstate')
    elem = MpoBuilder(
        k_max,
        tb.system.size,
        ht.M1_modes_per_state,
        tb.mode.n_l2,
        tb.system.param['HAMILTONIAN'],
        tb.system.state_list,
        tb.mode,
        'homps',
        n_states_full=tb.system.param['NSTATES'],
        flag_nearest_neighbor_ham=True,
    )
    # Zero noise → only Hamiltonian terms in the MPO
    z_conj_t = np.zeros(tb.system.size, dtype=np.complex128)
    L_conj_avg = np.zeros(tb.system.size, dtype=np.complex128)
    norm_corr = 0.0
    list_cores_op = elem.build_fullstate_mpo(
        z_conj_t,
        L_conj_avg,
        norm_corr,
    )
    psi_result = _apply_ham_mpo_and_extract(ht, tb, list_cores_op)
    psi_input = extract_psi(
        ht.list_cores_phi,
        ht.method,
        ht.M1_modes_per_state,
    )
    # Analytical: with zero noise, the fullstate MPO applies -i/hbar * H
    # to the state. Result should be proportional to H @ psi.
    psi_Hpsi = hs @ psi_input
    # Find proportionality constant from first nonzero component
    nonzero = np.argmax(np.abs(psi_Hpsi))
    assert np.abs(psi_Hpsi[nonzero]) > 1e-14, (
        'H @ psi should be nonzero for this setup'
    )
    ratio = psi_result[nonzero] / psi_Hpsi[nonzero]
    psi_expected = ratio * psi_Hpsi
    np.testing.assert_allclose(
        psi_result,
        psi_expected,
        atol=1e-8,
        err_msg='Fullstate MPO (zero noise) not proportional to H @ psi',
    )


# ============================================================
# TEST SUITE: build_statenumber_operator_mpo()
# ============================================================


# ------------------------------------------------------------
# TEST: Raise operator MPO moves population correctly
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_statenumber_operator_mpo_raise():
    # This case tests that a raise operator |1><0| applied via MPO
    # moves population from site 0 to site 1.
    psi_site0 = np.zeros(nsite, dtype=np.complex128)
    psi_site0[0] = 1.0
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    tb = _make_tb(psi=psi_site0)
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_site0, tb.system)

    op_raise = np.zeros((nsite, nsite), dtype=np.complex128)
    op_raise[1, 0] = 1.0
    list_cores_op = build_statenumber_operator_mpo(
        op_raise, nsite, k_max, ht.M1_modes_per_state,
    )
    result_cores_flat, _ = tensor_matvec_prod(
        ht.flat_cores, list_cores_op, ht.mps_epsilon, ht.bond_dim_max,
    )
    result_cores = unflatten_cores(result_cores_flat, ht.M1_modes_per_state)
    phi0_after = extract_psi(result_cores, ht.method, ht.M1_modes_per_state)
    expected = np.zeros(nsite, dtype=np.complex128)
    expected[1] = 1.0
    np.testing.assert_allclose(phi0_after, expected, atol=1e-10)


# ------------------------------------------------------------
# TEST: General dense operator MPO matches direct matrix-vector
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_statenumber_operator_mpo_general():
    # This case tests that a general dense operator applied via MPO
    # gives the same phi_0 as direct matrix-vector multiplication.
    ht, tb = _make_tensor_pair('number')
    phi0_before = extract_psi(
        ht.list_cores_phi, ht.method, ht.M1_modes_per_state
    )
    # General operator with all entries nonzero
    H2_op = np.array([
        [0.5, 0.1, 0.2, 0.0],
        [0.1, 0.3, 0.0, 0.4],
        [0.2, 0.0, 0.7, 0.1],
        [0.0, 0.4, 0.1, 0.6],
    ], dtype=np.complex128)
    list_cores_op = build_statenumber_operator_mpo(
        H2_op, nsite, k_max, ht.M1_modes_per_state,
    )
    result_cores_flat, _ = tensor_matvec_prod(
        ht.flat_cores, list_cores_op, ht.mps_epsilon, ht.bond_dim_max,
    )
    result_cores = unflatten_cores(result_cores_flat, ht.M1_modes_per_state)
    phi0_after = extract_psi(result_cores, ht.method, ht.M1_modes_per_state)
    expected = H2_op @ phi0_before
    np.testing.assert_allclose(phi0_after, expected, atol=1e-10)


# ------------------------------------------------------------
# TEST: Single-site operator MPO applies scalar correctly
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_statenumber_operator_mpo_single_site():
    # This case tests the n_state == 1 special case (bond dim 1).
    # A scalar operator c * |0><0| should scale the occupied component by c
    # and leave the unoccupied component as identity (Q_site).
    n = 1
    k = 2
    M1_modes = np.array([2])
    c = 3.0 + 1.0j
    H2_op = np.array([[c]], dtype=np.complex128)
    list_cores = build_statenumber_operator_mpo(H2_op, n, k, M1_modes)

    # This case tests core count: 1 state core + 2 mode cores
    assert len(list_cores) == 3

    # This case tests bond dim 1 throughout
    for core in list_cores:
        assert core.shape[0] == 1
        assert core.shape[3] == 1

    # This case tests state core content: c * P + Q = [[1,0],[0,c]]
    T2_state = list_cores[0][0, :, :, 0]
    expected_state = np.array([[1, 0], [0, c]], dtype=np.complex128)
    np.testing.assert_allclose(T2_state, expected_state, atol=1e-12)

    # This case tests mode cores are identity
    for core in list_cores[1:]:
        np.testing.assert_allclose(
            core[0, :, :, 0], np.eye(k + 1, dtype=np.complex128), atol=1e-12
        )


# ------------------------------------------------------------
# TEST: Two-site operator MPO matches matrix-vector product
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_statenumber_operator_mpo_two_site():
    # This case tests the n_state == 2 path (bond dim 4, no daisy-chain).
    # Verifies end-to-end correctness via contraction against a 2-site MPS.
    psi_2site = np.zeros(2, dtype=np.complex128)
    psi_2site[0] = 0.6
    psi_2site[1] = 0.8

    n = 2
    k = 2
    M1_modes = np.array([2, 2])
    H2_op = np.array([[0.5, 0.3], [0.3, 0.7]], dtype=np.complex128)
    list_cores = build_statenumber_operator_mpo(H2_op, n, k, M1_modes)

    # This case tests bond dimension: first core (1, 2, 2, 4),
    # last state core (4, 2, 2, 1).
    assert list_cores[0].shape == (1, 2, 2, 4)
    assert list_cores[3].shape == (4, 2, 2, 1)

    # This case tests core count: 2 state cores + 2 + 2 mode cores
    assert len(list_cores) == 6


# ------------------------------------------------------------
# TEST: Diagonal operator MPO scales each state independently
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_statenumber_operator_mpo_diagonal():
    # This case tests that a purely diagonal operator scales each
    # state component without mixing, verifying off-diagonal channels
    # don't leak.
    ht, tb = _make_tensor_pair('number')
    phi0_before = extract_psi(
        ht.list_cores_phi, ht.method, ht.M1_modes_per_state
    )
    diag_vals = np.array([0.5, 1.5, 2.0, 0.3], dtype=np.complex128)
    H2_op = np.diag(diag_vals)
    list_cores_op = build_statenumber_operator_mpo(
        H2_op, nsite, k_max, ht.M1_modes_per_state,
    )
    result_cores_flat, _ = tensor_matvec_prod(
        ht.flat_cores, list_cores_op, ht.mps_epsilon, ht.bond_dim_max,
    )
    result_cores = unflatten_cores(result_cores_flat, ht.M1_modes_per_state)
    phi0_after = extract_psi(result_cores, ht.method, ht.M1_modes_per_state)
    expected = diag_vals * phi0_before
    np.testing.assert_allclose(phi0_after, expected, atol=1e-10)


# ------------------------------------------------------------
# TEST: Operator MPO core count matches state + mode structure
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_statenumber_operator_mpo_core_count():
    # This case tests that the MPO has exactly n_state state cores
    # plus sum(M1_modes_per_state) mode cores.
    ht, tb = _make_tensor_pair('number')
    H2_op = np.eye(nsite, dtype=np.complex128)
    list_cores = build_statenumber_operator_mpo(
        H2_op, nsite, k_max, ht.M1_modes_per_state,
    )
    expected_count = nsite + int(np.sum(ht.M1_modes_per_state))
    assert len(list_cores) == expected_count


# ------------------------------------------------------------
# TEST: Fullstate MPO mode-core frequencies and bond taper
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_fullstate_mpo_mode_frequencies():
    # This case tests that the fullstate mode cores carry the mode
    # exponents of their own state, and that the bond closes one
    # L-operator channel per state.
    #
    # To expose mode-indexing bugs we use nonzero noise (activates L-operator bond
    # channels) and compare mode core damping terms against expected
    # values computed from the state mode exponents. The builder relies on
    # the sorted one-to-one state/L-operator map, so state_list covers every
    # state.

    # Full 4-state system provides mode frequencies and L-operators
    tb_full = _make_tb()
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    ht_full = HopsTensorWavefunction(
        k_max, tensor_param, integrator_param, eom_param,
    )
    ht_full.initialize(psi_0, tb_full.system)

    # M1_modes_per_state covers all 4 states (absolute indexing).
    # modes_per_state = [2, 2, 2, 2] and mode_offset = [0, 2, 4, 6, 8].
    M1_modes_per_state = ht_full.M1_modes_per_state  # shape (4,)
    subset_state_list = np.arange(nsite)

    n_state_sub = len(subset_state_list)

    # Restrict L-operators to the 2x2 active subspace for the state core
    class _MockMode:
        """Lightweight stand-in providing list_L2_coo sliced to subset."""
        def __init__(self, real_mode, subset):
            self.list_w = real_mode.list_w
            self.list_g = real_mode.list_g
            self.n_hmodes = real_mode.n_hmodes
            self.list_L2_coo = [
                sparse.coo_matrix(
                    L.toarray()[np.ix_(subset, subset)]
                )
                for L in real_mode.list_L2_coo
            ]
            self.list_L2_masks = [
                [sorted(set(L.row)), sorted(set(L.col)), None]
                for L in self.list_L2_coo
            ]
            self.list_index_L2_by_hmode = real_mode.list_index_L2_by_hmode
    mock_mode = _MockMode(tb_full.mode, subset_state_list)

    elem_sub = MpoBuilder(
        k_max,
        n_state_sub,
        M1_modes_per_state,
        tb_full.mode.n_l2,
        tb_full.system.param['HAMILTONIAN'],
        subset_state_list,
        mock_mode,
        'homps',
        n_states_full=tb_full.system.param['NSTATES'],
        flag_nearest_neighbor_ham=True,
    )

    np.random.seed(99)
    n_l2 = tb_full.mode.n_l2
    z_conj_t = np.random.randn(n_l2) + 1j * np.random.randn(n_l2)
    L_conj_avg = np.random.randn(n_l2) + 1j * np.random.randn(n_l2)
    norm_corr = 0.1

    list_cores_sub = elem_sub.build_fullstate_mpo(
        z_conj_t, L_conj_avg, norm_corr,
    )

    # The MPO should have 1 state core + one mode core per mode.
    expected_n_cores = 1 + int(np.sum(M1_modes_per_state))
    assert len(list_cores_sub) == expected_n_cores, (
        f'Expected {expected_n_cores} cores, got {len(list_cores_sub)}'
    )

    # Each state closes its own L-operator channel at its last mode core, so
    # the left bond steps down once per state and bottoms out at 3. The
    # layout is the one build_fullstate_mpo documents: the L-op channels
    # still open, plus the damping channel and the Hamiltonian sink.
    list_bond_left = [core.shape[0] for core in list_cores_sub[1:]]
    list_bond_expected = [
        n_l2 + 2 - state
        for state in range(nsite)
        for _ in range(M1_modes_per_state[state])
    ]
    assert list_bond_left == list_bond_expected, (
        f'Bond profile {list_bond_left} is not the minimal taper '
        f'{list_bond_expected}'
    )

    # Every mode core carries -w * N_occ on its damping channel. B2_lower has
    # only a superdiagonal so B2_lower[1, 1] = 0, and N_occ[1, 1] = 1, which
    # leaves entry [1, 1] of that block equal to -w for the mode the core
    # belongs to. A core reading a relative state index instead of an
    # absolute one picks up the wrong w here.
    M1_mode_offset = np.concatenate([[0], np.cumsum(M1_modes_per_state)])
    n_hmodes = int(np.sum(M1_modes_per_state))
    for state in range(nsite):
        for i in range(M1_modes_per_state[state]):
            idx_mode = M1_mode_offset[state] + i
            # Channels still open at this state, its own closing at its last
            # mode core, which shifts the survivors down one index.
            idx_damp = n_l2 - state
            shift = 1 if i == M1_modes_per_state[state] - 1 else 0
            # The terminal core contracts every channel to a scalar output.
            idx_out = 0 if idx_mode == n_hmodes - 1 else idx_damp + 1 - shift
            T4_mode = list_cores_sub[1 + idx_mode]
            np.testing.assert_allclose(
                -T4_mode[idx_damp, 1, 1, idx_out],
                tb_full.mode.list_w[idx_mode],
                atol=1e-10,
                err_msg=(
                    f'state {state} mode {i}: damping channel does not carry '
                    f'mode.list_w[{idx_mode}]. build_fullstate_mpo may be '
                    'using relative instead of absolute state indices.'
                ),
            )


# ============================================================
# TEST SUITE: refresh_state_data()
# ============================================================


# ------------------------------------------------------------
# TEST: refresh_state_data updates state-dependent attributes
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_refresh_state_data():
    # This case tests that refresh_state_data updates n_state,
    # list_state_list, and I2_state while leaving constant attributes unchanged.
    tb = _make_tb()
    ht, _ = _make_tensor_pair('fullstate')
    elem = MpoBuilder(
        k_max,
        tb.system.size,
        ht.M1_modes_per_state,
        tb.mode.n_l2,
        tb.system.param['HAMILTONIAN'],
        tb.system.state_list,
        tb.mode,
        'homps',
        n_states_full=tb.system.param['NSTATES'],
        flag_nearest_neighbor_ham=tb.system.flag_nearest_neighbor_ham,
    )
    B2_raise_before = elem.B2_raise.copy()
    C1_coupling_raise_before = elem.C1_coupling_raise.copy()

    new_state_list = [0, 2]
    elem.refresh_state_data(len(new_state_list), new_state_list)

    assert elem.n_state == 2
    assert elem.list_state_list == [0, 2]
    np.testing.assert_array_equal(elem.I2_state, np.eye(2, dtype=np.complex128))
    np.testing.assert_array_equal(elem.B2_raise, B2_raise_before)
    np.testing.assert_array_equal(elem.C1_coupling_raise, C1_coupling_raise_before)


# ============================================================
# TEST SUITE: Hierarchy MPO — interior mode, guards, and warnings
# ============================================================


# ------------------------------------------------------------
# TEST: Interior mode core has identity pass-through for L-op channel
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_statenumber_hierarchy_mpo_interior_mode_identity():
    # Analytical: for a state with n_modes >= 2, the non-final mode
    # cores must carry the L-op / noise channel (bond index 1) through
    # to the next mode core intact. This requires an identity block at
    # core[1:2, :, :, 1:2]. Without it the channel is silently
    # zeroed and no bath coupling reaches the final mode.
    g1, g2 = 1.5 + 0j, 0.8 + 0j
    w1, w2 = 5.0, 15.0
    mode = _mock_mode(list_g=[g1, g2], list_w=[w1, w2])
    mode.n_hmodes = 2

    # 1 active state, 2 bath modes → modes_per_state = [2]
    elem = MpoBuilder(
        k_max,
        1,
        np.array([2]),
        1,
        np.array([[0.0]], dtype=np.complex128),
        np.array([0]),
        mode,
        'adhops',
        n_states_full=1,
        flag_nearest_neighbor_ham=True,
    )

    z_conj_t = np.array([0.2 + 0.1j], dtype=np.complex128)
    L_conj_avg = np.array([0.3 - 0.05j], dtype=np.complex128)
    cores = elem.build_statenumber_hierarchy_mpo(z_conj_t, L_conj_avg, 0.5)

    # cores layout: [state_core, mode_core_0, mode_core_1]
    # mode_core_0 is the non-final mode; mode_core_1 is the final mode.
    assert len(cores) == 3, f'Expected 3 cores, got {len(cores)}'

    interior_mode_core = cores[1]  # non-final mode core
    # The L-op channel (bond index 1) must pass through via identity.
    identity_block = interior_mode_core[1:1+1, :, :, 1:1+1]
    np.testing.assert_allclose(
        identity_block.reshape(k_max + 1, k_max + 1),
        np.eye(k_max + 1, dtype=np.complex128),
        atol=1e-14,
        err_msg='Interior mode core missing identity pass-through at bond (1,1)',
    )


# ------------------------------------------------------------
# TEST: Multiple L-operators per state raises NotImplementedError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_statenumber_hierarchy_mpo_multi_lop_raises():
    # Error guard: the statenumber hierarchy MPO requires at most one
    # L-operator per state (bond dimension 5). Two L-operators whose masks
    # put them on the same site must raise NotImplementedError before any
    # core is built.
    mode = _mock_mode(list_g=[1.0 + 0j, 2.0 + 0j], list_w=[10.0, 20.0])
    mode.n_hmodes = 2
    # Both L-operators act on site 0
    mode.list_L2_masks = [[[0], [0], None], [[0], [0], None]]

    elem = MpoBuilder(
        k_max,
        1,
        np.array([2]),
        2,  # n_lop_full = 2
        np.array([[0.0]], dtype=np.complex128),
        np.array([0]),
        mode,
        'homps',
        n_states_full=1,
        flag_nearest_neighbor_ham=True,
    )

    z_conj_t = np.zeros(2, dtype=np.complex128)
    L_conj_avg = np.zeros(2, dtype=np.complex128)
    with pytest.raises(NotImplementedError):
        elem.build_statenumber_hierarchy_mpo(z_conj_t, L_conj_avg, 0.0)


# ------------------------------------------------------------
# TEST: Zero coupling constant produces no NaN in ladder operators
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_mpo_builder_g_zero_no_nan():
    # Limiting case: g=0 means no bath coupling for that mode.
    # The homps prefactors C1_coupling_raise = g/sqrt(|g|) and C1_coupling_lower = sqrt(|g|)
    # both vanish identically — no division-by-zero or NaN should appear.
    mode = _mock_mode(list_g=[0.0 + 0j], list_w=[10.0])
    mode.n_hmodes = 1

    elem = MpoBuilder(
        k_max,
        1,
        np.array([1]),
        1,
        np.array([[0.0]], dtype=np.complex128),
        np.array([0]),
        mode,
        'homps',
        n_states_full=1,
        flag_nearest_neighbor_ham=True,
    )

    assert not np.any(np.isnan(elem.C1_coupling_raise)), 'C1_coupling_raise contains NaN for g=0'
    assert not np.any(np.isnan(elem.C1_coupling_lower)), 'C1_coupling_lower contains NaN for g=0'
    # Both prefactors must be exactly zero when g=0
    np.testing.assert_allclose(elem.C1_coupling_raise, [0.0], atol=1e-15)
    np.testing.assert_allclose(elem.C1_coupling_lower, [0.0], atol=1e-15)


# ============================================================
# TEST SUITE: dipole MPO builders (operator action on the
# ground + single-excitation manifold)
# ============================================================
#
# These tests verify what each dipole MPO *does* to a wavefunction rather than
# matching its constructed cores to a hand-built operator. A vector-form input
# phi[state, aux] (rows = ground + single-excitation states, like the vector
# HOPS layout; columns = auxiliaries kept populated so the MPS is non-trivial)
# is converted to a statenumber MPS, the MPO is applied, the result is read
# back to vector form, and compared to the dipole operator's manifold action
# O @ phi (applied per auxiliary column, since the dipole acts as identity on
# the modes). Single->double-excitation outputs (from `raise`) leave the
# manifold and are intentionally not asserted.


def _dipole_layout(n_l2, list_modes_per_site, k_max):
    '''Build the index bookkeeping that maps a manifold wavefunction to
    MPS tensor indices and back.

    The dipole tests work with a small physics-level wavefunction indexed
    by (manifold state, auxiliary), but the MPS stores amplitudes on raw
    tensor indices, one per core. This helper builds the maps needed to
    translate between the two: the MPS axis sizes, the per-site occupation
    for each manifold state, the per-mode occupation for each auxiliary,
    and a function that stitches a site pattern and a mode pattern into a
    full MPS index.

    Parameters
    ----------
    1. n_l2: int
             Number of physical sites.
    2. list_modes_per_site: np.ndarray(int)
                            Mode cores per site.
    3. k_max: int
              Maximum hierarchy depth (mode physical dim = k_max + 1).

    Returns
    -------
    1. size: list(int)
             Size of each MPS core's physical leg, in core order
             (site_0, its modes, site_1, its modes, ...). A state core
             has size 2 (site unoccupied or singly occupied); a mode
             core has size k_max + 1 (occupation 0 to k_max).
    2. state_to_sites: dict(int -> tuple(int))
                       Manifold state to per-site occupation. State 0 is
                       the ground state (no site occupied); state j+1 puts
                       the single excitation on site j. For n_l2 = 2:
                       {0: (0, 0), 1: (1, 0), 2: (0, 1)}.
    3. aux_patterns: list(tuple(int))
                     Auxiliary index to per-mode occupation, over the mode
                     axes only. Entry 0 is all modes unexcited; entry m+1
                     is a single first-order excitation on mode m. This is
                     the same one-hot construction as state_to_sites applied
                     to the mode axes instead of the site axes; the two are
                     structurally parallel, not otherwise related.
    4. build_index: callable(sites, mode_pattern) -> tuple(int)
                    Combines a per-site occupation (a state_to_sites value)
                    and a per-mode occupation (an aux_patterns entry) into a
                    full MPS index tuple. See its own docstring.
    '''
    m_dim = k_max + 1
    size = []
    for k in range(n_l2):
        size.append(2)
        size += [m_dim] * int(list_modes_per_site[k])
    state_to_sites = {0: (0,) * n_l2}
    for j in range(n_l2):
        state_to_sites[j + 1] = tuple(1 if i == j else 0 for i in range(n_l2))
    n_modes = int(np.sum(list_modes_per_site))
    aux_patterns = [(0,) * n_modes]
    for m in range(n_modes):
        aux_patterns.append(tuple(1 if i == m else 0 for i in range(n_modes)))

    def build_index(sites, mode_pattern):
        '''Interleave a site pattern and a mode pattern into one MPS index.

        Parameters
        ----------
        1. sites: tuple(int)
                  Per-site occupation, one entry per site (a value from
                  state_to_sites).
        2. mode_pattern: tuple(int)
                         Per-mode occupation, one entry per mode across all
                         sites (an entry from aux_patterns).

        Returns
        -------
        1. idx: tuple(int)
                Full MPS index in core order (site_0, its modes, site_1,
                its modes, ...), for indexing a tensor of shape size.
        '''
        idx = []
        i_mode = 0
        for k in range(n_l2):
            idx.append(sites[k])
            for _ in range(int(list_modes_per_site[k])):
                idx.append(mode_pattern[i_mode])
                i_mode += 1
        return tuple(idx)

    return size, state_to_sites, aux_patterns, build_index


def _tensor_train_construction(input_tensor, size, epsilon):
    '''Convert a dense tensor into MPS cores exactly, by a left-to-right sweep
    of SVDs.

    This is an independent dense-tensor-to-MPS builder written for the tests;
    it shares no code with the production MPS construction, so it can serve as
    a trusted reference the production code is checked against.

    Parameters
    ----------
    1. input_tensor: np.ndarray
                     Dense tensor, one axis per MPS core.
    2. size: list(int)
             Physical dimension of each axis, in MPS order.
    3. epsilon: float
                Relative singular-value truncation threshold.

    Returns
    -------
    1. list_cores: list(np.ndarray)
                   Flat TT cores, shape (chi_left, size[i], chi_right).
    '''
    list_cores = []
    # C is the not-yet-factored remainder. Each pass peels one core off its
    # front and leaves a smaller C behind. rank is the bond dimension coming
    # in from the left; it starts at 1 for the open left boundary.
    C = input_tensor
    rank = 1
    # One pass per core except the last; whatever is left after the loop is
    # the final core.
    for i in range(len(size) - 1):
        # Reshape the remainder into a matrix. The rows bundle the incoming
        # left bond with this core's physical leg; the columns hold every
        # later axis.
        C = np.reshape(C, (int(rank * size[i]), int(C.size / (rank * size[i]))))
        # SVD separates this core and its left side, U, from everything still
        # to come, Vt. S holds the singular values across that cut.
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        # Truncate small singular values, keeping the discarded weight under
        # epsilon^2 relative to the total norm.
        normalized_S = S / np.linalg.norm(S)
        thr = determine_error_thresh(np.flip(normalized_S), epsilon * epsilon)
        S[normalized_S <= thr] = 0.0
        # The count of surviving singular values is the new bond dimension:
        # this core's right bond, which is also the next core's left bond.
        prev_rank, rank = rank, len(np.nonzero(S)[0])
        # Keep U's surviving columns as this core, shaped
        # left_bond x physical x right_bond.
        list_cores.append(
            U[:, :rank].astype(np.complex128).reshape(prev_rank, int(size[i]), rank)
        )
        # Fold the singular values and Vt back into the remainder so the next
        # pass factors the rest of the chain.
        C = np.diag(S[:rank]).astype(np.complex128) @ Vt[:rank, :]
    # Whatever remains is the last core; its right bond is 1, the open boundary.
    list_cores.append(C.reshape(C.shape[0], C.shape[1], 1))
    return list_cores


def _vector_to_mps_flat(V2_phi, layout, epsilon=1e-12):
    '''Convert a manifold wavefunction to flat MPS cores.

    Scatters each amplitude V2_phi[state, aux] into a dense tensor at the
    MPS index that (state, aux) maps to, then TT-SVDs that tensor into MPS
    cores. This is the inverse of _mps_flat_to_vector.

    Parameters
    ----------
    1. V2_phi: np.ndarray(complex)
               Manifold wavefunction, shape (n_state, n_aux): row = manifold
               state, column = auxiliary pattern.
    2. layout: tuple
               The (size, state_to_sites, aux_patterns, build_index) tuple
               from _dipole_layout.
    3. epsilon: float
                Relative singular-value truncation threshold for the TT-SVD.

    Returns
    -------
    1. flat_cores: list(np.ndarray)
                   Flat MPS cores representing V2_phi exactly (up to epsilon).
    '''
    size, state_to_sites, aux_patterns, build_index = layout
    # Lay the amplitudes into the full dense tensor first. Only the
    # ground-plus-single-excitation configurations carry amplitude; every
    # other entry stays zero.
    T = np.zeros(size, dtype=np.complex128)
    for state, sites in state_to_sites.items():
        for a, mode_pattern in enumerate(aux_patterns):
            # Each (manifold state, auxiliary) maps to one fixed MPS index.
            # build_index turns this configuration's site and mode patterns
            # into that full index, and the amplitude is dropped there.
            T[build_index(sites, mode_pattern)] = V2_phi[state, a]
    # Factor the filled dense tensor into MPS cores. This is the inverse of
    # the index-and-contract read in _mps_flat_to_vector.
    return _tensor_train_construction(T, size, epsilon)


def _mps_flat_to_vector(flat_cores, layout):
    '''Read flat MPS cores back to a manifold wavefunction.

    Contracts the MPS down to one amplitude for each (manifold state,
    auxiliary) configuration, filling in phi[state, aux]. This is the
    inverse of _vector_to_mps_flat.

    Parameters
    ----------
    1. flat_cores: list(np.ndarray)
                   Flat MPS cores, one per axis in size order.
    2. layout: tuple
               The (size, state_to_sites, aux_patterns, build_index) tuple
               from _dipole_layout.

    Returns
    -------
    1. V2_phi: np.ndarray(complex)
               Manifold wavefunction, shape (n_state, n_aux).
    '''
    size, state_to_sites, aux_patterns, build_index = layout
    V2_phi = np.zeros((len(state_to_sites), len(aux_patterns)), dtype=np.complex128)
    # One amplitude per (manifold state, auxiliary): each maps to a single
    # fixed MPS index, so reading it is a straight contraction of the cores
    # at that index rather than a full tensor sum.
    for state, sites in state_to_sites.items():
        for a, mode_pattern in enumerate(aux_patterns):
            # build_index gives the physical index to take on each core for
            # this (state, aux) configuration.
            # Each core has shape (left_bond, physical, right_bond). Fixing the
            # physical index with core[:, idx, :] leaves a (left_bond,
            # right_bond) matrix. Each core's right_bond is the next core's
            # left_bond, so multiplying these matrices in chain order contracts
            # the whole MPS. The open MPS boundaries have bond dimension 1, so
            # the running product starts as the 1x1 identity and ends as a 1x1
            # matrix whose single entry is the amplitude for this configuration.
            M2_chain = np.ones((1, 1), dtype=np.complex128)
            for core, idx in zip(flat_cores, build_index(sites, mode_pattern)):
                # Absorb this core's bond matrix into the running product.
                M2_chain = M2_chain @ core[:, idx, :]
            V2_phi[state, a] = M2_chain[0, 0]
    return V2_phi


def _dipole_manifold_operator(list_mu, kind, n_l2):
    '''Dipole operator on the (ground + single-excitation) manifold.

    Rows/cols ordered [ground, e_0, ..., e_{n_l2-1}]. Single->double
    excitation transitions (from `raise`) fall outside the manifold and are
    projected out (the corresponding columns are zero).
    '''
    O2 = np.zeros((n_l2 + 1, n_l2 + 1), dtype=np.complex128)
    if kind == 'raise':
        for k in range(n_l2):
            O2[k + 1, 0] = list_mu[k]
    elif kind == 'lower':
        for j in range(n_l2):
            O2[0, j + 1] = list_mu[j]
    elif kind == 'lower_plus_ident':
        for j in range(n_l2):
            O2[0, j + 1] = list_mu[j]
            O2[j + 1, j + 1] = 1.0
    elif kind == 'raise_plus_ground_ident':
        O2[0, 0] = 1.0
        for k in range(n_l2):
            O2[k + 1, 0] = list_mu[k]
    else:
        raise ValueError(f'unknown kind {kind!r}')
    return O2


def _build_dipole_mpo(list_mu, kind, n_l2, k_max, list_modes_per_site):
    '''Dispatch to the dipole MPO builder for the given kind.'''
    if kind == 'raise':
        return build_statenumber_dipole_mpo(
            list_mu, n_l2, k_max, list_modes_per_site, 'raise')
    if kind == 'lower':
        return build_statenumber_dipole_mpo(
            list_mu, n_l2, k_max, list_modes_per_site, 'lower')
    if kind == 'lower_plus_ident':
        return build_statenumber_dipole_lower_plus_ident_mpo(
            list_mu, n_l2, k_max, list_modes_per_site)
    if kind == 'raise_plus_ground_ident':
        return build_statenumber_dipole_raise_plus_ground_ident_mpo(
            list_mu, n_l2, k_max, list_modes_per_site)
    raise ValueError(f'unknown kind {kind!r}')


_DIPOLE_KINDS = ['raise', 'lower', 'lower_plus_ident', 'raise_plus_ground_ident']


# ------------------------------------------------------------
# TEST: dipole MPO action matches the manifold operator
# ------------------------------------------------------------
@pytest.mark.level(1)
@pytest.mark.parametrize('kind', _DIPOLE_KINDS)
@pytest.mark.parametrize('n_l2', [1, 2, 3])
@pytest.mark.parametrize('bond_dim_max', [20, 64])
def test_dipole_mpo_action_on_manifold(kind, n_l2, bond_dim_max):
    # This case feeds a fixed ground+single-excitation wavefunction (with
    # populated auxiliaries so the MPS is non-trivial), applies the dipole
    # MPO, reads the result back to vector form, and checks it equals the
    # manifold operator O applied per auxiliary column. n_l2 in {1, 2, 3}
    # exercises the single-core, first/last, and first/interior/last paths.
    # bond_dim_max=20 is the production-typical cap; the manifold MPS bonds
    # top out at 8 here so neither cap truncates, but the 20 variant guards
    # the operator action against future regressions at the production cap.
    # Layout holds the MPS axis sizes and the maps between manifold
    # (state, auxiliary) indices and full MPS tensor indices. n_phys is the
    # number of manifold states (ground + one per site); n_hier is the number
    # of auxiliary patterns (ground + one first-order excitation per mode).
    # k_max = 2 gives mode cores dimension 3, distinct from the dimension-2
    # state cores, so a state-vs-mode core mix-up in the builders would be
    # caught rather than masked by equal dimensions.
    k_max = 2
    list_modes_per_site = np.ones(n_l2, dtype=int)
    layout = _dipole_layout(n_l2, list_modes_per_site, k_max)
    _, state_to_sites, aux_patterns, _ = layout
    n_phys = len(state_to_sites)
    n_hier = len(aux_patterns)

    # Fixed complex amplitudes that follow no pattern so the check is not
    # accidentally trivial; the pools are sliced to the (n_phys, n_hier)
    # and n_l2 shapes this case needs.
    phi_pool = np.array([
        0.37 - 1.12j, -0.85 + 0.44j, 1.23 + 0.09j, -0.51 - 0.78j,
        0.66 + 1.41j, -1.30 + 0.22j, 0.18 - 0.63j, 0.94 + 0.55j,
        -0.29 + 1.07j, 0.72 - 0.38j, -1.15 - 0.91j, 0.41 + 0.83j,
        1.06 - 0.24j, -0.68 + 0.59j, 0.33 + 1.19j, -0.97 - 0.46j,
    ], dtype=np.complex128)
    V2_phi = phi_pool[:n_phys * n_hier].reshape(n_phys, n_hier)
    mu_pool = np.array(
        [0.80 - 0.30j, -0.45 + 1.10j, 0.60 + 0.70j], dtype=np.complex128,
    )
    list_mu = mu_pool[:n_l2].copy()
    # Zero one site's amplitude to confirm excluded sites drop from the sum
    # (skipped for n_l2 == 1, where that would make the operator trivial).
    if n_l2 >= 2:
        list_mu[0] = 0.0

    # Encode the manifold wavefunction as flat MPS cores.
    flat_cores = _vector_to_mps_flat(V2_phi, layout)
    # Build the dipole MPO under test.
    mpo_cores = _build_dipole_mpo(list_mu, kind, n_l2, k_max, list_modes_per_site)
    # Apply the MPO to the MPS and compress the result.
    result_flat, _ = tensor_matvec_prod(
        flat_cores, mpo_cores, 1e-12, bond_dim_max,
    )
    # Read the MPS result back to manifold vector form.
    V2_result = _mps_flat_to_vector(result_flat, layout)

    # Reference: apply the dense manifold operator directly (per auxiliary
    # column) and require the MPO path to match it.
    V2_expected = _dipole_manifold_operator(list_mu, kind, n_l2) @ V2_phi
    np.testing.assert_allclose(
        V2_result, V2_expected, atol=1e-10,
        err_msg=f'{kind} MPO action mismatch for n_l2={n_l2}',
    )


# ------------------------------------------------------------
# TEST: dipole builders reject a mismatched list_mu length
# ------------------------------------------------------------
@pytest.mark.level(1)
@pytest.mark.parametrize('kind', _DIPOLE_KINDS)
def test_dipole_mpo_list_mu_length_mismatch(kind):
    # This case tests that each builder validates len(list_mu) == n_state.
    n_l2 = 2
    bad_mu = np.ones(n_l2 + 1, dtype=np.complex128)  # one entry too many
    with pytest.raises(
        ValueError, match=r'list_mu must have length n_state = 2, got 3'
    ):
        _build_dipole_mpo(bad_mu, kind, n_l2, 1, np.ones(n_l2, dtype=int))


# ------------------------------------------------------------
# TEST: invalid raise_or_lower selector raises
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_dipole_mpo_invalid_raise_or_lower():
    # This case tests build_statenumber_dipole_mpo rejects an unknown
    # raise_or_lower selector.
    with pytest.raises(
        ValueError,
        match=r"raise_or_lower must be 'raise' or 'lower', got 'sideways'",
    ):
        build_statenumber_dipole_mpo(
            np.ones(2, dtype=np.complex128), 2, 1, np.ones(2, dtype=int),
            'sideways',
        )



# ============================================================
# TEST SUITE: combined per-topology generator MPOs
# ============================================================
# Each builder must encode the same operator that the two-MPO path
# (hierarchy MPO + Hamiltonian MPO, added and compressed) encodes, on the
# one-excitation manifold, and must do so at the bond dimension claimed for
# its topology.


def _make_generator_builder(H2_site, n_mode_per_site, k_max, flag_nn):
    """Helper: MpoBuilder for an n_site system with one L-operator per site."""
    n_site = H2_site.shape[0]
    n_mode = n_site * n_mode_per_site

    class MockMode:
        pass

    mode = MockMode()
    # Distinct g and w per mode so a mode-indexing error cannot pass.
    # L-operators are listed in site order, which the builders require.
    mode.list_g = np.array(
        [0.5 + 0.1 * m + 0.2j * (m + 1) for m in range(n_mode)],
    )
    mode.list_w = np.array([10.0 + m for m in range(n_mode)])
    mode.n_l2 = n_site
    mode.n_hmodes = n_mode
    mode.list_L2_coo = []
    mode.list_L2_masks = [[[i], [i], None] for i in range(n_site)]
    mode.list_index_L2_by_hmode = [
        m // n_mode_per_site for m in range(n_mode)
    ]
    return MpoBuilder(
        k_max,
        n_site,
        np.full(n_site, n_mode_per_site, dtype=int),
        n_site,
        H2_site,
        np.arange(n_site),
        mode,
        'homps',
        n_states_full=n_site,
        flag_nearest_neighbor_ham=flag_nn,
    )


def _mpo_to_dense(list_cores):
    """Helper: contract an MPO's cores into a dense matrix."""
    T_op = list_cores[0]
    for T4_core in list_cores[1:]:
        T_op = np.tensordot(T_op, T4_core, axes=([-1], [0]))
    n_core = len(list_cores)
    # Cores contract to (left bond, out_0, in_0, out_1, in_1, ..., right bond);
    # gather every output index ahead of every input index before reshaping.
    list_perm = (
        [0]
        + [1 + 2 * i for i in range(n_core)]
        + [2 + 2 * i for i in range(n_core)]
        + [T_op.ndim - 1]
    )
    T_op = T_op.transpose(list_perm)
    dim = int(np.prod([T4_core.shape[1] for T4_core in list_cores]))
    return T_op.reshape(dim, dim)


def _one_excitation_indices(n_site, n_mode_per_site, dim_mode):
    """Helper: composite indices holding exactly one excited site."""
    list_dim = []
    for _ in range(n_site):
        list_dim.append(2)
        list_dim.extend([dim_mode] * n_mode_per_site)
    list_idx = []
    for tup_idx in np.ndindex(*list_dim):
        # State cores sit at every (1 + n_mode_per_site)-th position.
        n_excited = sum(
            tup_idx[i * (1 + n_mode_per_site)] for i in range(n_site)
        )
        if n_excited != 1:
            continue
        idx_flat = 0
        for dim, idx in zip(list_dim, tup_idx):
            idx_flat = idx_flat * dim + idx
        list_idx.append(idx_flat)
    return np.array(list_idx, dtype=int)


def _topology_hamiltonian(topology, n_site):
    """Helper: a Hamiltonian of the named topology, with on-site energies."""
    H2_site = np.diag(
        np.array([100.0 + 7.0 * i for i in range(n_site)], dtype=np.complex128)
    )
    if topology == 'chain':
        for i in range(n_site - 1):
            H2_site[i, i + 1] = 40.0 - 3.0 * i
            H2_site[i + 1, i] = np.conj(H2_site[i, i + 1])
    elif topology == 'ring':
        for i in range(n_site - 1):
            H2_site[i, i + 1] = 40.0 - 3.0 * i
            H2_site[i + 1, i] = np.conj(H2_site[i, i + 1])
        H2_site[0, n_site - 1] = 17.0
        H2_site[n_site - 1, 0] = 17.0
    elif topology in ('star', 'star_mid'):
        # 'star_mid' puts the hub in the interior of the site ordering, so the
        # star builder's leaves-left and leaves-right branches are both used.
        site_hub = 0 if topology == 'star' else n_site // 2
        for i in range(n_site):
            if i == site_hub:
                continue
            H2_site[site_hub, i] = 25.0 + 2.0 * i
            H2_site[i, site_hub] = np.conj(H2_site[site_hub, i])
    elif topology == 'general_dense':
        # Every pair coupled with a distinct amplitude, so no bond's coupling
        # block is rank deficient. This is the worst case for the general
        # builder, whose width is then set by the shape of the bonds alone.
        for i in range(n_site):
            for j in range(i + 1, n_site):
                H2_site[i, j] = 40.0 - 3.0 * i + 5.0 * j
                H2_site[j, i] = np.conj(H2_site[i, j])
    elif topology == 'general_exp':
        # Couplings decaying with distance. Every pair is coupled, but each
        # bond's block factorizes as exp(-|i - bond|) * exp(-|j - bond|) and
        # so has rank one, which is the case the factorization exists to find.
        for i in range(n_site):
            for j in range(n_site):
                if i != j:
                    H2_site[i, j] = 60.0 * np.exp(-abs(i - j))
    else:
        raise ValueError(topology)
    return H2_site


# ------------------------------------------------------------
# TEST: combined generator matches the added-and-compressed MPO
# ------------------------------------------------------------


@pytest.mark.parametrize(
    'topology,list_bond_expected',
    [('chain', [5, 4, 5, 4, 5, 4, 3, 1]),
     ('ring', [5, 4, 7, 6, 5, 4, 3, 1]),
     ('star', [5, 4, 5, 4, 5, 4, 3, 1]),
     ('star_mid', [5, 4, 5, 4, 5, 4, 3, 1]),
     ('general_dense', [5, 4, 7, 6, 5, 4, 3, 1]),
     ('general_exp', [5, 4, 5, 4, 5, 4, 3, 1])],
)
def test_generator_mpo_matches_two_mpo_path(topology, list_bond_expected):
    # This case tests that the combined generator MPO for each coupling graph
    # encodes the same operator on the one-excitation manifold as the
    # hierarchy MPO added to the Hamiltonian MPO, at the per-bond width the
    # coupling-block ranks call for, while the two-MPO path needs more.
    # The width is checked bond by bond rather than at its peak: a too-wide
    # MPO encodes the same operator, so the operator comparison cannot see
    # excess channels anywhere but the maximum.
    n_site = 4
    n_mode_per_site = 1
    k_max = 2
    H2_site = _topology_hamiltonian(topology, n_site)
    builder = _make_generator_builder(
        H2_site, n_mode_per_site, k_max, flag_nn=(topology == 'chain'),
    )
    list_z_hat = np.array([0.3 + 0.4j, -0.2 + 0.1j, 0.5 - 0.6j, 0.1 + 0.2j])
    list_expect_L2 = np.array([0.7, 0.2 + 0.1j, -0.3, 0.45 - 0.2j])
    norm_corr = 0.37

    list_cores_generator = builder.build_general_generator_mpo(
        list_z_hat, list_expect_L2, norm_corr,
    )

    list_cores_hier = builder.build_statenumber_hierarchy_mpo(
        list_z_hat, list_expect_L2, norm_corr,
    )
    list_cores_ham = builder.build_statenumber_ham_mpo()
    # Adding two MPOs adds their bond dimensions, so this bounds the width
    # of the uncompressed sum. Passing it as bond_dim_max keeps tensor_add
    # from truncating, which is what makes the comparison below exact.
    bond_dim_uncompressed = (
        max(c.shape[-1] for c in list_cores_hier)
        + max(c.shape[-1] for c in list_cores_ham)
    )
    list_cores_two_mpo = tensor_add(
        list_cores_hier, list_cores_ham,
        epsilon=0.0, bond_dim_max=bond_dim_uncompressed,
    )

    assert len(list_cores_generator) == len(list_cores_two_mpo)
    V1_idx_exc = _one_excitation_indices(n_site, n_mode_per_site, k_max + 1)
    O2_generator = _mpo_to_dense(
        list_cores_generator
    )[np.ix_(V1_idx_exc, V1_idx_exc)]
    O2_two_mpo = _mpo_to_dense(
        list_cores_two_mpo
    )[np.ix_(V1_idx_exc, V1_idx_exc)]
    # Couplings and energies are O(10^2), so agreement is at roundoff:
    # the largest deviation observed across the three topologies is 1.1e-12.
    assert np.allclose(O2_generator, O2_two_mpo, rtol=0, atol=2e-12), (
        f'{topology}: max deviation '
        f'{np.abs(O2_generator - O2_two_mpo).max():.3e}'
    )

    list_bond_generator = [c.shape[-1] for c in list_cores_generator]
    assert list_bond_generator == list_bond_expected, (
        f'{topology}: bond profile {list_bond_generator} is not the minimal '
        f'{list_bond_expected}'
    )
    assert (max(c.shape[-1] for c in list_cores_two_mpo)
            > max(list_bond_expected))
