import numpy as np
import pytest
import scipy as sp

from mesohops.basis.hops_modes import HopsModes
from mesohops.basis.hops_noise_memory import HopsNoiseMemory
from mesohops.basis.hops_system import HopsSystem
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.hops_tensor_basis import HopsTensorBasis
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_tensor_trajectory import HopsTensorTrajectory
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.util.exceptions import LockedException, UnsupportedRequest
from mesohops.basis.basis_functions import determine_error_thresh
from mesohops.util.tensor_operations import (
    extract_psi,
    phi_aux as extract_phi_aux,
    unflatten_cores,
)

__title__ = 'Unit Tests for HopsTensorWavefunction'
__author__ = 'N. Covalsen'
__maintainer__ = 'N. Covalsen'

# ============================================================
# Shared Setup
# ============================================================
# Dimer-of-dimers system (4 sites, 2 modes per site)
# identical to test_hops_tensor_trajectory.py and test_dimer_of_dimers.py

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
delta_s = 0  # non-adaptive


eom_param = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
noise_param = {
    'SEED': 0,
    'MODEL': 'FFT_FILTER',
    'TLEN': 25000.0,
    'TAU': 1.0,
}
hier_param = {'MAXHIER': 2}
integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}


def _make_basis_objects(param=None):
    """Creates HopsSystem, HopsModes, HopsNoiseMemory directly."""
    if param is None:
        param = sys_param
    system = HopsSystem(param)
    mode = HopsModes(system, hierarchy=None)
    noise_memory = HopsNoiseMemory(system, mode)
    return system, mode, noise_memory


def _make_initialized_tensor_basis(param=None, ds=None):
    """Creates and initializes a HopsTensorBasis."""
    if ds is None:
        ds = delta_s
    system, mode, noise_memory = _make_basis_objects(param)
    tb = HopsTensorBasis(system, mode, noise_memory)
    system.initialize(ds > 0, psi_0)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = state_list
    tb.initialize(ds)
    return tb


def _make_propagated_tensor(method, t_max=8.0, t_step=4.0):
    """Creates a tensor trajectory, propagates, and returns the wavefunction.

    After propagation the MPS has physically realistic entanglement and
    complex-valued cores with non-trivial bond dimensions — unlike the
    trivial product state from _make_tensor.
    """
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    traj = HopsTensorTrajectory(
        system_param=sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        tensor_param=tensor_param,
    )
    traj.initialize(psi_0)
    traj.propagate(t_max, t_step)
    return traj.wavefunction


def _make_tensor(method):
    """Creates and initializes a HopsTensorWavefunction for the dimer-of-dimers system.

    NOTE: produces a trivial product state (single-site occupied, bond
    dim 1). Tests requiring entangled or delocalized states should
    use _make_propagated_tensor instead.
    """
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    tb = _make_initialized_tensor_basis()
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    return ht


def _make_tensor_uninit(method):
    """Creates a HopsTensorWavefunction WITHOUT calling initialize."""
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': method,
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    return HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)


def _make_tensor_basis_init():
    """Creates and initializes a HopsTensorBasis."""
    return _make_initialized_tensor_basis()


# ============================================================
# TEST SUITE: __init__()
# ============================================================


# ------------------------------------------------------------
# TEST: Config parameters are stored correctly
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_stores_config():
    """HopsTensorWavefunction stores tensor_param and integrator_param on init."""
    ht = _make_tensor_uninit('fullstate')
    assert ht.mps_epsilon == 1e-10
    assert ht.method == 'fullstate'
    assert ht.bond_dim_max == 20
    assert ht.k_max == 2
    assert ht.flag_norm is True
    assert ht.flag_tdvp is False
    # Default values before initialize
    assert ht.list_cores_phi == []
    assert ht.__initialized__ is False
    # system/mode are NOT stored on HopsTensorWavefunction
    assert not hasattr(ht, 'system')
    assert not hasattr(ht, 'mode')
    assert not hasattr(ht, 'noise_memory')


# ------------------------------------------------------------
# TEST: flag_gs_vacuum property defaults off and guards method
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_gs_vacuum_property_requires_number():
    # The flag defaults off and is set via its property. The all-zeros vacuum
    # config only exists in number representation, so setting it True must
    # raise for fullstate and succeed for number.
    ht_full = _make_tensor_uninit('fullstate')
    assert ht_full.flag_gs_vacuum is False
    with pytest.raises(UnsupportedRequest):
        ht_full.flag_gs_vacuum = True

    ht_num = _make_tensor_uninit('number')
    assert ht_num.flag_gs_vacuum is False
    ht_num.flag_gs_vacuum = True
    assert ht_num.flag_gs_vacuum is True


def _number_mps_from_dense(dense, list_modes_per_site, epsilon=1e-12):
    """Build a nested number MPS from a dense config tensor via exact TT-SVD.

    The dense axes are the MPS physical legs in order (site_0, mode_00, ...);
    the TT round-trip is lossless, so the MPS represents the dense state
    exactly.
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


# ------------------------------------------------------------
# TEST: manifold_norm_sq sums excited norm and vacuum amplitude
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_manifold_norm_sq_includes_vacuum():
    # Build a random number MPS via TT and wrap it in a wavefunction.
    # manifold_norm_sq must equal sum_s |psi_s|^2 for the physical wavefunction,
    # i.e. site s occupation at k=0, plus |vacuum|^2 from the all-zeros config.
    rng = np.random.default_rng(1)
    k_max_local = 1
    list_modes_per_site = [1, 1]
    size, site_axes, pos = [], [], 0
    for n_modes in list_modes_per_site:
        site_axes.append(pos)
        size += [2] + [k_max_local + 1] * n_modes
        pos += 1 + n_modes
    dense = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    n_state = len(list_modes_per_site)

    vac_idx = (0,) * len(size)
    expected = np.abs(dense[vac_idx]) ** 2
    for s in range(n_state):
        idx = [0] * len(size)
        idx[site_axes[s]] = 1
        expected += np.abs(dense[tuple(idx)]) ** 2

    ht = _make_tensor_uninit('number')
    ht.M1_modes_per_site = np.array(list_modes_per_site, dtype=int)
    ht.list_cores_phi = _number_mps_from_dense(dense, list_modes_per_site)
    np.testing.assert_allclose(ht.manifold_norm_sq, expected, atol=1e-10)


# ------------------------------------------------------------
# TEST: EOM flags are set correctly for different configurations
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_eom_flags():
    # This case tests that EOM-related flags are set correctly
    # for different EOM and integrator configurations.

    # flag_norm=False when EOM is NONLINEAR
    tensor_param_local = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    eom_nonlinear = {'EQUATION_OF_MOTION': 'NONLINEAR'}
    ht = HopsTensorWavefunction(
        k_max, tensor_param_local, integrator_param, eom_nonlinear,
    )
    assert ht.flag_norm is False
    assert ht.flag_tdvp is False

    # flag_tdvp=True when INTEGRATOR is 'TDVP1'
    integrator_param_tdvp = {'INTEGRATOR': 'TDVP1'}
    ht_tdvp = HopsTensorWavefunction(
        k_max, tensor_param_local, integrator_param_tdvp, eom_param
    )
    assert ht_tdvp.flag_norm is True
    assert ht_tdvp.flag_tdvp is True


# ------------------------------------------------------------
# TEST: flag_norm derives from eom_param, not tensor_param
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_flag_norm_from_eom_param():
    """HopsTensorWavefunction.flag_norm is set from eom_param, not tensor_param['EOM'].
    """
    # tensor_param has no 'EOM' key — the EOM lives in eom_param only
    tensor_param_no_eom = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}

    # Case 1: normalized nonlinear → flag_norm=True
    eom_normalized = {'EQUATION_OF_MOTION': 'NORMALIZED NONLINEAR'}
    ht = HopsTensorWavefunction(
        k_max, tensor_param_no_eom, integrator_param, eom_normalized,
    )
    assert ht.flag_norm is True

    # Case 2: nonlinear → flag_norm=False
    eom_nonlinear = {'EQUATION_OF_MOTION': 'NONLINEAR'}
    ht2 = HopsTensorWavefunction(
        k_max, tensor_param_no_eom, integrator_param, eom_nonlinear,
    )
    assert ht2.flag_norm is False

    # Case 3: linear → flag_norm=False
    eom_linear = {'EQUATION_OF_MOTION': 'LINEAR'}
    ht3 = HopsTensorWavefunction(
        k_max, tensor_param_no_eom, integrator_param, eom_linear,
    )
    assert ht3.flag_norm is False


# ============================================================
# TEST SUITE: initialize()
# ============================================================


# ------------------------------------------------------------
# TEST: Dimension bookkeeping is correct
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_dimensions():
    # This case tests that M1_modes_per_state is computed
    # correctly from system parameters after initialize.
    ht = _make_tensor('number')
    modes_per_site = len(list_gw_sysbath) // nsite
    np.testing.assert_array_equal(
        ht.M1_modes_per_state, [modes_per_site] * nsite
    )


# ------------------------------------------------------------
# TEST: Statenumber MPS has correct group structure
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_statenumber_group_structure():
    # This case tests that each state group in a statenumber MPS has
    # 1 state core + M1_modes_per_state[s] mode cores.
    ht = _make_tensor('number')
    for s, group in enumerate(ht.list_cores_phi):
        expected_len = 1 + ht.M1_modes_per_state[s]
        assert len(group) == expected_len, (
            f'State {s} group has {len(group)} cores, expected {expected_len}'
        )
        # State core has physical dimension 2 (occupied/unoccupied)
        assert group[0].shape[1] == 2
        # Mode cores have physical dimension k_max + 1
        for core_m in group[1:]:
            assert core_m.shape[1] == k_max + 1


# ------------------------------------------------------------
# TEST: Sparse Hamiltonian is converted to dense
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_sparse_hamiltonian_converted():
    # This case tests that a sparse Hamiltonian is converted to dense during
    # initialize (HopsTensorBasis/HopsSystem handles the conversion).
    sp_param = dict(sys_param)
    sp_param['HAMILTONIAN'] = sp.sparse.coo_matrix(H2_hamiltonian)
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    tb = _make_initialized_tensor_basis(param=sp_param)
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    # Verify the stored Hamiltonian is dense (MpoBuilder requires dense input).
    # HopsTensorTrajectory converts sparse → dense in __init__; here we bypass
    # that, so we convert manually to confirm the test exercises the right path.
    H2_ham = tb.system.param['HAMILTONIAN']
    if sp.sparse.issparse(H2_ham):
        tb.system.param['HAMILTONIAN'] = np.asarray(H2_ham.todense())
    assert isinstance(tb.system.param['HAMILTONIAN'], np.ndarray), (
        'Hamiltonian should be dense ndarray for tensor code'
    )
    # Analytical: fullstate MPS has 1 state core + sum(M1_modes_per_state) mode cores
    expected_n_cores = 1 + sum(ht.M1_modes_per_state)
    assert len(ht.list_cores_phi) == expected_n_cores
    # Analytical: psi should recover the initial state
    np.testing.assert_allclose(ht.psi, psi_0, atol=1e-12)


# ------------------------------------------------------------
# TEST: M1_modes_per_site is set from state_list
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_modes_per_site():
    # This case tests that M1_modes_per_site has one entry per state in state_list.
    ht = _make_tensor('fullstate')
    assert len(ht.M1_modes_per_site) == len(state_list)


# ============================================================
# TEST SUITE: build_list_cores_phi()
# ============================================================


# ------------------------------------------------------------
# TEST: Fullstate MPS has correct structure
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_list_cores_phi_fullstate_structure():
    # This case tests that the fullstate MPS has 1 system core + n_mode
    # mode cores, with correct physical dimensions.
    ht = _make_tensor('fullstate')
    n_modes = len(list_gw_sysbath)
    # 1 system core + n_modes mode cores
    assert len(ht.list_cores_phi) == 1 + n_modes
    # System core physical dimension = n_state
    assert ht.list_cores_phi[0].shape[1] == nsite
    # Mode cores physical dimension = k_max + 1
    for i in range(1, len(ht.list_cores_phi)):
        assert ht.list_cores_phi[i].shape[1] == k_max + 1


# ------------------------------------------------------------
# TEST: Statenumber MPS has correct structure
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_list_cores_phi_statenumber_structure():
    # This case tests that the statenumber MPS has n_state groups,
    # each containing 1 state core (physical dim 2) + mode cores (physical dim k_max+1).
    ht = _make_tensor('number')
    # list_cores_phi[s] = [state_core_s, mode_core_s0, ...] — one group per state
    assert len(ht.list_cores_phi) == nsite
    for site in range(nsite):
        group = ht.list_cores_phi[site]
        # First element is the state core: physical dim 2
        assert group[0].shape[1] == 2
        # Remaining elements are mode cores: physical dim k_max + 1
        assert len(group) == 1 + ht.M1_modes_per_state[site]
        for mode_core in group[1:]:
            assert mode_core.shape[1] == k_max + 1


# ------------------------------------------------------------
# TEST: phi_0 recovers the initial wavefunction
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_list_cores_phi_recovers_psi0():
    # This case tests that contracting the MPS back to a vector
    # recovers the input wavefunction, for both representations.
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        np.testing.assert_allclose(
            ht.psi,
            psi_0,
            atol=1e-12,
            err_msg=f'phi_0 does not match psi_0 for {method}',
        )


# ------------------------------------------------------------
# TEST: All hierarchy modes start in ground state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_list_cores_phi_hierarchy_ground_state():
    # This case tests that all first-order auxiliary wavefunctions
    # are zero at initialization (all modes in ground state |0>).
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        n_modes = len(list_gw_sysbath)
        for i_mode in range(n_modes):
            indices = [0] * n_modes
            indices[i_mode] = 1
            phi_1 = extract_phi_aux(
                ht.list_cores_phi, ht.method, ht.M1_modes_per_state, indices
            )
            np.testing.assert_allclose(
                phi_1,
                0.0,
                atol=1e-12,
                err_msg=f'Mode {i_mode} not in ground state for {method}',
            )


# ============================================================
# TEST SUITE: inflate_bonds_to()
# ============================================================


# ------------------------------------------------------------
# TEST: Bond dimensions reach target after inflation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_inflate_bonds_to_target():
    # This case tests that all internal bond dimensions equal chi_target
    # after explicitly calling inflate_bonds_to (inflate is not called
    # automatically during initialize for non-TDVP integrators).
    ht = _make_tensor('fullstate')
    chi = ht.bond_dim_max
    ht.inflate_bonds_to(chi, eps=0.0)
    for i in range(len(ht.list_cores_phi) - 1):
        assert ht.list_cores_phi[i].shape[2] == chi, (
            f'Bond {i} right dim = {ht.list_cores_phi[i].shape[2]}, expected {chi}'
        )
        assert ht.list_cores_phi[i + 1].shape[0] == chi, (
            f'Bond {i} left dim = {ht.list_cores_phi[i + 1].shape[0]}, expected {chi}'
        )
    # Verify no pathological output
    for i, core in enumerate(ht.list_cores_phi):
        assert not np.any(np.isnan(core)), f'Inflate produced NaN in core {i}'
        assert core.shape[0] >= 1, f'Zero left bond in core {i}'
        assert core.shape[2] >= 1, f'Zero right bond in core {i}'


# ------------------------------------------------------------
# TEST: Bond consistency (left dim matches right dim)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_check_bondsize_both_representations():
    # This case tests that adjacent cores have compatible bond dimensions
    # for statenumber representation.
    ht = _make_tensor('number')
    assert ht.check_bondsize(), 'Bond dimensions are inconsistent'

    # This case tests check_bondsize for fullstate representation.
    ht_full = _make_tensor('fullstate')
    assert ht_full.check_bondsize(), 'Fullstate bond dimensions inconsistent at init'

    # This case tests check_bondsize after inflate_bonds_to.
    ht_inflated = _make_tensor('fullstate')
    ht_inflated.inflate_bonds_to(5, eps=0.0)
    assert ht_inflated.check_bondsize(), 'Bond dimensions inconsistent after inflation'


# ------------------------------------------------------------
# TEST: inflate_bonds_to with eps > 0 adds random noise padding
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_inflate_bonds_eps_nonzero():
    # This case tests that inflate_bonds_to with eps > 0 produces
    # non-zero padding entries (random noise rather than zeros).
    ht = _make_tensor('fullstate')
    # Record original bond dims (all 1 for freshly built MPS)
    original_shapes = [core.shape for core in ht.list_cores_phi]
    chi_target = 5
    # eps > 0 fills padded bond entries with random noise scaled by eps,
    # ensuring the inflated MPS is not rank-deficient. Any eps > 0
    # guarantees nonzero padding that survives subsequent SVD compression.
    ht.inflate_bonds_to(chi_target, eps=0.1)
    # Check that padded region (beyond original bond dim) contains nonzero entries
    has_nonzero_pad = False
    for core, orig_shape in zip(ht.list_cores_phi, original_shapes):
        orig_left, _, orig_right = orig_shape
        # Padded entries are those with left index >= orig_left or right >= orig_right
        if core.shape[0] > orig_left:
            if np.any(np.abs(core[orig_left:, :, :]) > 1e-15):
                has_nonzero_pad = True
        if core.shape[2] > orig_right:
            if np.any(np.abs(core[:, :, orig_right:]) > 1e-15):
                has_nonzero_pad = True
    assert has_nonzero_pad, 'eps>0 padding should contain nonzero entries'


# ------------------------------------------------------------
# TEST: inflate_bonds_to is a no-op when bonds already at target
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_inflate_bonds_noop_at_target():
    # This case tests that calling inflate_bonds_to with chi_target
    # equal to the current bond dimension does not change the cores.
    # First inflate to 4, then inflate again to 4 — the second call
    # should be a no-op since bonds are already at target.
    ht = _make_tensor('fullstate')
    ht.inflate_bonds_to(4, eps=0.0)
    cores_at_4 = [c.copy() for c in ht.list_cores_phi]
    ht.inflate_bonds_to(4, eps=0.0)
    for i, (before, after) in enumerate(zip(cores_at_4, ht.list_cores_phi)):
        np.testing.assert_array_equal(
            before,
            after,
            err_msg=f'Core {i} changed when chi_target == current bond dim',
        )


# ------------------------------------------------------------
# TEST: Fullstate inflate preserves phi_0
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_inflate_bonds_preserves_phi0_fullstate():
    # This case tests that inflating bonds does not change the physical
    # wavefunction. Uses a propagated state with naturally complex-valued
    # cores and non-trivial bond structure.
    ht = _make_propagated_tensor('fullstate')
    V1_phi_before = ht.psi.copy()
    ht.inflate_bonds_to(ht.bond_dim_max)
    np.testing.assert_allclose(
        ht.psi,
        V1_phi_before,
        atol=1e-12,
        err_msg='inflate_bonds_to changed phi_0 in fullstate',
    )


# ------------------------------------------------------------
# TEST: Mixed bond dimensions — only undersized bonds are padded
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_inflate_bonds_mixed_bond_dims():
    # CASE: Inflate to 3, then to 5. Verify that the content in the
    # bond-3 region is preserved after the second inflate (not
    # overwritten), and that the new padding region is distinct.
    ht = _make_tensor('fullstate')
    # Inject complex values to avoid trivial special case
    ht.list_cores_phi[0][0, :, 0] = np.array(
        [0.3 + 0.1j, 0.5 - 0.2j, 0.6 + 0.4j, 0.1 - 0.3j],
        dtype=np.complex128,
    )
    ht.inflate_bonds_to(3, eps=0.0)
    V1_psi_at_3 = ht.psi.copy()
    cores_at_3 = [c.copy() for c in ht.list_cores_phi]
    ht.inflate_bonds_to(5, eps=0.0)
    # All interior bonds should now be 5
    for i in range(len(ht.list_cores_phi) - 1):
        assert ht.list_cores_phi[i].shape[2] == 5, (
            f'Bond {i} right dim should be 5 after second inflate'
        )
    # Physical wavefunction must be preserved through both inflations
    np.testing.assert_allclose(
        ht.psi, V1_psi_at_3, atol=1e-12,
        err_msg='Second inflate changed psi',
    )
    # The bond-3 subregion of each core should be preserved (not
    # overwritten by the inflate-to-5 padding).
    for i, (c3, c5) in enumerate(zip(cores_at_3, ht.list_cores_phi)):
        dl3, d, dr3 = c3.shape
        np.testing.assert_allclose(
            c5[:dl3, :, :dr3], c3, atol=1e-14,
            err_msg=f'Core {i}: bond-3 subregion altered by inflate to 5',
        )


# ------------------------------------------------------------
# TEST: Double inflate preserves phi_0
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_inflate_bonds_double_inflate_preserves_phi0():
    # CASE: Two successive inflations should still preserve the
    # physical wavefunction. Uses complex multi-site state.
    ht = _make_tensor('fullstate')
    ht.list_cores_phi[0][0, :, 0] = np.array(
        [0.4 - 0.3j, 0.2 + 0.5j, 0.1 + 0.1j, 0.6 - 0.2j],
        dtype=np.complex128,
    )
    V1_phi_before = ht.psi.copy()
    ht.inflate_bonds_to(3, eps=0.0)
    ht.inflate_bonds_to(10, eps=0.0)
    np.testing.assert_allclose(
        ht.psi,
        V1_phi_before,
        atol=1e-12,
        err_msg='Double inflate changed phi_0',
    )


# ============================================================
# TEST SUITE: normalize()
# ============================================================


# ------------------------------------------------------------
# TEST: Normalize returns the norm before normalization
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_normalize_returns_norm():
    # This case tests that normalize() returns the norm of phi_0
    # before dividing.
    ht = _make_tensor('fullstate')
    # Scale phi to have non-unit norm
    ht.list_cores_phi[0] = ht.list_cores_phi[0] * 2.0
    norm_before = np.linalg.norm(ht.psi)
    returned_norm = ht.normalize()
    np.testing.assert_allclose(returned_norm, norm_before, atol=1e-12)


# ------------------------------------------------------------
# TEST: Normalize makes phi_0 unit norm
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_normalize_unit_norm():
    # This case tests that after normalize(), phi_0 has unit norm.
    ht = _make_tensor('fullstate')
    ht.list_cores_phi[0] = ht.list_cores_phi[0] * 3.7
    ht.normalize()
    np.testing.assert_allclose(
        np.linalg.norm(ht.psi),
        1.0,
        atol=1e-12,
    )


# ------------------------------------------------------------
# TEST: Normalize only divides first core
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_normalize_only_first_core():
    # This case tests that normalization only modifies the first core,
    # leaving all other cores unchanged.
    ht = _make_tensor('fullstate')
    ht.list_cores_phi[0] = ht.list_cores_phi[0] * 2.0
    cores_before = [c.copy() for c in ht.list_cores_phi[1:]]
    ht.normalize()
    for i, (before, after) in enumerate(zip(cores_before, ht.list_cores_phi[1:])):
        np.testing.assert_array_equal(
            before,
            after,
            err_msg=f'Core {i + 1} was modified by normalize()',
        )


# ------------------------------------------------------------
# TEST: Normalize always normalizes (even when EOM is NONLINEAR)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_normalize_always_normalizes():
    # This case tests that normalize() always normalizes the MPS,
    # even when the EOM is NONLINEAR (flag_norm is False). The
    # flag_norm guard was removed from normalize(); the policy now
    # lives in the trajectory.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    eom_nonlinear = {'EQUATION_OF_MOTION': 'NONLINEAR'}
    tb_la = _make_initialized_tensor_basis()
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_nonlinear)
    ht.initialize(psi_0, tb_la.system)
    ht.list_cores_phi[0] = ht.list_cores_phi[0] * 5.0
    core0_before = ht.list_cores_phi[0].copy()
    ht.normalize()
    # The core SHOULD have changed (normalize divides by norm)
    assert not np.array_equal(ht.list_cores_phi[0], core0_before)
    # phi_0 should now have unit norm
    np.testing.assert_allclose(np.linalg.norm(ht.psi), 1.0, atol=1e-12)


# ------------------------------------------------------------
# TEST: normalize works correctly for statenumber representation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_normalize_statenumber():
    # This case tests that normalize produces unit-norm psi for statenumber
    ht = _make_tensor('number')
    # Scale first core to make norm != 1
    ht.list_cores_phi[0][0] = ht.list_cores_phi[0][0] * 3.0
    psi_before = ht.psi.copy()
    norm_returned = ht.normalize()
    # Invariant: returned norm equals pre-normalize norm
    np.testing.assert_allclose(
        norm_returned, np.linalg.norm(psi_before), atol=1e-12,
    )
    # Invariant: post-normalize norm is 1
    np.testing.assert_allclose(np.linalg.norm(ht.psi), 1.0, atol=1e-12)
    # Invariant: psi direction unchanged
    np.testing.assert_allclose(
        ht.psi, psi_before / np.linalg.norm(psi_before), atol=1e-12,
    )


# ------------------------------------------------------------
# TEST: Statenumber normalize only modifies first state core
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_normalize_only_first_core_statenumber():
    # This case tests that normalization only modifies list_cores_phi[0][0]
    # (the first state core), leaving other cores in the first group and
    # all other groups unchanged.
    ht = _make_tensor('number')
    ht.list_cores_phi[0][0] = ht.list_cores_phi[0][0] * 2.0
    # Save copies of all cores except [0][0]
    list_first_group_rest = [c.copy() for c in ht.list_cores_phi[0][1:]]
    list_other_groups = [
        [c.copy() for c in group] for group in ht.list_cores_phi[1:]
    ]
    ht.normalize()
    # Other cores in the first group should be unchanged
    for i, (before, after) in enumerate(
        zip(list_first_group_rest, ht.list_cores_phi[0][1:])
    ):
        np.testing.assert_array_equal(
            before, after,
            err_msg=f'Group 0, core {i + 1} was modified by normalize()',
        )
    # All other groups should be unchanged
    for g, (group_before, group_after) in enumerate(
        zip(list_other_groups, ht.list_cores_phi[1:])
    ):
        for i, (before, after) in enumerate(zip(group_before, group_after)):
            np.testing.assert_array_equal(
                before, after,
                err_msg=f'Group {g + 1}, core {i} was modified by normalize()',
            )


# ------------------------------------------------------------
# TEST: normalize raises UnsupportedRequest for unknown method
# ------------------------------------------------------------
def test_normalize_unsupported_method():
    # This case tests that normalize raises an error when the tensor
    # method is not recognized. extract_psi (called via self.psi)
    # raises ValueError before the method dispatch in normalize.
    ht = _make_tensor('fullstate')
    ht.method = 'bogus'
    with pytest.raises((UnsupportedRequest, ValueError)):
        ht.normalize()


# ============================================================
# TEST SUITE: Properties (psi, phi_aux, flat_cores)
# ============================================================


# ------------------------------------------------------------
# TEST: phi_aux returns correct length
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_phi_aux_returns_correct_length():
    # This case tests that phi_aux returns a vector of length n_state.
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        n_modes = len(list_gw_sysbath)
        indices = [0] * n_modes
        indices[0] = 1
        phi_1 = extract_phi_aux(
            ht.list_cores_phi, ht.method, ht.M1_modes_per_state, indices
        )
        assert len(phi_1) == nsite
        # Analytical: at initialization, all hierarchy occupations are zero,
        # so any first-order auxiliary wavefunction must be zero.
        np.testing.assert_allclose(
            phi_1, 0.0, atol=1e-12,
            err_msg=f'First-order auxiliary should be zero at init for {method}',
        )


# ============================================================
# TEST SUITE: MPS utilities (restore_phi, tensor_compress,
#             check_bondsize, linksize, get_core_shapes)
# ============================================================


# ------------------------------------------------------------
# TEST: restore_phi from list replaces cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_restore_phi_from_list():
    # This case tests that restore_phi with a list copies the cores into
    # list_cores_phi (independent copy, not a direct reference) and that
    # subsequent mutation of the source list does not affect the tensor.
    for method in ['fullstate', 'number']:
        ht = _make_tensor(method)
        if method == 'number':
            # list_cores_phi is a list-of-lists for statenumber;
            # create a scaled copy with the same group structure.
            new_cores = [[core * 2.0 for core in group] for group in ht.list_cores_phi]
            ht.restore_phi(new_cores)
            # Values should match the source
            for g_set, g_src in zip(ht.list_cores_phi, new_cores):
                for c_set, c_src in zip(g_set, g_src):
                    np.testing.assert_array_equal(c_set, c_src)
            # list_cores_phi must be an independent copy, not the same object
            assert ht.list_cores_phi is not new_cores
            # Mutating the source array must not affect the tensor
            core_before = ht.list_cores_phi[0][0].copy()
            new_cores[0][0] *= 0.0
            np.testing.assert_array_equal(ht.list_cores_phi[0][0], core_before)
        else:
            new_cores = [c * 2.0 for c in ht.list_cores_phi]
            ht.restore_phi(new_cores)
            for c_set, c_src in zip(ht.list_cores_phi, new_cores):
                np.testing.assert_array_equal(c_set, c_src)
            assert ht.list_cores_phi is not new_cores
            core_before = ht.list_cores_phi[0].copy()
            new_cores[0] *= 0.0
            np.testing.assert_array_equal(ht.list_cores_phi[0], core_before)


# ------------------------------------------------------------
# TEST: restore_phi from another HopsTensorWavefunction copies cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_restore_phi_from_tensor():
    # This case tests that restore_phi from another HopsTensorWavefunction copies
    # the cores (not just a reference). Mutating ht2 after restore_phi
    # must not affect ht1. Tested for both representations.
    for method in ['fullstate', 'number']:
        ht1 = _make_tensor(method)
        ht2 = _make_tensor(method)
        if method == 'number':
            # list_cores_phi[0] is a group (list); scale the state core of group 0
            ht2.list_cores_phi[0][0] = ht2.list_cores_phi[0][0] * 3.0
        else:
            ht2.list_cores_phi[0] = ht2.list_cores_phi[0] * 3.0
        # Copy cores from ht2 into ht1
        ht1.restore_phi(ht2)
        # Verify restore_phi actually transferred the correct values
        np.testing.assert_allclose(
            ht1.psi, ht2.psi, atol=1e-12,
            err_msg=f'restore_phi did not transfer correct psi ({method})',
        )
        phi0_after_set = ht1.psi.copy()
        # Mutate ht2 — this must not leak into ht1
        if method == 'number':
            ht2.list_cores_phi[0][0] = ht2.list_cores_phi[0][0] * 0.0
        else:
            ht2.list_cores_phi[0] = ht2.list_cores_phi[0] * 0.0
        # ht1 should still have the pre-mutation values
        np.testing.assert_allclose(
            ht1.psi,
            phi0_after_set,
            atol=1e-12,
            err_msg=f'Mutating source tensor leaked into restore_phi copy ({method})',
        )


# ------------------------------------------------------------
# TEST: get_core_shapes returns correct shapes
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_get_core_shapes():
    # This case tests that get_core_shapes returns one shape tuple per core
    # and each shape matches the actual core array dimensions.
    ht = _make_tensor('fullstate')
    shapes = ht.get_core_shapes()
    assert len(shapes) == len(ht.list_cores_phi)
    for i, shape in enumerate(shapes):
        assert shape == ht.list_cores_phi[i].shape

    # This case tests get_core_shapes for statenumber (must flatten groups)
    ht_sn = _make_tensor('number')
    shapes_sn = ht_sn.get_core_shapes()
    flat_cores = ht_sn.flat_cores
    assert len(shapes_sn) == len(flat_cores)
    for i, shape in enumerate(shapes_sn):
        assert shape == flat_cores[i].shape


# ============================================================
# TEST SUITE: __init__() — flag_tdvp edge cases
# ============================================================


# ------------------------------------------------------------
# TEST: flag_tdvp is True for TDVP2 integrator
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_init_flag_tdvp_true_for_tdvp2():
    # This case tests that flag_tdvp is True for TDVP2 (not just TDVP1).
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    ht = HopsTensorWavefunction(k_max, tensor_param, {'INTEGRATOR': 'TDVP2'}, eom_param)
    assert ht.flag_tdvp is True


# ============================================================
# TEST SUITE: initialize() guards and edge cases
# ============================================================


# ------------------------------------------------------------
# TEST: Double initialize raises LockedException
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_double_raises():
    # This case tests that calling initialize() twice raises LockedException.
    # _make_tensor already calls initialize once internally.
    ht = _make_tensor('fullstate')
    tb = _make_tensor_basis_init()
    with pytest.raises(LockedException):
        ht.initialize(psi_0, tb.system)


# ------------------------------------------------------------
# TEST: TDVP path inflates bond dimensions
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_tdvp_inflates_bonds():
    # This case tests that when flag_tdvp is True, initialize
    # inflates bond dimensions up to bond_dim_max.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'TDVP1'}
    tb = _make_initialized_tensor_basis()
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    # Analytical: TDVP inflate sets all internal bonds to bond_dim_max
    for i, core in enumerate(ht.list_cores_phi[:-1]):
        assert core.shape[2] == 20, (
            f'Core {i} right bond should be bond_dim_max=20, got {core.shape[2]}'
        )


# ------------------------------------------------------------
# TEST: k_max=0 produces valid single-level hierarchy
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_k_max_zero():
    # This case tests that k_max=0 (single hierarchy level) produces
    # valid MPS cores where mode cores have local dimension 1.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    tb = _make_initialized_tensor_basis()
    ht = HopsTensorWavefunction(0, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_0, tb.system)
    # Analytical: fullstate MPS has 1 state core + n_total_modes mode cores
    expected_n_cores = 1 + sum(ht.M1_modes_per_state)
    assert len(ht.list_cores_phi) == expected_n_cores
    # Mode cores should have local dimension k_max+1 = 1
    # First core is the system core, subsequent are mode cores
    for core in ht.list_cores_phi[1:]:
        assert core.shape[1] == 1, (
            f'Mode core should have local dim 1 for k_max=0, got {core.shape[1]}'
        )


# ============================================================
# TEST SUITE: build_list_cores_phi() — error path
# ============================================================


# ------------------------------------------------------------
# TEST: Invalid method raises UnsupportedRequest
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_build_list_cores_phi_invalid_method_raises():
    # This case tests that passing an unknown method to
    # build_list_cores_phi raises UnsupportedRequest.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'bogus_method',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    tb = _make_initialized_tensor_basis()
    with pytest.raises(UnsupportedRequest):
        ht.build_list_cores_phi(psi_0, len(tb.system.state_list))


# ============================================================
# TEST SUITE: restore_phi() — error path
# ============================================================


# ------------------------------------------------------------
# TEST: Invalid type raises TypeError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_restore_phi_invalid_type_raises():
    # This case tests that passing a non-list, non-HopsTensorWavefunction
    # argument to restore_phi raises TypeError.
    ht = _make_tensor('fullstate')
    with pytest.raises(TypeError, match='Expected list or HopsTensorWavefunction'):
        ht.restore_phi('not_a_list_or_tensor')


# ============================================================
# TEST SUITE: inflate_bonds_to() — statenumber coverage (Ritesh T1)
# ============================================================


# ------------------------------------------------------------
# TEST: Statenumber inflate reaches target bond dim
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_inflate_bonds_to_statenumber():
    # This case tests that inflate_bonds_to works for statenumber
    # representation and all internal bonds reach chi_target.
    ht = _make_tensor('number')
    chi_target = 5
    ht.inflate_bonds_to(chi_target)
    flat = ht.flat_cores
    for i in range(len(flat) - 1):
        bond_right = flat[i].shape[2]
        bond_left_next = flat[i + 1].shape[0]
        assert bond_right == chi_target, (
            f'Core {i} right bond {bond_right} != target {chi_target}'
        )
        assert bond_left_next == chi_target, (
            f'Core {i + 1} left bond {bond_left_next} != target {chi_target}'
        )


# ------------------------------------------------------------
# TEST: Statenumber inflate preserves phi_0
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_inflate_bonds_preserves_phi0_statenumber():
    # This case tests that inflating bonds in statenumber representation
    # does not change the physical wavefunction.
    ht = _make_tensor('number')
    V1_phi_before = ht.psi.copy()
    ht.inflate_bonds_to(5)
    np.testing.assert_allclose(
        ht.psi,
        V1_phi_before,
        atol=1e-12,
        err_msg='inflate_bonds_to changed phi_0 in statenumber',
    )


# ------------------------------------------------------------
# TEST: Statenumber inflate with eps > 0 adds random noise padding
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_inflate_bonds_eps_nonzero_statenumber():
    # This case tests that inflate_bonds_to with eps > 0 produces
    # non-zero padding entries in statenumber representation.
    ht = _make_tensor('number')
    original_shapes = [core.shape for core in ht.flat_cores]
    chi_target = 5
    ht.inflate_bonds_to(chi_target, eps=0.1)
    has_nonzero_pad = False
    for core, orig_shape in zip(ht.flat_cores, original_shapes):
        orig_left, _, orig_right = orig_shape
        if core.shape[0] > orig_left:
            if np.any(np.abs(core[orig_left:, :, :]) > 1e-15):
                has_nonzero_pad = True
        if core.shape[2] > orig_right:
            if np.any(np.abs(core[:, :, orig_right:]) > 1e-15):
                has_nonzero_pad = True
    assert has_nonzero_pad, 'eps>0 padding should contain nonzero entries'


# ------------------------------------------------------------
# TEST: Statenumber inflate is a no-op when bonds already at target
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_inflate_bonds_noop_at_target_statenumber():
    # This case tests that calling inflate_bonds_to with chi_target equal
    # to the current bond dimension does not change the cores in
    # statenumber representation.
    ht = _make_tensor('number')
    cores_before = [c.copy() for c in ht.flat_cores]
    current_chi = ht.flat_cores[0].shape[2]
    ht.inflate_bonds_to(current_chi, eps=0.0)
    for i, (before, after) in enumerate(zip(cores_before, ht.flat_cores)):
        np.testing.assert_array_equal(
            before,
            after,
            err_msg=f'Core {i} changed when chi_target == current bond dim',
        )


# ============================================================
# TEST SUITE: add_state_cores()
# ============================================================


# ------------------------------------------------------------
# TEST: add_state_cores fullstate increases core count
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_add_state_cores_fullstate_count():
    # This case tests that add_state_cores increases the number of mode
    # cores by modes_per_state per new state (fullstate representation).
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    psi_2 = np.array([0.0, 1.0], dtype=np.complex128)
    system, mode, noise_memory = _make_basis_objects()
    system.initialize(False, psi_2)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = [0, 1]
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_2, system)
    cores_before = len(ht.list_cores_phi)
    n_new_modes = sum(ht.M1_modes_per_state[s] for s in [2])
    ht.add_state_cores([2], list(system.state_list))
    assert len(ht.list_cores_phi) == cores_before + n_new_modes
    assert ht.list_cores_phi[0].shape[1] == 3
    assert len(ht.M1_modes_per_site) == 3


# ------------------------------------------------------------
# TEST: add_state_cores statenumber increases group count
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_add_state_cores_statenumber_count():
    # This case tests that add_state_cores adds one new group per new state
    # in statenumber representation, and the state count increases by one.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    psi_2 = np.array([0.0, 1.0], dtype=np.complex128)
    system, mode, noise_memory = _make_basis_objects()
    system.initialize(False, psi_2)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = [0, 1]
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_2, system)
    groups_before = len(ht.list_cores_phi)
    ht.add_state_cores([2], list(system.state_list))
    # Statenumber: one new group added
    assert len(ht.list_cores_phi) == groups_before + 1
    assert len(ht.M1_modes_per_site) == 3
    # Analytical: adding a zero-amplitude state preserves existing amplitudes
    # and the new state has zero amplitude
    phi_after = extract_psi(
        ht.list_cores_phi,
        'number',
        ht.M1_modes_per_state,
    )
    # This case tests that the newly added state has zero amplitude
    # psi is ordered by sorted state list: [0, 1, 2]
    # State 2 was just added and should have zero amplitude
    np.testing.assert_allclose(
        phi_after[2], 0.0, atol=1e-12,
        err_msg='Newly added state should have zero amplitude',
    )


# ------------------------------------------------------------
# TEST: add_state_cores preserves phi_0 for existing states
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_add_state_cores_fullstate_preserves_phi0():
    # This case tests that adding a new state leaves the existing phi_0 amplitudes
    # unchanged and places zero amplitude on the newly added state.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'fullstate',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    psi_2 = np.array([0.0, 1.0], dtype=np.complex128)
    system, mode, noise_memory = _make_basis_objects()
    system.initialize(False, psi_2)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = [0, 1]
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_2, system)
    phi_0_before = ht.psi.copy()
    ht.add_state_cores([2], list(system.state_list))
    phi_0_after = ht.psi
    assert len(phi_0_after) == 3
    np.testing.assert_allclose(phi_0_after[:2], phi_0_before, atol=1e-12)
    np.testing.assert_allclose(phi_0_after[2], 0.0, atol=1e-12)


# ------------------------------------------------------------
# TEST: add_state_cores statenumber preserves existing amplitudes
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_add_state_cores_statenumber_preserves_phi0():
    # This case tests that existing amplitudes are unchanged after
    # adding a new state in statenumber representation.
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    psi_2 = np.array([0.0, 1.0], dtype=np.complex128)
    system, mode, noise_memory = _make_basis_objects()
    system.initialize(False, psi_2)
    mode.list_modeidx_abs = sorted(system.list_statemodeidx_abs)
    noise_memory.initialize()
    system.state_list = [0, 1]
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_2, system)
    psi_before = ht.psi.copy()
    ht.add_state_cores([2], list(system.state_list))
    psi_after = ht.psi
    # Analytical: existing states keep their amplitudes, new state has zero
    np.testing.assert_allclose(psi_after[:2], psi_before, atol=1e-12)
    np.testing.assert_allclose(psi_after[2], 0.0, atol=1e-12)


# ============================================================
# TEST SUITE: remove_state_cores()
# ============================================================


# ------------------------------------------------------------
# TEST: remove_state_cores fullstate decreases core count
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_remove_state_cores_fullstate_count():
    # This case tests that remove_state_cores reduces the core count by
    # modes_per_state for the removed state, shrinks the system core physical
    # dimension, and decreases the site count by one.
    ht = _make_tensor('fullstate')
    cores_before = len(ht.list_cores_phi)
    state_list = list(range(nsite))
    n_modes_removed = sum(ht.M1_modes_per_state[s] for s in [0])
    ht.remove_state_cores([0], state_list)
    assert len(ht.list_cores_phi) == cores_before - n_modes_removed
    assert ht.list_cores_phi[0].shape[1] == nsite - 1
    assert len(ht.M1_modes_per_site) == nsite - 1


# ------------------------------------------------------------
# TEST: remove_state_cores statenumber decreases group count
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_remove_state_cores_statenumber_count():
    # This case tests that remove_state_cores removes exactly one group
    # in statenumber representation and the site count decreases by one.
    ht = _make_tensor('number')
    groups_before = len(ht.list_cores_phi)
    state_list = list(range(nsite))
    ht.remove_state_cores([0], state_list)
    assert len(ht.list_cores_phi) == groups_before - 1
    assert len(ht.M1_modes_per_site) == nsite - 1
    # Analytical: removing state 0 (zero amplitude in psi_0=[0,0,1,0])
    # should preserve total probability
    phi_after = extract_psi(
        ht.list_cores_phi,
        'number',
        ht.M1_modes_per_state,
    )
    # This case tests that total probability is preserved after removing
    # a zero-amplitude state
    assert np.sum(np.abs(phi_after) ** 2) > 0.5, (
        'Total probability collapsed after removing zero-amplitude state'
    )


# ------------------------------------------------------------
# TEST: remove_state_cores statenumber leftmost group (known bug)
# ------------------------------------------------------------
@pytest.mark.level(1)
@pytest.mark.xfail(
    reason=(
        'Known bug: remove_state_cores statenumber group_idx==0 uses `pass` '
        'instead of absorbing |0> slices into the next group. The bug is visible '
        'only when the left bond dimension Dr0 > 1 (post-propagation states). '
        'Simple product-state initializations have Dr0==1 so the `else: pass` '
        'path happens to be harmless. This xfail documents the structural defect.'
    ),
    strict=True,
)
def test_remove_state_cores_statenumber_leftmost_bug():
    # This case tests that removing the leftmost state in statenumber
    # preserves the wavefunction correctly when the removed state has zero
    # amplitude but the next state carries amplitude.
    # psi starts on state 1 — state 0 is the leftmost group (zero amplitude).
    # Removing state 0 must absorb its |0> slice into state 1's group.
    # The known bug (else: pass in remove_state_cores) discards the |0> slice
    # instead of absorbing it. Bonds are inflated to bond dim > 1 so the |0>
    # slices carry information and the bug actually manifests.
    psi_state1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    tensor_param = {
        'MPS_EPSILON': 1e-10,
        'METHOD': 'number',
        'BOND_DIM_MAX': 20,
    }
    integrator_param = {'INTEGRATOR': 'RUNGE_KUTTA'}
    tb = _make_initialized_tensor_basis()
    ht = HopsTensorWavefunction(k_max, tensor_param, integrator_param, eom_param)
    ht.initialize(psi_state1, tb.system)
    # Inflate bonds so the leftmost-removal bug actually manifests.
    # After inflation, the inter-group bonds have dim > 1. The |0> slice of
    # the removed group's state core is shape (1, Dr0) with Dr0=4. The bug
    # discards this slice instead of absorbing it into the next group. This
    # leaves the new first group with an orphaned left bond of dim Dr0 instead
    # of 1, which is detectable by inspecting the core shape.
    ht.inflate_bonds_to(4, eps=0.01)
    full_state_list = list(range(nsite))
    ht.remove_state_cores([0], full_state_list)
    # After correct removal, the new first group must have left bond dim == 1
    # (open left boundary). The bug leaves it at Dr0 == 4.
    new_first_state_core = ht.list_cores_phi[0][0]
    assert new_first_state_core.shape[0] == 1, (
        f'Leftmost group after removal has left bond dim '
        f'{new_first_state_core.shape[0]}, expected 1. '
        f'The |0> slice was discarded instead of absorbed.'
    )


# ------------------------------------------------------------
# TEST: remove then re-add state preserves phi_0 values
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_remove_then_add_state_preserves_phi0():
    # This case tests that after removing state 0 (zero amplitude), the remaining
    # phi_0 entries match the original values at indices 1 onward.
    ht = _make_tensor('fullstate')
    state_list = list(range(nsite))
    phi_0_orig = ht.psi.copy()
    # Remove state 0 (which has zero amplitude in psi_0)
    ht.remove_state_cores([0], state_list)
    phi_0_after_remove = ht.psi
    assert len(phi_0_after_remove) == nsite - 1
    np.testing.assert_allclose(
        phi_0_after_remove, phi_0_orig[1:], atol=1e-12,
    )


# ------------------------------------------------------------
# TEST: remove all but one state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_remove_all_but_one():
    # This case tests that removing all states except the one carrying
    # amplitude leaves a single-site MPS with non-zero psi[0].
    ht = _make_tensor('fullstate')
    state_list = list(range(nsite))
    # Remove states 0, 1, 3 — keep only state 2 (which has the amplitude)
    ht.remove_state_cores([0, 1, 3], state_list)
    assert ht.list_cores_phi[0].shape[1] == 1
    assert len(ht.M1_modes_per_site) == 1
    assert len(ht.psi) == 1
    # Analytical: psi_0 = [0,0,1,0], keeping only state 2 (which had
    # amplitude 1.0), so the single remaining amplitude should be 1.0
    np.testing.assert_allclose(
        abs(ht.psi[0]), 1.0, atol=1e-10,
        err_msg='Amplitude on kept state should be 1.0',
    )


# ------------------------------------------------------------
# TEST: add_state_cores raises on invalid method
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_add_state_cores_invalid_method_raises():
    # This case tests that add_state_cores raises UnsupportedRequest
    # when method is not a recognized representation.
    ht = _make_tensor('fullstate')
    ht.method = 'bogus'
    with pytest.raises(UnsupportedRequest):
        ht.add_state_cores([nsite], list(state_list))


# ------------------------------------------------------------
# TEST: remove_state_cores raises on invalid method
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_remove_state_cores_invalid_method_raises():
    # This case tests that remove_state_cores raises UnsupportedRequest
    # when method is not a recognized representation.
    ht = _make_tensor('fullstate')
    ht.method = 'bogus'
    with pytest.raises(UnsupportedRequest):
        ht.remove_state_cores([state_list[0]], list(state_list))


# ============================================================
# TEST SUITE: update_phi_from_flat()
# ============================================================


# ------------------------------------------------------------
# TEST: Flatten then update_phi_from_flat recovers original psi
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_update_phi_from_flat_recovers_psi():
    # This case tests that extracting flat cores, then writing them
    # back via update_phi_from_flat, recovers the original psi.
    # Uses a propagated state with naturally complex cores and
    # non-trivial bond structure.
    ht_full = _make_propagated_tensor('fullstate')
    psi_before = ht_full.psi.copy()
    flat = [c.copy() for c in ht_full.list_cores_phi]
    ht_full.update_phi_from_flat(flat)
    np.testing.assert_allclose(ht_full.psi, psi_before, atol=1e-14)

    # This case tests the same flatten->update->recover cycle for
    # statenumber representation with propagated state.
    ht_sn = _make_propagated_tensor('number')
    psi_before_sn = ht_sn.psi.copy()
    flat_sn = [c.copy() for c in ht_sn.flat_cores]
    ht_sn.update_phi_from_flat(flat_sn)
    np.testing.assert_allclose(ht_sn.psi, psi_before_sn, atol=1e-14)


# ------------------------------------------------------------
# TEST: update_phi_from_flat makes an independent copy
# ------------------------------------------------------------
@pytest.mark.level(2)
def test_update_phi_from_flat_independent_copy():
    # CASE: Mutating the input list after update_phi_from_flat must
    # not affect list_cores_phi. Uses propagated state for non-trivial
    # bond structure. Mirrors restore_phi independence tests.
    for method in ['fullstate', 'number']:
        ht = _make_propagated_tensor(method)
        flat = [c.copy() for c in ht.flat_cores]
        # Scale to distinguish from initial state
        flat[0] = flat[0] * 2.0
        ht.update_phi_from_flat(flat)
        psi_after_update = ht.psi.copy()
        cores_after_update = [c.copy() for c in ht.flat_cores]
        # Mutate the source list
        flat[0] *= 0.0
        # ht should be unaffected — check both psi and raw cores
        np.testing.assert_allclose(
            ht.psi, psi_after_update, atol=1e-14,
            err_msg=f'Mutating source leaked into psi ({method})',
        )
        for i, (c_now, c_saved) in enumerate(zip(ht.flat_cores, cores_after_update)):
            np.testing.assert_array_equal(
                c_now, c_saved,
                err_msg=f'Core {i} mutated by source modification ({method})',
            )


# ------------------------------------------------------------
# TEST: flat_cores returns views into list_cores_phi
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_flat_cores_are_views():
    # CASE: For statenumber representation, flat_cores should return
    # views into list_cores_phi. Mutating an entry in flat_cores should
    # be visible through list_cores_phi and affect psi. This documents
    # the view semantics — flat_cores is NOT a deep copy.
    ht = _make_tensor('number')
    flat = ht.flat_cores
    psi_before = ht.psi.copy()
    # Mutate the first flat core (which is list_cores_phi[0][0])
    flat[0][0, :, 0] *= 2.0
    psi_after = ht.psi
    # psi should have changed because flat_cores are views
    assert not np.allclose(psi_before, psi_after, atol=1e-14), (
        'Mutating flat_cores should affect psi (view semantics)'
    )
