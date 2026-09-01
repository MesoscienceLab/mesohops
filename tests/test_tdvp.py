import numpy as np
import pytest

from mesohops.tensor.tdvp import (
    _apply_heff_bond,
    _apply_heff_site,
    _apply_heff_twosite,
    _arnoldi_expm,
    _ivp_solve,
    _lanczos_expm,
    _solve_local,
    _split_qr,
    _split_rq,
    _split_svd,
    contract_left,
    contract_right,
    initialize,
    recenter_to_zero,
    sweep_left_1tdvp,
    sweep_left_2tdvp,
    sweep_right_1tdvp,
    sweep_right_2tdvp,
    timestep,
)

__title__ = 'Unit Tests for TDVP Internals'
__author__ = 'N. Covalsen'
__maintainer__ = 'N. Covalsen'


# ============================================================
# TEST SUITE: _split_qr()
# ============================================================


# ------------------------------------------------------------
# TEST: Q is left-orthogonal and Q @ R recovers input
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_qr_orthogonal_and_recovers():
    # This case tests that Q has orthonormal columns (Q†Q = I)
    # and Q @ R recovers the original tensor.
    np.random.seed(10)
    core_input = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
    q, r = _split_qr(core_input)
    # Q†Q = I (left-orthogonal over combined left-physical index)
    Dl, d, Dr_q = q.shape
    Q_mat = q.reshape(Dl * d, Dr_q)
    np.testing.assert_allclose(
        Q_mat.conj().T @ Q_mat,
        np.eye(Dr_q),
        atol=1e-12,
        err_msg='Q is not left-orthogonal',
    )
    # Q @ R recovers original
    recovered = np.tensordot(q, r, axes=((2,), (0,)))
    np.testing.assert_allclose(
        recovered, core_input, atol=1e-12, err_msg='Q @ R does not recover input'
    )


# ============================================================
# TEST SUITE: _split_rq()
# ============================================================


# ------------------------------------------------------------
# TEST: Q is right-orthogonal and R @ Q recovers input
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_rq_orthogonal_and_recovers():
    # This case tests that Q has orthonormal rows (QQ† = I over
    # combined physical-right index) and R @ Q recovers the original.
    np.random.seed(11)
    core_input = np.random.randn(4, 3, 2) + 1j * np.random.randn(4, 3, 2)
    r, q = _split_rq(core_input)
    # Q is right-orthogonal: reshape (Dl_q, d, Dr) -> (Dl_q, d*Dr), check M M† = I
    Dl_q, d, Dr = q.shape
    Q_mat = q.reshape(Dl_q, d * Dr)
    np.testing.assert_allclose(
        Q_mat @ Q_mat.conj().T,
        np.eye(Dl_q),
        atol=1e-12,
        err_msg='Q is not right-orthogonal',
    )
    # R @ Q recovers original
    recovered = np.tensordot(r, q, axes=((1,), (0,)))
    np.testing.assert_allclose(
        recovered, core_input, atol=1e-12, err_msg='R @ Q does not recover input'
    )


# ============================================================
# TEST SUITE: _split_svd()
# ============================================================


# ------------------------------------------------------------
# TEST: eps threshold keeps only SVs above cutoff
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_svd_eps_threshold():
    # This case tests that eps=0.5 drops the singular value 0.01 < 0.5,
    # leaving only the single SV 10.0 → chi_new=1.
    # Build theta by embedding a known 2x2 diagonal matrix in shape (1, 4, 1):
    # _split_svd reshapes (1, 4, 1) → (2, 2), giving SVs [10, 0.01].
    d = 2
    mat = np.diag([10.0, 0.01]).astype(np.complex128)
    theta = mat.reshape(1, d * d, 1)
    U, S, Vt, chi_new = _split_svd(theta, 0, 0.5)
    assert chi_new == 1, f'Expected chi_new=1, got {chi_new}'
    assert U.shape == (1, d, 1)
    assert Vt.shape == (1, d, 1)
    np.testing.assert_allclose(S[0], 10.0, atol=1e-12)


# ------------------------------------------------------------
# TEST: chi_max caps rank even when SVs are above eps
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_svd_rank_cap():
    # This case tests that chi_max=2 limits the rank even when
    # all singular values exceed eps.
    np.random.seed(13)
    d = 3
    theta = np.random.randn(1, d * d, 1) + 1j * np.random.randn(1, d * d, 1)
    U, S, Vt, chi_new = _split_svd(theta, 2, 0.0)
    assert chi_new == 2, f'Expected chi_new=2, got {chi_new}'
    assert U.shape == (1, d, 2)
    assert Vt.shape == (2, d, 1)
    # Analytical: the retained singular values should be the 2 largest
    _, S_full, _, _ = _split_svd(theta, 0, 0.0)
    S_sorted = np.sort(S_full)[::-1]
    np.testing.assert_allclose(
        np.sort(S)[::-1], S_sorted[:2], atol=1e-12,
        err_msg='Retained SVs should be the 2 largest',
    )


# ------------------------------------------------------------
# TEST: U @ diag(S) @ Vt recovers the input tensor
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_svd_recovers_input():
    # This case tests that contracting U, S, Vt back together
    # recovers the original two-site tensor (up to numerical precision).
    np.random.seed(14)
    d = 3
    Dl, Dr = 2, 2
    theta = np.random.randn(Dl, d * d, Dr) + 1j * np.random.randn(Dl, d * d, Dr)
    U, S, Vt, chi_new = _split_svd(theta, 0, 1e-14)
    # U: (Dl, d, chi), S: (chi,), Vt: (chi, d, Dr)
    # recovery: sum_k U[a,s1,k] * S[k] * Vt[k,s2,b] → (Dl, d, d, Dr) → reshape
    recovered = np.tensordot(U * S[None, None, :], Vt, axes=((2,), (0,))).reshape(
        Dl, d * d, Dr
    )
    np.testing.assert_allclose(
        recovered, theta, atol=1e-10, err_msg='U @ diag(S) @ Vt does not recover input'
    )


# ------------------------------------------------------------
# TEST: U is left-orthogonal after truncation
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_svd_u_left_orthogonal():
    # This case tests that the U factor satisfies U†U = I (left-orthogonal).
    np.random.seed(15)
    d = 3
    Dl, Dr = 2, 2
    theta = np.random.randn(Dl, d * d, Dr) + 1j * np.random.randn(Dl, d * d, Dr)
    U, S, Vt, chi_new = _split_svd(theta, 0, 1e-14)
    U_mat = U.reshape(Dl * d, chi_new)
    np.testing.assert_allclose(
        U_mat.conj().T @ U_mat,
        np.eye(chi_new),
        atol=1e-12,
        err_msg='U is not left-orthogonal',
    )


# ------------------------------------------------------------
# TEST: Vt factor from _split_svd is right-orthogonal
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_svd_vt_right_orthogonal():
    # This case tests that the Vt factor satisfies Vt Vt† = I.
    np.random.seed(16)
    d = 3
    Dl, Dr = 2, 2
    theta = np.random.randn(Dl, d * d, Dr) + 1j * np.random.randn(Dl, d * d, Dr)
    U, S, Vt, chi_new = _split_svd(theta, 0, 1e-14)
    Vt_mat = Vt.reshape(chi_new, d * Dr)
    np.testing.assert_allclose(
        Vt_mat @ Vt_mat.conj().T,
        np.eye(chi_new),
        atol=1e-12,
        err_msg='Vt is not right-orthogonal',
    )


# ============================================================
# TEST SUITE: _ivp_solve()
# ============================================================


# ------------------------------------------------------------
# TEST: Identity RHS scales state by exp(dt)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_ivp_solve_identity_rhs():
    # This case tests that dy/dt = y gives y(t) = exp(t) * y(0).
    state = np.array([1.0 + 0j, 0.5 + 0j])
    dt = 0.1
    result = _ivp_solve(state, lambda v: v, dt)
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-6)


# ============================================================
# TEST SUITE: _arnoldi_expm()
# ============================================================


def _trivial_envs():
    """Build trivial (1,1,1) left and right environment tensors."""
    env_left = np.ones((1, 1, 1), dtype=np.complex128)
    env_right = np.ones((1, 1, 1), dtype=np.complex128)
    return env_left, env_right


def _identity_mpo_core(d):
    """Build an identity MPO core of shape (1, d, d, 1)."""
    core_mpo = np.zeros((1, d, d, 1), dtype=np.complex128)
    for k in range(d):
        core_mpo[0, k, k, 0] = 1.0
    return core_mpo


def _diagonal_mpo_core(diag):
    """Build a diagonal MPO core from a 1-D array of eigenvalues."""
    d = len(diag)
    core_mpo = np.zeros((1, d, d, 1), dtype=np.complex128)
    for k in range(d):
        core_mpo[0, k, k, 0] = diag[k]
    return core_mpo


# ------------------------------------------------------------
# TEST: Identity MPO with trivial environments scales by exp(dt)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_identity_mpo_scales_by_exp():
    # This case tests that the effective operator is the identity, so
    # exp(dt * I) * state = exp(dt) * state.
    d = 3
    state = np.random.RandomState(31).randn(1, d, 1) + 0j
    env_left, env_right = _trivial_envs()
    core_mpo = _identity_mpo_core(d)
    dt = 0.05
    result = _arnoldi_expm(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        1e-12,
    )
    assert result.shape == state.shape
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-10)


# ------------------------------------------------------------
# TEST: Bond update with trivial environments scales by exp(dt)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_bond_update_trivial_envs():
    # This case tests the bond-update path. With trivial (ones) environments,
    # _apply_heff_bond is the identity on (1,1) bond tensors, so
    # exp(dt * I) * state = exp(dt) * state.
    state = np.array([[3.0 + 1j]], dtype=np.complex128)
    env_left, env_right = _trivial_envs()
    dt = 0.1
    result = _arnoldi_expm(
        state,
        lambda v: _apply_heff_bond(v, env_left, env_right),
        dt,
        1e-12,
    )
    assert result.shape == state.shape
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-10)


# ------------------------------------------------------------
# TEST: Known diagonal Hamiltonian matches analytical result
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_known_diagonal_hamiltonian():
    # This case tests _arnoldi_expm against an analytical result for a diagonal
    # Hamiltonian (Pauli Z). The effective operator with trivial environments
    # is diag(+1, -1). For state with both components nonzero, the Krylov
    # subspace spans the full 2D space, giving an exact result:
    # exp(dt * Z) @ state = [exp(dt)*a, exp(-dt)*b]
    d = 2
    rng = np.random.RandomState(34)
    state = (rng.randn(1, d, 1) + 1j * rng.randn(1, d, 1)).astype(np.complex128)
    env_left, env_right = _trivial_envs()
    core_mpo = _diagonal_mpo_core([1.0, -1.0])
    dt = 0.05
    result = _arnoldi_expm(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        1e-12,
    )
    expected = state.copy()
    expected[0, 0, 0] *= np.exp(dt)
    expected[0, 1, 0] *= np.exp(-dt)
    np.testing.assert_allclose(result, expected, atol=1e-10)


# ------------------------------------------------------------
# TEST: Convergence on rank-1 effective Hamiltonian
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_early_termination():
    # This case tests that the convergence check terminates iteration
    # after one step for the identity MPO (1D Krylov space): h_{2,1} = 0
    # → the error estimate is exactly zero → conv_tol check passes
    # immediately. Result must equal exp(dt) * state.
    d = 3
    state = np.random.RandomState(35).randn(1, d, 1) + 0j
    env_left, env_right = _trivial_envs()
    core_mpo = _identity_mpo_core(d)
    dt = 0.02
    matvec_heff = lambda v: _apply_heff_site(v, env_left, core_mpo, env_right)  # noqa: E731
    result = _arnoldi_expm(state, matvec_heff, dt, 1e-12)
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-10)


# ------------------------------------------------------------
# TEST: Full Krylov loop with non-Hermitian off-diagonal MPO
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_full_krylov_loop():
    # This case tests _arnoldi_expm with a non-trivial off-diagonal MPO
    # that does NOT trigger early breakdown, forcing the full Krylov loop.
    # We use a 3x3 nilpotent-shift matrix as the effective Hamiltonian.
    d = 3
    # Build an MPO core that acts as a shift: |i> -> |i+1 mod d>
    core_mpo = np.zeros((1, d, d, 1), dtype=complex)
    for i in range(d):
        core_mpo[0, (i + 1) % d, i, 0] = 1.0
    env_left = np.ones((1, 1, 1), dtype=complex)
    env_right = np.ones((1, 1, 1), dtype=complex)
    state = np.zeros((1, d, 1), dtype=complex)
    state[0, 0, 0] = 1.0
    dt = 0.1
    matvec = lambda v: _apply_heff_site(v, env_left, core_mpo, env_right)  # noqa: E731
    result = _arnoldi_expm(state, matvec, dt, conv_tol=1e-14)
    # Reference: build the d x d shift matrix and compute expm directly
    H_shift = np.zeros((d, d), dtype=complex)
    for i in range(d):
        H_shift[(i + 1) % d, i] = 1.0
    from scipy.linalg import expm as scipy_expm
    expected_vec = scipy_expm(dt * H_shift) @ np.array([1, 0, 0], dtype=complex)
    expected = expected_vec.reshape(1, d, 1)
    np.testing.assert_allclose(result, expected, atol=1e-10)


# ------------------------------------------------------------
# TEST: _arnoldi_expm returns zeros for zero-vector input
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_zero_vector():
    # Analytical: if input is zero, output must be zero (beta < tol path)
    phys_dims = [2, 3]
    list_cores_mpo = _make_list_mpo(phys_dims)
    L = np.ones((1, 1, 1), dtype=np.complex128)
    R = np.ones((1, 1, 1), dtype=np.complex128)
    W = list_cores_mpo[0]
    v = np.zeros((1, phys_dims[0], 1), dtype=np.complex128)
    matvec = lambda x, _L=L, _W=W, _R=R: _apply_heff_site(x, _L, _W, _R)  # noqa: E731
    result = _arnoldi_expm(v, matvec, 0.01, 1e-12)
    np.testing.assert_allclose(result, 0.0, atol=1e-14)


# ============================================================
# TEST SUITE: contract_left() / contract_right()
# ============================================================


# ------------------------------------------------------------
# TEST: Output shape is correct for both directions
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_right_output_shape_and_value():
    # contract_right(R_next, W, B): R_next (Dr,bR,Dr), W (bL,d,d,bR),
    # B (Dl,d,Dr) → output (Dl, bL, Dl)
    d = 2
    Dl, Dr, bL, bR = 3, 4, 2, 2
    rng = np.random.RandomState(41)
    core_mps = rng.randn(Dl, d, Dr) + 1j * rng.randn(Dl, d, Dr)
    core_mpo = rng.randn(bL, d, d, bR) + 1j * rng.randn(bL, d, d, bR)
    env_init = rng.randn(Dr, bR, Dr) + 1j * rng.randn(Dr, bR, Dr)
    # Compute via contract_right
    result = contract_right(env_init, core_mpo, core_mps)
    assert result.shape == (Dl, bL, Dl), f'Expected {(Dl, bL, Dl)}, got {result.shape}'
    # Verify value against direct einsum
    expected = np.einsum(
        'ijk,lmi,nmoj,pok->lnp',
        env_init,
        core_mps.conj(),
        core_mpo,
        core_mps,
    )
    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-12,
        err_msg='contract_right value mismatch vs direct einsum',
    )


# ------------------------------------------------------------
# TEST: Output shape and value are correct for contract_left
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_left_output_shape_and_value():
    # contract_left(L_prev, W, A): L_prev (Dl,bL,Dl), W (bL,d,d,bR),
    # A (Dl,d,Dr) → output (Dr, bR, Dr)
    d = 2
    Dl, Dr, bL, bR = 3, 4, 2, 2
    rng = np.random.RandomState(42)
    core_mps = rng.randn(Dl, d, Dr) + 1j * rng.randn(Dl, d, Dr)
    core_mpo = rng.randn(bL, d, d, bR) + 1j * rng.randn(bL, d, d, bR)
    env_init = rng.randn(Dl, bL, Dl) + 1j * rng.randn(Dl, bL, Dl)
    # Compute via contract_left
    result = contract_left(env_init, core_mpo, core_mps)
    assert result.shape == (Dr, bR, Dr), f'Expected {(Dr, bR, Dr)}, got {result.shape}'
    # Verify value against direct einsum
    expected = np.einsum(
        'ijk,ilm,jlno,knp->mop',
        env_init,
        core_mps.conj(),
        core_mpo,
        core_mps,
    )
    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-12,
        err_msg='contract_left value mismatch vs direct einsum',
    )


# ------------------------------------------------------------
# TEST: Full sweep builds consistent environments
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_contract_right_full_sweep_consistency():
    # This case tests that a right-to-left sweep with identity MPO
    # on a random 3-site MPS produces a final environment equal to
    # the MPS norm squared.
    rng = np.random.RandomState(44)
    phys_dims = [2, 3, 2]
    list_cores = [rng.randn(1, d, 1) + 1j * rng.randn(1, d, 1) for d in phys_dims]
    list_cores_mpo = [_identity_mpo_core(d) for d in phys_dims]

    phi = np.ones((1, 1, 1), dtype=np.complex128)
    for i in range(len(list_cores) - 1, -1, -1):
        phi = contract_right(phi, list_cores_mpo[i], list_cores[i])

    # Compute norm squared by full contraction
    state = list_cores[0]
    for c in list_cores[1:]:
        state = np.tensordot(state, c, axes=([-1], [0]))
    norm_sq = np.vdot(state, state)
    np.testing.assert_allclose(
        phi.item(),
        norm_sq,
        atol=1e-10,
        err_msg='Full R-to-L sweep should give norm squared',
    )


def _identity_mpo_core_2site(d):
    """Build identity MPO list_cores for two adjacent sites of dim d."""
    core_mpo_site1 = np.zeros((1, d, d, 1), dtype=np.complex128)
    core_mpo_site2 = np.zeros((1, d, d, 1), dtype=np.complex128)
    for k in range(d):
        core_mpo_site1[0, k, k, 0] = 1.0
        core_mpo_site2[0, k, k, 0] = 1.0
    return core_mpo_site1, core_mpo_site2


def _make_list_cores(phys_dims, seed=0):
    """Build random bond-dim-1 MPS list_cores as a plain list."""
    rng = np.random.RandomState(seed)
    return [rng.randn(1, d, 1) + 1j * rng.randn(1, d, 1) for d in phys_dims]


def _make_list_mpo(phys_dims):
    """Build an identity MPO as a plain list of list_cores."""
    return [_identity_mpo_core(d) for d in phys_dims]


def _mps_norm(cores):
    """Compute <psi|psi> from MPS cores via full contraction."""
    env = np.ones((1, 1), dtype=np.complex128)
    for c in cores:
        env = np.einsum('ij,ikl,jkm->lm', env, c, np.conj(c))
    return np.sqrt(np.abs(env[0, 0]))


# ============================================================
# TEST SUITE: _lanczos_expm()
# ============================================================


# ------------------------------------------------------------
# TEST: Identity MPO with trivial environments scales by exp(dt)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_lanczos_expm_identity_mpo_scales_by_exp():
    # This case tests that the Hermitian identity operator gives
    # exp(dt * I) * state = exp(dt) * state.
    d = 3
    state = np.random.RandomState(71).randn(1, d, 1) + 0j
    env_left, env_right = _trivial_envs()
    core_mpo = _identity_mpo_core(d)
    dt = 0.05
    result = _lanczos_expm(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        1e-12,
    )
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-10)


# ------------------------------------------------------------
# TEST: Known diagonal Hamiltonian matches analytical result
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_lanczos_expm_known_diagonal_hamiltonian():
    # This case tests _lanczos_expm against an analytical result for
    # the Hermitian diagonal Hamiltonian diag(+1, -1) (Pauli Z).
    # exp(dt * Z) @ [a, b] = [exp(dt)*a, exp(-dt)*b]
    d = 2
    rng = np.random.RandomState(72)
    state = (rng.randn(1, d, 1) + 1j * rng.randn(1, d, 1)).astype(np.complex128)
    env_left, env_right = _trivial_envs()
    core_mpo = _diagonal_mpo_core([1.0, -1.0])
    dt = 0.05
    result = _lanczos_expm(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        1e-12,
    )
    expected = state.copy()
    expected[0, 0, 0] *= np.exp(dt)
    expected[0, 1, 0] *= np.exp(-dt)
    np.testing.assert_allclose(result, expected, atol=1e-10)


# ------------------------------------------------------------
# TEST: Convergence on rank-1 effective Hamiltonian
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_lanczos_expm_early_termination():
    # This case tests that the convergence check terminates iteration
    # after one step for the identity MPO (1D Krylov space): β_1 = 0
    # → the error estimate is exactly zero → conv_tol check passes
    # immediately. Result must equal exp(dt) * state.
    d = 3
    state = np.random.RandomState(73).randn(1, d, 1) + 0j
    env_left, env_right = _trivial_envs()
    core_mpo = _identity_mpo_core(d)
    dt = 0.02
    matvec_heff = lambda v: _apply_heff_site(v, env_left, core_mpo, env_right)  # noqa: E731
    result = _lanczos_expm(state, matvec_heff, dt, 1e-12)
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-10)


# ------------------------------------------------------------
# TEST: _lanczos_expm returns zeros for zero-vector input
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_lanczos_expm_zero_vector():
    # Analytical: if input is zero, output must be zero (beta < tol path)
    phys_dims = [2, 3]
    list_cores_mpo = _make_list_mpo(phys_dims)
    L = np.ones((1, 1, 1), dtype=np.complex128)
    R = np.ones((1, 1, 1), dtype=np.complex128)
    W = list_cores_mpo[0]
    v = np.zeros((1, phys_dims[0], 1), dtype=np.complex128)
    matvec = lambda x, _L=L, _W=W, _R=R: _apply_heff_site(x, _L, _W, _R)  # noqa: E731
    result = _lanczos_expm(v, matvec, 0.01, 1e-12)
    np.testing.assert_allclose(result, 0.0, atol=1e-14)


# ============================================================
# TEST SUITE: _solve_local() — dispatch branches
# ============================================================


# ------------------------------------------------------------
# TEST: Lanczos dispatch produces correct result
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_solve_local_lanczos_dispatch():
    # This case tests that solver='lanczos' dispatches to _lanczos_expm
    # and produces exp(dt * I) * state = exp(dt) * state with identity operator.
    d = 2
    state = np.random.RandomState(80).randn(1, d, 1) + 0j
    env_left, env_right = _trivial_envs()
    core_mpo = _identity_mpo_core(d)
    dt = 0.05
    result = _solve_local(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        solver='lanczos',
    )
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-10)


# ------------------------------------------------------------
# TEST: IVP dispatch produces correct result
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_solve_local_ivp_dispatch():
    # This case tests that solver='ivp' dispatches to _ivp_solve
    # and produces exp(dt) * state with identity operator.
    d = 2
    state = np.random.RandomState(81).randn(1, d, 1) + 0j
    dt = 0.05
    result = _solve_local(state, lambda v: v, dt, solver='ivp')
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-6)


# ------------------------------------------------------------
# TEST: Unknown solver raises ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_solve_local_unknown_solver_raises():
    # This case tests that an invalid solver string raises ValueError.
    d = 2
    state = np.random.RandomState(82).randn(1, d, 1) + 0j
    with pytest.raises(ValueError, match='Unknown solver'):
        _solve_local(state, lambda v: v, 0.01, solver='bogus')


# ============================================================
# TEST SUITE: _ivp_solve() — negative dt and custom kwargs
# ============================================================


# ------------------------------------------------------------
# TEST: Negative dt gives backward integration
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_ivp_solve_negative_dt():
    # This case tests that dt < 0 activates the sign = -1.0 branch,
    # giving dy/dt = -y → y(|dt|) = exp(-|dt|) * y(0), equivalent
    # to exp(dt) * y(0) with dt < 0.
    state = np.array([1.0 + 0j, 0.5 + 0j])
    dt = -0.1
    result = _ivp_solve(state, lambda v: v, dt)
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-6)


# ------------------------------------------------------------
# TEST: Custom kwargs forwarded to solve_ivp
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_ivp_solve_custom_kwargs():
    # This case tests that method, rtol, atol kwargs are forwarded
    # to solve_ivp. Uses RK45 with tighter tolerances.
    state = np.array([1.0 + 0j, 0.5 + 0j])
    dt = 0.1
    result = _ivp_solve(state, lambda v: v, dt, method='RK45', rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-8)


# ============================================================
# TEST SUITE: _split_svd() — edge cases
# ============================================================


# ------------------------------------------------------------
# TEST: Non-square middle axis raises AssertionError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_svd_nonsquare_middle_axis_raises():
    # This case tests that a middle axis that is not a perfect square
    # (d1 != d2) raises ValueError when d1 is not provided.
    theta = np.random.RandomState(83).randn(1, 6, 1) + 0j
    with pytest.raises(ValueError, match='not a perfect square'):
        _split_svd(theta, 0, 0.0)


# ------------------------------------------------------------
# TEST: Non-square middle axis succeeds when d1 is provided
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_svd_nonsquare_with_d1():
    # This case tests that _split_svd handles d1 != d2 when d1 is
    # explicitly provided. This is required for tensor HOPS where
    # adjacent MPS sites have different physical dimensions.
    np.random.seed(84)
    d1, d2, Dl, Dr = 2, 3, 4, 5
    theta_4d = np.random.randn(Dl, d1, d2, Dr) + 1j * np.random.randn(Dl, d1, d2, Dr)
    theta = theta_4d.reshape(Dl, d1 * d2, Dr)
    U, S, Vt, chi_new = _split_svd(theta, 0, 0.0, d1=d1)
    assert U.shape[0] == Dl
    assert U.shape[1] == d1
    assert Vt.shape[1] == d2
    assert Vt.shape[2] == Dr
    # Verify recovery: U @ diag(S) @ Vt ≈ theta
    recovered = np.einsum('ijk,k,klm->ijlm', U, S, Vt).reshape(Dl, d1 * d2, Dr)
    np.testing.assert_allclose(recovered, theta, atol=1e-12,
                               err_msg='SVD with d1 provided does not recover input')


# ------------------------------------------------------------
# TEST: All SVs below eps still keeps chi_new=1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_svd_all_svs_below_eps_floor():
    # This case tests that when all singular values fall below eps,
    # chi_new is floored to 1 (max(0, 1) = 1) rather than 0.
    d = 2
    mat = np.diag([0.001, 0.002]).astype(np.complex128)
    theta = mat.reshape(1, d * d, 1)
    U, S, Vt, chi_new = _split_svd(theta, 0, 1.0)
    assert chi_new == 1, f'Expected chi_new=1 (floor), got {chi_new}'
    assert U.shape == (1, d, 1)
    assert Vt.shape == (1, d, 1)
    # Analytical: when floored to chi=1, the retained SV should be the largest
    assert S[0] == pytest.approx(0.002, abs=1e-14), (
        'Retained SV should be the largest (0.002)'
    )


# ============================================================
# TEST SUITE: timestep() — unknown method
# ============================================================


# ------------------------------------------------------------
# TEST: Unknown method raises ValueError
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_unknown_method_raises():
    # This case tests that an invalid method string raises ValueError.
    phys_dims = [2, 2]
    list_cores = _make_list_cores(phys_dims, seed=84)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(list_cores, list_cores_mpo)
    with pytest.raises(ValueError, match='Unknown method'):
        timestep(
            0.01,
            L0,
            list_envs_R,
            list_cores_mpo,
            core_M,
            list_cores_B,
            method='3tdvp',
        )


# ============================================================
# TEST SUITE: initialize() — correctness
# ============================================================


# ------------------------------------------------------------
# TEST: B list_cores are right-orthogonal after initialization
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_b_cores_right_orthogonal():
    # This case tests that each B core returned by initialize() satisfies
    # the right-orthogonality condition: Q @ Q† = I.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=85)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(list_cores, list_cores_mpo)
    for idx, B in enumerate(list_cores_B):
        chi_l, d, Dr = B.shape
        Q_mat = B.reshape(chi_l, d * Dr)
        np.testing.assert_allclose(
            Q_mat @ Q_mat.conj().T,
            np.eye(chi_l),
            atol=1e-12,
            err_msg=f'B core {idx} is not right-orthogonal',
        )


# ------------------------------------------------------------
# TEST: Norm preserved through initialization
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_norm_preserved():
    # This case tests that the MPS norm is preserved through the
    # gauge transformation in initialize(). Contracts all list_cores
    # before and after and compares ⟨ψ|ψ⟩.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=86)
    list_cores_mpo = _make_list_mpo(phys_dims)

    # norm before
    state_before = list_cores[0]
    for c in list_cores[1:]:
        state_before = np.tensordot(state_before, c, axes=([-1], [0]))
    norm_sq_before = np.vdot(state_before, state_before)

    # norm after
    core_M, list_cores_B, L0, list_envs_R = initialize(list_cores, list_cores_mpo)
    all_cores = [core_M] + list(list_cores_B)
    state_after = all_cores[0]
    for c in all_cores[1:]:
        state_after = np.tensordot(state_after, c, axes=([-1], [0]))
    norm_sq_after = np.vdot(state_after, state_after)

    np.testing.assert_allclose(
        norm_sq_after,
        norm_sq_before,
        atol=1e-10,
        err_msg='Norm not preserved through initialization',
    )


# ------------------------------------------------------------
# TEST: Environment tensors consistent with contract_right
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_envs_consistent_with_contract_right():
    # This case tests that each right environment R_j returned by
    # initialize() equals contract_right(R_{j+1}, W_j, B_j), verifying
    # the environment sweep is self-consistent.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=88)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores, list_cores_mpo,
    )
    n_sites = len(phys_dims)
    # list_envs_R[n_sites-1] = R_{L+1} = boundary (1,1,1)
    # list_envs_R[j-1] = R_j for j=2..L+1
    # Verify: list_envs_R[j-1] == contract_right(list_envs_R[j], W_j, B_{j-1})
    for j in range(n_sites - 1, 0, -1):
        R_next = list_envs_R[j]
        W_j = list_cores_mpo[j]
        B_j = list_cores_B[j - 1]
        R_expected = contract_right(R_next, W_j, B_j)
        np.testing.assert_allclose(
            list_envs_R[j - 1],
            R_expected,
            atol=1e-12,
            err_msg=f'Environment R_{j} inconsistent with contract_right',
        )


# ============================================================
# TEST SUITE: L=2 boundary condition
# ============================================================


# ------------------------------------------------------------
# TEST: 1TDVP timestep with L=2
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_1tdvp_l2():
    # This case tests that 1TDVP runs without error on L=2 and
    # produces correct output structure.
    phys_dims = [2, 2]
    list_cores = _make_list_cores(phys_dims, seed=88)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(list_cores, list_cores_mpo)
    norm_before = _mps_norm(list_cores)
    core_M, list_cores_B, L0, list_envs_R = timestep(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        method='1tdvp',
    )
    all_cores = [core_M] + list(list_cores_B)
    assert len(all_cores) == 2
    assert all_cores[0].shape[0] == 1, 'Left boundary bond dim should be 1'
    assert all_cores[-1].shape[2] == 1, 'Right boundary bond dim should be 1'
    # Invariant: TDVP preserves the norm of the state
    norm_after = _mps_norm(all_cores)
    np.testing.assert_allclose(
        norm_after, norm_before, atol=1e-10,
        err_msg='1TDVP should preserve norm',
    )


# ------------------------------------------------------------
# TEST: 2TDVP timestep with L=2 and R_2 fixup
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_2tdvp_l2_r2_fixup():
    # This case tests that 2TDVP on L=2 exercises the R_2 fixup branch
    # in sweep_left_2tdvp (source:1100-1102). Verifies list_envs_R[0]
    # is not None and has the correct shape after the timestep.
    phys_dims = [2, 2]
    list_cores = _make_list_cores(phys_dims, seed=89)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(list_cores, list_cores_mpo)
    core_M, list_cores_B, L0, list_envs_R = timestep(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        method='2tdvp',
    )
    all_cores = [core_M] + list(list_cores_B)
    assert len(all_cores) == 2
    assert all_cores[0].shape[0] == 1, 'Left boundary bond dim should be 1'
    assert all_cores[-1].shape[2] == 1, 'Right boundary bond dim should be 1'
    # Verify R_2 fixup: list_envs_R[0] must not be None
    assert list_envs_R[0] is not None, 'R_2 fixup failed: list_envs_R[0] is None'
    assert list_envs_R[0].ndim == 3, 'R_2 should be a rank-3 environment tensor'


# ============================================================
# TEST SUITE: timestep() — alternate solvers
# ============================================================


# ------------------------------------------------------------
# TEST: 1TDVP with lanczos solver
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_1tdvp_lanczos_solver():
    # This case tests that the lanczos solver propagates through
    # the full timestep pipeline without error.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=90)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(list_cores, list_cores_mpo)
    norm_before = _mps_norm(list_cores)
    core_M, list_cores_B, L0, list_envs_R = timestep(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        method='1tdvp',
        solver='lanczos',
    )
    all_cores = [core_M] + list(list_cores_B)
    assert len(all_cores) == len(phys_dims)
    assert all_cores[0].shape[0] == 1, 'Left boundary bond dim should be 1'
    assert all_cores[-1].shape[2] == 1, 'Right boundary bond dim should be 1'
    # Invariant: TDVP preserves the norm of the state
    norm_after = _mps_norm(all_cores)
    np.testing.assert_allclose(
        norm_after, norm_before, atol=1e-10,
        err_msg='1TDVP should preserve norm',
    )


# ------------------------------------------------------------
# TEST: 1TDVP with ivp solver raises on complex dt
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_1tdvp_ivp_solver_complex_dt_raises():
    # This case tests that the ivp solver cannot handle the complex
    # dt = -1j * delta that the TDVP sweeps pass to _solve_local.
    # _ivp_solve compares dt >= 0 which fails for complex numbers.
    # This documents a known limitation of the ivp backend.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=91)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(list_cores, list_cores_mpo)
    with pytest.raises(TypeError):
        timestep(
            0.01,
            L0,
            list_envs_R,
            list_cores_mpo,
            core_M,
            list_cores_B,
            method='1tdvp',
            solver='ivp',
        )


# ============================================================
# TEST SUITE: 2TDVP chi_max truncation
# ============================================================


# ------------------------------------------------------------
# TEST: chi_max caps bond dimension
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_2tdvp_chi_max_caps_bond():
    # This case tests that chi_max=2 limits all output bond dimensions
    # to at most 2, even when the input MPS has bond dimension 4.
    d = 2
    rng = np.random.RandomState(92)
    # Build chi=4 MPS: (1,d,4), (4,d,4), (4,d,1)
    list_cores = [
        rng.randn(1, d, 4) + 1j * rng.randn(1, d, 4),
        rng.randn(4, d, 4) + 1j * rng.randn(4, d, 4),
        rng.randn(4, d, 1) + 1j * rng.randn(4, d, 1),
    ]
    list_cores_mpo = _make_list_mpo([d, d, d])
    core_M, list_cores_B, L0, list_envs_R = initialize(list_cores, list_cores_mpo)
    core_M, list_cores_B, L0, list_envs_R = timestep(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        method='2tdvp',
        chi_max=2,
    )
    all_cores = [core_M] + list(list_cores_B)
    for idx, c in enumerate(all_cores):
        Dl, d_phys, Dr = c.shape
        assert Dl <= 2, f'Core {idx}: left bond dim {Dl} exceeds chi_max=2'
        assert Dr <= 2, f'Core {idx}: right bond dim {Dr} exceeds chi_max=2'


# ============================================================
# TEST SUITE: Complex dt for Krylov solvers
# ============================================================


# ------------------------------------------------------------
# TEST: _arnoldi_expm with imaginary dt
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_imaginary_dt():
    # This case tests _arnoldi_expm with dt = -0.05j (the actual TDVP
    # use case). For diagonal H = diag(+1, -1):
    # exp(-0.05j * H) @ [a, b] = [exp(-0.05j)*a, exp(+0.05j)*b]
    # This is a unitary rotation — magnitudes preserved, phases shifted.
    d = 2
    rng = np.random.RandomState(93)
    state = (rng.randn(1, d, 1) + 1j * rng.randn(1, d, 1)).astype(np.complex128)
    env_left, env_right = _trivial_envs()
    core_mpo = _diagonal_mpo_core([1.0, -1.0])
    dt = -0.05j
    result = _arnoldi_expm(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        1e-12,
    )
    expected = state.copy()
    expected[0, 0, 0] *= np.exp(dt)
    expected[0, 1, 0] *= np.exp(-dt)
    np.testing.assert_allclose(result, expected, atol=1e-10)
    # Verify magnitudes preserved (unitary evolution)
    np.testing.assert_allclose(
        np.abs(result),
        np.abs(state),
        atol=1e-10,
        err_msg='Imaginary dt should preserve magnitudes',
    )


# ------------------------------------------------------------
# TEST: _lanczos_expm with imaginary dt
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_lanczos_expm_imaginary_dt():
    # This case tests _lanczos_expm with dt = -0.05j for the same
    # Hermitian diagonal H = diag(+1, -1). Same analytical result.
    d = 2
    rng = np.random.RandomState(94)
    state = (rng.randn(1, d, 1) + 1j * rng.randn(1, d, 1)).astype(np.complex128)
    env_left, env_right = _trivial_envs()
    core_mpo = _diagonal_mpo_core([1.0, -1.0])
    dt = -0.05j
    result = _lanczos_expm(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        1e-12,
    )
    expected = state.copy()
    expected[0, 0, 0] *= np.exp(dt)
    expected[0, 1, 0] *= np.exp(-dt)
    np.testing.assert_allclose(result, expected, atol=1e-10)
    # Verify magnitudes preserved (unitary evolution)
    np.testing.assert_allclose(
        np.abs(result),
        np.abs(state),
        atol=1e-10,
        err_msg='Imaginary dt should preserve magnitudes',
    )


# ============================================================
# TEST SUITE: _apply_heff_site() — non-trivial MPO
# ============================================================


# ------------------------------------------------------------
# TEST: Diagonal MPO applies eigenvalues to input
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_heff_site_diagonal_mpo_applies_eigenvalues():
    # This case tests that _apply_heff_site with a diagonal MPO diag(2, -1)
    # and trivial environments gives H|v⟩ = [2*a, -1*b].
    d = 2
    Dl, Dr = 1, 1
    state = np.array([[[3.0 + 1j]], [[0.5 - 2j]]], dtype=np.complex128)
    state = state.reshape(Dl, d, Dr)
    env_left, env_right = _trivial_envs()
    core_mpo = _diagonal_mpo_core([2.0, -1.0])
    result = _apply_heff_site(state, env_left, core_mpo, env_right)
    expected = state.copy()
    expected[0, 0, 0] *= 2.0
    expected[0, 1, 0] *= -1.0
    np.testing.assert_allclose(
        result, expected, atol=1e-12, err_msg='Diagonal MPO should scale each component'
    )


# ============================================================
# TEST SUITE: _apply_heff_bond() — non-trivial environments
# ============================================================


# ------------------------------------------------------------
# TEST: Scaled environments multiply bond tensor
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_heff_bond_scaled_envs():
    # This case tests that _apply_heff_bond with L=[[[3.0]]] and R=[[[2.0]]]
    # gives HC = 3 * 2 * C = 6 * C. Verifies both environments
    # contribute multiplicatively.
    H2_bond = np.array([[1.5 + 0.5j]], dtype=np.complex128)
    env_left_scaled = np.array([[[3.0]]], dtype=np.complex128)
    env_right_scaled = np.array([[[2.0]]], dtype=np.complex128)
    result = _apply_heff_bond(H2_bond, env_left_scaled, env_right_scaled)
    np.testing.assert_allclose(
        result,
        6.0 * H2_bond,
        atol=1e-12,
        err_msg='Scaled envs should multiply bond tensor',
    )


# ============================================================
# TEST SUITE: _apply_heff_bond() — multi-bond-dimension
# ============================================================


# ------------------------------------------------------------
# TEST: Bond contraction with chi > 1 and w > 1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_heff_bond_multi_bond_dim():
    # This case tests that _apply_heff_bond contracts correctly when
    # the bond tensor and environments have dimensions > 1.
    # Uses chi_l=2, chi_r=3, w=2 and verifies against direct einsum.
    chi_l, chi_r, w = 2, 3, 2
    rng = np.random.RandomState(100)
    H2_bond = rng.randn(chi_l, chi_r) + 1j * rng.randn(chi_l, chi_r)
    env_left = rng.randn(chi_l, w, chi_l) + 1j * rng.randn(chi_l, w, chi_l)
    env_right = rng.randn(chi_r, w, chi_r) + 1j * rng.randn(chi_r, w, chi_r)
    # Compute via function
    result = _apply_heff_bond(H2_bond, env_left, env_right)
    # Compute via direct einsum for reference
    expected = np.einsum('ijk,ljm,km->il', env_left, env_right, H2_bond)
    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-12,
        err_msg='_apply_heff_bond with chi>1 does not match einsum',
    )


# ============================================================
# TEST SUITE: _apply_heff_twosite() — non-trivial MPO
# ============================================================


# ------------------------------------------------------------
# TEST: Diagonal MPO scales two-site tensor by eigenvalues
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_heff_twosite_diagonal_mpo():
    # This case tests _apply_heff_twosite with diagonal MPO diag(2, -1)
    # on both sites. With trivial environments, the effective operator
    # is diag(2,-1) ⊗ diag(2,-1), so each (s1,s2) component is
    # scaled by eigenvalue[s1] * eigenvalue[s2].
    d = 2
    eigenvalues = [2.0, -1.0]
    rng = np.random.RandomState(101)
    # Build theta as (1, d*d, 1) with known components
    theta = rng.randn(1, d * d, 1) + 1j * rng.randn(1, d * d, 1)
    core_mpo_site1 = _diagonal_mpo_core(eigenvalues)
    core_mpo_site2 = _diagonal_mpo_core(eigenvalues)
    env_left = np.ones((1, 1, 1), dtype=np.complex128)
    env_right = np.ones((1, 1, 1), dtype=np.complex128)
    # Compute result
    result = _apply_heff_twosite(
        theta,
        env_left,
        core_mpo_site1,
        core_mpo_site2,
        env_right,
    )
    # Build expected: each (s1, s2) scaled by eigenvalues[s1]*eigenvalues[s2]
    theta_4d = theta.reshape(1, d, d, 1)
    expected_4d = theta_4d.copy()
    for s1 in range(d):
        for s2 in range(d):
            expected_4d[0, s1, s2, 0] *= eigenvalues[s1] * eigenvalues[s2]
    expected = expected_4d.reshape(1, d * d, 1)
    np.testing.assert_allclose(result, expected, atol=1e-12)


# ------------------------------------------------------------
# TEST: Two-site contraction with chi > 1 and w > 1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_heff_twosite_multi_bond_dim():
    # This case tests _apply_heff_twosite with non-trivial bond and MPO
    # dimensions: Dl=2, Dr=3, d=2 (so d*d=4), w_l=2, w_m=2, w_r=2.
    # Verifies against direct einsum.
    Dl, Dr, d = 2, 3, 2
    w_l, w_m, w_r = 2, 2, 2
    rng = np.random.RandomState(201)
    theta = rng.randn(Dl, d * d, Dr) + 1j * rng.randn(Dl, d * d, Dr)
    L = rng.randn(Dl, w_l, Dl) + 1j * rng.randn(Dl, w_l, Dl)
    W1 = rng.randn(w_l, d, d, w_m) + 1j * rng.randn(w_l, d, d, w_m)
    W2 = rng.randn(w_m, d, d, w_r) + 1j * rng.randn(w_m, d, d, w_r)
    R = rng.randn(Dr, w_r, Dr) + 1j * rng.randn(Dr, w_r, Dr)
    result = _apply_heff_twosite(theta, L, W1, W2, R)
    # Direct einsum reference
    theta_4d = theta.reshape(Dl, d, d, Dr)
    expected_4d = np.einsum(
        'ijk,jlmn,nopq,rqs,kmps->ilor',
        L, W1, W2, R, theta_4d,
    )
    expected = expected_4d.reshape(Dl, d * d, Dr)
    np.testing.assert_allclose(
        result, expected, atol=1e-12,
        err_msg='_apply_heff_twosite with chi>1/w>1 does not match einsum',
    )


# ------------------------------------------------------------
# TEST: Two-site contraction with non-uniform physical dims (d1 != d2)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_heff_twosite_nonuniform_phys_dims():
    # This case tests _apply_heff_twosite where the two sites have
    # different physical dimensions (d1=2, d2=3), as occurs in tensor
    # HOPS with statenumber representation.
    Dl, Dr = 2, 3
    d1, d2 = 2, 3
    w_l, w_m, w_r = 2, 2, 2
    rng = np.random.RandomState(202)
    theta = rng.randn(Dl, d1 * d2, Dr) + 1j * rng.randn(Dl, d1 * d2, Dr)
    L = rng.randn(Dl, w_l, Dl) + 1j * rng.randn(Dl, w_l, Dl)
    W1 = rng.randn(w_l, d1, d1, w_m) + 1j * rng.randn(w_l, d1, d1, w_m)
    W2 = rng.randn(w_m, d2, d2, w_r) + 1j * rng.randn(w_m, d2, d2, w_r)
    R = rng.randn(Dr, w_r, Dr) + 1j * rng.randn(Dr, w_r, Dr)
    result = _apply_heff_twosite(theta, L, W1, W2, R)
    # Direct einsum reference
    theta_4d = theta.reshape(Dl, d1, d2, Dr)
    expected_4d = np.einsum(
        'ijk,jlmn,nopq,rqs,kmps->ilor',
        L, W1, W2, R, theta_4d,
    )
    expected = expected_4d.reshape(Dl, d1 * d2, Dr)
    np.testing.assert_allclose(
        result, expected, atol=1e-12,
        err_msg='_apply_heff_twosite with d1!=d2 does not match einsum',
    )


# ============================================================
# TEST SUITE: _apply_heff_site() — multi-bond-dimension
# ============================================================


# ------------------------------------------------------------
# TEST: Site contraction with chi > 1 and MPO bond w > 1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_apply_heff_site_multi_bond_dim():
    # This case tests _apply_heff_site with non-trivial bond dimensions:
    # Dl=2, Dr=3, d=2, w_l=2, w_r=2. Verifies against direct einsum.
    Dl, Dr, d, w_l, w_r = 2, 3, 2, 2, 2
    rng = np.random.RandomState(102)
    core_M = rng.randn(Dl, d, Dr) + 1j * rng.randn(Dl, d, Dr)
    env_left = rng.randn(Dl, w_l, Dl) + 1j * rng.randn(Dl, w_l, Dl)
    core_mpo = rng.randn(w_l, d, d, w_r) + 1j * rng.randn(w_l, d, d, w_r)
    env_right = rng.randn(Dr, w_r, Dr) + 1j * rng.randn(Dr, w_r, Dr)
    # Compute via function
    result = _apply_heff_site(core_M, env_left, core_mpo, env_right)
    # Compute via direct einsum
    expected = np.einsum(
        'ijk,jlmn,onp,kmp->ilo',
        env_left,
        core_mpo,
        env_right,
        core_M,
    )
    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-12,
        err_msg='_apply_heff_site with chi>1 does not match einsum',
    )


# ============================================================
# TEST SUITE: _split_svd() — combined truncation
# ============================================================


# ------------------------------------------------------------
# TEST: chi_max and eps both active simultaneously
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_split_svd_chi_max_and_eps_combined():
    # This case tests that chi_max AND eps both constrain together.
    # SVs = [10, 5, 0.01]. With eps=0.5 → keeps [10, 5] (drops 0.01).
    # With chi_max=1 → keeps only [10]. Both active → chi_new=1.
    d = 3
    H2_diag = np.diag([10.0, 5.0, 0.01]).astype(np.complex128)
    theta = H2_diag.reshape(1, d * d, 1)
    U, S, Vt, chi_new = _split_svd(theta, chi_max=1, eps=0.5)
    # chi_max=1 is the binding constraint
    assert chi_new == 1
    np.testing.assert_allclose(S[0], 10.0, atol=1e-12)

    # Now chi_max=2, eps=0.5 → eps drops 0.01, chi_max allows 2 → chi_new=2
    U, S, Vt, chi_new = _split_svd(theta, chi_max=2, eps=0.5)
    assert chi_new == 2
    np.testing.assert_allclose(S[0], 10.0, atol=1e-12)
    np.testing.assert_allclose(S[1], 5.0, atol=1e-12)


# ============================================================
# TEST SUITE: _ivp_solve() — rank-3 input and max_step
# ============================================================


# ------------------------------------------------------------
# TEST: Rank-3 input reshape round-trip
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_ivp_solve_rank3_input():
    # This case tests that _ivp_solve correctly handles rank-3 tensors
    # by flattening to 1D, integrating, and reshaping back.
    # dy/dt = y → y(t) = exp(t) * y(0).
    d = 3
    state = np.random.RandomState(103).randn(1, d, 1) + 0j
    dt = 0.05
    result = _ivp_solve(state, lambda v: v, dt)
    assert result.shape == (1, d, 1), 'Output shape must match input'
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-6)


# ============================================================
# TEST SUITE: _arnoldi_expm() — negative real dt
# ============================================================


# ------------------------------------------------------------
# TEST: Negative real dt shrinks state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_negative_real_dt():
    # This case tests exp(-|dt| * I) * state = exp(-|dt|) * state.
    # With identity MPO and trivial envs, the operator is I.
    d = 2
    rng = np.random.RandomState(104)
    state = (rng.randn(1, d, 1) + 1j * rng.randn(1, d, 1)).astype(
        np.complex128,
    )
    env_left, env_right = _trivial_envs()
    core_mpo = _identity_mpo_core(d)
    dt = -0.05
    result = _arnoldi_expm(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        1e-12,
    )
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-10)


# ============================================================
# TEST SUITE: _lanczos_expm() — negative real dt
# ============================================================


# ------------------------------------------------------------
# TEST: Negative real dt shrinks state
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_lanczos_expm_negative_real_dt():
    # This case tests exp(-|dt| * I) * state = exp(-|dt|) * state.
    d = 2
    rng = np.random.RandomState(105)
    state = (rng.randn(1, d, 1) + 1j * rng.randn(1, d, 1)).astype(
        np.complex128,
    )
    env_left, env_right = _trivial_envs()
    core_mpo = _identity_mpo_core(d)
    dt = -0.05
    result = _lanczos_expm(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        1e-12,
    )
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-10)


# ============================================================
# TEST SUITE: _solve_local() — custom kwargs passthrough
# ============================================================


# ------------------------------------------------------------
# TEST: conv_tol forwarded through arnoldi dispatch
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_solve_local_custom_conv_tol():
    # This case tests that conv_tol propagates through _solve_local to
    # _arnoldi_expm. With identity MPO the result must be exp(dt) * state
    # regardless of conv_tol (convergence is trivial for rank-1 operators).
    d = 2
    state = np.random.RandomState(106).randn(1, d, 1) + 0j
    env_left, env_right = _trivial_envs()
    core_mpo = _identity_mpo_core(d)
    dt = 0.05
    result = _solve_local(
        state,
        lambda v: _apply_heff_site(v, env_left, core_mpo, env_right),
        dt,
        solver='arnoldi',
        conv_tol=1e-8,
    )
    np.testing.assert_allclose(result, np.exp(dt) * state, atol=1e-8)


# ============================================================
# TEST SUITE: initialize() — boundary and bond-dim cases
# ============================================================


# ------------------------------------------------------------
# TEST: L=1 single-site chain
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_l1_single_site():
    # This case tests the degenerate L=1 case. With one site:
    # - list_cores_B should be empty
    # - list_envs_R should have 1 element (the boundary R_2 = [[1]])
    # - core_M equals the input core (no RQ sweep needed)
    phys_dims = [3]
    list_cores = _make_list_cores(phys_dims, seed=107)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    assert len(list_cores_B) == 0, 'L=1 should have no B cores'
    assert len(list_envs_R) == 1, 'L=1 should have 1 env (boundary)'
    # core_M should equal the input (no gauge transform needed)
    np.testing.assert_allclose(core_M, list_cores[0], atol=1e-12)
    # Boundary environment should be [[1]]
    np.testing.assert_allclose(
        list_envs_R[0],
        np.ones((1, 1, 1)),
        atol=1e-12,
    )


# ------------------------------------------------------------
# TEST: Higher bond-dimension MPS
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_initialize_high_bond_dim():
    # This case tests initialize with a 3-site MPS with bond dims > 1:
    # (1,d,3), (3,d,2), (2,d,1). Verifies norm preservation and
    # B cores right-orthogonal after the RQ gauge sweep.
    d = 2
    rng = np.random.RandomState(108)
    list_cores = [
        rng.randn(1, d, 3) + 1j * rng.randn(1, d, 3),
        rng.randn(3, d, 2) + 1j * rng.randn(3, d, 2),
        rng.randn(2, d, 1) + 1j * rng.randn(2, d, 1),
    ]
    list_cores_mpo = _make_list_mpo([d, d, d])
    # Compute norm before
    state_before = list_cores[0]
    for c in list_cores[1:]:
        state_before = np.tensordot(state_before, c, axes=([-1], [0]))
    norm_sq_before = np.vdot(state_before, state_before)
    # Initialize
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    # Verify norm preservation
    all_cores = [core_M] + list(list_cores_B)
    state_after = all_cores[0]
    for c in all_cores[1:]:
        state_after = np.tensordot(state_after, c, axes=([-1], [0]))
    norm_sq_after = np.vdot(state_after, state_after)
    np.testing.assert_allclose(
        norm_sq_after,
        norm_sq_before,
        atol=1e-10,
        err_msg='Norm not preserved for high bond-dim MPS',
    )
    # Verify B cores are right-orthogonal
    for idx, B in enumerate(list_cores_B):
        chi_l, d_phys, Dr = B.shape
        Q_mat = B.reshape(chi_l, d_phys * Dr)
        np.testing.assert_allclose(
            Q_mat @ Q_mat.conj().T,
            np.eye(chi_l),
            atol=1e-12,
            err_msg=f'B core {idx} not right-orthogonal',
        )


# ============================================================
# TEST SUITE: timestep() — delta=0, eps, diagonal MPO
# ============================================================


# ------------------------------------------------------------
# TEST: delta=0 returns state unchanged
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_delta_zero_unchanged():
    # This case tests that timestep with delta=0 returns the
    # state and environments unchanged (exp(0) = I).
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=109)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    # Save copies of the initial state
    core_M_before = core_M.copy()
    list_B_before = [b.copy() for b in list_cores_B]
    # Timestep with delta=0
    core_M_out, list_B_out, L0_out, list_R_out = timestep(
        0.0,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        method='1tdvp',
    )
    # Core M should be unchanged
    np.testing.assert_allclose(core_M_out, core_M_before, atol=1e-12)
    # B cores should be unchanged
    for idx in range(len(list_B_before)):
        np.testing.assert_allclose(
            list_B_out[idx],
            list_B_before[idx],
            atol=1e-12,
            err_msg=f'B core {idx} changed with delta=0',
        )


# ------------------------------------------------------------
# TEST: 2TDVP with eps > 0 truncates small singular values
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_2tdvp_eps_truncation():
    # This case tests that eps > 0 in 2TDVP triggers SVD truncation.
    # Uses a high bond-dim MPS so there are singular values to drop.
    d = 2
    rng = np.random.RandomState(110)
    list_cores = [
        rng.randn(1, d, 4) + 1j * rng.randn(1, d, 4),
        rng.randn(4, d, 4) + 1j * rng.randn(4, d, 4),
        rng.randn(4, d, 1) + 1j * rng.randn(4, d, 1),
    ]
    list_cores_mpo = _make_list_mpo([d, d, d])
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    # Run with large eps to force aggressive truncation
    core_M_out, list_B_out, L0_out, list_R_out = timestep(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        method='2tdvp',
        eps=1.0,
    )
    # With eps=1.0 many SVs should be dropped, reducing bond dims
    all_cores = [core_M_out] + list(list_B_out)
    max_bond = max(max(c.shape[0], c.shape[2]) for c in all_cores)
    assert max_bond < 4, f'eps=1.0 should truncate bond dim below 4, got {max_bond}'


# ------------------------------------------------------------
# TEST: Diagonal MPO through full 1TDVP pipeline
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_1tdvp_diagonal_mpo():
    # This case tests the full pipeline with a non-identity Hamiltonian.
    # Uses diagonal on-site energies diag(ε₁, ε₂) on each site.
    # After time δ, each component picks up exp(-i δ εₖ) per site.
    d = 2
    delta = 0.01
    eigenvalues = [1.0, -0.5]
    # Build product state: site 1 in state 0, site 2 in state 1
    core_site1 = np.zeros((1, d, 1), dtype=np.complex128)
    core_site1[0, 0, 0] = 1.0  # state |0⟩
    core_site2 = np.zeros((1, d, 1), dtype=np.complex128)
    core_site2[0, 1, 0] = 1.0  # state |1⟩
    list_cores = [core_site1, core_site2]
    # Diagonal MPO: each site has diag(1.0, -0.5)
    list_cores_mpo = [_diagonal_mpo_core(eigenvalues) for _ in range(2)]
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    # Timestep
    core_M_out, list_B_out, L0_out, list_R_out = timestep(
        delta,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        method='1tdvp',
    )
    # For product state |0⟩⊗|1⟩ under diagonal H:
    # E_total = ε[0] + ε[1] = 1.0 + (-0.5) = 0.5
    # TDVP Strang splitting convention: dt = -1j*delta passed to
    # _solve_local → net evolution is exp(+i delta E_total)
    expected_phase = np.exp(+1j * delta * (eigenvalues[0] + eigenvalues[1]))
    # Contract output MPS to get scalar amplitude
    all_cores_out = [core_M_out] + list(list_B_out)
    psi_out = all_cores_out[0]
    for c in all_cores_out[1:]:
        psi_out = np.tensordot(psi_out, c, axes=([-1], [0]))
    # The (0,1) component should carry the phase
    amplitude = psi_out[0, 0, 1, 0]  # |0⟩⊗|1⟩ component
    np.testing.assert_allclose(
        amplitude,
        expected_phase,
        atol=1e-6,
        err_msg='Diagonal MPO did not produce correct phase',
    )


# ============================================================
# TEST SUITE: timestep() — L=4 multi-iteration 2TDVP
# ============================================================


# ------------------------------------------------------------
# TEST: 2TDVP with L=4 exercises multi-iteration sweep body
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_2tdvp_l4_multi_iteration():
    # This case tests that 2TDVP with L=4 correctly exercises
    # the multi-iteration backward evolution inside both sweeps.
    # Verifies output structure and norm preservation.
    d = 2
    phys_dims = [d, d, d, d]
    list_cores = _make_list_cores(phys_dims, seed=111)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    # Compute norm before
    all_before = [core_M] + list(list_cores_B)
    psi_before = all_before[0]
    for c in all_before[1:]:
        psi_before = np.tensordot(psi_before, c, axes=([-1], [0]))
    norm_sq_before = np.vdot(psi_before, psi_before)
    # Timestep
    core_M_out, list_B_out, L0_out, list_R_out = timestep(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        method='2tdvp',
    )
    # Verify structure: 4 cores, boundary bond dims = 1
    all_out = [core_M_out] + list(list_B_out)
    assert len(all_out) == 4
    assert all_out[0].shape[0] == 1, 'Left boundary should be 1'
    assert all_out[-1].shape[2] == 1, 'Right boundary should be 1'
    # Verify norm preservation (identity MPO → unitary evolution)
    psi_after = all_out[0]
    for c in all_out[1:]:
        psi_after = np.tensordot(psi_after, c, axes=([-1], [0]))
    norm_sq_after = np.vdot(psi_after, psi_after)
    np.testing.assert_allclose(
        norm_sq_after,
        norm_sq_before,
        rtol=1e-6,
        err_msg='Norm not preserved for L=4 2TDVP',
    )


# ============================================================
# TEST SUITE: timestep() — 2TDVP diagonal MPO physics
# ============================================================


# ------------------------------------------------------------
# TEST: 2TDVP diagonal MPO produces correct phase and preserves norm
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_2tdvp_diagonal_mpo_norm():
    # Invariant: norm should be preserved under diagonal MPO evolution
    phys_dims = [2, 2]
    list_cores = _make_list_cores(phys_dims, seed=200)
    list_cores_mpo = []
    for d in phys_dims:
        mpo = np.zeros((1, d, d, 1), dtype=np.complex128)
        for k in range(d):
            mpo[0, k, k, 0] = float(k + 1)
        list_cores_mpo.append(mpo)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores, list_cores_mpo,
    )
    norm_before = _mps_norm([core_M] + list(list_cores_B))
    dt = 0.005
    core_M2, list_cores_B2, L02, list_envs_R2 = timestep(
        dt, L0, list_envs_R, list_cores_mpo, core_M, list_cores_B,
        method='2tdvp', chi_max=4, eps=1e-12,
    )
    # Invariant: norm preserved
    norm_after = _mps_norm([core_M2] + list(list_cores_B2))
    np.testing.assert_allclose(norm_after, norm_before, atol=1e-8)


# ============================================================
# TEST SUITE: timestep() — multi-step norm preservation
# ============================================================


# ------------------------------------------------------------
# TEST: Two consecutive 1TDVP timesteps preserve norm
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_timestep_1tdvp_multi_step_norm():
    # Invariant: norm preserved across multiple steps
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=210)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores, list_cores_mpo,
    )
    norm_init = _mps_norm([core_M] + list(list_cores_B))
    for _ in range(2):
        core_M, list_cores_B, L0, list_envs_R = timestep(
            0.005, L0, list_envs_R, list_cores_mpo,
            core_M, list_cores_B, method='1tdvp',
        )
    norm_final = _mps_norm([core_M] + list(list_cores_B))
    np.testing.assert_allclose(norm_final, norm_init, atol=1e-8)


# ============================================================
# TEST SUITE: sweep_right_1tdvp() — direct tests
# ============================================================


# ------------------------------------------------------------
# TEST: A cores are left-orthogonal after right sweep
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_sweep_right_1tdvp_a_cores_left_orthogonal():
    # This case tests that A cores from sweep_right_1tdvp satisfy
    # the left-orthogonality condition: Q†Q = I.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=112)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    list_envs_L, list_cores_A, core_M_last = sweep_right_1tdvp(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
    )
    # Verify each A core is left-orthogonal
    for idx, A in enumerate(list_cores_A):
        Dl, d, Dr = A.shape
        Q_mat = A.reshape(Dl * d, Dr)
        np.testing.assert_allclose(
            Q_mat.conj().T @ Q_mat,
            np.eye(Dr),
            atol=1e-10,
            err_msg=f'A core {idx} not left-orthogonal after right sweep',
        )


# ------------------------------------------------------------
# TEST: Left environments consistent with A cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_sweep_right_1tdvp_envs_consistent():
    # This case tests that list_envs_L[j+1] = contract_left(
    # list_envs_L[j], W_j, A_j) for each j.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=113)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    list_envs_L, list_cores_A, core_M_last = sweep_right_1tdvp(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
    )
    # Check consistency: L_{j+1} = contract_left(L_j, W_j, A_j)
    for j in range(len(list_cores_A)):
        L_rebuilt = contract_left(
            list_envs_L[j],
            list_cores_mpo[j],
            list_cores_A[j],
        )
        np.testing.assert_allclose(
            list_envs_L[j + 1],
            L_rebuilt,
            atol=1e-10,
            err_msg=f'Left env at j={j + 1} inconsistent with A core',
        )


# ------------------------------------------------------------
# TEST: Norm preserved through right sweep
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_sweep_right_1tdvp_norm_preserved():
    # This case tests that the MPS norm is preserved through the
    # right sweep. Contracts A cores + core_M_last and compares.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=114)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    # Norm before
    all_before = [core_M] + list(list_cores_B)
    psi_before = all_before[0]
    for c in all_before[1:]:
        psi_before = np.tensordot(psi_before, c, axes=([-1], [0]))
    norm_sq_before = np.vdot(psi_before, psi_before)
    # Right sweep
    list_envs_L, list_cores_A, core_M_last = sweep_right_1tdvp(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
    )
    # Norm after: A cores + core_M_last
    all_after = list(list_cores_A) + [core_M_last]
    psi_after = all_after[0]
    for c in all_after[1:]:
        psi_after = np.tensordot(psi_after, c, axes=([-1], [0]))
    norm_sq_after = np.vdot(psi_after, psi_after)
    np.testing.assert_allclose(
        norm_sq_after,
        norm_sq_before,
        rtol=1e-6,
        err_msg='Norm not preserved through right sweep',
    )


# ============================================================
# TEST SUITE: sweep_left_1tdvp() — direct tests
# ============================================================


# ------------------------------------------------------------
# TEST: B cores right-orthogonal and norm preserved
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_sweep_left_1tdvp_b_cores_and_norm():
    # This case tests that sweep_left_1tdvp produces right-orthogonal
    # B cores and preserves the MPS norm.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=115)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    # Norm before
    all_before = [core_M] + list(list_cores_B)
    psi_before = all_before[0]
    for c in all_before[1:]:
        psi_before = np.tensordot(psi_before, c, axes=([-1], [0]))
    norm_sq_before = np.vdot(psi_before, psi_before)
    # Right sweep first (needed as input)
    list_envs_L, list_cores_A, core_M_last = sweep_right_1tdvp(
        0.005,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
    )
    R_boundary = list_envs_R[-1]
    # Left sweep
    core_M_first, list_B_out, list_R_out = sweep_left_1tdvp(
        0.005,
        list_envs_L,
        R_boundary,
        list_cores_mpo,
        list_cores_A,
        core_M_last,
    )
    # Verify B cores are right-orthogonal
    for idx, B in enumerate(list_B_out):
        chi_l, d, Dr = B.shape
        Q_mat = B.reshape(chi_l, d * Dr)
        np.testing.assert_allclose(
            Q_mat @ Q_mat.conj().T,
            np.eye(chi_l),
            atol=1e-10,
            err_msg=f'B core {idx} not right-orthogonal after left sweep',
        )
    # Verify norm preserved through full right+left sweep
    all_after = [core_M_first] + list(list_B_out)
    psi_after = all_after[0]
    for c in all_after[1:]:
        psi_after = np.tensordot(psi_after, c, axes=([-1], [0]))
    norm_sq_after = np.vdot(psi_after, psi_after)
    np.testing.assert_allclose(
        norm_sq_after,
        norm_sq_before,
        rtol=1e-6,
        err_msg='Norm not preserved through right+left sweep',
    )


# ------------------------------------------------------------
# TEST: sweep_left_1tdvp with lanczos solver runs without error
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_sweep_left_1tdvp_lanczos_solver():
    # This case tests that sweep_left_1tdvp works with the lanczos
    # solver (non-default). Identity MPO makes the problem Hermitian,
    # which is the Lanczos requirement.
    phys_dims = [2, 3, 2]
    list_cores = _make_list_cores(phys_dims, seed=116)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores, list_cores_mpo,
    )
    list_envs_L, list_cores_A, core_M_last = sweep_right_1tdvp(
        0.005, L0, list_envs_R, list_cores_mpo, core_M, list_cores_B,
        solver='lanczos',
    )
    R_boundary = list_envs_R[-1]
    core_M_first, list_B_out, list_R_out = sweep_left_1tdvp(
        0.005, list_envs_L, R_boundary, list_cores_mpo,
        list_cores_A, core_M_last, solver='lanczos',
    )
    # Basic sanity: output shapes match input
    assert core_M_first.shape[1] == phys_dims[0]
    assert len(list_B_out) == len(phys_dims) - 1
    # Invariant: B cores from left sweep should be right-orthogonal
    for i, B in enumerate(list_B_out):
        Dl, d, Dr = B.shape
        M = B.reshape(Dl, d * Dr)
        eye_check = M @ M.conj().T
        np.testing.assert_allclose(
            eye_check, np.eye(Dl), atol=1e-10,
            err_msg=f'B core {i} not right-orthogonal after Lanczos left sweep',
        )


# ============================================================
# TEST SUITE: sweep_right_2tdvp() / sweep_left_2tdvp()
# ============================================================


# ------------------------------------------------------------
# TEST: 2TDVP right sweep produces left-orthogonal A cores
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_sweep_right_2tdvp_a_cores_left_orthogonal():
    # This case tests that A cores from sweep_right_2tdvp satisfy
    # the left-orthogonality condition.
    d = 2
    phys_dims = [d, d, d]
    list_cores = _make_list_cores(phys_dims, seed=116)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    list_envs_L, list_cores_A, core_M_last = sweep_right_2tdvp(
        0.01,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        chi_max=0,
        eps=0.0,
    )
    for idx, A in enumerate(list_cores_A):
        Dl, d_phys, Dr = A.shape
        Q_mat = A.reshape(Dl * d_phys, Dr)
        np.testing.assert_allclose(
            Q_mat.conj().T @ Q_mat,
            np.eye(Dr),
            atol=1e-10,
            err_msg=f'A core {idx} not left-orthogonal (2TDVP)',
        )


# ------------------------------------------------------------
# TEST: 2TDVP full sweep preserves norm
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_sweep_2tdvp_full_norm_preserved():
    # This case tests norm preservation through a full 2TDVP
    # right+left sweep pair.
    d = 2
    phys_dims = [d, d, d]
    list_cores = _make_list_cores(phys_dims, seed=117)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    # Norm before
    all_before = [core_M] + list(list_cores_B)
    psi_before = all_before[0]
    for c in all_before[1:]:
        psi_before = np.tensordot(psi_before, c, axes=([-1], [0]))
    norm_sq_before = np.vdot(psi_before, psi_before)
    # Right sweep
    list_envs_L, list_cores_A, core_M_last = sweep_right_2tdvp(
        0.005,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        chi_max=0,
        eps=0.0,
    )
    R_boundary = list_envs_R[-1]
    # Left sweep
    core_M_first, list_B_out, list_R_out = sweep_left_2tdvp(
        0.005,
        list_envs_L,
        R_boundary,
        list_cores_mpo,
        list_cores_A,
        core_M_last,
        chi_max=0,
        eps=0.0,
    )
    # Norm after
    all_after = [core_M_first] + list(list_B_out)
    psi_after = all_after[0]
    for c in all_after[1:]:
        psi_after = np.tensordot(psi_after, c, axes=([-1], [0]))
    norm_sq_after = np.vdot(psi_after, psi_after)
    np.testing.assert_allclose(
        norm_sq_after,
        norm_sq_before,
        rtol=1e-6,
        err_msg='Norm not preserved through 2TDVP right+left sweep',
    )


# ------------------------------------------------------------
# TEST: 2TDVP left sweep produces right-orthogonal B cores and preserves norm
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_sweep_left_2tdvp_b_cores_and_norm():
    # This case tests that sweep_left_2tdvp produces right-orthogonal
    # B cores and preserves the MPS norm through a full right+left sweep.
    # 2TDVP requires uniform physical dimensions (d1==d2 for two-site merge).
    phys_dims = [2, 2, 2]
    list_cores = _make_list_cores(phys_dims, seed=120)
    list_cores_mpo = _make_list_mpo(phys_dims)
    core_M, list_cores_B, L0, list_envs_R = initialize(
        list_cores,
        list_cores_mpo,
    )
    # Norm before
    all_before = [core_M] + list(list_cores_B)
    psi_before = all_before[0]
    for c in all_before[1:]:
        psi_before = np.tensordot(psi_before, c, axes=([-1], [0]))
    norm_sq_before = np.vdot(psi_before, psi_before)
    # Right sweep first (needed as input to left sweep)
    list_envs_L, list_cores_A, core_M_last = sweep_right_2tdvp(
        0.005,
        L0,
        list_envs_R,
        list_cores_mpo,
        core_M,
        list_cores_B,
        chi_max=0,
        eps=0.0,
    )
    R_boundary = list_envs_R[-1]
    # Left sweep
    core_M_first, list_B_out, list_R_out = sweep_left_2tdvp(
        0.005,
        list_envs_L,
        R_boundary,
        list_cores_mpo,
        list_cores_A,
        core_M_last,
        chi_max=0,
        eps=0.0,
    )
    # Verify B cores are right-orthogonal
    for idx, B in enumerate(list_B_out):
        chi_l, d, Dr = B.shape
        Q_mat = B.reshape(chi_l, d * Dr)
        np.testing.assert_allclose(
            Q_mat @ Q_mat.conj().T,
            np.eye(chi_l),
            atol=1e-10,
            err_msg=f'B core {idx} not right-orthogonal after left sweep',
        )
    # Verify norm preserved through full right+left sweep
    all_after = [core_M_first] + list(list_B_out)
    psi_after = all_after[0]
    for c in all_after[1:]:
        psi_after = np.tensordot(psi_after, c, axes=([-1], [0]))
    norm_sq_after = np.vdot(psi_after, psi_after)
    np.testing.assert_allclose(
        norm_sq_after,
        norm_sq_before,
        rtol=1e-6,
        err_msg='Norm not preserved through 2TDVP right+left sweep',
    )


# ============================================================
# Helpers for recenter_to_zero tests
# ============================================================


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


def _contract_mps(c_list):
    """Full contraction of an MPS into a state vector."""
    s = c_list[0]
    for c in c_list[1:]:
        s = np.tensordot(s, c, axes=([-1], [0]))
    return s.squeeze().ravel()


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
    state_before = _contract_mps([c.copy() for c in cores])
    result = recenter_to_zero(cores)
    state_after = _contract_mps(result)
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
    state_orig = _contract_mps(cores)
    result = recenter_to_zero(cores, normalize=True)
    norm_A0 = np.linalg.norm(result[0].ravel())
    np.testing.assert_allclose(norm_A0, 1.0, atol=1e-12)
    # The contracted state should be proportional to the original
    state_norm = _contract_mps(result)
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
    np.random.seed(55)
    cores = _make_simple_mps(3, [2, 3, 2], [1, 4, 4, 1])
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

    # copy=True should return an independent list
    np.random.seed(56)
    cores2 = _make_simple_mps(3, [2, 3, 2], [1, 4, 4, 1])
    result2 = recenter_to_zero(cores2, copy=True)
    assert result2 is not cores2


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
    state_before = _contract_mps(cores)
    result = recenter_to_zero(cores)
    state_after = _contract_mps(result)
    np.testing.assert_allclose(state_after, state_before, atol=1e-12)


# ------------------------------------------------------------
# TEST: recenter_to_zero on already-canonical MPS is near-no-op
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_already_canonical():
    # This case tests that passing an already-canonical MPS returns the
    # same state (within numerical precision).
    np.random.seed(99)
    cores = _make_simple_mps(3, [2, 3, 2], [1, 3, 3, 1])
    canonical = recenter_to_zero(cores)
    state_first = _contract_mps(canonical)
    result = recenter_to_zero(canonical)
    state_second = _contract_mps(result)
    np.testing.assert_allclose(state_second, state_first, atol=1e-12)


# ------------------------------------------------------------
# TEST: recenter_to_zero with all bond dims = 1
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_trivial_bonds():
    # This case tests the degenerate geometry where every bond is 1,
    # so QR reduces to scalar extraction at every site.
    np.random.seed(11)
    cores = _make_simple_mps(4, [2, 3, 2, 2], [1, 1, 1, 1, 1])
    state_before = _contract_mps(cores)
    result = recenter_to_zero(cores)
    state_after = _contract_mps(result)
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


# ------------------------------------------------------------
# TEST: empty list returns empty
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_recenter_to_zero_empty_list():
    # This case tests that an empty list input returns an empty list
    # without error.
    result = recenter_to_zero([])
    assert result == []


# ============================================================
# TEST SUITE: Krylov edge cases — breakdown and fallback
# ============================================================


# ------------------------------------------------------------
# TEST: Arnoldi explicit breakdown guard (conv_tol=0)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_explicit_breakdown_guard():
    # This case tests the explicit breakdown guard (lines 412-416)
    # by setting conv_tol=0 so the convergence check (h * |phi| < 0)
    # can never fire. The rank-1 projector P = |e_0><e_0| terminates
    # the Krylov space at m=1 via the h < 1e-14 guard.
    n = 10
    u = np.zeros(n, dtype=complex)
    u[0] = 1.0
    P = np.outer(u, u.conj())
    matvec = lambda v: P @ v
    dt = 0.5
    result = _arnoldi_expm(u.copy(), matvec, dt=dt, conv_tol=0.0)
    expected = np.exp(dt * 1.0) * u
    np.testing.assert_allclose(result, expected, atol=1e-12)


# ------------------------------------------------------------
# TEST: Arnoldi max-iteration fallback (random matrix, conv_tol=0)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_arnoldi_expm_max_iteration_fallback():
    # This case tests the post-loop expm fallback (lines 420-426)
    # by using a random 70x70 matrix with conv_tol=0. The Krylov
    # loop exhausts _KRYLOV_DIM_MAX=60 iterations without converging,
    # then exponentiates the full 60x60 Hessenberg matrix.
    from scipy.linalg import expm as scipy_expm
    rng = np.random.default_rng(42)
    n = 70
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    v = rng.standard_normal(n).astype(complex)
    matvec = lambda x: H @ x
    dt = 0.01  # small dt for accuracy with truncated Krylov
    result = _arnoldi_expm(v, matvec, dt=dt, conv_tol=0.0)
    expected = scipy_expm(dt * H) @ v
    np.testing.assert_allclose(result, expected, rtol=1e-4)


# ------------------------------------------------------------
# TEST: Lanczos explicit breakdown guard (conv_tol=0)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_lanczos_expm_explicit_breakdown_guard():
    # This case tests the Lanczos explicit breakdown guard (lines
    # 495-497) by setting conv_tol=0 on a rank-1 Hermitian projector.
    # Lanczos terminates at m=1 via the beta_j < 1e-14 guard.
    n = 10
    u = np.zeros(n, dtype=complex)
    u[0] = 1.0
    P = np.outer(u, u.conj())  # Hermitian rank-1 projector
    matvec = lambda v: P @ v
    dt = 0.5
    result = _lanczos_expm(u.copy(), matvec, dt=dt, conv_tol=0.0)
    expected = np.exp(dt * 1.0) * u
    np.testing.assert_allclose(result, expected, atol=1e-12)


# ------------------------------------------------------------
# TEST: Lanczos max-iteration fallback (conv_tol=0)
# ------------------------------------------------------------
@pytest.mark.level(1)
def test_lanczos_expm_max_iteration_fallback():
    # This case tests the Lanczos post-loop tridiagonal expm fallback
    # (lines 502-511) by using a Hermitian random matrix with
    # conv_tol=0 and dimension > _KRYLOV_DIM_MAX=60.
    from scipy.linalg import expm as scipy_expm
    rng = np.random.default_rng(42)
    n = 70
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = (H + H.conj().T) / 2  # symmetrize for Lanczos
    v = rng.standard_normal(n).astype(complex)
    matvec = lambda x: H @ x
    dt = 0.01  # small dt for accuracy with truncated Krylov
    result = _lanczos_expm(v, matvec, dt=dt, conv_tol=0.0)
    expected = scipy_expm(dt * H) @ v
    np.testing.assert_allclose(result, expected, rtol=1e-4)
