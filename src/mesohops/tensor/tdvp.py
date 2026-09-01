"""TDVP time-stepping for MPS wavefunctions following Paeckel et al. (2019).

All functions operate on bare lists of np.ndarray — no MPS container.
MPS cores have shape (Dl, d, Dr); MPO cores have shape (wl, d_out, d_in, wr);
environment tensors have shape (chi, w, chi).

State convention
----------------
The MPS is held in mixed canonical form with the non-canonical center
at site 1:
  core_M        : ndarray(Dl, d, Dr)        — center core (site 1)
  list_cores_B  : list(ndarray(Dl, d, Dr))   — right-canonical cores,
                  sites 2..L
  L0            : ndarray(1, 1, 1)           — left boundary environment
  list_envs_R   : list(ndarray(chi, w, chi)) — right environments
                  R_2..R_{L+1}; list_envs_R[0]=R_2,
                  list_envs_R[j-2]=R_j, list_envs_R[L-1]=R_{L+1}

Public API
----------
initialize(list_cores_mps, list_cores_mpo)
    → core_M, list_cores_B, L0, list_envs_R

timestep(delta, L0, list_envs_R, list_cores_mpo, core_M, list_cores_B,
         method='1tdvp', solver='arnoldi', **kwargs)
    → core_M, list_cores_B, L0, list_envs_R
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import svd as scipy_svd

__title__ = 'TDVP Integrator'
__author__ = 'B. Z. Citty'
__maintainer__ = 'B. Z. Citty'

# Hard cap on Krylov subspace dimension; convergence (conv_tol) is expected to
# terminate the iteration long before this limit is reached.
_KRYLOV_DIM_MAX = 60

# Below this threshold h_{m+1,m} is treated as zero (prevents division by
# zero when extending the Krylov basis; not exposed to callers).
_KRYLOV_BREAKDOWN_TOL = 1e-14


# ============================================================
# Section 1: Splitting utilities
# ============================================================


def _split_qr(core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left-orthogonalize a rank-3 MPS core via QR decomposition.

    Reshapes the core from (Dl, d, Dr) into a matrix (Dl*d, Dr), performs
    QR, then reshapes Q back to (Dl, d, chi).  The result satisfies
    sum_s Q[:,s,:]^dag Q[:,s,:] = I  (left-orthogonality condition).

    Parameters
    ----------
    1. core : np.ndarray(Dl, d, Dr)

    Returns
    -------
    1. Q : np.ndarray(Dl, d, chi)  — left-orthogonal
    2. R : np.ndarray(chi, Dr)
    """
    Dl, d, Dr = core.shape
    # merge physical and left-bond indices: (Dl*d, Dr)
    H2_core = core.reshape(Dl * d, Dr)
    Q, R = np.linalg.qr(H2_core, mode='reduced')
    chi = Q.shape[1]
    # restore physical index: (Dl, d, chi)
    Q = Q.reshape(Dl, d, chi)
    return Q, R


def _split_rq(core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-orthogonalize a rank-3 MPS core via RQ decomposition.

    Reshapes the core from (Dl, d, Dr) into a matrix (Dl, d*Dr), performs
    QR on the transpose, then reshapes Q back to (chi, d, Dr).  The result
    satisfies sum_s Q[:,s,:] Q[:,s,:]^dag = I (right-orthogonality condition).

    Parameters
    ----------
    1. core : np.ndarray(Dl, d, Dr)

    Returns
    -------
    1. R : np.ndarray(Dl, chi)
    2. Q : np.ndarray(chi, d, Dr)  — right-orthogonal
    """
    Dl, d, Dr = core.shape
    # merge physical and right-bond indices: (Dl, d*Dr)
    H2_core = core.reshape(Dl, d * Dr)
    # RQ via QR of transpose: H2_core = R @ Q  ↔  H2_core^T = Q^T @ R^T
    Qt, Rt = np.linalg.qr(H2_core.T.conj(), mode='reduced')
    R = Rt.T.conj()  # shape (Dl, chi)
    Q = Qt.T.conj()  # shape (chi, d*Dr)
    chi = Q.shape[0]
    # restore physical index: (chi, d, Dr)
    Q = Q.reshape(chi, d, Dr)
    return R, Q


def _split_svd(
    theta: np.ndarray,
    chi_max: int,
    eps: float,
    d1: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated SVD of a two-site tensor. Returns raw factors; callers absorb S.

    Reshapes theta from (Dl, d1*d2, Dr) into (Dl*d1, d2*Dr), performs full SVD,
    applies truncation by eps and chi_max, then reshapes U and Vt back to rank-3
    cores.

    Parameters
    ----------
    1. theta   : np.ndarray(Dl, d1*d2, Dr)
    2. chi_max : int    — maximum bond dimension (0 = unlimited)
    3. eps     : float  — singular value truncation threshold
    4. d1      : int | None
                 Physical dimension of the left site. If None, assumes
                 d1 == d2 and infers d1 = sqrt(d1*d2).

    Returns
    -------
    1. U       : np.ndarray(Dl, d1, chi_new)   — left-orthogonal
    2. S       : np.ndarray(chi_new,)           — singular values (not absorbed here)
    3. Vt      : np.ndarray(chi_new, d2, Dr)   — right-orthogonal
    4. chi_new : int

    Notes
    -----
    Callers are responsible for absorbing S into whichever side becomes the new
    center tensor:
      - sweep_right_2tdvp: C_j = S[:,None] * Vt  (S absorbed right)
      - sweep_left_2tdvp:  C_{j-1} = U * S[None,:]  (S absorbed left)
    """
    Dl, d1d2, Dr = theta.shape
    if d1 is None:
        d1 = round(math.sqrt(d1d2))
        if d1 * d1 != d1d2:
            raise ValueError(
                '_split_svd: d1 not provided and middle axis is not a perfect '
                f'square. Got d1*d2={d1d2}.'
            )
    d2 = d1d2 // d1
    # bipartition: (Dl, d1, d2, Dr) treated as (Dl*d1, d2*Dr)
    H2_theta = theta.reshape(Dl * d1, d2 * Dr)
    U, S, Vt = scipy_svd(H2_theta, full_matrices=False, lapack_driver='gesvd')
    # truncate by threshold
    mask = S > eps
    if chi_max > 0:
        # keep at most chi_max values
        mask[chi_max:] = False
    chi_new = int(mask.sum())
    chi_new = max(chi_new, 1)  # always keep at least one singular value
    U = U[:, :chi_new]  # (Dl*d1, chi_new)
    S = S[:chi_new]  # (chi_new,)
    Vt = Vt[:chi_new, :]  # (chi_new, d2*Dr)
    # reshape to rank-3 cores
    U = U.reshape(Dl, d1, chi_new)  # (Dl, d1, chi_new) — left-orthogonal
    Vt = Vt.reshape(chi_new, d2, Dr)  # (chi_new, d2, Dr) — right-orthogonal
    return U, S, Vt, chi_new


# ============================================================
# Section 2: Environment contractions  (Paeckel Alg. 3)
# ============================================================


def contract_left(L_prev: np.ndarray, W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Paeckel Alg. 3 CONTRACT-LEFT. Extend left environment one site to the right.

    Computes the partial sandwich L_j = <A| W |A> contracted with L_{j-1}:

        L[b', w', b] = sum_{a', w, a, s', s}
                       L_prev[a', w, a]  *  A*.conj()[a', s', b']
                       *  W[w, s', s, w']  *  A[a, s, b]

    Einsum index map:
      L_prev[i, j, k]  — i=a'(chi_l_bra), j=w_l, k=a(chi_l_ket)
      A.conj()[i, l, m] — i=a'(chi_l_bra), l=s'(d_bra), m=b'(chi_r_bra)
      W[j, l, n, o]    — j=w_l, l=s'(d_out), n=s(d_in), o=w'(w_r)
      A[k, n, p]       — k=a(chi_l_ket), n=s(d_in), p=b(chi_r_ket)
      → L[m, o, p]     — m=b'(chi_r_bra), o=w'(w_r), p=b(chi_r_ket)

    Parameters
    ----------
    1. L_prev : np.ndarray(chi_l, w_l, chi_l)
    2. W      : np.ndarray(w_l, d_out, d_in, w_r)  — MPO core at site j
    3. A      : np.ndarray(Dl, d, Dr)               — left-canonical MPS core at site j

    Returns
    -------
    1. L : np.ndarray(chi_r, w_r, chi_r)
    """
    return np.einsum('ijk,ilm,jlno,knp->mop', L_prev, A.conj(), W, A,
                     optimize=True)


def contract_right(R_next: np.ndarray, W: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Paeckel Alg. 3 CONTRACT-RIGHT. Extend right environment one site to the left.

    Computes the partial sandwich R_j = <B| W |B> contracted with R_{j+1}:

        R[l, n, p] = sum_{i, j, k, m, o}
                     R_next[i, j, k]  *  B*.conj()[l, m, i]
                     *  W[n, m, o, j]  *  B[p, o, k]

    Einsum index map:
      R_next[i, j, k]  — i=b'(chi_r_bra), j=w_r, k=b(chi_r_ket)
      B.conj()[l, m, i] — l=a'(chi_l_bra), m=s'(d_bra), i=b'(chi_r_bra)
      W[n, m, o, j]    — n=w'(w_l), m=s'(d_out), o=s(d_in), j=w_r
      B[p, o, k]       — p=a(chi_l_ket), o=s(d_in), k=b(chi_r_ket)
      → R[l, n, p]     — l=a'(chi_l_bra), n=w'(w_l), p=a(chi_l_ket)

    Parameters
    ----------
    1. R_next : np.ndarray(chi_r, w_r, chi_r)
    2. W      : np.ndarray(w_l, d_out, d_in, w_r)  — MPO core at site j
    3. B      : np.ndarray(Dl, d, Dr)               — right-canonical MPS core at site j

    Returns
    -------
    1. R : np.ndarray(chi_l, w_l, chi_l)
    """
    return np.einsum('ijk,lmi,nmoj,pok->lnp', R_next, B.conj(), W, B,
                     optimize=True)


# ============================================================
# Section 3: Effective Hamiltonian actions
# ============================================================


def _apply_heff_site(
    M: np.ndarray, L: np.ndarray, W: np.ndarray, R: np.ndarray,
) -> np.ndarray:
    """One-site effective Hamiltonian action: H_j^eff |M⟩ = L · W · R |M⟩.

    Implements the tensor contraction that appears in Paeckel Alg. 5 TIMESTEP
    (single-site forward/backward updates).  The effective Hamiltonian acts on
    the center core M as

        HM[i, l, o] = sum_{j,k,m,n,p}
                      L[i, j, k] * W[j, l, m, n] * R[o, n, p] * M[k, m, p]

    Einsum index map:
      L[i, j, k]    — i=a'(chi_l_bra), j=w_l, k=a(chi_l_ket)
      W[j, l, m, n] — j=w_l, l=s'(d_out), m=s(d_in), n=w_r
      R[o, n, p]    — o=b'(chi_r_bra), n=w_r, p=b(chi_r_ket)
      M[k, m, p]    — k=a(chi_l_ket), m=s(d_in), p=b(chi_r_ket)
      → HM[i, l, o] — i=a', l=s', o=b'

    Parameters
    ----------
    1. M : np.ndarray(Dl, d, Dr)
    2. L : np.ndarray(chi_l, w_l, chi_l)
    3. W : np.ndarray(w_l, d_out, d_in, w_r)
    4. R : np.ndarray(chi_r, w_r, chi_r)

    Returns
    -------
    1. HM : np.ndarray(Dl, d, Dr)
    """
    return np.einsum('ijk,jlmn,onp,kmp->ilo', L, W, R, M, optimize=True)


def _apply_heff_bond(C: np.ndarray, L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Zero-site (bond) effective Hamiltonian action: H_{j+}^eff |C⟩ = L · R |C⟩.

    Used for the backward bond step in 1TDVP (Paeckel Alg. 5, Step 6).  The
    zero-site effective Hamiltonian has no MPO core — the environments are
    contracted directly over the bond matrix C:

        HC[i, l] = sum_{j, k, m} L[i, j, k] * R[l, j, m] * C[k, m]

    The sum over the MPO bond j collapses because both environments already
    encode the full MPO up to their respective boundaries.

    Einsum index map:
      L[i, j, k] — i=a'(chi_l_bra), j=w, k=a(chi_l_ket)
      R[l, j, m] — l=b'(chi_r_bra), j=w, m=b(chi_r_ket)
      C[k, m]    — k=a(chi_l_ket), m=b(chi_r_ket)
      → HC[i, l] — i=a', l=b'

    Parameters
    ----------
    1. C : np.ndarray(chi_l, chi_r)
    2. L : np.ndarray(chi_l, w, chi_l)
    3. R : np.ndarray(chi_r, w, chi_r)

    Returns
    -------
    1. HC : np.ndarray(chi_l, chi_r)
    """
    return np.einsum('ijk,ljm,km->il', L, R, C, optimize=True)


def _apply_heff_twosite(
    theta: np.ndarray,
    L: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Two-site effective Hamiltonian action: H_{j,j+1}^eff |θ⟩ = L · W1 · W2 · R |θ⟩.

    Used for the forward two-site step in 2TDVP (Paeckel Alg. 6).  theta is
    passed as (Dl, d*d, Dr) and reshaped to (Dl, d, d, Dr) internally:

        Htheta[i, l, o, r] = sum_{j,k,m,n,p,q,s}
                              L[i,j,k] * W1[j,l,m,n] * W2[n,o,p,q]
                              * R[r,q,s] * th[k,m,p,s]

    Einsum index map:
      L[i, j, k]       — i=a', j=w_l, k=a
      W1[j, l, m, n]   — j=w_l, l=s1', m=s1, n=w_m
      W2[n, o, p, q]   — n=w_m, o=s2', p=s2, q=w_r
      R[r, q, s]       — r=b', q=w_r, s=b
      th[k, m, p, s]   — k=a, m=s1, p=s2, s=b
      → Htheta[i,l,o,r]

    Parameters
    ----------
    1. theta : np.ndarray(Dl, d1*d2, Dr)
    2. L     : np.ndarray(chi_l, w_l, chi_l)
    3. W1    : np.ndarray(w_l, d_out, d_in, w_m)  — left MPO core
    4. W2    : np.ndarray(w_m, d_out, d_in, w_r)  — right MPO core
    5. R     : np.ndarray(chi_r, w_r, chi_r)

    Returns
    -------
    1. Htheta : np.ndarray(Dl, d1*d2, Dr)
    """
    Dl, d1d2, Dr = theta.shape
    d1 = W1.shape[1]  # physical dim of left site (from MPO core)
    d2 = W2.shape[1]  # physical dim of right site (from MPO core)
    theta_4d = theta.reshape(Dl, d1, d2, Dr)
    result_4d = np.einsum(
        'ijk,jlmn,nopq,rqs,kmps->ilor',
        L,
        W1,
        W2,
        R,
        theta_4d,
        optimize=True,
    )
    return result_4d.reshape(Dl, d1 * d2, Dr)


# ============================================================
# Section 4: Local solvers
# ============================================================


def _arnoldi_expm(v, matvec, dt, conv_tol):
    """Krylov-Arnoldi matrix exponential for general (non-Hermitian) H.

    Computes exp(dt * H) |v⟩ without forming H explicitly.  The algorithm:

    1. Build an orthonormal Krylov basis V_m = [v_1, ..., v_m] via Arnoldi
       iteration (modified Gram-Schmidt), recording the upper Hessenberg
       matrix H_m such that  H V_m ≈ V_m H_m + h_{m+1,m} v_{m+1} e_m^T.
    2. After each step, check the a posteriori relative error estimate
       (Hochbruck & Lubich 1997, §2.3):
           err_rel ≈ h_{m+1,m} · |[exp(dt H_m) e_1]_m|
       and terminate as soon as err_rel < conv_tol.  expm on the m × m
       Hessenberg is negligible compared to the cost of a matvec.
    3. Reconstruct the result:
          exp(dt * H) |v⟩ ≈ ‖v‖ · V_m · exp(dt * H_m) · e_1

    Reference: Hochbruck & Lubich (1997) SIAM J. Numer. Anal. 34(5).

    Parameters
    ----------
    1. v        : np.ndarray — state to evolve (any shape; flattened internally)
    2. matvec   : callable(v) → same shape as v
    3. dt       : complex  — time step (may be negative for backward evolution)
    4. conv_tol : float    — relative convergence tolerance on the output vector

    Returns
    -------
    1. v_out : np.ndarray, same shape as v
    """
    shape = v.shape
    v_flat = v.ravel().astype(complex)
    n = len(v_flat)
    m = min(_KRYLOV_DIM_MAX, n)

    beta = np.linalg.norm(v_flat)
    if beta < _KRYLOV_BREAKDOWN_TOL:
        return np.zeros_like(v)

    # H2_krylov: columns are Krylov basis vectors; H2_hess: upper Hessenberg
    H2_krylov = np.zeros((n, m + 1), dtype=complex)
    H2_hess = np.zeros((m + 1, m), dtype=complex)
    H2_krylov[:, 0] = v_flat / beta

    m_actual = m
    for j in range(m):
        w = matvec(H2_krylov[:, j].reshape(shape)).ravel()
        # modified Gram-Schmidt orthogonalization
        for i in range(j + 1):
            H2_hess[i, j] = np.dot(H2_krylov[:, i].conj(), w)
            w = w - H2_hess[i, j] * H2_krylov[:, i]
        H2_hess[j + 1, j] = np.linalg.norm(w)
        # Convergence check: relative error ≈ h_{j+2,j+1} · |φ_j|
        # where φ = expm(dt H_{j+1}) e_1.  Also catches happy breakdown
        # (h ≈ 0) without a separate code path.
        V1_unit = np.zeros(j + 1, dtype=complex)
        V1_unit[0] = 1.0
        phi = expm(dt * H2_hess[:j + 1, :j + 1]) @ V1_unit
        if H2_hess[j + 1, j] * abs(phi[j]) < conv_tol:
            m_actual = j + 1
            return (beta * H2_krylov[:, :m_actual] @ phi).reshape(shape)
        if H2_hess[j + 1, j] < _KRYLOV_BREAKDOWN_TOL:
            # exact happy breakdown not caught by conv_tol (e.g. conv_tol=0);
            # guard against division by zero before storing next basis vector
            m_actual = j + 1
            break
        H2_krylov[:, j + 1] = w / H2_hess[j + 1, j]

    # exponentiate the (m_actual × m_actual) Hessenberg matrix
    H2_hessenberg = H2_hess[:m_actual, :m_actual]
    V1_unit = np.zeros(m_actual, dtype=complex)
    V1_unit[0] = 1.0
    # exp(dt * H2_hessenberg) e_1  →  first column of matrix exponential
    V1_expm = expm(dt * H2_hessenberg) @ V1_unit
    v_out = beta * H2_krylov[:, :m_actual] @ V1_expm
    return v_out.reshape(shape)


def _lanczos_expm(v, matvec, dt, conv_tol):
    """Krylov-Lanczos matrix exponential for Hermitian H.

    Computes exp(dt * H) |v⟩ for Hermitian H without forming H explicitly.
    Uses the three-term Lanczos recurrence to build a tridiagonal T_m and
    an orthonormal basis Q_m, then reconstructs the result as:

        exp(dt * H) |v⟩ ≈ ‖v‖ · Q_m · exp(dt · T_m) · e_1

    For Hermitian H the Lanczos recurrence is cheaper than Arnoldi (two inner
    products per step instead of j+1), and T_m is real symmetric when H is
    Hermitian, so `expm` operates on a real matrix.

    Convergence is checked after each step via the same relative estimate as
    _arnoldi_expm: β_{j+1} · |[exp(dt T_{j+1}) e_1]_j| < conv_tol.

    Reference: Park & Light (1986) J. Chem. Phys. 85(10), 5870.

    Parameters
    ----------
    1. v        : np.ndarray
    2. matvec   : callable(v) → same shape as v
    3. dt       : complex
    4. conv_tol : float — relative convergence tolerance on the output vector

    Returns
    -------
    1. v_out : np.ndarray, same shape as v
    """
    shape = v.shape
    v_flat = v.ravel().astype(complex)
    n = len(v_flat)
    m = min(_KRYLOV_DIM_MAX, n)

    beta = np.linalg.norm(v_flat)
    if beta < _KRYLOV_BREAKDOWN_TOL:
        return np.zeros_like(v)

    # alpha: diagonal of T_m; beta_vec: off-diagonal (beta_vec[j] = T[j+1,j])
    alpha = np.zeros(m, dtype=complex)
    beta_vec = np.zeros(m, dtype=complex)
    # Q: Lanczos basis vectors stored as columns
    Q = np.zeros((n, m + 1), dtype=complex)
    Q[:, 0] = v_flat / beta

    m_actual = m
    v_prev = np.zeros(n, dtype=complex)
    for j in range(m):
        w = matvec(Q[:, j].reshape(shape)).ravel()
        alpha[j] = np.dot(Q[:, j].conj(), w).real
        w = w - alpha[j] * Q[:, j] - (beta_vec[j - 1] if j > 0 else 0.0) * v_prev
        beta_j = np.linalg.norm(w)
        beta_vec[j] = beta_j
        # Convergence check: relative error ≈ β_{j+1} · |φ_j|
        # where φ = expm(dt T_{j+1}) e_1.
        H2_tridiag_cur = (
            np.diag(alpha[:j + 1])
            + np.diag(beta_vec[:j].real, 1)
            + np.diag(beta_vec[:j].real, -1)
        )
        V1_unit = np.zeros(j + 1, dtype=complex)
        V1_unit[0] = 1.0
        phi = expm(dt * H2_tridiag_cur) @ V1_unit
        if beta_j * abs(phi[j]) < conv_tol:
            m_actual = j + 1
            return (beta * Q[:, :m_actual] @ phi).reshape(shape)
        if beta_j < _KRYLOV_BREAKDOWN_TOL:
            m_actual = j + 1
            break
        v_prev = Q[:, j].copy()
        Q[:, j + 1] = w / beta_j

    # build tridiagonal matrix T_m (m_actual × m_actual)
    H2_tridiag = (
        np.diag(alpha[:m_actual])
        + np.diag(beta_vec[: m_actual - 1].real, 1)
        + np.diag(beta_vec[: m_actual - 1].real, -1)
    )
    V1_unit = np.zeros(m_actual, dtype=complex)
    V1_unit[0] = 1.0
    V1_expm = expm(dt * H2_tridiag) @ V1_unit
    v_out = beta * Q[:, :m_actual] @ V1_expm
    return v_out.reshape(shape)


def _ivp_solve(
    v: np.ndarray,
    matvec: Callable,
    dt: float | complex,
    **ivp_kwargs,
) -> np.ndarray:
    """Evolve v under y' = matvec(y) via scipy.integrate.solve_ivp.

    Wraps scipy ODE integration for the local exponential problem dv/dt = H v.
    dt may be negative (backward TDVP evolution); in that case the rhs is
    negated and integrated forward over |dt|.

    Default solver is 'BDF' (stiff), suitable for the large-norm effective
    Hamiltonians that arise in HOPS.  Pass method='RK45' for non-stiff problems.

    Parameters
    ----------
    1. v           : np.ndarray
    2. matvec      : callable(v) → same shape as v
    3. dt          : float
    4. **ivp_kwargs passed to solve_ivp (method, rtol, atol, max_step)

    Returns
    -------
    1. v_out : np.ndarray, same shape as v
    """
    shape = v.shape
    V1_init = v.ravel().astype(complex)
    sign = 1.0 if dt >= 0 else -1.0
    t_span = (0.0, abs(dt))

    def rhs(_t, y):
        # sign accounts for backward integration (negate rhs when dt < 0)
        return sign * matvec(y.reshape(shape)).ravel()

    method = ivp_kwargs.pop('method', 'BDF')
    rtol = ivp_kwargs.pop('rtol', 1e-7)
    atol = ivp_kwargs.pop('atol', 1e-9)
    max_step = ivp_kwargs.pop('max_step', np.inf)

    sol = solve_ivp(
        rhs,
        t_span,
        V1_init,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
        **ivp_kwargs,
    )
    return sol.y[:, -1].reshape(shape)


def _solve_local(
    v: np.ndarray,
    matvec: Callable,
    dt: float | complex,
    solver: str = 'arnoldi',
    **kwargs,
) -> np.ndarray:
    """Dispatch local exponential solve: compute exp(dt * H_eff) |v⟩.

    Selects among three backends:
      - 'arnoldi' (default): Krylov-Arnoldi, suitable for non-Hermitian H.
        kwargs: conv_tol (float, default 1e-6) — relative convergence tolerance.
      - 'lanczos': Krylov-Lanczos, more efficient for Hermitian H.
        kwargs: conv_tol (float, default 1e-6) — relative convergence tolerance.
      - 'ivp': scipy.integrate.solve_ivp, fully adaptive step control.
        kwargs: method (str), rtol (float), atol (float), max_step (float).

    The matvec callable encodes H_eff (not -i/hbar * H_eff); the factor -i/hbar
    is absorbed into dt by the caller (i.e. dt = -i*delta/hbar for forward
    evolution and dt = +i*delta/hbar for the backward bond step).

    Parameters
    ----------
    1. v      : np.ndarray
    2. matvec : callable(v) → same shape as v
    3. dt     : float or complex
    4. solver : {'arnoldi', 'lanczos', 'ivp'}
    5. **kwargs forwarded to the selected solver

    Returns
    -------
    1. v_out : np.ndarray, same shape as v
    """
    if solver == 'arnoldi':
        conv_tol = kwargs.pop('conv_tol', 1e-6)
        return _arnoldi_expm(v, matvec, dt, conv_tol)
    elif solver == 'lanczos':
        conv_tol = kwargs.pop('conv_tol', 1e-6)
        return _lanczos_expm(v, matvec, dt, conv_tol)
    elif solver == 'ivp':
        return _ivp_solve(v, matvec, dt, **kwargs)
    else:
        raise ValueError(
            f"Unknown solver '{solver}'. Choose 'arnoldi', 'lanczos', or 'ivp'."
        )


# ============================================================
# Section 5: Initialization and timestep  (Paeckel Alg. 3)
# ============================================================


def recenter_to_zero(
    tensors: list[np.ndarray],
    normalize: bool = False,
    copy: bool = True,
) -> list[np.ndarray]:
    """
    Put an MPS into mixed canonical form with the orthogonality center at site 0.

    Sweeps left-to-right through sites 1..L-1, QR-factorizing each core to make
    it left-canonical (Q^dagger Q = I on the reshaped (Dl*d, Dr) matrix). The
    residual R factor is absorbed into the next site, with the final scalar
    folded into A[0]. Assumes open boundary conditions (Dr of the last site == 1).

    Parameters
    ----------
    1. tensors: list(np.ndarray)
                MPS cores, each shaped (Dl, d, Dr).
    2. normalize: bool
                  If True, rescale so that ||state|| = 1.
    3. copy: bool
             If True (default), copy the cores before modifying. If False,
             reuse the input list and replace its elements with the
             canonicalized cores (original array objects are not mutated).

    Returns
    -------
    1. list_cores_centered: list(np.ndarray)
       The canonicalized MPS cores with orthogonality center at site 0.
    """
    if len(tensors) == 0:
        return [] if copy else tensors

    list_cores = [t.copy() for t in tensors] if copy else tensors
    n_cores = len(list_cores)

    # Validate open boundary condition on the last core
    _, _, bond_dim_right_last = list_cores[-1].shape
    if bond_dim_right_last != 1:
        raise ValueError(
            f'Expected open boundaries with last right bond = 1, '
            f'got Dr[{n_cores - 1}] = {bond_dim_right_last}.'
        )

    # Validate all internal bond dimensions before sweeping
    for n in range(n_cores - 1):
        bond_dim_right_n = list_cores[n].shape[2]
        bond_dim_left_next = list_cores[n + 1].shape[0]
        if bond_dim_right_n != bond_dim_left_next:
            raise ValueError(
                f'Bond mismatch at link {n}: '
                f'right dim of site {n} is {bond_dim_right_n}, '
                f'but left dim of site {n + 1} is {bond_dim_left_next}.'
            )

    # Sweep left-to-right through sites 1..L-1, QR-factorizing each core
    # so that Q replaces the core (making it left-canonical) and R captures
    # the non-orthogonal content. R is then absorbed into the neighboring
    # core to preserve the overall state. After the full sweep, all content
    # that couldn't be made orthogonal has been pushed into site 0, which
    # becomes the orthogonality center.
    for n in range(1, n_cores):
        bond_dim_left, phys_dim, bond_dim_right = list_cores[n].shape
        H2_core_mat = list_cores[n].reshape(
            bond_dim_left * phys_dim, bond_dim_right,
        )
        Q, R = np.linalg.qr(H2_core_mat, mode='reduced')
        rank_qr = Q.shape[1]
        list_cores[n] = Q.reshape(bond_dim_left, phys_dim, rank_qr)

        if n < n_cores - 1:
            # R holds the part of this core that couldn't be made orthogonal.
            # Absorb it into the next core so the product Q @ R @ next is
            # unchanged, preserving the overall state.
            (bond_dim_left_next, phys_dim_next, bond_dim_right_next) = (
                list_cores[n + 1].shape
            )
            H2_core_next_mat = list_cores[n + 1].reshape(
                bond_dim_left_next,
                phys_dim_next * bond_dim_right_next,
            )
            H2_core_next_mat = R @ H2_core_next_mat
            list_cores[n + 1] = H2_core_next_mat.reshape(
                rank_qr,
                phys_dim_next,
                bond_dim_right_next,
            )
        else:
            # At the boundary under OBC the right bond is 1, so reduced QR
            # yields R as a (1,1) scalar. All remaining content is folded
            # into site 0 — the orthogonality center.
            list_cores[0] *= R[0, 0]

    # Optional normalization
    # (with n>=1 left-canonical, ||state|| = ||list_cores[0]||_F)
    if normalize:
        norm_phi = np.linalg.norm(list_cores[0].ravel())
        if norm_phi > 0:
            list_cores[0] /= norm_phi

    return list_cores


def initialize(
    list_cores_mps: list[np.ndarray],
    list_cores_mpo: list[np.ndarray],
) -> tuple[
    np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]
]:
    """Paeckel Alg. 3 INITIALIZE.

    Right-normalizes the MPS and builds all right-environment tensors needed
    before the first timestep. The input MPS is first recentered to site 0
    (mixed canonical form) for numerical stability before right-normalization.

    Procedure:
    1. Recenter the MPS to site 0 via a left-to-right QR sweep.
    2. Right-sweep from site L down to site 2: apply _split_rq to each core,
       absorb R into the core to the left, producing right-canonical B cores.
    3. Site 1 is left as the non-canonical center core_M (absorbs the last R).
    4. Build right environments by sweeping from right to left using contract_right.
       The boundary environment R_{L+1} = [[1]] (shape (1,1,1)).
       list_envs_R[j-2] = R_j for j in 2..L+1:
         list_envs_R[0] = R_2 (at bond between sites 1 and 2)
         ...
         list_envs_R[L-1] = R_{L+1} (right boundary, all ones)

    Parameters
    ----------
    1. list_cores_mps : list(np.ndarray(Dl, d, Dr))      — L MPS cores
    2. list_cores_mpo : list(np.ndarray(wl, d', d, wr))  — L MPO cores

    Returns
    -------
    1. core_M       : np.ndarray  — non-canonical center core at site 1
    2. list_cores_B : list(np.ndarray)  — right-canonical cores, sites 2..L
    3. L0           : np.ndarray(1, 1, 1)  — left boundary environment
    4. list_envs_R  : list(np.ndarray)  — right environments R_2..R_{L+1}
                      (L elements; list_envs_R[0]=R_2, list_envs_R[L-1]=R_{L+1})
    """
    n_sites = len(list_cores_mps)
    # Recenter to site 0 for numerical stability (copy=True preserves input)
    list_cores = recenter_to_zero(list_cores_mps, copy=True)

    # Right-normalize: sweep from last site down to site 1
    for j in range(n_sites - 1, 0, -1):
        R_mat, Q = _split_rq(list_cores[j])
        # Q is right-canonical; R_mat absorbed into core to the left
        list_cores[j] = Q
        list_cores[j - 1] = np.einsum(
            'ijk,kl->ijl',
            list_cores[j - 1],
            R_mat,
        )

    core_M = list_cores[0]
    list_cores_B = list_cores[1:]

    # Build right environments: R_{L+1}, R_L, ..., R_2
    # list_envs_R[j] corresponds to R_{j+2} (0-indexed j → site j+2)
    R_boundary = np.ones((1, 1, 1), dtype=complex)  # R_{L+1}
    list_envs_R = [None] * n_sites
    list_envs_R[n_sites - 1] = R_boundary

    # sweep right-to-left building R environments
    R_cur = R_boundary
    for j in range(n_sites - 1, 0, -1):
        # site j (0-indexed) → site j+1 (1-indexed)
        W_j = list_cores_mpo[j]
        B_j = list_cores_B[j - 1]
        R_cur = contract_right(R_cur, W_j, B_j)
        list_envs_R[j - 1] = R_cur

    # left boundary
    L0 = np.ones((1, 1, 1), dtype=complex)

    return core_M, list_cores_B, L0, list_envs_R


def timestep(
    delta: float | complex,
    L0: np.ndarray,
    list_envs_R: list[np.ndarray],
    list_cores_mpo: list[np.ndarray],
    core_M: np.ndarray,
    list_cores_B: list[np.ndarray],
    method: str = '1tdvp',
    solver: str = 'arnoldi',
    **kwargs,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Paeckel Alg. 3 TIMESTEP. Strang splitting: sweep_right(δ/2) + sweep_left(δ/2).

    The Strang splitting ensures second-order accuracy in the time step δ:

        exp(δ H) ≈ sweep_right(δ/2) ∘ sweep_left(δ/2)

    Each half-sweep is itself a sequence of local exponentials (the TDVP
    integrator); the composition cancels the leading-order splitting error.

    After the right half-sweep the state is in right-canonical form with center
    at site L.  The left half-sweep restores it to the initial canonical form
    (center at site 1) with updated environments.

    Parameters
    ----------
    1. delta          : float
    2. L0             : np.ndarray(1, 1, 1)
    3. list_envs_R    : list(np.ndarray)  — R_2..R_{L+1}
    4. list_cores_mpo : list(np.ndarray)
    5. core_M         : np.ndarray  — center MPS core at site 1
    6. list_cores_B   : list(np.ndarray)  — right-canonical cores, sites 2..L
    7. method         : {'1tdvp', '2tdvp'}
    8. solver         : {'arnoldi', 'lanczos', 'ivp'}

    Returns
    -------
    1. core_M       : np.ndarray  — updated center core at site 1
    2. list_cores_B : list(np.ndarray)  — updated right-canonical cores
    3. L0           : np.ndarray  — left boundary (unchanged)
    4. list_envs_R  : list(np.ndarray)  — updated right environments
    """
    half = delta / 2.0

    if method == '1tdvp':
        # right half-sweep: returns (list_envs_L, list_cores_A, core_M_last)
        list_envs_L, list_cores_A, core_M_last = sweep_right_1tdvp(
            half,
            L0,
            list_envs_R,
            list_cores_mpo,
            core_M,
            list_cores_B,
            solver=solver,
            **kwargs,
        )
        # boundary R_{L+1} is the last element of list_envs_R (unchanged by right sweep)
        R_boundary = list_envs_R[-1]
        # left half-sweep: returns (core_M_first, list_cores_B, list_envs_R)
        core_M, list_cores_B, list_envs_R = sweep_left_1tdvp(
            half,
            list_envs_L,
            R_boundary,
            list_cores_mpo,
            list_cores_A,
            core_M_last,
            solver=solver,
            **kwargs,
        )

    elif method == '2tdvp':
        chi_max = kwargs.pop('chi_max', 0)
        eps = kwargs.pop('eps', 0.0)
        list_envs_L, list_cores_A, core_M_last = sweep_right_2tdvp(
            half,
            L0,
            list_envs_R,
            list_cores_mpo,
            core_M,
            list_cores_B,
            chi_max,
            eps,
            solver=solver,
            **kwargs,
        )
        R_boundary = list_envs_R[-1]
        core_M, list_cores_B, list_envs_R = sweep_left_2tdvp(
            half,
            list_envs_L,
            R_boundary,
            list_cores_mpo,
            list_cores_A,
            core_M_last,
            chi_max,
            eps,
            solver=solver,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown method '{method}'. Choose '1tdvp' or '2tdvp'.")

    return core_M, list_cores_B, L0, list_envs_R


# ============================================================
# Section 6: 1TDVP sweeps  (Paeckel Alg. 5)
# ============================================================


def sweep_right_1tdvp(
    delta: float | complex,
    L0: np.ndarray,
    list_envs_R: list[np.ndarray],
    list_cores_mpo: list[np.ndarray],
    core_M: np.ndarray,
    list_cores_B: list[np.ndarray],
    solver: str = 'arnoldi',
    **kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Paeckel Alg. 5 SWEEP-RIGHT for 1TDVP.

    Implements the right-to-left half of the Strang-split 1TDVP integrator.
    For each site j = 1..L the algorithm:

    1. Forward-evolves the center core M_j under H_j^eff for time delta:
           M_j ← exp(-i * delta * H_j^eff) |M_j⟩
       (The factor -i is absorbed into dt = -1j * delta passed to _solve_local;
       H_j^eff is encoded via the matvec closure using _apply_heff_site.)

    2. QR-decomposes M_j → A_j (left-canonical) + C_j (non-unitary remainder).

    3. Updates L_j ← contract_left(L_{j-1}, W_j, A_j).

    4. For j < L: backward-evolves the bond matrix C_j for time -delta:
           C_j ← exp(+i * delta * H_{j+}^eff) |C_j⟩
       This backward step compensates for the gauge choice (Paeckel eq. 46).
       Then absorbs M_{j+1} = C_j · B_{j+1} for the next site.

    At the end the state is in mixed canonical form with center at site L.
    The left environments L_0..L_{L-1} and left-canonical cores A_1..A_{L-1}
    are returned for use by sweep_left_1tdvp.

    Parameters
    ----------
    1. delta          : float
    2. L0             : np.ndarray(1, 1, 1)
    3. list_envs_R    : list(np.ndarray)  — R_2..R_{L+1}
    4. list_cores_mpo : list(np.ndarray)
    5. core_M         : np.ndarray  — center core at site 1
    6. list_cores_B   : list(np.ndarray)  — right-canonical cores, sites 2..L
    7. solver         : {'arnoldi', 'lanczos', 'ivp'}

    Returns
    -------
    1. list_envs_L  : list(np.ndarray)  — L_0..L_{L-1} (L elements)
    2. list_cores_A : list(np.ndarray)  — left-canonical cores A_1..A_{L-1}
    3. core_M_last  : np.ndarray        — center core at site L
    """
    n_sites = len(list_cores_mpo)
    # list_envs_R[j] = R_{j+2}:
    #   list_envs_R[0] = R_2 (right of site 1)
    #   list_envs_R[j-1] = R_{j+1} (right of site j, 1-indexed)
    # We build list_envs_L[j] = L_{j+1} (left of site j+1, 1-indexed):
    #   list_envs_L[0] = L_0 (left boundary)
    #   list_envs_L[j] = L_j (left of site j+1)

    list_envs_L = [None] * n_sites
    list_cores_A = [None] * (n_sites - 1)

    list_envs_L[0] = L0
    M_cur = core_M  # current center core

    for j in range(n_sites):
        # site index j (0-indexed) → site j+1 (1-indexed, Paeckel notation)
        L_cur = list_envs_L[j]  # L_{j} (left env to the left of site j+1)
        # R_{j+2} = R_{(j+1)+1} (right env to right of site j+1)
        R_cur = list_envs_R[j]
        W_cur = list_cores_mpo[j]

        # Step 1: forward-evolve M_j under H_j^eff
        #   dt = -1j * delta  (forward TDVP: dM/dt = -i H_eff M)
        def matvec_site(v, _L=L_cur, _W=W_cur, _R=R_cur):
            return _apply_heff_site(v, _L, _W, _R)

        M_cur = _solve_local(M_cur, matvec_site, -1j * delta, solver=solver, **kwargs)

        # Step 2: QR decompose M_j → A_j (left-canonical) + C_j
        A_j, C_j = _split_qr(M_cur)  # A_j: (Dl, d, chi), C_j: (chi, Dr)

        # Step 3: update left environment L_j ← contract_left(L_{j-1}, W_j, A_j)
        L_new = contract_left(L_cur, W_cur, A_j)

        if j < n_sites - 1:
            list_cores_A[j] = A_j

            # Step 4: backward-evolve C_j under H_{j+}^eff (bond Hamiltonian)
            #   dt = +1j * delta  (backward step cancels gauge artifact)
            def matvec_bond(v, _L=L_new, _R=R_cur):
                return _apply_heff_bond(v, _L, _R)

            C_j = _solve_local(C_j, matvec_bond, +1j * delta, solver=solver, **kwargs)

            # Absorb C_j into next site: M_{j+1} = C_j · B_{j+1}
            B_next = list_cores_B[j]  # shape (chi_mid, d, Dr)
            M_cur = np.einsum('ij,jkl->ikl', C_j, B_next)

            list_envs_L[j + 1] = L_new
        else:
            # last site; M_cur is the final center core
            core_M_last = M_cur

    return list_envs_L, list_cores_A, core_M_last


def sweep_left_1tdvp(
    delta: float | complex,
    list_envs_L: list[np.ndarray],
    R_boundary: np.ndarray,
    list_cores_mpo: list[np.ndarray],
    list_cores_A: list[np.ndarray],
    core_M_last: np.ndarray,
    solver: str = 'arnoldi',
    **kwargs,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Paeckel Alg. 5 SWEEP-LEFT for 1TDVP.

    Implements the left-to-right half of the Strang-split 1TDVP integrator.
    Traverses sites j = L..1, restoring the canonical form with center at site 1
    and rebuilding the right environments list_envs_R.

    For each site j = L..1:

    1. Forward-evolves M_j under H_j^eff for time delta:
           M_j ← exp(-i * delta * H_j^eff) |M_j⟩

    2. RQ-decomposes M_j → C_{j-1} (non-unitary remainder) + B_j (right-canonical).

    3. Updates R_j ← contract_right(R_{j+1}, W_j, B_j).

    4. For j > 1: backward-evolves C_{j-1} for time -delta:
           C_{j-1} ← exp(+i * delta * H_{j-}^eff) |C_{j-1}⟩
       Then absorbs M_{j-1} = A_{j-1} · C_{j-1}.

    After the sweep the state has center at site 1 with core_M_first, and
    list_envs_R is ready for the next timestep.

    Parameters
    ----------
    1. delta          : float
    2. list_envs_L    : list(np.ndarray)  — L_0..L_{L-1} (L elements)
    3. R_boundary     : np.ndarray        — R_{L+1}
    4. list_cores_mpo : list(np.ndarray)
    5. list_cores_A   : list(np.ndarray)  — left-canonical cores A_1..A_{L-1}
    6. core_M_last    : np.ndarray        — center core at site L
    7. solver         : {'arnoldi', 'lanczos', 'ivp'}

    Returns
    -------
    1. core_M_first : np.ndarray        — center core at site 1
    2. list_cores_B : list(np.ndarray)  — right-canonical cores B_2..B_L
    3. list_envs_R  : list(np.ndarray)  — R_2..R_{L+1}
    """
    n_sites = len(list_cores_mpo)
    list_envs_R = [None] * n_sites
    list_cores_B = [None] * (n_sites - 1)

    list_envs_R[n_sites - 1] = R_boundary  # R_{L+1}
    M_cur = core_M_last  # start at last site

    for j in range(n_sites - 1, -1, -1):
        # site index j (0-indexed) → site j+1 (1-indexed)
        L_cur = list_envs_L[j]
        R_cur = list_envs_R[j]  # R_{j+2} (right env to right of site j+1)
        W_cur = list_cores_mpo[j]

        # Step 1: forward-evolve M_j under H_j^eff
        def matvec_site(v, _L=L_cur, _W=W_cur, _R=R_cur):
            return _apply_heff_site(v, _L, _W, _R)

        M_cur = _solve_local(M_cur, matvec_site, -1j * delta, solver=solver, **kwargs)

        # Step 2: RQ decompose M_j → C_{j-1} + B_j (right-canonical)
        C_j, B_j = _split_rq(M_cur)  # C_j: (Dl, chi), B_j: (chi, d, Dr)

        # Step 3: update right environment R_j ← contract_right(R_{j+1}, W_j, B_j)
        R_new = contract_right(R_cur, W_cur, B_j)

        if j > 0:
            # store B_j in the right-canonical list (0-indexed: B at site j+1)
            list_cores_B[j - 1] = B_j  # B at site j+1 (1-indexed); slot j-1

            # Step 4: backward-evolve C_j under H_{j-}^eff
            def matvec_bond(v, _L=L_cur, _R=R_new):
                return _apply_heff_bond(v, _L, _R)

            C_j = _solve_local(C_j, matvec_bond, +1j * delta, solver=solver, **kwargs)

            # Absorb: M_{j-1} = A_{j-1} · C_j
            # list_cores_A[j-1] = A at site j (0-indexed) = A_{j} (1-indexed)
            A_prev = list_cores_A[j - 1]  # shape (Dl, d, chi_l)
            M_cur = np.einsum('ijk,kl->ijl', A_prev, C_j)
            # A_prev: (Dl, d, chi_mid), C_j: (chi_mid, Dr) → M_cur: (Dl, d, Dr)

            list_envs_R[j - 1] = R_new  # R_{j+1} goes in slot j-1
        else:
            # j == 0: site 1 is the new center; M_cur (forward-evolved) becomes
            # core_M_first directly.  No RQ decomposition is needed or correct here:
            # the decomposed B_j would be right-canonical but list_cores_B[0] is
            # already right-canonical from the j==1 iteration above.
            # list_envs_R[0] = R_2 was already stored when j==1; do NOT overwrite.
            core_M_first = M_cur

    return core_M_first, list_cores_B, list_envs_R


# ============================================================
# Section 7: 2TDVP sweeps  (Paeckel Alg. 6)
# ============================================================


def sweep_right_2tdvp(
    delta: float | complex,
    L0: np.ndarray,
    list_envs_R: list[np.ndarray],
    list_cores_mpo: list[np.ndarray],
    core_M: np.ndarray,
    list_cores_B: list[np.ndarray],
    chi_max: int,
    eps: float,
    solver: str = 'arnoldi',
    **kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Paeckel Alg. 6 SWEEP-RIGHT for 2TDVP.

    Implements the right half of the Strang-split 2TDVP integrator.  Unlike
    1TDVP, the two-site algorithm operates on merged two-site tensors theta,
    which allows bond dimension to grow before being truncated by SVD.  This
    enables dynamical adaptation of the bond dimension during time evolution.

    For each pair of sites j, j+1 with j = 1..L-1:

    1. Contract the two-site tensor:
           theta_{j,j+1}[a, s1, s2, b] = M_j[a, s1, c] * B_{j+1}[c, s2, b]
       Then reshape to (Dl, d*d, Dr) for the two-site effective Hamiltonian.

    2. Forward-evolve theta under H_{j,j+1}^eff for time delta:
           theta ← exp(-i * delta * H_{j,j+1}^eff) |theta⟩

    3. Truncated SVD → A_j (left-canonical), S, Vt:
           C_j = diag(S) · Vt  (center tensor with bond weights absorbed right)

    4. For j < L-1:
       - Update L_j ← contract_left(L_{j-1}, W_j, A_j)
       - Backward single-site evolution of the center C_j under H_{j+1}^eff:
             C_j ← exp(+i * delta * H_{j+1}^eff) |C_j⟩
         This backward step exactly inverts the two-site forward step's
         contribution from site j+1, leaving only the net effect on site j.
       - M_{j+1} = C_j for the next iteration (center shifts right).

    Parameters
    ----------
    1. delta          : float
    2. L0             : np.ndarray(1, 1, 1)
    3. list_envs_R    : list(np.ndarray)  — R_2..R_{L+1}
    4. list_cores_mpo : list(np.ndarray)
    5. core_M         : np.ndarray  — center core at site 1
    6. list_cores_B   : list(np.ndarray)  — right-canonical cores, sites 2..L
    7. chi_max        : int    — maximum bond dimension (0 = unlimited)
    8. eps            : float  — SVD truncation threshold
    9. solver         : {'arnoldi', 'lanczos', 'ivp'}

    Returns
    -------
    1. list_envs_L  : list(np.ndarray)  — L_0..L_{L-2} (L-1 elements)
    2. list_cores_A : list(np.ndarray)  — left-canonical cores A_1..A_{L-1}
    3. core_M_last  : np.ndarray        — center core at site L
    """
    n_sites = len(list_cores_mpo)
    # list_envs_L[j] = L_j (left env for site j+1, 1-indexed)
    list_envs_L = [None] * (n_sites - 1)
    list_cores_A = [None] * (n_sites - 1)

    list_envs_L[0] = L0
    M_cur = core_M

    for j in range(n_sites - 1):
        # site j (0-indexed) = site j+1 (1-indexed)
        # site j+1 (0-indexed) = site j+2 (1-indexed)
        L_cur = list_envs_L[j]  # L_{j} (left of site j+1)
        R_next = list_envs_R[j + 1]  # R_{j+3} = R_{(j+2)+1}  — right of site j+2
        W_j = list_cores_mpo[j]
        W_jp1 = list_cores_mpo[j + 1]

        # Step 1: contract two-site tensor
        # M_cur: (Dl, d, chi_mid), B_{j+1}: (chi_mid, d, Dr)
        B_next = list_cores_B[j]  # B at site j+2 (0-indexed j → site j+2)
        # theta shape (Dl, d, d, Dr) → reshape to (Dl, d*d, Dr)
        theta = np.einsum('ijk,klm->ijlm', M_cur, B_next)
        Dl, d1, d2, Dr = theta.shape
        theta = theta.reshape(Dl, d1 * d2, Dr)

        # Step 2: forward-evolve theta under H_{j,j+1}^eff
        def matvec_two(v, _L=L_cur, _W1=W_j, _W2=W_jp1, _R=R_next):
            return _apply_heff_twosite(v, _L, _W1, _W2, _R)

        theta = _solve_local(theta, matvec_two, -1j * delta, solver=solver, **kwargs)

        # Step 3: truncated SVD
        A_j, S, Vt, _chi = _split_svd(theta, chi_max, eps, d1=d1)
        # bond weight absorbed right: C_j = S[:,None,None] * Vt
        # shape (chi_new, d2, Dr)
        C_j = S[:, None, None] * Vt

        list_cores_A[j] = A_j

        if j < n_sites - 2:
            # Step 4: update left environment
            L_new = contract_left(L_cur, W_j, A_j)
            list_envs_L[j + 1] = L_new

            # Backward single-site evolution of C_j (the new center at site j+2)
            # Uses L_j (just built) and R_{j+2} = list_envs_R[j+1]
            def matvec_back(v, _L=L_new, _W=W_jp1, _R=R_next):
                return _apply_heff_site(v, _L, _W, _R)

            C_j = _solve_local(C_j, matvec_back, +1j * delta, solver=solver, **kwargs)

        M_cur = C_j  # center core shifts to site j+2

    core_M_last = M_cur
    return list_envs_L, list_cores_A, core_M_last


def sweep_left_2tdvp(
    delta: float | complex,
    list_envs_L: list[np.ndarray],
    R_boundary: np.ndarray,
    list_cores_mpo: list[np.ndarray],
    list_cores_A: list[np.ndarray],
    core_M_last: np.ndarray,
    chi_max: int,
    eps: float,
    solver: str = 'arnoldi',
    **kwargs,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Paeckel Alg. 6 SWEEP-LEFT for 2TDVP.

    Implements the left half of the Strang-split 2TDVP integrator.  Traverses
    pairs of sites j-1, j for j = L..2, restoring canonical form with center
    at site 1 and rebuilding right environments.

    For each pair j-1, j with j = L..2:

    1. Contract the two-site tensor:
           theta_{j-1,j}[a, s1, s2, b] = A_{j-1}[a, s1, c] * M_j[c, s2, b]
       Then reshape to (Dl, d*d, Dr).

    2. Forward-evolve theta under H_{j-1,j}^eff for time delta.

    3. Truncated SVD → U, S, Vt:
           C_{j-1} = U * S[None,None,:]   (bond weight absorbed left)
           B_j = Vt (right-canonical)

    4. For j > 2:
       - Update R_j ← contract_right(R_{j+1}, W_j, B_j)
       - Backward single-site evolution of C_{j-1} under H_{j-1}^eff.
       - M_{j-1} = C_{j-1} (center shifts left).

    At the end the center is at site 1.

    Parameters
    ----------
    1. delta          : float
    2. list_envs_L    : list(np.ndarray)  — L_0..L_{L-2} (L-1 elements)
    3. R_boundary     : np.ndarray        — R_{L+1}
    4. list_cores_mpo : list(np.ndarray)
    5. list_cores_A   : list(np.ndarray)  — left-canonical cores A_1..A_{L-1}
    6. core_M_last    : np.ndarray        — center core at site L
    7. chi_max        : int
    8. eps            : float
    9. solver         : {'arnoldi', 'lanczos', 'ivp'}

    Returns
    -------
    1. core_M_first : np.ndarray        — center core at site 1
    2. list_cores_B : list(np.ndarray)  — right-canonical cores B_2..B_L
    3. list_envs_R  : list(np.ndarray)  — R_2..R_{L+1}
    """
    n_sites = len(list_cores_mpo)
    list_envs_R = [None] * n_sites
    list_cores_B = [None] * (n_sites - 1)

    list_envs_R[n_sites - 1] = R_boundary  # R_{L+1}
    M_cur = core_M_last

    for j in range(n_sites - 1, 0, -1):
        # site j (0-indexed) = site j+1 (1-indexed)
        # site j-1 (0-indexed) = site j (1-indexed)
        L_prev = list_envs_L[j - 1]  # L_{j-1} (left of site j, 1-indexed)
        R_cur = list_envs_R[j]  # R_{j+2} (right of site j+1, 1-indexed)
        W_j = list_cores_mpo[j]
        W_jm1 = list_cores_mpo[j - 1]

        # Step 1: contract two-site tensor
        # A_{j-1}: (Dl, d, chi_mid), M_cur: (chi_mid, d, Dr)
        A_prev = list_cores_A[j - 1]  # A at site j (1-indexed)
        theta = np.einsum('ijk,klm->ijlm', A_prev, M_cur)
        Dl, d1, d2, Dr = theta.shape
        theta = theta.reshape(Dl, d1 * d2, Dr)

        # Step 2: forward-evolve theta under H_{j-1,j}^eff
        def matvec_two(v, _L=L_prev, _W1=W_jm1, _W2=W_j, _R=R_cur):
            return _apply_heff_twosite(v, _L, _W1, _W2, _R)

        theta = _solve_local(theta, matvec_two, -1j * delta, solver=solver, **kwargs)

        # Step 3: truncated SVD
        U, S, B_j, _chi = _split_svd(theta, chi_max, eps, d1=d1)
        # bond weight absorbed left: U is (Dl, d1, chi_new),
        # S is (chi_new,); U * S[None, None, :] → (Dl, d, chi_new)
        C_jm1 = U * S[None, None, :]  # shape (Dl, d, chi_new)

        list_cores_B[j - 1] = B_j  # B at site j+1 (1-indexed) in slot j-1

        if j > 1:
            # Step 4: update right environment
            R_new = contract_right(R_cur, W_j, B_j)
            list_envs_R[j - 1] = R_new  # R_{j+1} in slot j-1

            # Backward single-site evolution of C_{j-1} under H_{j-1}^eff
            def matvec_back(v, _L=L_prev, _W=W_jm1, _R=R_new):
                return _apply_heff_site(v, _L, _W, _R)

            C_jm1 = _solve_local(
                C_jm1,
                matvec_back,
                +1j * delta,
                solver=solver,
                **kwargs,
            )

        M_cur = C_jm1  # center shifts left to site j (1-indexed)

    core_M_first = M_cur
    # R_2 = list_envs_R[0]; must have been set when j==1 in the loop above.
    # When j==1 we skip the "if j > 1" block, so list_envs_R[0] is still None.
    # Build R_2 from B at site 2 (list_cores_B[0]) and R_3 = list_envs_R[1]:
    if list_envs_R[0] is None and n_sites > 1:
        R2 = contract_right(list_envs_R[1], list_cores_mpo[1], list_cores_B[0])
        list_envs_R[0] = R2

    return core_M_first, list_cores_B, list_envs_R
