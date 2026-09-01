"""
Core MPS arithmetic and extraction routines used by HopsTensorWavefunction.

Functions
---------
tensor_add(list_tensor_cores_1, list_tensor_cores_2, epsilon, bond_dim_max)
    Add two MPS by block-concatenating their cores, then compress.

tensor_compress(list_cores, epsilon, bond_dim_max)
    Right-orthogonalize then left-sweep with SVD truncation (Oseledets
    rounding algorithm).

extract_psi(list_cores_phi, method, M1_modes_per_state)
    Extract the physical wavefunction (zero-auxiliary slice) from an MPS.

extract_gs_amp(list_cores_phi, method)
    Extract the amplitude of the all-phys-zero configuration of an MPS.
    In the vacuum convention this is the ground-state amplitude.

phi_aux(list_cores_phi, method, M1_modes_per_state, indices)
    Extract a specific auxiliary-state vector from an MPS given its
    per-mode occupation indices.

tensor_to_array(list_cores_phi, method, M1_modes_per_state, system, mode)
    Flatten the ground and first-order auxiliary wavefunctions into a
    single adHOPS-style array, for debugging purposes.

contract_down(list_cores_phi, method, n_state)
    Approximate per-state norm squared (sum over hierarchy) by contracting
    squared core elements independently.

contract_down_exact(list_cores_phi, method, n_state)
    Exact per-state norm squared via a double-layer right-to-left sweep.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd as scipy_svd

from mesohops.basis.basis_functions import determine_error_thresh
from mesohops.basis.hops_modes import HopsModes
from mesohops.basis.hops_system import HopsSystem

__title__ = 'Tensor Operations'
__author__ = 'B. Z. Citty'
__maintainer__ = 'B. Z. Citty'


def flatten_cores(list_cores_phi: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Flatten a statenumber list-of-lists MPS into a flat list of cores.

    Parameters
    ----------
    1. list_cores_phi: list(list(np.ndarray))
                        Nested MPS: list_cores_phi[s] = [state_core_s,
                        mode_core_s0, mode_core_s1, ...]

    Returns
    -------
    1. list_cores_flat: list(np.ndarray)
                         1D list of cores (one ndarray per element) in
                         statenumber MPS representation order:
                         state_0, mode_00, mode_01, ..., state_1, mode_10, ...
    """
    return [core for group in list_cores_phi for core in group]


def unflatten_cores(
    list_cores_flat: list[np.ndarray],
    list_modes_per_site: list[int],
) -> list[list[np.ndarray]]:
    """Restore list-of-lists MPS structure from a flat core list.

    Parameters
    ----------
    1. list_cores_flat: list(np.ndarray)
                         1D list of cores (one ndarray per element) in
                         statenumber MPS representation order.
    2. list_modes_per_site: list(int)
                             Number of mode cores per site s
                             (i.e. len(list_cores_phi[s]) - 1).

    Returns
    -------
    1. list_cores_phi: list(list(np.ndarray))
                        Nested MPS: list_cores_phi[s] = [state_core_s,
                        mode_core_s0, mode_core_s1, ...]
    """
    result = []
    idx = 0
    for n_modes in list_modes_per_site:
        result.append(list_cores_flat[idx: idx + 1 + int(n_modes)])
        idx += 1 + int(n_modes)
    return result


def _flat_cores_with_labels(
    list_cores_phi: list[list[np.ndarray]],
) -> list[tuple[np.ndarray, bool, int]]:
    """Return a list of (core, is_state_core, state_idx) tuples in
    statenumber MPS representation order.

    Parameters
    ----------
    1. list_cores_phi: list(list(np.ndarray))
                        Nested MPS: list_cores_phi[s] = [state_core_s,
                        mode_core_s0, mode_core_s1, ...]

    Returns
    -------
    1. list_labeled: list(tuple(np.ndarray, bool, int))
                      Each entry is (core, is_state_core, state_idx) where
                      is_state_core is True for state cores and False for
                      mode cores, and state_idx is the site index s.
    """
    result = []
    for s, group in enumerate(list_cores_phi):
        result.append((group[0], True, s))
        for core_m in group[1:]:
            result.append((core_m, False, s))
    return result


def _statenumber_offsets(M1_modes_per_state: np.ndarray) -> np.ndarray:
    """Compute the MPS core index of each state core.

    In number representation, the MPS layout is:
    [state_0][mode_0_0]...[mode_0_M0][state_1][mode_1_0]...
    This function returns the index of each state core.

    Parameters
    ----------
    1. M1_modes_per_state: array-like(int)
                            Number of mode cores per state.

    Returns
    -------
    1. M1_offsets: np.ndarray(int)
                    Core index of each state core.
    """
    # Stride per state is 1 (the state core) + the number of mode cores;
    # offsets are the cumulative stride prefix with a leading 0.
    M1_strides = 1 + np.asarray(M1_modes_per_state, dtype=int)
    M1_offsets = np.zeros(len(M1_strides), dtype=int)
    if len(M1_strides) > 0:
        M1_offsets[1:] = np.cumsum(M1_strides[:-1])
    return M1_offsets


def extract_gs_amp(
    list_cores_phi: list[np.ndarray] | list[list[np.ndarray]],
    method: str,
) -> np.complex128:
    """
    Amplitude of the all-phys-zero configuration of the MPS.

    For number representation this is the amplitude of the
    configuration with every state core in |0> and every mode core in
    the hierarchy-ground state.  In the ground-state-as-vacuum
    convention this is the physical ground-state amplitude; in the
    GS-as-state-core convention it is an unphysical "no state core
    occupied" configuration whose amplitude is typically ~0 for
    single-occupancy physical states.

    Parameters
    ----------
    1. list_cores_phi : list(np.ndarray) | list(list(np.ndarray))
                        MPS cores.  Nested statenumber groups are
                        flattened before contraction.

    2. method : str
                'number' or 'fullstate'.

    Returns
    -------
    1. amp : np.complex128
             <0,0,...,0|psi>.
    """
    if method == 'number':
        list_cores = []
        for group in list_cores_phi:
            list_cores.extend(group)
    elif method == 'fullstate':
        list_cores = list_cores_phi
    else:
        raise ValueError(f'Unknown method {method!r}.')

    if not list_cores:
        raise ValueError(
            'extract_gs_amp requires at least one core; '
            'got empty list_cores after flattening.'
        )

    M2_env = np.array([[1.0]], dtype=np.complex128)
    for core in list_cores:
        # core shape (l, p, r); slice at p=0 to get (l, r) and chain.
        M2_env = M2_env @ core[:, 0, :]
    return M2_env[0, 0]


def extract_psi(
    list_cores_phi: list[np.ndarray] | list[list[np.ndarray]],
    method: str,
    M1_modes_per_state: np.ndarray,
) -> np.ndarray:
    """
    Extracts the physical (zero-auxiliary) wavefunction from an MPS.

    Slices each mode core at occupation index 0 and contracts the
    resulting bond matrices. For number representation, each state
    core is sliced at physical index 1 (occupied) or 0 (unoccupied)
    according to a one-hot encoding.

    Parameters
    ----------
    1. list_cores_phi: list(np.ndarray) | list(list(np.ndarray))
                        MPS cores of the current wavefunction. For
                        fullstate, a flat list
                        [state_core, mode_core_0, ...]. For
                        number, nested:
                        list_cores_phi[s] = [state_core_s, mode_core_s0, ...]
    2. method: str
                Tensor encoding type ('fullstate' or 'number').
    3. M1_modes_per_state: np.ndarray(int)
                           Number of bath modes per state.

    Returns
    -------
    1. V1_phi_result: np.ndarray(complex)
                       Physical wavefunction, shape (n_state,).
    """
    M1_modes_per_state = np.asarray(M1_modes_per_state, dtype=int)
    n_total_modes = int(np.sum(M1_modes_per_state))

    if method == 'fullstate':
        list_cores_sliced = [list_cores_phi[0]]
        for i in range(n_total_modes):
            core_m = list_cores_phi[i + 1]
            list_cores_sliced.append(core_m[:, 0, :])
        M2_env = list_cores_sliced[0]
        for i in range(n_total_modes):
            M2_env = np.tensordot(M2_env, list_cores_sliced[i + 1], 1)
        # M2_env shape: (1, n_state, 1) — collapse trivial OBC boundary dims
        V1_phi_result = M2_env[0, :, 0]

    elif method == 'number':
        # list_cores_phi[s] = [state_core_s, mode_core_s0, ...]
        n_state = len(list_cores_phi)

        # Pre-contract each group's mode-core chain (occ=0 slices) into a
        # bond matrix. These products don't depend on target_state, so
        # computing them once drops the per-target work from
        # O(n_total_cores * chi^2) to O(n_state * chi^3).
        list_mode_envs = []
        for group in list_cores_phi:
            dim = group[0].shape[-1]
            M2_mode_env = np.eye(dim, dtype=np.complex128)
            for core_m in group[1:]:
                M2_mode_env = M2_mode_env @ core_m[:, 0, :]
            list_mode_envs.append(M2_mode_env)

        # Per-target contraction only touches the state cores and the
        # pre-contracted mode envs. The one-hot slicing selects phys=1
        # at target_state's axis and phys=0 elsewhere.
        V1_phi_result = np.zeros(n_state, dtype=np.complex128)
        for target_state in range(n_state):
            phys = 1 if target_state == 0 else 0
            M2_env = list_cores_phi[0][0][:, phys, :] @ list_mode_envs[0]
            for s in range(1, n_state):
                phys = 1 if s == target_state else 0
                M2_env = (
                    M2_env
                    @ list_cores_phi[s][0][:, phys, :]
                    @ list_mode_envs[s]
                )
            V1_phi_result[target_state] = M2_env[0, 0]
    else:
        raise ValueError(f'Unknown method {method!r}.')
    return V1_phi_result

def phi_aux(
    list_cores_phi: list[np.ndarray] | list[list[np.ndarray]],
    method: str,
    M1_modes_per_state: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    """
    Extracts an auxiliary-state wavefunction from an MPS.

    Slices each mode core at the occupation number given by indices,
    then contracts the resulting bond matrices to yield a state vector.
    For indices = [0, 0, ..., 0], this returns phi_0.

    Parameters
    ----------
    1. list_cores_phi: list(np.ndarray) | list(list(np.ndarray))
                        MPS cores of the current wavefunction. For
                        fullstate, a flat list
                        [state_core, mode_core_0, ...]. For
                        number, nested:
                        list_cores_phi[s] = [state_core_s, mode_core_s0, ...]
    2. method: str
                Tensor encoding type ('fullstate' or
                'number').
    3. M1_modes_per_state: np.ndarray(int)
                           Number of bath modes per state.
    4. indices: list(int)
                 Per-mode occupation numbers selecting which auxiliary
                 to extract. Length must equal total number of modes.

    Returns
    -------
    1. phi_aux: np.ndarray(complex)
                 Auxiliary wavefunction, shape (n_state,).
    """
    M1_modes_per_state = np.asarray(M1_modes_per_state, dtype=int)
    n_total_modes = int(np.sum(M1_modes_per_state))

    if method == 'fullstate':
        list_cores_sliced = [list_cores_phi[0]]
        for i in range(n_total_modes):
            core_m = list_cores_phi[i + 1]
            list_cores_sliced.append(core_m[:, indices[i], :])
        M2_env = list_cores_sliced[0]
        for i in range(n_total_modes):
            M2_env = np.tensordot(M2_env, list_cores_sliced[i + 1], 1)
        # M2_env shape: (1, n_state, 1) — collapse trivial OBC boundary dims
        V1_phi_result = M2_env[0, :, 0]

    elif method == 'number':
        # list_cores_phi[s] = [state_core_s, mode_core_s0, ...]
        n_state = len(list_cores_phi)

        # Pad indices with zeros for padded mode cores (non-bath states
        # have extra cores beyond the real mode count).
        n_total_padded = sum(len(g) - 1 for g in list_cores_phi)
        if len(indices) < n_total_padded:
            indices = list(indices) + [0] * (n_total_padded - len(indices))

        # Build sliced list from the labeled-core view: state cores kept
        # as-is (Dl, 2, Dr), mode cores sliced at the requested occupation
        # index. mode_flat_idx tracks the running position into `indices`
        # across all mode cores.
        list_cores_sliced = []
        mode_flat_idx = 0
        for core, is_state_core, _ in _flat_cores_with_labels(list_cores_phi):
            if is_state_core:
                list_cores_sliced.append(core)
            else:
                list_cores_sliced.append(core[:, indices[mode_flat_idx], :])
                mode_flat_idx += 1

        # Contract full chain. State cores contribute a dim-2 axis each;
        # mode slices are 2D and contract purely over bond indices.
        # Pre-collapse shape: (1, 2, 2, ..., 2, 1). The trailing indexing
        # drops the trivial OBC boundary dims, leaving (2, 2, ..., 2) with
        # one axis per system state.
        M2_env = list_cores_sliced[0]
        for core in list_cores_sliced[1:]:
            M2_env = np.tensordot(M2_env, core, 1)
        M2_env = M2_env[0, ..., 0]

        # Extract amplitude for each state by selecting index 1 at that state's
        # axis and 0 everywhere else. Reset the previous state's index before
        # advancing to avoid stale 1s — starting from i=1 so i=0 is never
        # "reset" (it is freshly set on the first iteration).
        list_state_idx = [0] * n_state
        V1_phi_result = np.zeros(n_state, dtype=np.complex128)
        for i in range(n_state):
            if i > 0:
                list_state_idx[i - 1] = 0
            list_state_idx[i] = 1
            V1_phi_result[i] = M2_env[tuple(list_state_idx)]
    else:
        raise ValueError(f'Unknown method {method!r}.')

    return V1_phi_result

def tensor_add(
    list_tensor_cores_1: list,
    list_tensor_cores_2: list,
    epsilon: float,
    bond_dim_max: int,
) -> list:
    """
    Adds two MPS (or MPO) tensors and compresses the result.

    Accepts flat lists of cores or nested statenumber MPS (list-of-lists).
    For nested input the structure is flattened internally and restored on
    output.  For MPO inputs (rank-4 cores) the two physical indices are fused
    before addition and restored afterward.

    Block-concatenates corresponding cores, then calls tensor_compress.

    Reference: Oseledets, 'Tensor-Train Decomposition', SIAM J. Sci.
    Comput. 33(5), 2295-2317 (2011), Section 2.3.

    Parameters
    ----------
    1. list_tensor_cores_1: list
                            First MPS/MPO.  Either a flat list of cores
                            or a nested list-of-lists (statenumber format).
    2. list_tensor_cores_2: list
                            Second MPS/MPO, same format as list_tensor_cores_1.
    3. epsilon: float
                SVD truncation threshold for compression.
    4. bond_dim_max: int
                     Maximum bond dimension after compression.

    Returns
    -------
    1. list_cores_sum: list
                       Compressed cores of the sum, same structure as inputs.
    """
    # Detect nested statenumber MPS (list-of-lists) and flatten before algebra.
    is_nested = isinstance(list_tensor_cores_1[0], list)
    if is_nested:
        list_modes_per_site = [len(g) - 1 for g in list_tensor_cores_1]
        list_tensor_cores_1 = flatten_cores(list_tensor_cores_1)
        list_tensor_cores_2 = flatten_cores(list_tensor_cores_2)

    if len(list_tensor_cores_1) != len(list_tensor_cores_2):
        raise ValueError(
            f'MPS/MPO core count mismatch: '
            f'{len(list_tensor_cores_1)} != {len(list_tensor_cores_2)}.'
        )
    # Verify open boundary conditions: left bond of first core and right
    # bond of last core must each be 1 (no dangling virtual indices).
    if list_tensor_cores_1[0].shape[0] != 1:
        raise ValueError(
            f'list_tensor_cores_1: left bond of first core must be 1, '
            f'got {list_tensor_cores_1[0].shape[0]}.'
        )
    if list_tensor_cores_1[-1].shape[-1] != 1:
        raise ValueError(
            f'list_tensor_cores_1: right bond of last core must be 1, '
            f'got {list_tensor_cores_1[-1].shape[-1]}.'
        )
    if list_tensor_cores_2[0].shape[0] != 1:
        raise ValueError(
            f'list_tensor_cores_2: left bond of first core must be 1, '
            f'got {list_tensor_cores_2[0].shape[0]}.'
        )
    if list_tensor_cores_2[-1].shape[-1] != 1:
        raise ValueError(
            f'list_tensor_cores_2: right bond of last core must be 1, '
            f'got {list_tensor_cores_2[-1].shape[-1]}.'
        )
    list_cores_sum = []
    # Direct-sum (block-diagonal) concatenation of MPS cores.
    # Boundary cores (first/last) keep bond dim 1 on the open end;
    # interior cores are block-diagonal in both bond indices.
    for core_idx in range(len(list_tensor_cores_1)):
        core_1 = list_tensor_cores_1[core_idx]
        core_2 = list_tensor_cores_2[core_idx]
        # Bond dims are always the first and last axes; physical dims are
        # everything in between (one axis for MPS, two for MPO, etc.)
        bond_left_1, bond_right_1 = core_1.shape[0], core_1.shape[-1]
        bond_left_2, bond_right_2 = core_2.shape[0], core_2.shape[-1]
        phys_shape = core_1.shape[1:-1]
        if core_1.shape[1:-1] != core_2.shape[1:-1]:
            raise ValueError(
                f'Physical dimension mismatch at core {core_idx}: '
                f'{core_1.shape[1:-1]} != {core_2.shape[1:-1]}.'
            )
        if core_idx == 0:
            # Left boundary: left bond stays 1, right bond concatenated
            core = np.zeros(
                (1, *phys_shape, bond_right_1 + bond_right_2),
                dtype=np.complex128,
            )
            core[..., :bond_right_1] = core_1
            core[..., bond_right_1:] = core_2
        elif core_idx == len(list_tensor_cores_1) - 1:
            # Right boundary: left bond concatenated, right bond stays 1
            core = np.zeros(
                (bond_left_1 + bond_left_2, *phys_shape, 1),
                dtype=np.complex128,
            )
            core[:bond_left_1, ...] = core_1
            core[bond_left_1:, ...] = core_2
        else:
            # Interior: block-diagonal in both bond indices
            core = np.zeros(
                (bond_left_1 + bond_left_2, *phys_shape,
                 bond_right_1 + bond_right_2),
                dtype=np.complex128,
            )
            core[:bond_left_1, ..., :bond_right_1] = core_1
            core[bond_left_1:, ..., bond_right_1:] = core_2
        list_cores_sum.append(core)

    list_cores_sum = tensor_compress(list_cores_sum, epsilon, bond_dim_max)
    if is_nested:
        list_cores_sum = unflatten_cores(list_cores_sum, list_modes_per_site)
    return list_cores_sum


def calc_mps_complexity(list_cores: list[np.ndarray]) -> int:
    """
    Computes a scalar complexity proxy for an MPS.

    Each core of shape (D_left, d_phys, D_right) contributes
        D_left * D_right * max(D_left, D_right) * d_phys
    and the total is summed across cores.  This matches the leading cost
    of a truncated SVD on the core reshaped as a (D_left * d_phys) x D_right
    matrix (or its transpose, whichever orientation is larger), and serves
    as a proxy for the overall work per matvec-then-compress cycle.

    Parameters
    ----------
    1. list_cores: list(np.ndarray)
                    MPS cores, each shaped (D_left, d_phys, D_right).

    Returns
    -------
    1. complexity: int
                    Sum of per-core complexity scores.
    """
    complexity = 0
    for core in list_cores:
        D_left, d_phys, D_right = core.shape
        complexity += D_left * D_right * max(D_left, D_right) * d_phys
    return complexity


def scale_mps(
    cores: list[np.ndarray] | list[list[np.ndarray]],
    factor: complex,
) -> None:
    """Scale an MPS in-place by multiplying its first core by factor.

    Works for both flat and nested (statenumber list-of-lists) MPS structures.
    Multiplying a single boundary core is equivalent to scaling the whole MPS
    because bond-contracted MPS cores form a product.

    Parameters
    ----------
    1. cores: list(np.ndarray) | list(list(np.ndarray))
               MPS cores. For fullstate, a flat list
               [state_core, mode_core_0, ...]. For
               number, nested:
               cores[s] = [state_core_s, mode_core_s0, ...]
    2. factor: complex
                Scalar multiplier.

    Returns
    -------
    None
    """
    if isinstance(cores[0], list):
        cores[0][0] = cores[0][0] * factor
    else:
        cores[0] = cores[0] * factor


def tensor_compress(
    list_cores: list[np.ndarray],
    epsilon: float,
    bond_dim_max: int,
) -> list[np.ndarray]:
    """
    Compresses an MPS via the Oseledets rounding algorithm.

    Right-orthogonalizes, then left-sweeps with truncated SVD to
    reduce bond dimensions while preserving accuracy up to epsilon.

    Reference: Oseledets, 'Tensor-Train Decomposition', SIAM J. Sci.
    Comput. 33(5), 2295-2317 (2011), Algorithm 2 (TT-rounding), p. 2305.

    Parameters
    ----------
    1. list_cores: list(np.ndarray)
                     MPS cores to compress.
    2. epsilon: float
                SVD truncation threshold (applied to normalized
                singular values).
    3. bond_dim_max: int
                     Maximum bond dimension after compression.

    Returns
    -------
    1. list_cores_compressed: list(np.ndarray)
                              Compressed MPS cores.
    """
    n_cores = len(list_cores)
    list_cores_compressed = []
    list_cores_ortho = []
    core_cur = list_cores[n_cores - 1]

    # === Step 1: Right-orthogonalization (Oseledets Alg. 2, p. 2305, line 1) ===
    # Sweep right-to-left, factoring each core into R @ Q via RQ
    # decomposition. Q (right-orthogonal) is stored; R is absorbed
    # into the neighboring core to the left.
    #
    # Right-orthogonalizing first ensures that during the subsequent
    # left-to-right SVD sweep, the singular values at each bond give the
    # exact truncation error for that bipartition of the chain. Without
    # this gauge fix, the SVD threshold would not have a well-defined
    # relationship to the overall approximation error.
    #
    # RQ for complex matrices uses QR of the conjugate transpose:
    #   core^H = Qt @ Rt   (np.linalg.qr)
    #   core   = Rt^H @ Qt^H  =  R @ Q
    # Qt^H has orthonormal rows: (Qt^H)(Qt^H)^H = Qt^H Qt* = I
    # because Qt has orthonormal columns (Qt^H Qt = I).
    for i in range(n_cores - 1, 0, -1):
        shape = core_cur.shape
        # Reshape (Dl, ..., Dr) → (Dl, phys*Dr) to treat as a matrix;
        # the left bond index is separated for the RQ factorization
        core_cur = core_cur.reshape(shape[0], -1)
        Qt, Rt = np.linalg.qr(core_cur.conj().T, mode='reduced')
        R = Rt.conj().T   # (Dl, chi): factor passed left
        Q = Qt.conj().T   # (chi, phys*Dr): right-orthogonal factor
        # Restore physical indices: (chi, ..., Dr)
        Q = Q.reshape(Q.shape[0], *shape[1:])
        list_cores_ortho.append(Q)
        # Absorb R into the core to the left, propagating gauge freedom
        # leftward so the next iteration operates on the updated core
        core_cur = np.tensordot(list_cores[i - 1], R, 1)
    list_cores_ortho.append(core_cur)
    # Reverse: list was built right-to-left, need left-to-right order
    list_cores_ortho.reverse()

    # === Step 2: SVD compression sweep (Oseledets Alg. 2, p. 2305, line 2) ===
    # Sweep left-to-right through the right-orthogonalized MPS. At each
    # bond, the current core is unfolded into a matrix and its SVD
    # gives the Schmidt decomposition across that bipartition. Singular
    # values below the threshold (or beyond bond_dim_max) are discarded,
    # reducing the bond dimension. U is kept as a new left-orthogonal
    # core; the remaining weight S*Vh is absorbed into the next core
    # to maintain the gauge and propagate the truncation error rightward.
    core_cur = list_cores_ortho[0]
    for i in range(n_cores - 1):
        shape = core_cur.shape
        # Reshape (Dl, ..., Dr) → (Dl*phys, Dr) for SVD bipartition;
        # left bond and physical indices are merged into the row index
        core_cur = core_cur.reshape(-1, shape[-1])
        # scipy.linalg.svd returns the decomposition A = U diag(S) Vh,
        # where Vh = V† is the Hermitian conjugate (conjugate transpose)
        # of the right singular vector matrix V (Oseledets eq. 2.1,
        # p. 2296). The gesvd driver is used for better numerical
        # stability than the default gesdd.
        U, S, Vh = scipy_svd(core_cur, full_matrices=False, lapack_driver='gesvd')
        # Truncation step 1: drop singular values below a relative
        # threshold. S is normalized so that the threshold is scale-
        # invariant; epsilon^2 is used because the error in the state
        # norm is quadratic in the singular values (Frobenius norm).
        # determine_error_thresh finds the largest cutoff such that
        # the discarded squared norm stays within epsilon^2.
        normalized_S = S / np.linalg.norm(S)
        singular_value_threshold = determine_error_thresh(
            np.flip(normalized_S), epsilon * epsilon,
        )
        S[normalized_S <= singular_value_threshold] = 0.0
        # Truncation step 2: hard cap at bond_dim_max to prevent
        # bond dimensions from growing beyond the MPS budget,
        # independent of the accuracy-based threshold above
        if S.shape[0] > bond_dim_max:
            S[bond_dim_max:] = 0.0
        rank = np.count_nonzero(S)
        U_trunc = U[:, :rank]
        Vh_trunc = Vh[:rank, :]
        # Reshape U back: (Dl, ..., rank) — left-orthogonal core
        list_cores_compressed.append(U_trunc.reshape(*shape[:-1], rank))
        # Absorb singular values into Vh and contract with the next
        # right-orthogonal core to form the new center core. This keeps
        # all discarded weight local to the current bond.
        M2_weighted_vh = S[:rank, None] * Vh_trunc
        core_cur = np.tensordot(
            M2_weighted_vh, list_cores_ortho[i + 1], 1,
        )
    # Last core carries all remaining bond weights accumulated from the
    # left sweep; it is appended as-is without further decomposition
    list_cores_compressed.append(core_cur)
    return list_cores_compressed

def contract_down(
    list_cores_phi: list[np.ndarray] | list[list[np.ndarray]],
    method: str,
    n_state: int,
) -> np.ndarray:
    """
    Computes approximate per-state norm squared by independent core contraction.

    Parameters
    ----------
    1. list_cores_phi: list(np.ndarray) | list(list(np.ndarray))
                        MPS cores. For fullstate representation, a flat
                        list [state_core, mode_core_0, ...]. For
                        number representation, nested:
                        list_cores_phi[s] = [state_core_s, mode_core_s0, ...]
    2. method: str
                'fullstate' or 'number'.
    3. n_state: int
                 Number of active system states.

    Returns
    -------
    1. V1_approx_norm_sq: np.ndarray(complex), shape (n_state,).
    """

    if method == 'fullstate':
        # Approximate per-state norm by treating each core independently:
        # replace each mode core A_m with sum_k |A_m[:,k,:]|^2, dropping
        # cross-bond interference terms. This is cheaper than the exact
        # double-layer contraction in contract_down_exact but overestimates
        # the norm when bond correlations are significant.
        # Sweep right-to-left, accumulating the squared transfer matrices.
        for i in range(len(list_cores_phi) - 1, 0, -1):
            outer_core = np.abs(list_cores_phi[i]) ** 2
            # Sum over physical index to get a bond-to-bond transfer matrix
            outer_core_contracted = np.sum(outer_core, axis=1)
            if i == len(list_cores_phi) - 1:
                result = outer_core_contracted
            else:
                result = outer_core_contracted @ result
        # Contract the state core (keeps physical index) with the
        # accumulated mode transfer matrices to get per-state values
        outer_core = np.abs(list_cores_phi[0]) ** 2
        result = outer_core @ result
        V1_approx_norm_sq = np.sum(result, axis=-1)[0]

    elif method == 'number':
        # list_cores_phi[s] = [state_core_s, mode_core_s0, ...]
        # Approximation: replace each mode core A_m with sum(|A_m|^2, axis=phys),
        # dropping cross-bond interference terms.
        list_labeled = _flat_cores_with_labels(list_cores_phi)
        # Precompute target_state-independent mode-core transfer matrices
        # (|A_m|^2 summed over the physical axis). State cores stay 3-D
        # to be sliced per target_state below.
        list_mode_transfer = [
            np.sum(np.abs(core) ** 2, axis=1) if not is_state_core else None
            for core, is_state_core, _ in list_labeled
        ]

        V1_approx_norm_sq = np.zeros(n_state, dtype=np.complex128)
        for target_state in range(n_state):
            M2_env = None
            for idx, (core, is_state_core, state_idx) in enumerate(
                list_labeled,
            ):
                if is_state_core:
                    phys = 1 if state_idx == target_state else 0
                    M2_slice = np.abs(core[:, phys, :]) ** 2
                    M2_env = (
                        M2_slice if M2_env is None
                        else np.tensordot(M2_env, M2_slice, 1)
                    )
                else:
                    M2_env = np.tensordot(
                        M2_env, list_mode_transfer[idx], 1,
                    )
            V1_approx_norm_sq[target_state] = M2_env[0][0]
    else:
        raise ValueError(f'Unknown method {method!r}.')

    return V1_approx_norm_sq

def contract_down_exact(
    list_cores_phi: list[np.ndarray] | list[list[np.ndarray]],
    method: str,
    n_state: int,
) -> np.ndarray:
    """
    Computes exact per-state norm squared via double-layer contraction.

    Parameters
    ----------
    1. list_cores_phi: list(np.ndarray) | list(list(np.ndarray))
                        MPS cores. For fullstate representation, a flat
                        list [state_core, mode_core_0, ...]. For
                        number representation, nested:
                        list_cores_phi[s] = [state_core_s, mode_core_s0, ...]
    2. method: str
                'fullstate' or 'number'.
    3. n_state: int
                 Number of active system states.

    Returns
    -------
    1. V1_norm_sq: np.ndarray(complex), shape (n_state,).
    """

    if method not in ('fullstate', 'number'):
        raise ValueError(
            f'Unknown method {method!r}. Expected '
            f"'fullstate' or 'number'."
        )

    if method == 'fullstate':
        M2_boundary = np.sum(list_cores_phi[-1], axis=-1)
        M2_boundary_conj = np.conj(M2_boundary)
        M2_env = np.einsum(
            'ij, kj -> ik', M2_boundary, M2_boundary_conj,
        )
        for i in range(len(list_cores_phi) - 2, 0, -1):
            T3_core = list_cores_phi[i]
            T3_core_conj = np.conj(T3_core)
            M2_env = np.einsum(
                'ijk, il, mjl -> im', T3_core, M2_env, T3_core_conj,
                optimize=True,
            )
        M2_first_core = np.sum(list_cores_phi[0], axis=0)
        M2_first_core_conj = np.conj(M2_first_core)
        V1_norm_sq = np.diag(np.einsum(
            'ij,jl,kl -> ik',
            M2_first_core, M2_env, M2_first_core_conj,
            optimize=True,
        ))
    elif method == 'number':
        # list_cores_phi[s] = [state_core_s, mode_core_s0, ...]
        # One right-to-left double-layer sweep per target state.
        # list_labeled is precomputed once; the inner loop reuses it for
        # each target_state to avoid rebuilding per iteration.
        if len(list_cores_phi) < n_state:
            raise ValueError(
                f'contract_down_exact number representation expects at least '
                f'{n_state} groups, got {len(list_cores_phi)}.'
            )
        for s in range(len(list_cores_phi) - 1):
            right_bond = list_cores_phi[s][-1].shape[2]
            left_bond = list_cores_phi[s + 1][0].shape[0]
            if right_bond != left_bond:
                raise ValueError(
                    f'Environment/bond mismatch at group {s}: right bond '
                    f'{right_bond} != left bond {left_bond} of group {s + 1}.'
                )
        list_labeled = _flat_cores_with_labels(list_cores_phi)
        V1_norm_sq = np.zeros(n_state, dtype=np.complex128)

        for target_state in range(n_state):
            # R is the right environment matrix, shape (chi, chi).
            # Initialized to scalar 1 for the right boundary (OBC).
            R = np.ones((1, 1), dtype=np.complex128)

            for core, is_state_core, state_idx in reversed(list_labeled):
                core_conj = np.conj(core)
                if is_state_core:
                    # One-hot selection: project onto occupied (1) or
                    # unoccupied (0) physical index for this state.
                    phys = 1 if state_idx == target_state else 0
                    M2_slice = core[:, phys, :]
                    R = np.einsum(
                        'ai,ij,bj->ab',
                        M2_slice, R, core_conj[:, phys, :],
                        optimize=True,
                    )
                else:
                    # Mode core: sum over physical index in double layer.
                    R = np.einsum(
                        'ami,ij,bmj->ab', core, R, core_conj,
                        optimize=True,
                    )

            V1_norm_sq[target_state] = R[0, 0]
    return V1_norm_sq


def tensor_to_array(
    list_cores_phi: list[np.ndarray] | list[list[np.ndarray]],
    method: str,
    M1_modes_per_state: np.ndarray,
    system: HopsSystem,
    mode: HopsModes,
) -> np.ndarray:
    """
    Returns the ground and first-order auxiliary wavefunctions as a flat array
    in adHOPS form, for debugging purposes.

    Parameters
    ----------
    1. list_cores_phi: list(np.ndarray) | list(list(np.ndarray))
                   MPS cores of the current wavefunction. For
                   fullstate, a flat list
                   [state_core, mode_core_0, ...]. For
                   number, nested:
                   list_cores_phi[s] = [state_core_s, mode_core_s0, ...]

    2. method: str
                Tensor encoding type ('fullstate' or
                'number').

    3. M1_modes_per_state: np.ndarray(int)
                           Number of bath modes per state.

    4. system: HopsSystem
                System object providing current state count.

    5. mode: HopsModes
              Mode object providing list_modeidx_abs.

    Returns
    -------
    1. V1_arrayform: np.array(complex)
                   Array of length n_state * (n_mode + 1) containing phi_0
                   followed by each first-order auxiliary wavefunction.
    """
    n_state = system.size
    n_mode = len(mode.list_modeidx_abs)
    V1_arrayform = np.zeros(n_state * (n_mode + 1), dtype=np.complex128)
    V1_arrayform[0:n_state] = extract_psi(list_cores_phi, method, M1_modes_per_state)
    for i_mode in range(n_mode):
        list_occ_idx = [0] * n_mode
        list_occ_idx[i_mode] = 1
        V1_arrayform[(i_mode + 1) * n_state:(i_mode + 2) * n_state] = (
            phi_aux(list_cores_phi, method, M1_modes_per_state, list_occ_idx)
        )
    return V1_arrayform
