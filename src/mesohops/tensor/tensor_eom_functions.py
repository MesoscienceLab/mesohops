"""
EOM helper functions for HopsTensorEOM.

These routines sit between the pure MPS algebra (tensor_operations.py) and the
physics layer (hops_tensor_eom.py). They handle MPO-MPS contraction and the
normalization correction factor.

Functions
---------
tensor_matvec_prod(list_cores_vec, list_cores_mpo, epsilon, bond_dim_max)
    Apply an MPO to an MPS and compress the result via SVD.

calc_norm_corr_tensor(wavefunction, psi, z_hat, list_avg_L2, mode,
    list_index_L2_by_mode)
    Compute the normalization correction factor needed for propagating
    the normalized nonlinear wave function.

apply_system_operator(list_cores_phi, O2_op_trimmed, method, k_max,
    M1_modes_per_state, epsilon, bond_dim_max)
    Apply a pre-trimmed system-space operator to an MPS.
"""

from __future__ import annotations

import numpy as np

from mesohops.basis.hops_modes import HopsModes
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.mpo_constructors import build_statenumber_operator_mpo
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.tensor_operations import (
    calc_mps_complexity,
    flatten_cores,
    phi_aux,
    tensor_compress,
    unflatten_cores,
)

__title__ = 'Tensor EOM Functions'
__author__ = 'B. Z. Citty'
__maintainer__ = 'B. Z. Citty'


def tensor_matvec_prod(
    list_cores_vec: list[np.ndarray],
    list_cores_mpo: list[np.ndarray],
    epsilon: float,
    bond_dim_max: int,
) -> tuple[list[np.ndarray], int]:
    """
    Performs the matrix-vector product in MPS form, then compresses the result.

    Parameters
    ----------
    1. list_cores_vec: list(np.ndarray)
                       MPS cores of the vector, each shaped
                       (dim_left, dim_phys, dim_right).
    2. list_cores_mpo: list(np.ndarray)
                       MPO cores of the operator, each shaped
                       (dim_left, dim_out, dim_in, dim_right).
    3. epsilon: float
                SVD truncation threshold for compression.
    4. bond_dim_max: int
                     Maximum bond dimension after compression.

    Returns
    -------
    1. list_cores_compressed: list(np.ndarray)
                              Compressed MPS cores of the result.
    2. complexity: int
                    Scalar complexity proxy (see calc_mps_complexity) of the
                    uncompressed contracted MPS, i.e. its peak size before
                    SVD truncation.
    """
    if len(list_cores_vec) == 0:
        raise ValueError(
            'tensor_matvec_prod requires at least one core; '
            'got empty list_cores_vec.'
        )
    # Detect nested statenumber MPS (list/tuple-of-list/tuple) and flatten
    # before contraction. Both list and tuple are accepted for the outer
    # and inner containers so callers aren't forced to wrap tuples; the
    # non-nested branch keeps identifying flat input by the ndarray type
    # of the first element.
    is_nested = len(list_cores_vec) > 0 and isinstance(
        list_cores_vec[0], (list, tuple)
    )
    if is_nested:
        # Each group has 1 state core + N mode cores; subtract 1 to get mode count.
        list_modes_per_site = [len(g) - 1 for g in list_cores_vec]
        list_cores_vec = flatten_cores(list_cores_vec)

    if len(list_cores_mpo) != len(list_cores_vec):
        raise ValueError('MPO and MPS must have the same number of cores.')

    list_cores_compressed = []
    for core_mpo, core_vec in zip(list_cores_mpo, list_cores_vec):
        # Contract over the shared physical index (operator applied to state),
        # interleaving MPO and MPS bond indices. Bond dimensions multiply,
        # requiring compression afterward.
        # L,R = MPO bond left/right; l,r = MPS bond left/right;
        # o = phys out; i = phys in (contracted)
        mpo_dim_left, dim_out, _, mpo_dim_right = core_mpo.shape
        vec_dim_left, _, vec_dim_right = core_vec.shape
        #TODO: Cite an equation
        core_contracted = np.einsum('LoiR,lir->LloRr', core_mpo, core_vec).reshape(
            mpo_dim_left * vec_dim_left, dim_out, mpo_dim_right * vec_dim_right
        )
        list_cores_compressed.append(core_contracted)

    # Peak-size proxy: sum of per-core complexity on the uncompressed MPS,
    # where MPO and MPS bond dimensions have multiplied. This is the largest
    # the tensor becomes in one matvec-then-compress cycle.
    complexity = calc_mps_complexity(list_cores_compressed)

    # Compress contracted MPS back to tractable bond dimension with SVD
    list_cores_result = tensor_compress(list_cores_compressed, epsilon, bond_dim_max)
    if is_nested:
        list_cores_result = unflatten_cores(list_cores_result, list_modes_per_site)
    return list_cores_result, complexity


def calc_norm_corr_tensor(
    wavefunction: HopsTensorWavefunction,
    psi: np.ndarray,
    z_hat: np.ndarray,
    list_avg_L2: list[complex],
    mode: HopsModes,
    list_index_L2_by_mode: list[int],
) -> float:
    """
    Computes the correction factor for propagating the normalized wave function.

    Parameters
    ----------
    1. wavefunction: HopsTensorWavefunction
                     MPS wavefunction container (provides list_cores_phi,
                     method, M1_modes_per_state).
    2. psi: np.ndarray(complex)
            Physical (system) wavefunction already extracted from the
            MPS. Passed in rather than re-extracted via extract_psi so
            a single RK4 step doesn't redo the full contraction on an
            unchanged wavefunction.
    3. z_hat: np.ndarray(complex)
              Combined noise + memory term, indexed by L2 operator.
    4. list_avg_L2: list(complex)
                    Expectation values <L_m> for each L2 operator.
    5. mode: HopsModes
             Mode object providing list_g, list_L2_coo.
    6. list_index_L2_by_mode: list(int)
                         L2 operator index for each hierarchy mode, ordered
                         by mode position.

    Returns
    -------
    1. delta: float
              Norm correction factor.
    """
    list_L2 = mode.list_L2_coo
    # z-component: sum_m z_hat_m * <L_m>
    delta = np.dot(z_hat, list_avg_L2)
    V1_psi = psi
    list_g = mode.list_g
    n_modes = len(mode.list_modeidx_abs)

    V1_psi_conj = np.conj(V1_psi)

    # Per-mode correction: for each hierarchy mode m, subtract <phi_0|L_m|phi_1_m>
    # and add <phi_0|phi_1_m> * <L_m>, where phi_1_m is the first-order auxiliary
    # along mode m scaled by V_m^- = sqrt(|g_m|) (Gao rescaling; see
    # X. Gao, J. Ren, A. Eisfeld, Z. Shuai, "Non-Markovian stochastic
    # Schrodinger equation: Matrix-product-state approach to the hierarchy of
    # pure states," Phys. Rev. A 105, L030202 (2022),
    # DOI: 10.1103/PhysRevA.105.L030202).
    for (mode_idx, l2_idx) in enumerate(list_index_L2_by_mode):
        # Unit vector in auxiliary space selecting mode_idx
        list_aux_idx = [0] * n_modes
        list_aux_idx[mode_idx] = 1
        # First-order auxiliary wavefunction scaled by V_m^- = sqrt(|g_m|),
        # the same prefactor the b rail carries in MpoBuilder.
        V1_phi_aux1 = (
            np.sqrt(np.abs(list_g[mode_idx]))
            * phi_aux(
                wavefunction.list_cores_phi,
                wavefunction.method,
                wavefunction.M1_modes_per_state,
                list_aux_idx,
            )
        )
        H2_lop = list_L2[l2_idx]
        avg_lop = list_avg_L2[l2_idx]
        # -<phi_0 | L_m | phi_1_m>
        delta -= V1_psi_conj @ (H2_lop @ V1_phi_aux1)
        # +<phi_0 | phi_1_m> * <L_m>
        delta += (V1_psi_conj @ V1_phi_aux1) * avg_lop
    return np.real(delta)


def apply_system_operator(
    list_cores_phi: list[np.ndarray],
    O2_op_trimmed: np.ndarray,
    method: str,
    k_max: int,
    M1_modes_per_state: np.ndarray,
    epsilon: float,
    bond_dim_max: int,
) -> list[np.ndarray]:
    """
    Apply a pre-trimmed system-space operator to an MPS.

    The operator must already be trimmed to the active state_list.

    Parameters
    ----------
    1. list_cores_phi: list(np.ndarray)
                       MPS cores of the wavefunction.
    2. O2_op_trimmed: np.ndarray(complex)
                      Operator in active basis, shape (n_active, n_active).
    3. method: str
               Tensor encoding type ('fullstate' or
               'number').
    4. k_max: int
              Maximum hierarchy depth.
    5. M1_modes_per_state: np.ndarray(int)
                           Number of bath modes per physical state.
    6. epsilon: float
                SVD truncation threshold.
    7. bond_dim_max: int
                     Maximum MPS bond dimension.

    Returns
    -------
    1. list_cores_phi: list(np.ndarray)
                       Updated MPS cores.
    """
    if hasattr(O2_op_trimmed, 'toarray'):
        O2_op_trimmed = O2_op_trimmed.toarray()
    if O2_op_trimmed.ndim != 2 or O2_op_trimmed.shape[0] != O2_op_trimmed.shape[1]:
        raise ValueError(
            f'O2_op_trimmed must be a square 2-D array, got shape '
            f'{O2_op_trimmed.shape}'
        )
    n_op = O2_op_trimmed.shape[0]

    if method == 'fullstate':
        # The first core has the full system Hilbert space as its physical
        # dimension (1, n_state, bond_right), so the operator can be applied
        # directly via matrix multiplication on the physical index.
        n_phys = list_cores_phi[0].shape[1]
        if n_op != n_phys:
            raise ValueError(
                f'Operator dimension ({n_op}) does not match physical '
                f'dimension of core 0 ({n_phys})'
            )
        # a = bond left; b = bond right;
        # i = phys out; j = phys in (contracted)
        T3_core_new = np.einsum(
            'ij,ajb->aib', O2_op_trimmed, list_cores_phi[0],
        )
        return [T3_core_new] + list_cores_phi[1:]
    elif method == 'number':
        # Each state occupies a separate binary core, so the operator
        # cannot be applied to a single core. Instead, build a full
        # operator MPO (with transfer matrices for off-diagonal elements)
        # and apply it to the MPS via MPO-MPS contraction + compression.
        list_cores_op = build_statenumber_operator_mpo(
            O2_op_trimmed, n_op, k_max, M1_modes_per_state,
        )
        # Observable/operator application path — peak-size tracking is
        # only meaningful inside the RK4 derivative, so discard the scalar.
        list_cores_phi_new, _ = tensor_matvec_prod(
            list_cores_phi, list_cores_op, epsilon, bond_dim_max,
        )
        return list_cores_phi_new
    else:
        raise NotImplementedError(
            f'apply_system_operator not implemented for method={method!r}'
        )


def build_physical_correction_mps(
    H1_psi_corr: np.ndarray,
    method: str,
    k_max: int,
    M1_modes_per_state: np.ndarray,
) -> list:
    """
    Build an MPS representing a correction to only the physical wavefunction
    (zero-auxiliary component) of the hierarchy.

    The result is a low-bond-dimension MPS where all mode cores project onto
    the m=0 occupation state. For fullstate this is rank-1 (bond dim 1). For
    statenumber, each state group gets a state core with the correction
    amplitude on the occupied index and zero on unoccupied, with m=0 projector
    mode cores.

    Parameters
    ----------
    1. H1_psi_corr: np.ndarray(complex)
                    The correction vector, shape (n_state,). Typically
                    C2_LT_corr_physical @ psi or C2_LT_corr_linear @ psi.

    2. method: str
               'fullstate' or 'number'.

    3. k_max: int
              Maximum hierarchy depth (mode core physical dim = k_max + 1).

    4. M1_modes_per_state: np.ndarray(int)
                           Number of bath modes per system state.

    Returns
    -------
    1. list_cores: list
                   MPS cores representing the correction. Structure matches
                   the wavefunction format (nested lists for statenumber,
                   flat list for fullstate).
    """
    n_state = len(H1_psi_corr)

    # Mode core projecting onto m=0: shape (1, k_max+1, 1)
    T3_mode_proj = np.zeros((1, k_max + 1, 1), dtype=np.complex128)
    T3_mode_proj[0, 0, 0] = 1.0

    if method == 'fullstate':
        T3_core_state = H1_psi_corr.reshape(1, n_state, 1)
        n_total_modes = int(M1_modes_per_state.sum())
        return [T3_core_state] + [T3_mode_proj.copy() for _ in range(n_total_modes)]

    elif method == 'number':
        list_cores = []
        for site in range(n_state):
            T3_core_state = np.zeros((1, 2, 1), dtype=np.complex128)
            T3_core_state[0, 1, 0] = H1_psi_corr[site]  # occupied
            # unoccupied index stays 0 — correction only contributes
            # when this site is occupied
            group = [T3_core_state] + [
                T3_mode_proj.copy() for _ in range(M1_modes_per_state[site])
            ]
            list_cores.append(group)
        return list_cores

    else:
        raise UnsupportedRequest(method, 'build_physical_correction_mps')


def build_physical_correction_mpo(
    C2_op: np.ndarray,
    method: str,
    k_max: int,
    M1_modes_per_state: np.ndarray,
) -> list[np.ndarray]:
    """
    Build a rank-1 MPO for a system-space operator that acts only on
    the physical wavefunction (k=0 hierarchy level).

    Mode cores carry |0><0| projectors so the operator is zero on all
    auxiliary (k>0) components. For fullstate the result is a flat list;
    for statenumber it mirrors the interleaved state/mode core layout
    of the main MPO.

    Parameters
    ----------
    1. C2_op: np.ndarray(complex)
              System-space operator, shape (n_state, n_state).
    2. method: str
               'fullstate' or 'number'.
    3. k_max: int
              Maximum hierarchy depth.
    4. M1_modes_per_state: np.ndarray(int)
                           Number of bath modes per system state.

    Returns
    -------
    1. list_cores_mpo: list(np.ndarray)
                       Rank-1 MPO cores, each shaped (1, d, d, 1).
    """
    n_state = C2_op.shape[0]
    d_mode = k_max + 1

    # |0><0| projector on mode space
    T4_mode_proj = np.zeros((1, d_mode, d_mode, 1), dtype=np.complex128)
    T4_mode_proj[0, 0, 0, 0] = 1.0

    if method == 'fullstate':
        T4_state = C2_op.reshape(1, n_state, n_state, 1)
        n_total_modes = int(M1_modes_per_state.sum())
        return [T4_state] + [T4_mode_proj.copy() for _ in range(n_total_modes)]

    elif method == 'number':
        # Build via the general operator MPO, then replace identity
        # mode cores with |0><0| projectors.
        list_cores = build_statenumber_operator_mpo(
            C2_op, n_state, k_max, M1_modes_per_state,
        )
        for i, core in enumerate(list_cores):
            # Mode cores have physical dim k_max+1; state cores have dim 2
            if core.shape[1] == d_mode:
                T4_proj = np.zeros_like(core)
                # Keep bond structure but zero out k>0
                T4_proj[:, 0, 0, :] = core[:, 0, 0, :]
                list_cores[i] = T4_proj
        return list_cores

    else:
        raise UnsupportedRequest(method, 'build_physical_correction_mpo')
