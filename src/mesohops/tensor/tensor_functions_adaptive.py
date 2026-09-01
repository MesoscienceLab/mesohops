from __future__ import annotations

from collections.abc import Callable

__title__ = 'Tensor Functions Adaptive'
__author__ = 'B. Z. Citty'
__maintainer__ = 'B. Z. Citty'

import numpy as np
import scipy as sp

from mesohops.basis.basis_functions import determine_error_thresh
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.physical_constants import hbar
from mesohops.util.tensor_operations import contract_down_exact


def tensor_state_adaptive_check_add_state(
    list_cores_phi: list,
    ham: np.ndarray,
    old_states: list[int],
    n_state_full: int,
    n_state: int,
    delta_s: float,
    state_list: list[int] | np.ndarray,
    method: str,
    M1_modes_per_state: np.ndarray,
) -> list[int]:
    """
    Adaptive algorithm which checks to see if new states and their corresponding modes
    should be added to the tensor train.

    Parameters
    ----------
    1. list_cores_phi: list
                       MPS cores of the wavefunction.

    2. ham: np.ndarray
            Full Hamiltonian matrix, shape (n_state_full, n_state_full).

    3. old_states: list(int)
                   Absolute state indices already marked for removal
                   (excluded from the new-state candidates).

    4. n_state_full: int
                     Total number of states in the full Hilbert space.

    5. n_state: int
                Number of currently active states.

    6. delta_s: float
                Adaptive threshold for state inclusion.

    7. state_list: list(int) | np.ndarray
                   Currently active absolute state indices.

    8. method: str
               Tensor encoding type ('fullstaterepresentation' or
               'statenumberrepresentation').

    9. M1_modes_per_state: np.ndarray(int)
                           Number of bath modes per state.

    Returns
    -------
    1. list_new_states: list(int)
                        Absolute state indices to add.
    """

    if method == 'fullstate':
        # Embed the adaptive-basis state core into the full Hilbert space,
        # apply H to estimate flux into states outside the current basis.
        state_core = list_cores_phi[0]
        extended_state_core = np.zeros(
            shape=(1, n_state_full, state_core.shape[2]), dtype=np.complex128
        )
        # Scatter adaptive-basis entries into full-space positions
        for i in range(state_core.shape[2]):
            extended_state_core[np.ix_(np.array([0]), state_list, np.array([i]))] = (
                state_core[np.ix_(np.array([0]), np.arange(n_state), np.array([i]))]
            )
        # H @ psi gives the boundary flux into all states
        boundary_state_core = np.tensordot(ham, extended_state_core, axes=([1], [1]))
        boundary_state_core = np.swapaxes(boundary_state_core, 0, 1)

        # Contract mode indices to get per-state flux magnitude.
        # Division by hbar^2 converts to the dimensionless error metric.
        list_cores_tmp = list_cores_phi.copy()
        list_cores_tmp[0] = boundary_state_core
        V1_flux = (
            contract_down_exact(list_cores_tmp, method, n_state)
            / hbar**2
        )
        # Zero flux for states already in basis (only check new states)
        V1_flux[state_list] = 0
        adaptive_thresh = determine_error_thresh(np.sort(V1_flux), delta_s * delta_s)
        list_new_states = np.where(V1_flux > adaptive_thresh)[0]
        list_new_states = list(set(list_new_states) - set(old_states))
        return list_new_states
    elif method == 'number':
        # Contract mode indices to get per-state population sum_k |phi[k,s]|^2
        V1_phi_0 = contract_down_exact(list_cores_phi, method, n_state)
        # Embed into full Hilbert space, then compute flux:
        # |H|^2 @ |psi|^2 / hbar^2 estimates outgoing flux per state
        V1_phi_0_full = np.zeros(n_state_full, dtype=np.complex128)
        V1_phi_0_full[state_list] = V1_phi_0[:]
        V1_flux = np.abs(ham**2) @ V1_phi_0_full / hbar**2
        # set the flux to states in the basis to 0.
        V1_flux[state_list] = 0.0
        adaptive_thresh = determine_error_thresh(np.sort(V1_flux), delta_s * delta_s)
        list_new_states = np.where(V1_flux > adaptive_thresh)[0]
        list_new_states = list(set(list_new_states) - set(old_states))
        return list_new_states
    else:
        raise UnsupportedRequest(method, 'tensor_state_adaptive_check_add_state')


def tensor_state_adaptive_check_remove_state(
    list_cores_phi: list,
    ham: np.ndarray,
    z_step: np.ndarray,
    n_state_full: int,
    n_state: int,
    delta_s: float,
    state_list: list[int] | np.ndarray,
    method: str,
    M1_modes_per_state: np.ndarray,
    dsystem_dt: Callable,
) -> np.ndarray:
    """
    Adaptive algorithm which checks to see if states and their corresponding modes
    should be removed from the tensor train.

    Parameters
    ----------
    1. list_cores_phi: list
                       MPS cores of the wavefunction.

    2. ham: np.ndarray
            Full Hamiltonian matrix, shape (n_state_full, n_state_full).

    3. z_step: np.ndarray
               Noise values for the current time step, passed to
               dsystem_dt as (z_mem, z_rnd, z_rnd2).

    4. n_state_full: int
                     Total number of states in the full Hilbert space.

    5. n_state: int
                Number of currently active states.

    6. delta_s: float
                Adaptive threshold for state removal.

    7. state_list: list(int) | np.ndarray
                   Currently active absolute state indices.

    8. method: str
               Tensor encoding type ('fullstaterepresentation' or
               'statenumberrepresentation').

    9. M1_modes_per_state: np.ndarray(int)
                           Number of bath modes per state.

    10. dsystem_dt: Callable
                    Derivative closure that takes (z_mem, z_rnd, z_rnd2)
                    and returns the MPS derivative cores.

    Returns
    -------
    1. old_state_indices: np.ndarray(int)
                          Relative indices (into state_list) of states
                          to remove.
    """
    list_states = state_list
    list_cores_phi = list_cores_phi.copy()

    if method == 'fullstate':
        # --- Flux-in: time derivative contribution ---
        # Compute d(phi)/dt, divide by hbar, then contract mode indices
        # to get per-state derivative magnitude
        list_cores_d_phi = dsystem_dt(z_step[2], z_step[0], z_step[1])
        list_cores_d_phi[0] = list_cores_d_phi[0] / hbar
        V1_error = contract_down_exact(
            list_cores_d_phi, method, n_state
        )

        # --- Flux-out: norm contribution ---
        # Contract phi to get per-state norm squared sum_k |phi[k,s]|^2
        V1_norm_sq_by_state = contract_down_exact(
            list_cores_phi, method, n_state
        )

    elif method == 'number':
        # --- Flux-in: time derivative contribution ---
        list_cores_d_phi = dsystem_dt(z_step[2], z_step[0], z_step[1])
        list_cores_d_phi[0] = list_cores_d_phi[0] / hbar
        V1_error = contract_down_exact(
            list_cores_d_phi, method, n_state
        )

        # --- Flux-out: norm contribution ---
        V1_norm_sq_by_state = contract_down_exact(
            list_cores_phi, method, n_state
        )
    else:
        raise UnsupportedRequest(method, 'tensor_state_adaptive_check_remove_state')

    # Combine flux-in and flux-out into total error per state.
    # Extract off-diagonal couplings (remove on-site energies),
    # then compute sum_j |H_js|^2 * |psi_s|^2 / hbar^2 for each
    # basis state s — this estimates outgoing coupling flux.
    H2_sparse_hamiltonian = sp.sparse.coo_array(ham)
    H2_sparse_couplings = H2_sparse_hamiltonian - sp.sparse.diags(
        H2_sparse_hamiltonian.diagonal(0),
        format='csc',
        shape=H2_sparse_hamiltonian.shape,
    )
    # Keep only columns for states in the current basis
    H2_sparse_hamiltonian = H2_sparse_couplings[:, list_states]
    # sum_j |H_js|^2 for each state s (column-wise squared sum)
    V1_norm_sq = np.array(np.sum(np.abs(H2_sparse_hamiltonian).power(2), axis=0))
    V1_error += V1_norm_sq * V1_norm_sq_by_state / hbar**2
    # States with total error below threshold are candidates for removal
    adaptive_thresh = determine_error_thresh(np.sort(V1_error), delta_s * delta_s)
    old_state_indices = np.where(V1_error <= adaptive_thresh)[0]
    return old_state_indices
