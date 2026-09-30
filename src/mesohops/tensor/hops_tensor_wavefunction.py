"""
MPS wavefunction container for tensor HOPS.

Defines HopsTensorWavefunction, which stores the hierarchy wavefunction as a Matrix
Product State (MPS) and provides initialization, normalization, operator
application, and bond-dimension management.
"""
from __future__ import annotations

import bisect

import numpy as np

from mesohops.basis.hops_system import HopsSystem
from mesohops.util.exceptions import LockedException, UnsupportedRequest
from mesohops.util.tensor_operations import (
    extract_gs_amp,
    extract_psi,
    flatten_cores,
    unflatten_cores,
)

__title__ = 'HopsTensorWavefunction Class'
__author__ = 'B. Z. Citty'
__maintainer__ = 'B. Z. Citty'


class HopsTensorWavefunction:
    """
    Matrix Product State (MPS) representation of the HOPS wavefunction.

    In the standard HOPS formalism, the full state vector is a flat array
    indexed by (physical state, auxiliary index). HopsTensorWavefunction re-encodes this
    as an MPS whose sites correspond to system states and bath modes. The
    exact local structure depends on the encoding method: in the statenumber
    encoding each system state is represented by a binary (dimension-2) core,
    while mode cores carry a local dimension of ``k_max + 1``, reflecting the
    hierarchy depth along that mode axis. The bond dimensions between adjacent
    sites are determined by SVD compression and are bounded by ``bond_dim_max``.

    Time evolution is driven either by applying a Matrix Product Operator
    (MPO) and then truncating via SVD (Runge-Kutta path), or by a
    single-site TDVP sweep that naturally respects the MPS geometry.

    Responsibilities
    ----------------
    - Store and update the MPS cores (``list_cores_phi``).
    - Provide norm and bond-dimension utilities.
    - Apply system-space operators to the MPS.

    MPO construction and time-derivative computation are owned by
    ``HopsTensorEOM``, which reads config flags from this class.

    Key attributes
    --------------
    list_cores_phi: list(np.ndarray)
        MPS cores for the current wavefunction.
    k_max: int
        Maximum hierarchy depth; local dimension of each mode core = k_max + 1.
    mps_epsilon: float
        SVD truncation threshold used during MPS compression
        (mirrors tensor_param['MPS_EPSILON']).
    mpo_epsilon: float
        SVD truncation threshold for MPO assembly, applied where MPO parts
        are summed and on the coupling-block factorization behind the
        general generator (mirrors tensor_param['MPO_EPSILON']).
    flag_mpo_optimize: bool
        True (default) builds the number-method generator as one MPO at the
        rank of each bond's coupling block; False takes the two-MPO
        reference path (mirrors tensor_param['FLAG_MPO_OPTIMIZE']).
    method: str
        MPS representation ('number' or 'fullstate').
    bond_dim_max: int
        Hard cap on MPS bond dimension.
    flag_tdvp: bool
        True when the TDVP integrator is active; False for Runge-Kutta.
    flag_norm: bool
        True when the normalized nonlinear EOM is used.
    _flag_gs_vacuum: bool
        True when the physical ground state is the all-zeros MPS
        configuration; its amplitude counts toward the nonlinear <L> norm.
    Notes
    -----
    This class is owned by ``HopsTensorTrajectory`` and is populated during
    ``HopsTensorTrajectory.initialize``. It does not hold references to
    ``HopsSystem`` or ``HopsModes``; those are owned by ``HopsTensorEOM``,
    which holds a reference to this object and reads its config flags.
    """

    __slots__ = (
        # --- Configuration ---
        'k_max',  # Maximum hierarchy depth (int)
        'mps_epsilon',  # MPS SVD truncation threshold (float)
        'mpo_epsilon',  # MPO SVD truncation threshold (float)
        'flag_mpo_optimize',  # Number path: combined generator (bool)
        'method',  # MPS representation, 'number' or 'fullstate' (str)
        'bond_dim_max',  # Maximum MPS bond dimension (int)
        'flag_norm',  # True for normalized nonlinear EOM (bool)
        'flag_tdvp',  # True when TDVP integrator is active (bool)
        '_flag_gs_vacuum',  # Ground state as all-zeros vacuum (bool)
        # --- MPS data (populated by initialize) ---
        'list_cores_phi',  # MPS cores of the wavefunction (list[np.ndarray])
        'M1_modes_per_site',  # Active modes per MPS site, sorted (np.ndarray[int])
        # --- System-derived data (set by initialize) ---
        'M1_modes_per_state',  # Bath modes per state (np.ndarray[int])
        'M1_mode_offset',  # Cumulative mode offset per state (np.ndarray[int])
        '__initialized__',  # Initialization status flag (bool)
    )

    def __init__(self, k_max: int, tensor_param: dict, integrator_param: dict,
                 eom_param: dict) -> None:
        """
        Initializes the HopsTensorWavefunction configuration. Does not
        create system, mode, or noise_memory objects — those live in
        HopsTensorBasis.

        Parameters
        ----------
        1. k_max: int
                  Maximum hierarchy depth.

        2. tensor_param: dict
                         Tensor configuration parameters.
            a. 'METHOD':            str   — tensor topology type
            b. 'MPS_EPSILON':       float — SVD truncation threshold for MPS
            c. 'MPO_EPSILON':       float — SVD truncation threshold for MPO
            d. 'FLAG_MPO_OPTIMIZE': bool  — use the combined generator MPO
            e. 'BOND_DIM_MAX':      int   — maximum MPS bond dimension

        3. integrator_param: dict
                             Integration parameters.
            a. 'INTEGRATOR': str — 'RUNGE_KUTTA', 'TDVP1', or 'TDVP2'

        4. eom_param: dict
                      Equation of motion parameters.
            a. 'EQUATION_OF_MOTION': str — 'NORMALIZED NONLINEAR',
               'NONLINEAR', or 'LINEAR'
        """

        self.k_max = k_max
        self.mps_epsilon = tensor_param.get('MPS_EPSILON', 0.0)
        self.mpo_epsilon = tensor_param.get('MPO_EPSILON', 0.0)
        self.flag_mpo_optimize = tensor_param.get('FLAG_MPO_OPTIMIZE', True)
        self.method = tensor_param['METHOD']
        self.bond_dim_max = tensor_param['BOND_DIM_MAX']
        # True when the normalized nonlinear equation of motion is active
        self.flag_norm = (
            eom_param['EQUATION_OF_MOTION'] == 'NORMALIZED NONLINEAR'
        )
        # True when a TDVP integrator is active (TDVP1 or TDVP2)
        self.flag_tdvp = integrator_param['INTEGRATOR'] in ('TDVP1', 'TDVP2')
        # Set by the spectroscopy layer; not a user tensor_param.
        self._flag_gs_vacuum = False
        # MPS data — populated by initialize
        self.list_cores_phi = []
        self.M1_modes_per_site = None
        # System-derived data — set by initialize
        self.M1_modes_per_state = None
        self.M1_mode_offset = None
        self.__initialized__ = False

    def initialize(self, phi_0: np.ndarray, system: HopsSystem) -> None:
        """
        Sets system-derived constants, initializes dimension scalars, and builds
        the initial MPS.

        Parameters
        ----------
        1. phi_0: np.array(complex)
                  Initial physical wavefunction.

        2. system: HopsSystem
                   Initialized system object from HopsTensorBasis.

        Returns
        -------
        None
        """
        if self.__initialized__:
            raise LockedException('initialize', 'HopsTensorWavefunction')
        # Per-state mode counts from the L-operator → state mapping built
        # during system initialization.
        self.M1_modes_per_state = np.array([
            len(system.param['LIST_HMODE_INDICES_BY_STATE'][s])
            for s in range(system.param['NSTATES'])
        ], dtype=int)
        # The number representation gives every state group the same mode cores,
        # so a state carrying no hierarchy modes has no representation here. The
        # ground state is the all-zeros configuration, not a state of its own.
        if self.method == 'number' and np.any(self.M1_modes_per_state == 0):
            raise UnsupportedRequest(
                'a state with no hierarchy modes',
                'HopsTensorWavefunction.initialize with METHOD=number',
            )
        self.M1_mode_offset = np.concatenate(
            [[0], np.cumsum(self.M1_modes_per_state)]
        )
        self.M1_modes_per_site = self.M1_modes_per_state[sorted(system.state_list)]
        # Restrict phi_0 to the active state_list before building MPS
        self.build_list_cores_phi(phi_0[system.state_list], len(system.state_list))
        # TDVP sweeps require all bond dimensions to match; pad with
        # zeros up to bond_dim_max so the first sweep can proceed
        if self.flag_tdvp:
            self.inflate_bonds_to(self.bond_dim_max, eps=0.0)
        self.__initialized__ = True

    def build_list_cores_phi(self, psi_0: np.ndarray, n_state: int) -> None:
        """
        Builds the initial MPS form of the wave function given the initial physical
        wave function psi_0 in array form.

        Precondition: self.M1_modes_per_state must be set before calling.

        Parameters
        ----------
        1. psi_0: np.array(complex)
                  Physical wavefunction restricted to the current state_list.

        2. n_state: int
                    Number of active states.

        Returns
        -------
        None
        """
        # Construct MPS form of Phi
        if self.method == 'number':
            # Each physical site s gets a group [state_core_s, mode_core_s0, ...]
            # State core: binary (dim-2), index 1 = occupied, 0 = unoccupied.
            # Mode cores: initialized to hierarchy ground state |0> (index 0 = 1).
            # States with no bath coupling get a group with only the state core.
            for site in range(n_state):
                T3_core_state = np.zeros(shape=(1, 2, 1), dtype=np.complex128)
                T3_core_state[0, 1, 0] = psi_0[site]
                T3_core_state[0, 0, 0] = 1.0
                T3_core_mode = np.zeros(
                    shape=(1, self.k_max + 1, 1), dtype=np.complex128
                )
                T3_core_mode[0, 0, 0] = 1.0
                # Build group: state core followed by one mode core per mode.
                # .copy() prevents all mode cores sharing the same array.
                group = [T3_core_state] + [
                    T3_core_mode.copy() for _ in range(self.M1_modes_per_state[site])
                ]
                self.list_cores_phi.append(group)

        elif self.method == 'fullstate':
            # Single system core with physical dim = n_state, then
            # one mode core per hierarchy mode across all states, each
            # initialized to hierarchy ground state |0>. States with no
            # bath coupling contribute 0 mode cores.
            T3_core_state = np.zeros(shape=(1, n_state, 1), dtype=np.complex128)
            T3_core_state[0, :, 0] = psi_0
            T3_core_mode = np.zeros(shape=(1, self.k_max + 1, 1), dtype=np.complex128)
            T3_core_mode[0, 0, 0] = 1.0
            n_total_modes = int(self.M1_mode_offset[-1])
            self.list_cores_phi = [T3_core_state] + [
                T3_core_mode.copy() for _ in range(n_total_modes)
            ]

        else:
            raise UnsupportedRequest(self.method, 'build_list_cores_phi')

    def normalize(self) -> float:
        """
        Normalizes the main wave function phi_0. Always normalizes — the
        decision of whether to normalize (based on EOM type) belongs to the
        calling trajectory, not the wavefunction.

        Parameters
        ----------
        None

        Returns
        -------
        1. norm_psi: float
                    Norm of psi before normalization.
        """
        norm_psi = np.linalg.norm(self.psi)
        # Dividing any single core by a scalar divides the full MPS
        # contraction result by that scalar, so normalizing via the
        # first core is exact regardless of canonical form.
        # For statenumber, list_cores_phi[0] is a group (list), so
        # index [0][0] to reach the first state core array.
        if self.method == 'number':
            self.list_cores_phi[0][0] = self.list_cores_phi[0][0] / norm_psi
        elif self.method == 'fullstate':
            self.list_cores_phi[0] = self.list_cores_phi[0] / norm_psi
        else:
            raise UnsupportedRequest(self.method, 'normalize')
        return norm_psi

    def add_state_cores(
        self, list_states_new: list[int], state_list: list[int],
    ) -> None:
        """
        Adds new state and mode cores to the MPS for newly activated states.

        For fullstate representation, expands the state core to include new
        states and appends mode cores at the end of the flat core list. For
        statenumber representation, inserts new state groups (state core +
        mode cores) at the sorted position within the list-of-lists structure.

        Parameters
        ----------
        1. list_states_new : list(int)
                             Absolute state indices to add.

        2. state_list : list(int)
                        Current list of active state indices before adding.

        Returns
        -------
        None
        """
        n_state = len(state_list)
        list_states_new = sorted(list(list_states_new))
        if self.method == 'fullstate':
            # Expand the state core to accommodate new states. The old
            # state data is scattered into a larger core at the positions
            # where those states appear in the combined sorted list.
            list_total_states = sorted(list(state_list) + list(list_states_new))
            new_state_list_len = len(list_total_states)
            # Where old states land in the expanded physical dimension
            list_old_idx = sorted(
                [list_total_states.index(state) for state in state_list]
            )
            state_core_shape = self.list_cores_phi[0].shape
            new_state_core = np.zeros(
                shape=(1, new_state_list_len, state_core_shape[2]),
                dtype=np.complex128,
            )
            # Source indices: all entries from the current (smaller) state core
            old_tensor_indices = np.ix_(
                np.array([0]),
                np.arange(n_state),
                np.arange(state_core_shape[2]),
            )
            # Destination indices: scatter into the expanded core at the
            # positions corresponding to the old states
            new_tensor_indices = np.ix_(
                np.array([0]),
                np.array(list_old_idx),
                np.arange(state_core_shape[2]),
            )
            new_state_core[new_tensor_indices] = self.list_cores_phi[0][
                old_tensor_indices
            ]
            self.list_cores_phi[0] = new_state_core

            # Insert mode cores at sorted positions (not appended at end).
            # This ensures list_cores_phi maintains sorted state ordering.
            current_sorted = sorted(state_list)
            for state in list_states_new:
                insert_site = bisect.bisect_left(current_sorted, state)
                # Flat core index: 1 (state core) + modes for all sites before insertion
                core_pos = 1
                for s in current_sorted[:insert_site]:
                    core_pos += self.M1_modes_per_state[s]
                for m in range(self.M1_modes_per_state[state]):
                    new_mode_core = np.zeros(
                        shape=(1, self.k_max + 1, 1), dtype=np.complex128
                    )
                    new_mode_core[0, 0, 0] = 1 + 0j
                    self.list_cores_phi.insert(core_pos + m, new_mode_core)
                current_sorted.insert(insert_site, state)

        elif self.method == 'number':
            current_state_order = sorted(list(state_list))
            for state in list_states_new:
                # Insert the new group at the sorted position so the
                # MPS site ordering matches the state index ordering
                group_idx = bisect.bisect_left(current_state_order, state)
                # Match the bond dimension of the neighboring group so
                # the MPS bonds remain compatible
                bond_dim = (
                    self.list_cores_phi[group_idx - 1][-1].shape[2]
                    if group_idx > 0
                    else 1
                )
                # State core: identity along bond dimension at the
                # |unoccupied> index (0), so the new state acts as a
                # pass-through and does not alter the existing MPS
                new_state_core = np.zeros(
                    (bond_dim, 2, bond_dim), dtype=np.complex128
                )
                for bond in range(bond_dim):
                    new_state_core[bond, 0, bond] = 1 + 0j
                new_group = [new_state_core]
                # Mode cores: identity at hierarchy ground state |0>
                for mode in range(self.M1_modes_per_state[state]):
                    new_mode_core = np.zeros(
                        (bond_dim, self.k_max + 1, bond_dim), dtype=np.complex128
                    )
                    for bond in range(bond_dim):
                        new_mode_core[bond, 0, bond] = 1 + 0j
                    new_group.append(new_mode_core)
                self.list_cores_phi.insert(group_idx, new_group)
                current_state_order.insert(group_idx, state)

        else:
            raise UnsupportedRequest(self.method, 'add_state_cores')

        new_state_list = sorted(set(state_list) | set(list_states_new))
        self.M1_modes_per_site = self.M1_modes_per_state[new_state_list]

        if self.flag_tdvp:
            self.inflate_bonds_to(self.bond_dim_max, eps=0.0)

    def remove_state_cores(
        self, list_states_old: list[int], state_list: list[int],
    ) -> None:
        """
        Removes state and mode cores from the MPS for deactivated states.

        For fullstate representation, shrinks the state core to exclude removed
        states and absorbs their mode cores into adjacent cores. For
        statenumber representation, absorbs each removed group's |0> slices
        into the previous group and pops it from the list-of-lists structure.

        Parameters
        ----------
        1. list_states_old : list(int)
                             Absolute state indices to remove.

        2. state_list : list(int)
                        Current list of active state indices before removing.

        Returns
        -------
        None
        """
        if self.method == 'fullstate':
            old_state_indices = [list(state_list).index(s)
                                 for s in list_states_old]
            list_remaining_state_idx = list(
                set(np.arange(len(state_list))) - set(old_state_indices)
            )
            new_length = len(state_list) - len(old_state_indices)

            # Modify state core for remaining states
            state_core_shape = self.list_cores_phi[0].shape
            new_state_core = np.zeros(
                shape=(1, new_length, state_core_shape[2]), dtype=np.complex128
            )
            old_tensor_indices = np.ix_(
                np.array([0]),
                list_remaining_state_idx,
                np.arange(state_core_shape[2]),
            )
            new_tensor_indices = np.ix_(
                np.array([0]),
                np.arange(new_length),
                np.arange(state_core_shape[2]),
            )
            new_state_core[new_tensor_indices] = self.list_cores_phi[0][
                old_tensor_indices
            ]
            self.list_cores_phi[0] = new_state_core

            # Build flat indices of mode cores to remove (1-indexed since
            # core 0 is the state core). Compute before any mutation.
            state_order = sorted(state_list)
            list_flat_mode_indices = []
            for old_state in list_states_old:
                order_idx = state_order.index(old_state)
                base = 1
                for j in range(order_idx):
                    base += self.M1_modes_per_state[state_order[j]]
                n_modes = self.M1_modes_per_state[old_state]
                for m in range(n_modes):
                    list_flat_mode_indices.append(base + m)

            # Remove in reverse order so earlier indices stay valid.
            # Each removed mode core is contracted into its left neighbor
            # at the hierarchy ground state slice [:, 0, :]. This is valid
            # because a removed state has no hierarchy excitations, so its
            # |0> slice carries all the weight and higher slices are zero.
            for flat_idx in sorted(list_flat_mode_indices, reverse=True):
                mode_core = self.list_cores_phi[flat_idx]
                mode_shape = np.shape(mode_core)
                # Extract the |0> slice as a 2D transfer matrix (bond x bond)
                new_mode_core = np.zeros(
                    shape=(mode_shape[0], mode_shape[2]),
                    dtype=np.complex128,
                )
                new_indices = np.ix_(
                    np.arange(mode_shape[0]), np.arange(mode_shape[2])
                )
                old_indices = np.ix_(
                    np.arange(mode_shape[0]),
                    [0],
                    np.arange(mode_shape[2]),
                )
                new_mode_core[new_indices] = mode_core[old_indices][:, 0, :]
                # Absorb the transfer matrix into the adjacent core
                self.list_cores_phi[flat_idx - 1] = (
                    self.list_cores_phi[flat_idx - 1] @ new_mode_core
                )
                self.list_cores_phi.pop(flat_idx)

        elif self.method == 'number':
            list_group_indices = sorted(
                [sorted(state_list).index(s)
                 for s in list_states_old],
                reverse=True,
            )
            for group_idx in list_group_indices:
                group = self.list_cores_phi[group_idx]
                if group_idx > 0:
                    prev_group = self.list_cores_phi[group_idx - 1]
                    # Absorb state core's |0> slice into previous group's last core
                    prev_group[-1] = prev_group[-1] @ group[0][:, 0, :]
                    # Chain-absorb each mode core's |0> slice
                    for mode_core in group[1:]:
                        prev_group[-1] = prev_group[-1] @ mode_core[:, 0, :]
                else:
                    # TODO: structural bug — when removing the leftmost group
                    # (group_idx == 0), the |0> slices are discarded instead
                    # of absorbed into the next group. The |0> slice is a
                    # (1, Dr0) row vector carrying normalization information.
                    # Discarding it leaves the next group with wrong left
                    # boundary bond dimension (Dr0 instead of 1). Fix: absorb
                    # into next group via
                    #   np.einsum('ij,jkl->ikl', group[0][:, 0, :],
                    #             next_group[0])
                    # and chain-absorb mode core |0> slices similarly.
                    pass
                self.list_cores_phi.pop(group_idx)

        else:
            raise UnsupportedRequest(self.method, 'remove_state_cores')

        new_state_list = sorted(set(state_list) - set(list_states_old))
        self.M1_modes_per_site = self.M1_modes_per_state[new_state_list]

    def inflate_bonds_to(self, chi_target: int, eps: float = 0.0) -> None:
        """
        Pads every internal bond of list_cores_phi up to chi_target by appending
        zero (or small random) columns/rows at each bond interface.

        Parameters
        ----------
        1. chi_target: int
                       Target bond dimension. Bonds already at or above this
                       value are left unchanged.

        2. eps: float
                If greater than zero, the padding entries are filled with
                complex Gaussian noise of standard deviation eps rather than
                zeros. Default: 0.0.

        Returns
        -------
        None
        """
        # Flatten statenumber groups into a single core list so the
        # padding loop can treat all representations uniformly.
        if self.method == 'number':
            list_modes_per_site = [len(g) - 1 for g in self.list_cores_phi]
            list_cores = flatten_cores(self.list_cores_phi)
        elif self.method == 'fullstate':
            list_modes_per_site = None
            list_cores = self.list_cores_phi
        else:
            raise UnsupportedRequest(self.method, 'inflate_bonds_to')
        # Walk each internal bond (shared between adjacent cores) and pad
        # any bond whose dimension is below chi_target. Each MPS core has
        # shape (bond_left, phys, bond_right), so the right bond of core i
        # must equal the left bond of core i+1. Padding appends zero
        # slices to both sides of the interface to keep them consistent:
        #   core[i]:   (bond_left, phys, bond_right) → (bond_left, phys, chi_target)
        #   core[i+1]: (bond_right, phys', bond_right')
        #           → (chi_target, phys', bond_right')
        n_cores = len(list_cores)
        for i in range(n_cores - 1):
            T3_core_cur = list_cores[i]
            T3_core_next = list_cores[i + 1]
            bond_dim_left, phys_dim, bond_dim_right = T3_core_cur.shape
            _, phys_dim_next, bond_dim_right_next = T3_core_next.shape
            if bond_dim_right < chi_target:
                n_pad_right = chi_target - bond_dim_right
                # Pad right bond of current core with zeros (or noise)
                T3_pad_cur = np.zeros(
                    (bond_dim_left, phys_dim, n_pad_right), dtype=T3_core_cur.dtype
                )
                if eps > 0:
                    T3_pad_cur += eps * (
                        np.random.randn(*T3_pad_cur.shape)
                        + 1j * np.random.randn(*T3_pad_cur.shape)
                    )
                list_cores[i] = np.concatenate([T3_core_cur, T3_pad_cur], axis=2)
                # Pad left bond of next core with zeros to match
                T3_pad_next = np.zeros(
                    (n_pad_right, phys_dim_next, bond_dim_right_next),
                    dtype=T3_core_next.dtype,
                )
                list_cores[i + 1] = np.concatenate([T3_core_next, T3_pad_next], axis=0)
        # Restore statenumber group structure from the padded flat list
        if self.method == 'number':
            self.list_cores_phi = unflatten_cores(list_cores, list_modes_per_site)

    def update_phi_from_flat(self, list_cores_flat: list[np.ndarray]) -> None:
        """Set list_cores_phi from a flat core list.

        For number representation, restores the list-of-lists structure
        using the current group sizes. For fullstate representation, assigns
        the flat list directly.

        Parameters
        ----------
        1. list_cores_flat: list(np.ndarray)
                            Cores in statenumber MPS representation order.

        Returns
        -------
        None
        """
        if self.method == 'number':
            list_modes_per_site = [len(g) - 1 for g in self.list_cores_phi]
            self.list_cores_phi = unflatten_cores(
                [c.copy() for c in list_cores_flat], list_modes_per_site
            )
        else:
            self.list_cores_phi = [c.copy() for c in list_cores_flat]

    def restore_phi(
        self, tensor_source: list[np.ndarray] | HopsTensorWavefunction,
    ) -> None:
        """
        Sets list_cores_phi by copying cores from a list or another
        HopsTensorWavefunction.

        Always makes an independent copy of the input cores so that subsequent
        mutations to tensor_source do not affect this instance.

        Parameters
        ----------
        1. tensor_source: list(np.ndarray) or HopsTensorWavefunction
                          Source to copy cores from. If a list, each element
                          must be a 3D ndarray with axes [bond_left, phys_dim,
                          bond_right]. If a HopsTensorWavefunction,
                          cores are copied from its list_cores_phi.

        Returns
        -------
        None
        """
        if isinstance(tensor_source, HopsTensorWavefunction):
            source = tensor_source.list_cores_phi
        elif isinstance(tensor_source, list):
            source = tensor_source
        else:
            raise TypeError(
                'Expected list or HopsTensorWavefunction, '
                f'got {type(tensor_source).__name__}'
            )
        # Deep copy on restore prevents the integrator from mutating
        # the checkpoint arrays through aliased references
        if self.method == 'number':
            # source is a list of groups; deep-copy each ndarray within each group
            self.list_cores_phi = [[arr.copy() for arr in g] for g in source]
        else:
            self.list_cores_phi = [c.copy() for c in source]

    def get_core_shapes(self) -> list[tuple]:
        """
        Returns the shape of every core in the wavefunction MPS.

        Parameters
        ----------
        None

        Returns
        -------
        1. list_shapes: list(tuple)
                        List of (left_bond, phys_dim, right_bond) for each core.
        """
        if self.method == 'number':
            list_shapes = [core.shape for core in flatten_cores(self.list_cores_phi)]
        else:
            list_shapes = [core.shape for core in self.list_cores_phi]
        return list_shapes

    def check_bondsize(self) -> bool:
        """
        Checks that adjacent MPS cores have compatible bond dimensions.

        Parameters
        ----------
        None

        Returns
        -------
        1. valid: bool
                  True if all internal bond dimensions are consistent.
        """
        # Adjacent cores must have matching bond dimensions: right bond of
        # core i must equal left bond of core i+1.
        if self.method == 'number':
            list_cores_flat = flatten_cores(self.list_cores_phi)
        else:
            list_cores_flat = self.list_cores_phi
        list_bond_dim_left = [core.shape[0] for core in list_cores_flat][1:]
        list_bond_dim_right = [core.shape[2] for core in list_cores_flat][:-1]
        return np.array_equal(list_bond_dim_left, list_bond_dim_right)

    @property
    def manifold_norm_sq(self) -> np.float64:
        """
        Physical-wavefunction norm-squared <psi|psi> + |<0,...,0|psi>|^2 on
        the GS + single-excitation manifold of a vacuum-convention MPS.

        Under the vacuum convention extract_psi only returns the single-
        excitation amplitudes; the physical ground-state amplitude lives at
        the all-zeros MPS configuration and is invisible to traj.psi.
        Adding |<0,...,0|psi>|^2 to <psi|psi> recovers the same physical-
        wavefunction norm the gs_core-convention trajectory has by
        construction.

        Returns
        -------
        1. norm_sq : np.float64
                     <psi|psi> + |<0,...,0|psi>|^2.
        """
        psi = self.psi
        norm_sq_excited = np.linalg.norm(psi) ** 2
        gs_amp = extract_gs_amp(self.list_cores_phi, self.method)
        return norm_sq_excited + np.abs(gs_amp) ** 2

    @property
    def flat_cores(self) -> list[np.ndarray]:
        """All MPS cores as a 1D flat list suitable for tensor algebra routines.

        For number representation this flattens the list-of-lists;
        for fullstate representation the list is already flat.

        Returns
        -------
        1. list_cores_flat: list(np.ndarray)
                            Cores in statenumber MPS representation order.
        """
        if self.method == 'number':
            return flatten_cores(self.list_cores_phi)
        return self.list_cores_phi

    @property
    def psi(self) -> np.ndarray:
        """
        Physical (system) wavefunction extracted from the MPS.

        Returns
        -------
        1. psi: np.array(complex)
                Physical wavefunction of length n_state.
        """
        return extract_psi(
            self.list_cores_phi, self.method,
            self.M1_modes_per_site,
        )

    @property
    def flag_gs_vacuum(self) -> bool:
        """True when the ground state is tracked as the all-zeros vacuum."""
        return self._flag_gs_vacuum

    @flag_gs_vacuum.setter
    def flag_gs_vacuum(self, value: bool) -> None:
        # The vacuum (all-zeros) config only exists in the number
        # representation, so the flag is meaningless under fullstate.
        if value and self.method != 'number':
            raise UnsupportedRequest(
                f"flag_gs_vacuum=True with method={self.method!r}",
                'HopsTensorWavefunction.flag_gs_vacuum',
            )
        self._flag_gs_vacuum = value
