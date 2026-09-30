"""
MPO construction routines and shared operator building blocks for
HopsTensorWavefunction.

Classes
-------
MpoBuilder
    Precomputed ladder operators, projectors, and Hamiltonian slices shared
    by all MPO builder methods. All elementary operators are computed during
    __init__; builder methods use self.X to access them.

Functions
---------
build_statenumber_operator_mpo(H2_op, n_state, k_max, M1_modes_per_state)
    Build an MPO that applies an arbitrary (n_state x n_state) system operator
    to a statenumberrepresentation MPS, acting as identity on all mode cores.

build_statenumber_dipole_mpo(list_mu, n_state, k_max, M1_modes_per_state,
                              raise_or_lower)
    Build the bond-dim-2 MPO for a sum-of-single-site dipole raise or
    lower operator in the statenumberrepresentation, under the
    ground-state-as-vacuum convention.

build_statenumber_dipole_lower_plus_ident_mpo(list_mu, n_state, k_max,
                                               M1_modes_per_state)
    Build the bond-dim-3 MPO for (I + mu^-) = identity-everywhere plus
    sum-of-single-site dipole lower operator, in the
    statenumberrepresentation under the ground-state-as-vacuum
    convention.  Used by the fluorescence pathway to preserve
    single-excitation content while injecting ground-state amplitude.

Methods on MpoBuilder
---------------------
build_statenumber_hierarchy_mpo(list_z_hat, list_expect_L2, norm_corr)
    Build the hierarchy-interaction MPO (noise, bath frequency, and
    coupling terms) for the statenumberrepresentation. Bond dimension 5.

build_statenumber_ham_mpo()
    Build the Hamiltonian MPO for the statenumberrepresentation.
    Dispatches to nearest-neighbor or general builder based on
    flag_nearest_neighbor_ham.

build_general_generator_mpo(list_z_hat, list_expect_L2, norm_corr)
    Build the combined Hamiltonian + hierarchy MPO for a Hamiltonian of any
    coupling pattern, in the number representation on the one-excitation
    manifold, so that no MPO addition or compression is needed. Each bond
    carries as many channels as the rank of the coupling block that crosses
    it, giving peak bond dimension 5 on a chain or star and 7 on a ring. The
    two-MPO path assembles 9 (nearest neighbor) or 5 + 2*n_state (general)
    before compressing, and is kept as the FLAG_MPO_OPTIMIZE=False reference.

build_fullstate_mpo(list_z_hat, list_expect_L2, norm_corr)
    Build the combined hierarchy+Hamiltonian MPO for the
    fullstaterepresentation. The first core covers the full system
    Hilbert space; subsequent cores handle each bath mode.
    Bond dimension tapers n_lop_full + 2, n_lop_full + 1, ..., 3 as each
    L-operator channel closes at its own last mode core.
"""

from __future__ import annotations

import numpy as np

# --- Shared 2x2 elementary operators for statenumber MPO construction ---
# Site projector |1><1| (occupied state)
_P2_SITE = np.array([[0, 0], [0, 1]], dtype=np.complex128)
# Unoccupied-site projector Q = 1-P = |0><0|
_Q2_SITE = np.array([[1, 0], [0, 0]], dtype=np.complex128)
# Raise operator sigma^+ = |1><0| (carries coupling rightward)
_T2_PLUS = np.array([[0, 0], [1, 0]], dtype=np.complex128)
# Lower operator sigma^- = |0><1| (carries coupling leftward)
_T2_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)

__title__ = 'MPO Constructors'
__author__ = 'B. Z. Citty'
__maintainer__ = 'B. Z. Citty'


def _identity_mode_cores(bond_dim, k_max, n_copies):
    """Build identity mode cores that pass all bond channels through.

    Parameters
    ----------
    1. bond_dim: int
                 MPO bond dimension (left = right).
    2. k_max: int
              Maximum hierarchy depth (physical dim = k_max + 1).
    3. n_copies: int
                 Number of identical cores to return.

    Returns
    -------
    1. list_cores: list(np.ndarray)
                   n_copies cores of shape (bond_dim, k_max+1, k_max+1, bond_dim).
    """
    T4_core = np.zeros(
        (bond_dim, k_max + 1, k_max + 1, bond_dim), dtype=np.complex128,
    )
    idx_bond = np.arange(bond_dim)
    T4_core[idx_bond, :, :, idx_bond] = np.eye(k_max + 1, dtype=np.complex128)
    return [T4_core.copy() for _ in range(n_copies)]


class MpoBuilder:
    """
    Precomputed building blocks shared by all MPO builder methods.

    All elementary operators are computed during __init__. A new instance
    is created on each call to HopsTensorWavefunction.build_op; there is
    no persistent state between timesteps.

    The number-representation builders carry one L-operator per site, ordered
    by site, so a site index is also its L2 index into list_z_hat and
    list_expect_L2.

    That ordering is a precondition, not a check: L_HIER listed out of site
    order gives wrong results in the number representation.
    """

    __slots__ = (
        # --- Scalars ---
        'k_max',  # Maximum hierarchy depth (int)
        'n_state',  # Number of active states (int)
        'n_lop_full',  # Number of L-operators (int)
        'M1_modes_per_state',  # Bath modes per state, padded (np.ndarray[int])
        'M1_mode_offset',  # Cumulative mode offset (np.ndarray[int])
        'list_state_list',  # Active state indices (list[int])
        'n_hmodes',  # Total hierarchy mode count (int)
        'n_states_full',  # Full system state count (int)
        'flag_nearest_neighbor_ham',  # bool
        'flag_gs_vacuum',  # Ground state is the all-zeros MPS config (bool)
        'mpo_epsilon',  # Relative SVD threshold for the bond factors (float)
        # --- Mode-dimension operators ---
        'I2_mode',  # Mode-space identity (k_max+1, k_max+1)
        'B2_raise',  # Raising (creation) operator (k_max+1, k_max+1)
        'B2_lower',  # Lowering (annihilation) operator (k_max+1, k_max+1)
        'N2_occ',  # Occupation number operator (k_max+1, k_max+1)
        'C1_coupling_raise',  # Raise (b^dagger) coupling prefactors (n_modes,)
        'C1_coupling_lower',  # Lower (b) coupling prefactors (n_modes,)
        'list_w',  # Mode frequencies (np.ndarray)
        'list_L2_coo',  # L-operator matrices (list[sparse])
        'list_L2_masks',  # Per-L2 [rows, cols, ix_]; [0][0] is the L2's site
        'list_index_L2_by_hmode',  # L2 index of each hierarchy mode (list[int])
        # --- State-dimension operators (4-D core shape) ---
        'T4_plus',  # Raise sigma^+ = |1><0| (1, 2, 2, 1)
        'T4_minus',  # Lower sigma^- = |0><1| (1, 2, 2, 1)
        'Q4_site',  # Unoccupied projector |0><0| (1, 2, 2, 1)
        'P4_site',  # Occupied projector |1><1| (1, 2, 2, 1)
        'I2_state',  # State-space identity (n_state, n_state)
        # --- Hamiltonian ---
        'H2_ham',  # Full Hamiltonian (n_state_full, n_state_full)
        # Per-bond SVD factors of the coupling graph, built on first use by
        # _bond_factors and dropped when the active basis changes.
        '_list_bond_factors',  # (Y2_left, Z2_coeff, n_bond) per bond (list|None)
    )

    def __init__(
        self,
        k_max,
        n_state,
        modes_per_state,
        n_lop_full,
        ham,
        state_list,
        mode,
        normalization,
        n_states_full=None,
        flag_nearest_neighbor_ham=False,
        flag_gs_vacuum=False,
        mpo_epsilon=0.0,
    ):
        """
        Builds and stores the elementary operators used by all MPO constructors.

        Inputs
        ------
        1. k_max: int
                  Maximum hierarchy depth.

        2. n_state: int
                    Number of active states (system.size).

        3. modes_per_state: np.ndarray(int)
                            Number of bath modes per state.

        4. n_lop_full: int
                       Number of L-operators (mode.n_l2).

        5. ham: np.ndarray
                Full Hamiltonian matrix.

        6. state_list: np.ndarray
                       Active state indices (system.state_list).

        7. mode: HopsModes
                 Mode object providing list_g and list_w.

        8. normalization: str
                          Scaling convention for the auxiliary vectors.
                          'homps' balances the coupling across the raising
                          and lowering operators for truncation stability;
                          'adhops' reproduces the vector HOPS scaling. Sets
                          B2_raise, B2_lower, C1_coupling_raise and
                          C1_coupling_lower below.

        9. n_states_full: int or None
                          Full system state count (self.n_states_full).

        10. flag_nearest_neighbor_ham: bool
                                      Whether the Hamiltonian is nearest-neighbor
                                      (default False).

        11. flag_gs_vacuum: bool
                                  Whether the physical ground state is the
                                  all-zeros MPS configuration (vacuum
                                  convention). Gates the site-0 damp1 widening
                                  in build_statenumber_hierarchy_mpo
                                  (default False).

        12. mpo_epsilon: float
                         Relative threshold on the singular values of each
                         bond's coupling block in _bond_factors, below which
                         no channel is opened. Zero (default) keeps every
                         coupling and gives the exact rank; raising it drops
                         weak couplings and narrows the general generator.

        Returns
        -------
        None
        """
        if normalization != 'homps' and normalization != 'adhops':
            raise ValueError(
                f"Unknown normalization '{normalization}'. Use 'homps' or 'adhops'."
            )

        # --- scalars ---
        self.k_max = k_max
        self.n_state = n_state
        self.n_lop_full = n_lop_full
        self.M1_modes_per_state = np.asarray(modes_per_state, dtype=int)
        self.M1_mode_offset = np.concatenate(
            [[0], np.cumsum(self.M1_modes_per_state)]
        )

        # --- Static data from system/mode (used by builder methods) ---
        self.list_state_list = list(state_list)
        self.list_w = np.asarray(mode.list_w)
        self.list_L2_coo = mode.list_L2_coo
        self.list_L2_masks = mode.list_L2_masks
        self.list_index_L2_by_hmode = mode.list_index_L2_by_hmode
        self.n_hmodes = mode.n_hmodes
        self.n_states_full = n_states_full
        self.flag_nearest_neighbor_ham = flag_nearest_neighbor_ham
        self.flag_gs_vacuum = flag_gs_vacuum
        self.mpo_epsilon = mpo_epsilon

        # --- mode-dimension operators ---
        # Complex dtype so these combine with complex coupling prefactors
        # and ladder operators without repeated upcasting.
        self.I2_mode = np.eye(k_max + 1, dtype=np.complex128)
        self.B2_raise = np.zeros((k_max + 1, k_max + 1), dtype=np.complex128)
        self.B2_lower = np.zeros((k_max + 1, k_max + 1), dtype=np.complex128)
        self.N2_occ = np.zeros((k_max + 1, k_max + 1), dtype=np.complex128)

        # Ladder operator values for occupation levels 1..k_max
        V1_levels = np.arange(1, k_max + 1, dtype=np.complex128)

        if normalization == 'homps':
            # sqrt(n+1) ladder operators; coupling split as g/sqrt|g| and sqrt|g|
            V1_sqrt_levels = np.sqrt(V1_levels)
            self.B2_raise += np.diag(V1_sqrt_levels, k=-1)
            self.B2_lower += np.diag(V1_sqrt_levels, k=1)
            self.N2_occ = self.B2_raise @ self.B2_lower
            # Prefactor convention of Gao et al., Phys. Rev. A 105, L030202
            # (2022), Eq. (10).
            # C1_coupling_raise pairs with B2_raise (b^dagger):
            #   V_m^+ = g_m / sqrt(|g_m|)
            # C1_coupling_lower pairs with B2_lower (b):
            #   V_m^- = sqrt(|g_m|)
            # Physical coupling split:
            #   C1_coupling_raise * C1_coupling_lower = g_m.
            list_g = np.asarray(mode.list_g)
            abs_g = np.abs(list_g)
            sqrt_abs_g = np.sqrt(abs_g)
            # Guard against g=0: both prefactors vanish, so all
            # bath coupling terms for that mode are zero.
            self.C1_coupling_lower = sqrt_abs_g
            with np.errstate(invalid='ignore', divide='ignore'):
                self.C1_coupling_raise = np.where(abs_g > 0, list_g / sqrt_abs_g, 0.0)
        elif normalization == 'adhops':
            # unit ladder operators; prefactors w and g/w
            self.C1_coupling_raise = mode.list_w
            self.C1_coupling_lower = mode.list_g / mode.list_w
            self.B2_raise += np.diag(V1_levels, k=-1)
            self.B2_lower += np.diag(np.ones(k_max, dtype=np.complex128), k=1)
            # N_occ diagonal: levels 1..k_max at positions 1..k_max
            self.N2_occ += np.diag(
                np.concatenate([[0], V1_levels]),
            )

        # --- State-dimension operators (pre-reshaped to 4-D core shape) ---
        # In the binary occupation basis {|0>, |1>}:
        #   P4_site = |1><1| occupied-site projector
        #   Q4_site = |0><0| unoccupied-site projector
        #   T4_plus = |1><0| carries coupling rightward
        #   T4_minus = |0><1| carries coupling leftward
        # .copy() ensures these are independent of the module-level
        # constants; without it, reshape returns a view that shares
        # memory, and any accidental mutation would silently corrupt
        # the constants for all future MpoBuilder instances.
        self.T4_plus = _T2_PLUS.reshape(1, 2, 2, 1).copy()
        self.T4_minus = _T2_MINUS.reshape(1, 2, 2, 1).copy()
        self.Q4_site = _Q2_SITE.reshape(1, 2, 2, 1).copy()
        self.P4_site = _P2_SITE.reshape(1, 2, 2, 1).copy()
        self.I2_state = np.eye(n_state, dtype=np.complex128)

        # --- Hamiltonian (full, for slicing inside constructors) ---
        self.H2_ham = ham
        self._list_bond_factors = None

    def refresh_state_data(self, n_state, state_list):
        """
        Updates only the state-dependent attributes after an adaptive basis
        change. Constant attributes (ladder operators, coupling prefactors,
        mode operators) are unchanged.

        Parameters
        ----------
        1. n_state: int
                    New number of active states.
        2. state_list: list(int)
                       New active state indices.

        Returns
        -------
        None
        """
        self.n_state = n_state
        self.list_state_list = list(state_list)
        self.I2_state = np.eye(n_state, dtype=np.complex128)
        self._list_bond_factors = None

    def build_statenumber_hierarchy_mpo(
        self, list_z_hat, list_expect_L2, norm_corr,
    ):
        """
        Builds the hierarchy interaction MPO cores for statenumberrepresentation.

        # TODO: replace with preprint reference and equation numbers
        # before merging to MesoHOPS.

        Parameters
        ----------
        1. list_z_hat: np.ndarray
                       Conjugate noise values at current timestep, shape (n_state,).

        2. list_expect_L2: np.ndarray
                           L-operator expectation values, shape (n_state,).

        3. norm_corr: float
                      Normalization correction term.

        Returns
        -------
        1. list_cores_op: list(np.ndarray)
                          MPO cores: one (l_bond, 2, 2, r_bond) per state site,
                          then one (l_bond, k_max+1, k_max+1, r_bond) per mode
                          site.
        """
        # Guard: statenumber hierarchy MPO supports one L2 per state (bond dim 5).
        list_states_seen = set()
        for list_mask in self.list_L2_masks:
            if list_mask[0][0] in list_states_seen:
                raise NotImplementedError(
                    'Multiple L-operators per state is not supported '
                    'for number representation.'
                )
            list_states_seen.add(list_mask[0][0])

        k_max = self.k_max
        n_state = self.n_state
        M1_modes_per_state = self.M1_modes_per_state
        M1_mode_offset = self.M1_mode_offset
        I2_mode = self.I2_mode
        B2_raise = self.B2_raise
        B2_lower = self.B2_lower
        C1_coupling_raise = self.C1_coupling_raise
        C1_coupling_lower = self.C1_coupling_lower

        Q4_site = self.Q4_site
        P4_site = self.P4_site

        list_cores_op = []
        shape_mode_core = (1, k_max + 1, k_max + 1, 1)
        # Pre-reshape mode-space operators to 4-D core shape to avoid
        # repeated inline reshaping throughout the mode core loop.
        I4_mode = I2_mode.reshape(shape_mode_core)
        B4_raise = B2_raise.reshape(shape_mode_core)
        B4_lower = B2_lower.reshape(shape_mode_core)
        N4_occ = self.N2_occ.reshape(shape_mode_core)

        # MPO bond index convention (w = 5):
        #   0 = identity channel (pass-through)
        #   1 = L-operator / noise channel
        #   2 = damping channel 1 (-w * N_occ + L^dagger_avg * B_lower)
        #   3 = damping channel 2 (multi-site coupling carry)
        #   4 = accumulator / output channel
        channel_ident = slice(0, 1)
        channel_lop = slice(1, 2)
        channel_damp1 = slice(2, 3)
        channel_damp2 = slice(3, 4)
        channel_accum = slice(4, 5)
        for site in range(n_state):
            if site == 0:
                T4_core_site = np.zeros((1, 2, 2, 5), dtype=np.complex128)
            else:
                T4_core_site = np.zeros((5, 2, 2, 5), dtype=np.complex128)
            T4_core_site[channel_ident, :, :, channel_ident] = Q4_site
            # Channels lop and damp1 are gated by P_site (occupied
            # projector): the L-op, noise, and damping terms are specific to
            # the occupied branch. Identity gating would let SVD-compressed
            # unoccupied-branch content contaminate the damping channel.
            I4_site = P4_site + Q4_site
            # The 1j prefactor on hierarchy state cores is required by
            # the TDVP integrator: the TDVP sweep applies
            # exp(-1j * delta * MPO), so hierarchy terms need 1j so
            # that (-1j)(1j * hier) = hier (real damping), while the
            # Hamiltonian MPO (no prefactor) gets (-1j)(H) = -iH
            # (Schrodinger rotation). For the RK4 path, derivative()
            # applies scale_mps(dphi, -1j) to achieve the same result.
            T4_core_site[channel_ident, :, :, channel_lop] = 1j * P4_site
            # In the vacuum convention the site-0 damp1 opener is widened
            # from P_site to I_site = P + Q. The Q-component opens the NL
            # feedback channel on the all-zeros MPS configuration so the
            # vacuum amplitude correctly tracks the physical
            # $\\overline{\\langle L\\rangle}\\,(g_n/w_n)\\,\\Phi[\\text{vac}, k=e_n]$
            # drift. The widening is a no-op algebraically on
            # single-excitation configurations (the P-gated downstream
            # Q-relay carries no contribution) but perturbs the MPS bond
            # singular-value spectrum, so it is gated to the vacuum
            # convention to keep non-vacuum truncation accuracy intact.
            if site == 0 and self.flag_gs_vacuum:
                T4_core_site[channel_ident, :, :, channel_damp1] = (
                    1j * I4_site
                )
            else:
                T4_core_site[channel_ident, :, :, channel_damp1] = (
                    1j * P4_site
                )
            if site != 0:
                T4_core_site[channel_damp1, :, :, channel_damp1] = Q4_site
                T4_core_site[channel_damp2, :, :, channel_damp2] = Q4_site
                T4_core_site[channel_accum, :, :, channel_accum] = Q4_site
                T4_core_site[channel_damp2, :, :, channel_accum] = 1j * P4_site
            list_cores_op.append(T4_core_site)

            # Mode cores for this state: hierarchy cores indexed via
            # M1_mode_offset into list_w / C1_coupling_raise /
            # C1_coupling_lower.
            n_modes_site = M1_modes_per_state[site]
            n_total_modes = int(M1_mode_offset[-1])
            l2_idx = site
            for i in range(n_modes_site):
                global_idx = M1_mode_offset[site] + i
                # Coupling channel (ch 1):
                #   C1_coupling_raise * b^dagger
                #   - C1_coupling_lower * b + (z_hat - norm_corr)/M * I
                # where C1_coupling_raise = g/sqrt(|g|) and
                #       C1_coupling_lower = sqrt(|g|) for homps, so that
                #       b^dagger pairs with V^+ and b pairs with V^-;
                #       for adhops the pair is w and g/w instead.
                M4_coupling = (
                    C1_coupling_raise[global_idx] * B4_raise
                    - C1_coupling_lower[global_idx] * B4_lower
                    + (list_z_hat[l2_idx] - norm_corr)
                    * I4_mode / n_modes_site
                )
                # Damping channel (ch 2):
                #   -w_m * N_occ + C1_coupling_lower * <L^dagger> * b
                # where -w_m * N_occ is the bath frequency decay and
                # C1_coupling_lower * <L^dagger> * b is the nonlinear feedback.
                M4_damping = (
                    -self.list_w[global_idx] * N4_occ
                    + C1_coupling_lower[global_idx]
                    * np.conj(list_expect_L2[l2_idx]) * B4_lower
                )
                if global_idx == n_total_modes - 1:
                    # Last mode core in the MPS: right bond collapses
                    # to dim 1, closing all channels to the output.
                    T4_core_mode = np.zeros(
                        (5, k_max + 1, k_max + 1, 1), dtype=np.complex128
                    )
                    # lop→output: coupling closes to output
                    T4_core_mode[channel_lop, :, :, 0:0+1] = M4_coupling
                    # accum→output: accumulator closes to output
                    T4_core_mode[channel_accum, :, :, 0:0+1] = I4_mode
                    # damp1→output: damping closes to output
                    T4_core_mode[channel_damp1, :, :, 0:0+1] = M4_damping
                else:
                    # Interior mode core: relay identity, coupling,
                    # and damping channels through the bond.
                    T4_core_mode = np.zeros(
                        (5, k_max + 1, k_max + 1, 5),
                        dtype=np.complex128,
                    )
                    # ident→ident: identity pass-through
                    T4_core_mode[channel_ident, :, :, channel_ident] = I4_mode
                    # accum→accum: accumulator pass-through
                    T4_core_mode[channel_accum, :, :, channel_accum] = I4_mode
                    # lop→accum: coupling into accumulator
                    T4_core_mode[channel_lop, :, :, channel_accum] = M4_coupling
                    # damp1→accum: damping into accumulator
                    T4_core_mode[channel_damp1, :, :, channel_accum] = M4_damping
                    # ident→damp2: damping relay (multi-site carry)
                    T4_core_mode[channel_ident, :, :, channel_damp2] = M4_damping
                    # damp1→damp1: damping channel relay
                    T4_core_mode[channel_damp1, :, :, channel_damp1] = I4_mode
                    # damp2→damp2: multi-site damping relay
                    T4_core_mode[channel_damp2, :, :, channel_damp2] = I4_mode
                    if i != n_modes_site - 1:
                        # lop→lop: L-op channel stays open for
                        # remaining modes of this state
                        T4_core_mode[channel_lop, :, :, channel_lop] = I4_mode
                list_cores_op.append(T4_core_mode)

        return list_cores_op

    def build_statenumber_ham_mpo(self):
        """Build the Hamiltonian MPO for number representation.

        Dispatches to nearest-neighbor or general builder based on
        self.flag_nearest_neighbor_ham.

        Returns
        -------
        1. list_cores_ham: list(np.ndarray)
                           Hamiltonian MPO cores.
        """
        if self.flag_nearest_neighbor_ham:
            return self._build_statenumber_ham_nn_mpo()
        return self._build_statenumber_ham_general_mpo()

    def _build_statenumber_ham_nn_mpo(self):
        """
        Builds Hamiltonian MPO cores for nearest-neighbor Hamiltonians
        (number representation). Bond dimension is 4: one channel each
        for left transfer, right transfer, identity pass-through, and
        accumulated diagonal energy.

        # TODO: replace with preprint reference and equation numbers
        # before merging to MesoHOPS.

        Returns
        -------
        1. list_cores_ham_full: list(np.ndarray)
                                MPO cores interleaved: one state core
                                (w_l, 2, 2, w_r) followed by modes_per_state
                                identity mode cores (w, k_max+1, k_max+1, w),
                                repeating for each physical site.
        """
        k_max = self.k_max
        n_state = self.n_state
        M1_modes_per_state = self.M1_modes_per_state
        H2_ham = self.H2_ham

        T4_plus = self.T4_plus
        T4_minus = self.T4_minus
        Q4_site = self.Q4_site
        P4_site = self.P4_site

        # Convention: H2_ham[i, j] couples state j into state i.
        # T4_plus = |1><0| raises occupation (carries coupling rightward).
        # T4_minus = |0><1| lowers occupation (carries coupling leftward).

        # 4-channel MPO structure for nearest-neighbor Hamiltonians:
        #   The bond dimension is 4 because a nearest-neighbor Hamiltonian
        #   decomposes into exactly four channels:
        #     index 0 -- left transfer channel (T4_plus, carries coupling rightward)
        #     index 1 -- right transfer channel (T4_minus, carries coupling leftward)
        #     index 2 -- identity pass-through (Q4_site, site unoccupied)
        #     index 3 -- accumulated diagonal Hamiltonian (on-site energy * P4_site)
        #   The first-site core dispatches into these 4 channels (row vector),
        #   interior cores relay and close channels, and the last-site core
        #   receives and contracts them to a single output (column vector).
        list_cores_ham_full = []

        # NOTE: `site` is a relative index into the active basis
        # (0..n_state-1), not an absolute site label. The absolute site
        # index is obtained via self.list_state_list[site] (was system.state_list).
        for site in range(n_state):
            state = self.list_state_list[site]
            if site == 0:
                # First site: row vector core (1, 2, 2, 4).
                T4_core_site = np.zeros((1, 2, 2, 4), dtype=np.complex128)
                # Slice notation i:i+1 preserves the bond axis (returns a
                # (1,m,m,1) view); plain integer indexing i would collapse
                # it to (m,m), mismatching the 4-D operator shapes.
                # Channel 0: left transfer
                T4_core_site[0:0+1, :, :, 0:0+1] = T4_plus
                # Channel 1: right transfer
                T4_core_site[0:0+1, :, :, 1:1+1] = T4_minus
                # Channel 2: identity pass-through
                T4_core_site[0:0+1, :, :, 2:2+1] = Q4_site
                # Channel 3: diagonal on-site energy
                T4_core_site[0:0+1, :, :, 3:3+1] = H2_ham[state, state] * P4_site
            elif site == n_state - 1:
                # Last site: column vector core (4, 2, 2, 1).
                T4_core_site = np.zeros((4, 2, 2, 1), dtype=np.complex128)
                state_prev = self.list_state_list[site - 1]
                # Channel 0: close left-transfer with nn coupling
                T4_core_site[0:0+1, :, :, 0:0+1] = H2_ham[state_prev, state] * T4_minus
                # Channel 1: close right-transfer with nn coupling
                T4_core_site[1:1+1, :, :, 0:0+1] = H2_ham[state, state_prev] * T4_plus
                # Channel 2: diagonal on-site energy
                T4_core_site[2:2+1, :, :, 0:0+1] = H2_ham[state, state] * P4_site
                # Channel 3: close identity pass-through
                T4_core_site[3:3+1, :, :, 0:0+1] = Q4_site
            else:
                # Interior site: full (4, 2, 2, 4) core.
                T4_core_site = np.zeros((4, 2, 2, 4), dtype=np.complex128)
                state_prev = self.list_state_list[site - 1]
                # Channel 0 -> 3: close left-transfer with nn coupling
                T4_core_site[0:0+1, :, :, 3:3+1] = H2_ham[state_prev, state] * T4_minus
                # Channel 1 -> 3: close right-transfer with nn coupling
                T4_core_site[1:1+1, :, :, 3:3+1] = H2_ham[state, state_prev] * T4_plus
                # Channel 2 -> 0: reopen left-transfer for next nn pair
                T4_core_site[2:2+1, :, :, 0:0+1] = T4_plus
                # Channel 2 -> 1: reopen right-transfer for next nn pair
                T4_core_site[2:2+1, :, :, 1:1+1] = T4_minus
                # Channel 2 -> 2: identity pass-through
                T4_core_site[2:2+1, :, :, 2:2+1] = Q4_site
                # Channel 2 -> 3: diagonal on-site energy
                T4_core_site[2:2+1, :, :, 3:3+1] = H2_ham[state, state] * P4_site
                # Channel 3 -> 3: identity relay for accumulated diagonal
                T4_core_site[3:3+1, :, :, 3:3+1] = Q4_site
            list_cores_ham_full.append(T4_core_site)

            # Mode cores: identity pass-through for all channels.
            list_cores_ham_full.extend(
                _identity_mode_cores(
                    T4_core_site.shape[3],
                    k_max,
                    M1_modes_per_state[site],
                )
            )

        return list_cores_ham_full

    def _build_statenumber_ham_general_mpo(self):
        """
        Builds Hamiltonian MPO cores for general (non-nearest-neighbor) Hamiltonians
        (number representation). Delegates to build_statenumber_operator_mpo.

        # TODO: replace with preprint reference and equation numbers
        # before merging to MesoHOPS.

        Returns
        -------
        1. list_cores_ham_full: list(np.ndarray)
                                MPO cores interleaved: one state core then
                                modes_per_state mode cores, for each state.
        """
        return build_statenumber_operator_mpo(
            self.H2_ham, self.n_state, self.k_max, self.M1_modes_per_state,
        )

    def _bond_factors(self):
        """
        Factorizes the coupling graph at each bond between state sites.

        The bond after state site l carries the hoppings that start at a site
        at or left of l and end at a site right of l. Their amplitudes are the
        block H[0:l+1, l+1:n] of the off-diagonal Hamiltonian, and the number
        of channels the bond needs is that block's rank. A thin SVD splits it
        into Y2_left, whose row l says with what weight site l's sigma^+ opens
        each channel, and Z2_coeff, whose first column says with what amplitude
        each channel closes on the next site. The charge-lowering family reuses
        the conjugates of both, since H is Hermitian.

        Taking the left factor with orthonormal columns is what lets the
        site-to-site relay be a projection rather than a solve; the width
        itself is set by the rank, and any rank factorization would attain it.

        The active basis and the Hamiltonian are fixed between basis changes,
        so the factors are built once and reused until refresh_state_data
        drops them.

        Returns
        -------
        1. list_bond_factors: list(tuple)
                              One (Y2_left, Z2_coeff, n_bond) per bond, in site
                              order, with n_bond = 0 on the last.
        """
        if self._list_bond_factors is not None:
            return self._list_bond_factors

        # Off-diagonal couplings in the active basis: the same view of the
        # Hamiltonian the generator's site cores are built from.
        list_state = self.list_state_list
        M2_coupling = np.array(
            self.H2_ham[np.ix_(list_state, list_state)], dtype=np.complex128,
        )
        np.fill_diagonal(M2_coupling, 0.0)

        list_bond_factors = []
        for site in range(self.n_state - 1):
            # Rows are the sites at or left of this bond, columns those right.
            M2_cross = M2_coupling[:site + 1, site + 1:]
            U2_left, V1_sv, V2_right = np.linalg.svd(
                M2_cross, full_matrices=False,
            )
            # A channel exists only where the singular value clears the
            # threshold. mpo_epsilon raises it above the floor, dropping
            # couplings the caller accepts losing; the floor is there because
            # a structurally zero block still returns values at roundoff.
            tol = max(self.mpo_epsilon, 1e-12)
            n_bond = int(np.sum(
                V1_sv > tol * (V1_sv[0] if V1_sv.size else 0.0)
            ))
            list_bond_factors.append((
                U2_left[:, :n_bond],
                V1_sv[:n_bond, None] * V2_right[:n_bond, :],
                n_bond,
            ))
        # The last state site has nothing to its right, so no channel is open.
        list_bond_factors.append((None, None, 0))

        self._list_bond_factors = list_bond_factors
        return list_bond_factors

    def build_general_generator_mpo(self, list_z_hat, list_expect_L2, norm_corr):
        """
        Builds the combined Hamiltonian + hierarchy MPO for a Hamiltonian of
        any coupling pattern, in the number representation on the
        one-excitation manifold.

        Peak bond dimension 3 + 2 * max(n_bond) over the bonds, which is
        minimal. A chain or star has coupling blocks of rank one and a ring of
        rank two, giving peak width 5 and 7 respectively.

        Parameters
        ----------
        1. list_z_hat: np.ndarray(complex)
                       Conjugate noise values at the current timestep, indexed
                       by L-operator.

        2. list_expect_L2: np.ndarray(complex)
                           L-operator expectation values, indexed by
                           L-operator.

        3. norm_corr: float | complex
                      Normalization correction term.

        Returns
        -------
        1. list_cores_op: list(np.ndarray)
                          MPO cores: one (w_l, 2, 2, w_r) per site, each
                          followed by that site's (w_l, k_max+1, k_max+1, w_r)
                          mode cores.
        """
        n_site = self.n_state
        list_state = self.list_state_list
        I2_site = _P2_SITE + _Q2_SITE
        n_mode_core = int(self.M1_mode_offset[-1])
        list_bond_factors = self._bond_factors()
        list_cores_op = []

        for site in range(n_site):
            l2_idx = site
            Y2_left, _, n_bond_out = list_bond_factors[site]
            n_bond_in = list_bond_factors[site - 1][2] if site > 0 else 0
            # Outgoing cut is identity, the sigma^+/sigma^- pairs still in
            # flight, this site's bath coupling, and the accumulator.
            w_in = 1 if site == 0 else 2 + 2 * n_bond_in
            w_out = 3 + 2 * n_bond_out
            # The accumulator is always the last channel, so its index is
            # whatever the width happens to be on that side.
            idx_accum_in = w_in - 1
            idx_accum_out = w_out - 1
            idx_coupling = 1 + 2 * n_bond_out

            # Indexed [bond_in, occupation', occupation, bond_out]; each
            # (bond_in, bond_out) block is a 2x2 operator on the site.
            T4_core_site = np.zeros((w_in, 2, 2, w_out), dtype=np.complex128)
            T4_core_site[0, :, :, 0] = I2_site
            # Site energy plus this site's noise, both diagonal in occupation.
            energy = (
                self.H2_ham[list_state[site], list_state[site]]
                + 1j * list_z_hat[l2_idx]
            )
            T4_core_site[0, :, :, idx_accum_out] = energy * _P2_SITE
            if site == 0:
                # The norm correction multiplies the identity on the whole
                # chain, so it is emitted once here.
                T4_core_site[0, :, :, idx_accum_out] += -1j * norm_corr * I2_site
            T4_core_site[0, :, :, idx_coupling] = 1j * _P2_SITE

            # Open: this site's sigma^+ enters each outgoing channel with the
            # weight carried by its own row of the left factor.
            for idx_chan_out in range(n_bond_out):
                amp_open = Y2_left[site, idx_chan_out]
                T4_core_site[0, :, :, 1 + idx_chan_out] = amp_open * _T2_PLUS
                T4_core_site[0, :, :, 1 + n_bond_out + idx_chan_out] = (
                    np.conj(amp_open) * _T2_MINUS
                )

            if site > 0:
                T4_core_site[idx_accum_in, :, :, idx_accum_out] = I2_site
                Y2_left_prev, Z2_coeff_prev, _ = list_bond_factors[site - 1]
                # Close: the hoppings ending on this site leave the incoming
                # channels, weighted by the first column of the previous
                # bond's coefficients.
                for idx_chan_in in range(n_bond_in):
                    amp_close = Z2_coeff_prev[idx_chan_in, 0]
                    T4_core_site[1 + idx_chan_in, :, :, idx_accum_out] = (
                        amp_close * _T2_MINUS
                    )
                    T4_core_site[
                        1 + n_bond_in + idx_chan_in, :, :, idx_accum_out
                    ] = np.conj(amp_close) * _T2_PLUS
                # Relay: channels that outlive this site are re-expressed in
                # the outgoing bond's basis. The incoming left factor has
                # orthonormal columns, so the change of basis is a projection.
                if n_bond_out:
                    M2_relay = Y2_left_prev.conj().T @ Y2_left[:site, :]
                    for idx_chan_in in range(n_bond_in):
                        for idx_chan_out in range(n_bond_out):
                            amp_relay = M2_relay[idx_chan_in, idx_chan_out]
                            T4_core_site[
                                1 + idx_chan_in, :, :, 1 + idx_chan_out
                            ] = amp_relay * I2_site
                            T4_core_site[
                                1 + n_bond_in + idx_chan_in, :, :,
                                1 + n_bond_out + idx_chan_out,
                            ] = np.conj(amp_relay) * I2_site
            list_cores_op.append(T4_core_site)

            # The identity channel emits each mode's damping, the coupling channel
            # closes on each mode in turn, and every other channel passes through.
            n_mode_site = self.M1_modes_per_state[site]
            for p in range(n_mode_site):
                idx_mode = self.M1_mode_offset[site] + p
                # The bath channel stays open until this site's last mode, so the
                # right bond loses it there while the left bond still matches the
                # site core's outgoing width.
                coupling_out = 1 if p < n_mode_site - 1 else 0
                w_mode_in = 3 + 2 * n_bond_out
                w_mode_out = 2 + 2 * n_bond_out + coupling_out
                if idx_mode == n_mode_core - 1:
                    # Final core of the chain contracts to the scalar boundary.
                    w_mode_out = 1
                T4_core_mode = np.zeros(
                    (w_mode_in, self.k_max + 1, self.k_max + 1, w_mode_out),
                    dtype=np.complex128,
                )
                if w_mode_out > 1:
                    T4_core_mode[0, :, :, 0] = self.I2_mode
                    for idx_bond in range(1, 1 + 2 * n_bond_out):
                        T4_core_mode[idx_bond, :, :, idx_bond] = self.I2_mode
                T4_core_mode[w_mode_in - 1, :, :, w_mode_out - 1] = self.I2_mode
                # Bath frequency decay plus the <L^dagger> feedback.
                T4_core_mode[0, :, :, w_mode_out - 1] += 1j * (
                    -self.list_w[idx_mode] * self.N2_occ
                    + self.C1_coupling_lower[idx_mode]
                    * np.conj(list_expect_L2[site]) * self.B2_lower
                )
                # Hierarchy raise/lower coupling closes the channel here.
                T4_core_mode[idx_coupling, :, :, w_mode_out - 1] += (
                    self.C1_coupling_raise[idx_mode] * self.B2_raise
                    - self.C1_coupling_lower[idx_mode] * self.B2_lower
                )
                if coupling_out:
                    T4_core_mode[idx_coupling, :, :, idx_coupling] = self.I2_mode
                list_cores_op.append(T4_core_mode)

        return list_cores_op

    def build_fullstate_mpo(self, list_z_hat, list_expect_L2, norm_corr,
                            C2_lt_corr_hier=None):
        """
        Builds combined hierarchy+Hamiltonian MPO cores for fullstate representation.

        # TODO: replace with preprint reference and equation numbers
        # before merging to MesoHOPS.

        Parameters
        ----------
        1. list_z_hat: np.ndarray
                       Conjugate noise values at current timestep, shape (n_lop_full,).

        2. list_expect_L2: np.ndarray
                           L-operator expectation values, shape (n_lop_full,).

        3. norm_corr: float
                      Normalization correction term.

        4. C2_lt_corr_hier: np.ndarray or None
                            LTC hierarchy correction matrix, shape
                            (n_state, n_state). Added to the Hamiltonian
                            channel in the state core.

        Returns
        -------
        1. list_cores_op: list(np.ndarray)
                          MPO cores: one (1, n_state, n_state, bond_dim) state
                          core followed by n_lop_full*modes_per_state mode cores.
        """
        k_max = self.k_max
        n_state = self.n_state
        n_lop_full = self.n_lop_full
        M1_modes_per_state = self.M1_modes_per_state
        M1_mode_offset = self.M1_mode_offset
        I2_mode = self.I2_mode
        I2_state = self.I2_state

        bond_dim = n_lop_full + 2
        state_shape = (1, n_state, n_state, 1)

        # Named MPO channel slices for the state core.
        # Using slices keeps the 4-D indexing axis intact (returns
        # a (1, d, d, 1) view) and replaces verbose n_lop_full
        # arithmetic throughout the method.
        channel_input = slice(0, 1)
        channel_damp = slice(n_lop_full, n_lop_full + 1)
        channel_ham = slice(n_lop_full + 1, n_lop_full + 2)

        # State core block structure (bond_dim = n_lop_full + 2):
        #   indices 0..n_lop_full-1 -- L-op channels (one per l-op)
        #   index n_lop_full        -- damping/noise channel
        #   index n_lop_full+1      -- Hamiltonian + norm correction
        list_cores_op = []
        T4_core_site = np.zeros(
            (1, n_state, n_state, bond_dim),
            dtype=np.complex128,
        )

        # Each l-op opens a bond channel carrying 1j * L_state into the mode
        # cores, where it will be combined with raise/lower/noise. Channels are
        # ordered by chain position so that the mode cores of a state find its
        # channel at the head of their left bond.
        channel = 0
        for abs_state in self.list_state_list:
            if M1_modes_per_state[abs_state] == 0:
                continue
            l2_idx = self.list_index_L2_by_hmode[M1_mode_offset[abs_state]]
            L2_lop_site = self.list_L2_coo[l2_idx]
            T4_core_site[0, L2_lop_site.row, L2_lop_site.col, channel] += (
                1j * L2_lop_site.data
            )
            channel += 1
        # Damping channel: identity into mode damping terms
        T4_core_site[channel_input, :, :, channel_damp] += (
            (1j) * I2_state.reshape(state_shape)
        )
        # Hamiltonian channel: on-site H minus norm correction
        H2_block = self.H2_ham - 1j * norm_corr * I2_state
        # LTC hierarchy correction: 1j factor matches
        # L-op/damping channels so that derivative()'s -1j
        # scaling produces the correct sign.
        if C2_lt_corr_hier is not None:
            H2_block = H2_block + 1j * C2_lt_corr_hier
        T4_core_site[channel_input, :, :, channel_ham] += (
            H2_block.reshape(state_shape)
        )
        list_cores_op.append(T4_core_site)

        # Mode cores carry the hierarchy coupling, the damping terms, and the
        # completed terms. Left bond of a core belonging to a state with
        # n_channel_open channels still live:
        #   index 0                     -- this state's own L-op channel
        #   indices 1..n_channel_open-1 -- later states' L-op channels
        #   index idx_damp              -- damping/noise channel
        #   index idx_ham               -- sink: Hamiltonian, norm correction,
        #                                  and every fully applied L-op term
        # A channel closes at its own state's last mode core, so the survivors
        # shift down one index there and the bond tapers n_lop_full + 2,
        # n_lop_full + 1, ..., 3 — minimal at every cut.
        # HopsModes sorts list_state_list and list_index_L2_by_hmode into chain
        # order, so this loop emits cores left to right along the MPS.
        n_channel_open = n_lop_full
        for abs_state in self.list_state_list:
            n_modes_l2 = M1_modes_per_state[abs_state]
            # A state with no hierarchy modes, such as the spectroscopy ground
            # state, occupies a slot in the state core and emits no mode cores.
            if n_modes_l2 == 0:
                continue
            mode_base = M1_mode_offset[abs_state]
            # Every hierarchy mode of a state carries the same L-operator, so
            # any of this state's modes names it.
            l2_idx = self.list_index_L2_by_hmode[mode_base]
            idx_damp = n_channel_open
            idx_ham = n_channel_open + 1
            for i in range(n_modes_l2):
                idx_mode = mode_base + i
                # This state's channel closes at its last mode core, dropping
                # one channel and shifting the survivors down one index.
                shift = 1 if i == n_modes_l2 - 1 else 0
                # Right-bond positions of the surviving channels.
                idx_damp_out = idx_damp - shift
                idx_out = idx_ham - shift

                # L-op channel: raise/lower + noise
                M2_coupling = (
                    -self.C1_coupling_lower[idx_mode] * self.B2_lower
                    + self.C1_coupling_raise[idx_mode] * self.B2_raise
                    + list_z_hat[l2_idx] * self.I2_mode / n_modes_l2
                )
                # Damping: <L> drift + frequency damping
                M2_damping = (
                    np.conj(list_expect_L2[l2_idx])
                    * self.C1_coupling_lower[idx_mode] * self.B2_lower
                    - self.list_w[idx_mode] * self.N2_occ
                )
                if idx_mode == self.n_hmodes - 1:
                    # Last mode core: contracts all bond channels
                    # to scalar output (idx_ham + 1, phys, phys, 1)
                    T4_core_mode = np.zeros(
                        (idx_ham + 1, k_max + 1, k_max + 1, 1),
                        dtype=np.complex128,
                    )
                    T4_core_mode[0, :, :, 0] += M2_coupling
                    T4_core_mode[idx_damp, :, :, 0] += M2_damping
                    # Hamiltonian: identity pass-through to output
                    T4_core_mode[idx_ham, :, :, 0] += I2_mode
                else:
                    T4_core_mode = np.zeros(
                        (idx_ham + 1, k_max + 1, k_max + 1, idx_out + 1),
                        dtype=np.complex128,
                    )
                    T4_core_mode[0, :, :, idx_out] += M2_coupling
                    T4_core_mode[idx_damp, :, :, idx_out] += M2_damping
                    # Damping identity: keep channel open
                    T4_core_mode[idx_damp, :, :, idx_damp_out] += I2_mode
                    # Hamiltonian identity: pass through
                    T4_core_mode[idx_ham, :, :, idx_out] += I2_mode
                    # L-op identities: pass the channels still in transit
                    for t in range(shift, n_channel_open):
                        T4_core_mode[t, :, :, t - shift] += I2_mode
                list_cores_op.append(T4_core_mode)
            # This state's channel closed at its last mode core.
            n_channel_open -= 1

        return list_cores_op

# Standalone function (not an MpoBuilder method) because callers
# (apply_system_operator) don't have an MpoBuilder instance and
# only need n_state, k_max, M1_modes_per_state — not the full
# builder configuration.
def build_statenumber_operator_mpo(H2_op, n_state, k_max, M1_modes_per_state):
    """
    Build an MPO that applies an arbitrary (n_state x n_state) system operator
    to a number representation MPS, acting as identity on all mode cores.

    Uses the same transfer-matrix structure as _build_statenumber_ham_general_mpo:
    diagonal elements via site projectors, off-diagonal elements via daisy-chained
    transfer matrices through intermediate sites. Bond dimension is
    4 + 2*(n_state - 2) for n_state >= 3, 4 for n_state == 2, and 2 for
    n_state == 1.

    Parameters
    ----------
    1. H2_op: np.ndarray(complex)
              System-space operator, shape (n_state, n_state). Already trimmed
              to the active basis by the caller.

    2. n_state: int
                Number of active system states.

    3. k_max: int
              Maximum hierarchy depth (mode core physical dim = k_max + 1).

    4. M1_modes_per_state: np.ndarray(int)
                           Number of bath-mode cores per system-state core.

    Returns
    -------
    1. list_cores_op: list(np.ndarray)
                      MPO cores interleaved: one state core (w_l, 2, 2, w_r)
                      then modes_per_state identity mode cores
                      (w, k_max+1, k_max+1, w), for each state.
    """
    # Pre-reshape 2x2 state operators to 4-D core shape once
    T4_plus = _T2_PLUS.reshape(1, 2, 2, 1)
    T4_minus = _T2_MINUS.reshape(1, 2, 2, 1)
    Q4_site = _Q2_SITE.reshape(1, 2, 2, 1)
    P4_site = _P2_SITE.reshape(1, 2, 2, 1)

    # Single-site special case: no off-diagonal terms, bond dim 1.
    if n_state == 1:
        T4_core_site = np.zeros((1, 2, 2, 1), dtype=np.complex128)
        T4_core_site[0, :, :, 0] = H2_op[0, 0] * _P2_SITE + _Q2_SITE
        list_cores_op = [T4_core_site]
        list_cores_op.extend(_identity_mode_cores(1, k_max, M1_modes_per_state[0]))
        return list_cores_op

    # Bond dim = 4 base channels (left-transfer, right-transfer,
    # identity, diagonal) + 2 long-range relay channels per
    # non-nearest-neighbor state pair.
    if n_state == 2:
        bond_dim = 4
    else:
        bond_dim = int(4 + 2 * (n_state - 2))

    # Base offset for the right-transfer relay block.
    # Left-transfer relays:  indices 4 .. 4+(n_state-3)
    # Right-transfer relays: indices relay_base .. end
    relay_base = 4 + (n_state - 2)

    list_cores_op = []

    for site in range(n_state):
        if site == 0:
            T4_core_site = np.zeros(
                (1, 2, 2, bond_dim), dtype=np.complex128,
            )
            if n_state > 1:
                T4_core_site[0:1, :, :, 0:1] = T4_plus
                T4_core_site[0:1, :, :, 1:2] = T4_minus
            T4_core_site[0:1, :, :, 2:3] = Q4_site
            T4_core_site[0:1, :, :, 3:4] = (
                H2_op[site, site] * P4_site
            )

        elif site == n_state - 1:
            T4_core_site = np.zeros(
                (bond_dim, 2, 2, 1), dtype=np.complex128,
            )
            T4_core_site[0:1, :, :, 0:1] = (
                H2_op[site - 1, site] * T4_minus
            )
            T4_core_site[1:2, :, :, 0:1] = (
                H2_op[site, site - 1] * T4_plus
            )
            T4_core_site[2:3, :, :, 0:1] = (
                H2_op[site, site] * P4_site
            )
            T4_core_site[3:4, :, :, 0:1] = Q4_site
            for i in range(n_state - 2):
                site_coupled = site - 2 - i
                idx_left_relay = 4 + i
                idx_right_relay = relay_base + i
                T4_core_site[
                    idx_left_relay:idx_left_relay+1,
                    :, :, 0:1,
                ] = H2_op[site_coupled, site] * T4_minus
                T4_core_site[
                    idx_right_relay:idx_right_relay+1,
                    :, :, 0:1,
                ] = H2_op[site, site_coupled] * T4_plus

        else:
            T4_core_site = np.zeros(
                (bond_dim, 2, 2, bond_dim),
                dtype=np.complex128,
            )
            T4_core_site[0:1, :, :, 3:4] = (
                H2_op[site - 1, site] * T4_minus
            )
            T4_core_site[1:2, :, :, 3:4] = (
                H2_op[site, site - 1] * T4_plus
            )
            T4_core_site[2:3, :, :, 0:1] = T4_plus
            T4_core_site[2:3, :, :, 1:2] = T4_minus
            T4_core_site[2:3, :, :, 2:3] = Q4_site
            T4_core_site[2:3, :, :, 3:4] = (
                H2_op[site, site] * P4_site
            )
            T4_core_site[3:4, :, :, 3:4] = Q4_site
            T4_core_site[0:1, :, :, 4:5] = Q4_site
            T4_core_site[
                1:2, :, :, relay_base:relay_base+1,
            ] = Q4_site
            for i in range(n_state - 3):
                idx_left_relay = 4 + i
                idx_right_relay = relay_base + i
                T4_core_site[
                    idx_left_relay:idx_left_relay+1,
                    :, :,
                    idx_left_relay+1:idx_left_relay+2,
                ] = Q4_site
                T4_core_site[
                    idx_right_relay:idx_right_relay+1,
                    :, :,
                    idx_right_relay+1:idx_right_relay+2,
                ] = Q4_site
            for i in range(site - 1):
                site_coupled = site - 2 - i
                idx_left_relay = 4 + i
                idx_right_relay = relay_base + i
                T4_core_site[
                    idx_left_relay:idx_left_relay+1,
                    :, :, 3:4,
                ] = H2_op[site_coupled, site] * T4_minus
                T4_core_site[
                    idx_right_relay:idx_right_relay+1,
                    :, :, 3:4,
                ] = H2_op[site, site_coupled] * T4_plus

        list_cores_op.append(T4_core_site)

        # Mode cores: identity pass-through for all bond channels.
        list_cores_op.extend(
            _identity_mode_cores(T4_core_site.shape[3], k_max, M1_modes_per_state[site])
        )

    return list_cores_op


def build_statenumber_dipole_mpo(
    list_mu, n_state, k_max, M1_modes_per_state, raise_or_lower,
):
    """
    Build the MPO for a sum-of-single-site dipole operator (raise or lower)
    in the statenumber representation under the ground-state-as-vacuum
    convention.  Standalone counterpart to
    MpoBuilder.build_statenumber_dipole_{raise,lower}_mpo for callers
    (e.g.\\ nondyadic_spectroscopy) that do not have a full MpoBuilder
    instance.

    Spectroscopically this is the bare transition-dipole operator that
    moves amplitude between the ground state and the single-excitation
    manifold: raise (a^dagger) is excitation by a photon (absorption),
    lower (a) is de-excitation (emission), with no ground-state or
    excited-manifold preservation.

    The operator is
        sum_k list_mu[k] * sigma_k,
    where sigma = a^dagger (T4_plus = |1><0|) for raise_or_lower='raise',
    sigma = a (T4_minus = |0><1|) for raise_or_lower='lower'.  Sites with
    list_mu[k] == 0 contribute identity on their state core and drop from
    the sum.  Acts as identity on all mode cores.  Bond dimension 2
    between state cores; 1 at the chain boundaries.

    Parameters
    ----------
    1. list_mu: np.ndarray(complex)
                Per-site dipole amplitudes, shape (n_state,).  Entries
                set to 0 mark sites excluded from the dipole's site
                selection.

    2. n_state: int
                Number of active system states (== number of excited
                states in the vacuum convention).

    3. k_max: int
              Maximum hierarchy depth (mode core physical dim = k_max + 1).

    4. M1_modes_per_state: np.ndarray(int)
                           Number of bath-mode cores per system-state core.

    5. raise_or_lower: str
                       'raise' for the raise operator, 'lower' for the
                       lower operator.

    Returns
    -------
    1. list_cores_op: list(np.ndarray)
                      MPO cores interleaved: one state core
                      (w_l, 2, 2, w_r) followed by M1_modes_per_state[site]
                      identity mode cores, for each state.
    """
    if raise_or_lower == 'raise':
        sigma = _T2_PLUS.reshape(1, 2, 2, 1)
    elif raise_or_lower == 'lower':
        sigma = _T2_MINUS.reshape(1, 2, 2, 1)
    else:
        raise ValueError(
            f"raise_or_lower must be 'raise' or 'lower', "
            f"got {raise_or_lower!r}."
        )

    if len(list_mu) != n_state:
        raise ValueError(
            f'list_mu must have length n_state = {n_state}, '
            f'got {len(list_mu)}.'
        )

    # State-core identity I_2 = P + Q reshaped as a (1, 2, 2, 1) core.
    I4_site = (_P2_SITE + _Q2_SITE).reshape(1, 2, 2, 1)

    list_cores_op = []

    for site in range(n_state):
        mu_site = list_mu[site]

        if n_state == 1:
            T4_core_site = np.zeros((1, 2, 2, 1), dtype=np.complex128)
            T4_core_site[0:1, :, :, 0:1] = mu_site * sigma
        elif site == 0:
            # First state core: row vector (1, 2, 2, 2).
            # Right-bond channel 0 means the local sigma operator has
            # already been placed on this site; channel 1 means it has
            # not been placed yet and the MPO should keep propagating.
            T4_core_site = np.zeros((1, 2, 2, 2), dtype=np.complex128)
            T4_core_site[0:1, :, :, 0:1] = mu_site * sigma
            T4_core_site[0:1, :, :, 1:2] = I4_site
        elif site == n_state - 1:
            # Last state core: column vector (2, 2, 2, 1).
            T4_core_site = np.zeros((2, 2, 2, 1), dtype=np.complex128)
            T4_core_site[0:1, :, :, 0:1] = I4_site
            T4_core_site[1:2, :, :, 0:1] = mu_site * sigma
        else:
            # Interior state core: (2, 2, 2, 2).
            T4_core_site = np.zeros((2, 2, 2, 2), dtype=np.complex128)
            T4_core_site[0:1, :, :, 0:1] = I4_site
            T4_core_site[1:2, :, :, 0:1] = mu_site * sigma
            T4_core_site[1:2, :, :, 1:2] = I4_site

        list_cores_op.append(T4_core_site)

        list_cores_op.extend(
            _identity_mode_cores(
                T4_core_site.shape[3], k_max, M1_modes_per_state[site],
            )
        )

    return list_cores_op


def build_statenumber_dipole_lower_plus_ident_mpo(
    list_mu, n_state, k_max, M1_modes_per_state,
):
    """
    Build the MPO for the operator
        sum_k list_mu[k] * a_k  +  I_excited,
    where I_excited = sum_k |e_k><e_k| is the single-excitation
    projector, in the number representation under the
    ground-state-as-vacuum convention.

    Spectroscopically this is the fluorescence detection-pulse operator:
    the lower term de-excites a single excitation to the ground state
    (forming the G/E coherence the signal reads out) while +I_excited
    keeps the surviving excited-manifold content.

    Action on the natural basis:
        |g>          -> 0          (killed by both terms)
        |e_j>        -> list_mu[j] |g>  +  |e_j>
        |e_j, e_k>   -> 0          (killed by I_excited)
        |e_j, e_k, ...> -> 0        (any multiple-excited state is
                                      annihilated by the excited-block
                                      projector)

    Bond dimension 4 = parallel bond-dim-2 mu^- path (sum-of-local
    sigma^- terms) and bond-dim-2 I_excited path (sum-of-local P_site
    terms with Q_site elsewhere).  W-matrix channels:
        0 = mu^- path, sigma^- has not yet fired -> apply I_site,
        1 = mu^- path, sigma^- already fired     -> apply I_site,
        2 = I_excited path, P has not yet fired  -> apply Q_site,
        3 = I_excited path, P already fired      -> apply Q_site to
            kill any later multiple-excited occupation.

    Here, "fired" means the corresponding branch has already been
    selected on an earlier site in the MPO walk.

    Parameters
    ----------
    1. list_mu: np.ndarray(complex)
                Per-site dipole amplitudes, shape (n_state,).  Entries
                set to 0 mark sites excluded from the lower's site
                selection.

    2. n_state: int
                Number of active system states.

    3. k_max: int
              Maximum hierarchy depth (mode core physical dim = k_max + 1).

    4. M1_modes_per_state: np.ndarray(int)
                           Number of bath-mode cores per system-state core.

    Returns
    -------
    1. list_cores_op: list(np.ndarray)
                      MPO cores interleaved: one state core then mode
                      cores, for each state.
    """
    if len(list_mu) != n_state:
        raise ValueError(
            f'list_mu must have length n_state = {n_state}, '
            f'got {len(list_mu)}.'
        )

    sigma_minus = _T2_MINUS.reshape(1, 2, 2, 1)
    P4_site = _P2_SITE.reshape(1, 2, 2, 1)
    Q4_site = _Q2_SITE.reshape(1, 2, 2, 1)
    I4_site = P4_site + Q4_site

    list_cores_op = []

    for site in range(n_state):
        mu_site = list_mu[site]

        if n_state == 1:
            # Single state core: mu_1 * sigma^- + P (=I_excited on a
            # single site reduces to projecting onto |1>).
            T4_core_site = np.zeros((1, 2, 2, 1), dtype=np.complex128)
            T4_core_site[0:1, :, :, 0:1] = (
                mu_site * sigma_minus + P4_site
            )
        elif site == 0:
            # First state core: row vector (1, 2, 2, 4).
            T4_core_site = np.zeros((1, 2, 2, 4), dtype=np.complex128)
            # mu^- path
            T4_core_site[0:1, :, :, 0:1] = I4_site          # I, not yet fired
            T4_core_site[0:1, :, :, 1:2] = mu_site * sigma_minus  # fire at site 1
            # I_excited path
            T4_core_site[0:1, :, :, 2:3] = Q4_site          # Q, not yet fired
            T4_core_site[0:1, :, :, 3:4] = P4_site          # fire P at site 1
        elif site == n_state - 1:
            # Last state core: column vector (4, 2, 2, 1).
            T4_core_site = np.zeros((4, 2, 2, 1), dtype=np.complex128)
            # mu^- path
            T4_core_site[0:1, :, :, 0:1] = mu_site * sigma_minus  # fire at last
            T4_core_site[1:2, :, :, 0:1] = I4_site          # already fired -> I
            # I_excited path
            T4_core_site[2:3, :, :, 0:1] = P4_site          # fire P at last
            T4_core_site[3:4, :, :, 0:1] = Q4_site          # already fired -> Q
        else:
            # Interior state core: (4, 2, 2, 4).
            T4_core_site = np.zeros((4, 2, 2, 4), dtype=np.complex128)
            # mu^- path
            T4_core_site[0:1, :, :, 0:1] = I4_site          # continue not-fired
            T4_core_site[0:1, :, :, 1:2] = mu_site * sigma_minus  # fire here
            T4_core_site[1:2, :, :, 1:2] = I4_site          # continue fired
            # I_excited path
            T4_core_site[2:3, :, :, 2:3] = Q4_site          # continue not-fired
            T4_core_site[2:3, :, :, 3:4] = P4_site          # fire P here
            T4_core_site[3:4, :, :, 3:4] = Q4_site          # continue fired

        list_cores_op.append(T4_core_site)

        list_cores_op.extend(
            _identity_mode_cores(
                T4_core_site.shape[3], k_max, M1_modes_per_state[site],
            )
        )

    return list_cores_op


def build_statenumber_dipole_raise_plus_ground_ident_mpo(
    list_mu, n_state, k_max, M1_modes_per_state,
):
    """
    Build the MPO for the operator
        sum_k list_mu[k] * a_k^dagger  +  I_g,
    where I_g is the ground-state projector |g><g|, i.e.
    |0,...,0><0,...,0|, the projector onto the all-zeros configuration
    of the system cores, in the number representation under the
    ground-state-as-vacuum convention.

    Spectroscopically this is the absorption excitation-pulse operator:
    the raise term promotes the ground state into the single-excitation
    manifold while +I_g preserves the ground-state amplitude the NL <L>
    denominator needs.

    Action on the natural basis:
        |g> = |0,...,0>  -> sum_k list_mu[k] |e_k>  +  |g>
        |e_j>            -> 0                       (killed by both terms)

    Counterpart to build_statenumber_dipole_lower_plus_ident_mpo: the
    raise term excites the all-zeros configuration into the single-
    excitation manifold; the +I_g term preserves the all-zeros amplitude
    after the raise, matching the gs_core dipole-raise's
    "(0,0)=1 keep |g>" behavior.  Without this preservation, the
    NL mean-field <L> denominator (which the EOM augments with
    |gs_amp|^2 under the vacuum convention) collapses to zero after
    the raise, breaking trajectory-by-trajectory equivalence with
    the gs_core path under NL absorption.

    Bond dimension 3 = parallel bond-dim-2 mu^+ path (sum-of-local
    sigma^+ terms) and bond-dim-1 I_g path (Q_site at every site, no
    before/after firing distinction).  W-matrix channels:
        0 = mu^+ path, sigma^+ has not yet fired -> apply I_site,
        1 = mu^+ path, sigma^+ already fired     -> apply I_site,
        2 = I_g path                             -> apply Q_site.

    Parameters
    ----------
    1. list_mu: np.ndarray(complex)
                Per-site dipole amplitudes, shape (n_state,).  Entries
                set to 0 mark sites excluded from the raise's site
                selection.

    2. n_state: int
                Number of active system states (excited states).

    3. k_max: int
              Maximum hierarchy depth (mode core physical dim = k_max + 1).

    4. M1_modes_per_state: np.ndarray(int)
                           Number of bath-mode cores per system-state core.

    Returns
    -------
    1. list_cores_op: list(np.ndarray)
                      MPO cores interleaved: one state core then mode
                      cores, for each state.
    """
    if len(list_mu) != n_state:
        raise ValueError(
            f'list_mu must have length n_state = {n_state}, '
            f'got {len(list_mu)}.'
        )

    sigma_plus = _T2_PLUS.reshape(1, 2, 2, 1)
    P4_site = _P2_SITE.reshape(1, 2, 2, 1)
    Q4_site = _Q2_SITE.reshape(1, 2, 2, 1)
    I4_site = P4_site + Q4_site

    list_cores_op = []

    for site in range(n_state):
        mu_site = list_mu[site]

        if n_state == 1:
            # Single state core: mu_1 * sigma^+ + Q (= I_g on one site).
            T4_core_site = np.zeros((1, 2, 2, 1), dtype=np.complex128)
            T4_core_site[0:1, :, :, 0:1] = (
                mu_site * sigma_plus + Q4_site
            )
        elif site == 0:
            # First state core: row vector (1, 2, 2, 3).
            T4_core_site = np.zeros((1, 2, 2, 3), dtype=np.complex128)
            # mu^+ path
            T4_core_site[0:1, :, :, 0:1] = I4_site               # not yet fired
            T4_core_site[0:1, :, :, 1:2] = mu_site * sigma_plus  # fire at site 0
            # I_g path
            T4_core_site[0:1, :, :, 2:3] = Q4_site               # I_g opener
        elif site == n_state - 1:
            # Last state core: column vector (3, 2, 2, 1).
            T4_core_site = np.zeros((3, 2, 2, 1), dtype=np.complex128)
            # mu^+ path
            T4_core_site[0:1, :, :, 0:1] = mu_site * sigma_plus  # fire at last
            T4_core_site[1:2, :, :, 0:1] = I4_site               # already fired
            # I_g path
            T4_core_site[2:3, :, :, 0:1] = Q4_site               # close I_g
        else:
            # Interior state core: (3, 2, 2, 3).
            T4_core_site = np.zeros((3, 2, 2, 3), dtype=np.complex128)
            # mu^+ path
            T4_core_site[0:1, :, :, 0:1] = I4_site               # continue not-fired
            T4_core_site[0:1, :, :, 1:2] = mu_site * sigma_plus  # fire here
            T4_core_site[1:2, :, :, 1:2] = I4_site               # continue fired
            # I_g path
            T4_core_site[2:3, :, :, 2:3] = Q4_site               # continue I_g

        list_cores_op.append(T4_core_site)

        list_cores_op.extend(
            _identity_mode_cores(
                T4_core_site.shape[3], k_max, M1_modes_per_state[site],
            )
        )

    return list_cores_op
