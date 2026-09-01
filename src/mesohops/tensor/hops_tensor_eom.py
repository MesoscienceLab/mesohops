from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mesohops.basis.hops_modes import HopsModes
from mesohops.basis.hops_noise_memory import HopsNoiseMemory
from mesohops.basis.hops_system import HopsSystem
from mesohops.eom.eom_functions import (
    calc_delta_zmem,
    calc_LT_corr,
    calc_LT_corr_linear,
    calc_LT_corr_to_norm_corr,
    compress_zmem,
    operator_expectation,
)
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.mpo_constructors import (
    MpoBuilder,
    build_statenumber_operator_mpo,
)
from mesohops.tensor.tensor_eom_functions import (
    build_physical_correction_mpo,
    calc_norm_corr_tensor,
    tensor_matvec_prod,
)
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.tensor_operations import (
    scale_mps,
    tensor_add,
)

__title__ = 'Tensor HOPS EOM'
__author__ = 'B. Z. Citty'
__maintainer__ = 'B. Z. Citty'


class HopsTensorEOM:
    """
    Builds and applies the time-evolution MPO for tensor HOPS.

    Sits between the MPS algebra (HopsTensorWavefunction) and the time-stepping
    algorithms (tensor_integrator.py). The class owns the MPO storage and the logic for
    computing the time-evolution operator and dz/dt at each integrator sub-step.
    This two-phase design (build_generator then derivative) differs from HopsEOM,
    which produces a single dsystem_dt derivative closure. The split is needed
    because multi-step integrators (RK4) rebuild the MPO at each sub-step noise
    point while reusing the derivative application.
    """

    __slots__ = (
        'wavefunction',  # MPS wavefunction container (HopsTensorWavefunction)
        'system',  # System parameters and Hamiltonian (HopsSystem)
        'mode',  # Bath mode indexing and coupling strengths (HopsModes)
        'noise_memory',  # Noise memory drift terms (HopsNoiseMemory)
        'adaptive',  # True when state adaptivity is active (bool)
        'mpo_cores',  # Full operator MPO (list[np.ndarray])
        # Hierarchy MPO on the general number path, the combined Hamiltonian +
        # hierarchy generator on a recognized topology, and the single MPO of
        # the fullstate method.
        '_list_cores_op',  # Operator MPO cores (list[np.ndarray])
        # Separate Hamiltonian MPO, built only on the general number path.
        # Empty everywhere else, which is how _construct_MPO knows whether
        # there is a second MPO to add.
        '_list_cores_ham',  # Hamiltonian MPO cores (list[np.ndarray])
        'normalization',  # Auxiliary scaling convention, 'homps' or 'adhops' (str)
        'mpo_builder',  # Persistent MPO builder (MpoBuilder)
        'flag_linear',  # True when EQUATION_OF_MOTION is LINEAR (bool)
        '_has_lt_corr',  # True when LTC corrections should be applied (bool)
        '_C2_LT_corr_phys',  # LTC physical correction matrix (np.ndarray|None)
        '_C2_LT_corr_hier',  # LTC hierarchy correction matrix (np.ndarray|None)
        # Diagnostic side-channel: complexity observed by the most recent
        # derivative() call; max across an integrator step (reset by the
        # step function, read by propagate). Avoids plumbing complexity
        # through every return value in the derivative → step → _step
        # → propagate chain.
        'last_matvec_complexity',  # Complexity of most recent derivative() (int)
        'max_complexity_step',  # Max complexity during the current step (int)
    )

    def __init__(
        self,
        wavefunction: HopsTensorWavefunction,
        system: HopsSystem,
        mode: HopsModes,
        noise_memory: HopsNoiseMemory,
        adaptive: bool,
        eom_param: dict,
        normalization: str = 'homps',
    ) -> None:
        """
        Inputs
        ------
        1. wavefunction: HopsTensorWavefunction
                         MPS wavefunction container.
        2. system: HopsSystem
                   System parameters and Hamiltonian.
        3. mode: HopsModes
                 Bath mode indexing and coupling strengths.
        4. noise_memory: HopsNoiseMemory
                         Noise memory drift terms.
        5. adaptive: bool
                     True when state adaptivity is active.
        6. eom_param: dict
                      Equation-of-motion parameter dictionary; must contain
                      'EQUATION_OF_MOTION'.
        7. normalization: str
                          Scaling convention for the auxiliary vectors
                          (default 'homps'). 'homps' splits the bath
                          coupling evenly between the raising and lowering
                          operators, g_m/sqrt(|g_m|) up and sqrt(|g_m|)
                          down, so neither direction dominates and the
                          auxiliary amplitudes stay comparable with
                          hierarchy depth, which keeps the MPS well
                          conditioned under truncation. 'adhops' is the
                          unbalanced scaling of the vector HOPS code, w_m up
                          and g_m/w_m down, kept so tensor results can be
                          compared against it term by term. The two differ
                          by a rescaling of each auxiliary, not by physics.
                          _build_mpo_parts accepts 'homps' only.

        Returns
        -------
        None
        """
        self.wavefunction = wavefunction
        self.system = system
        self.mode = mode
        self.noise_memory = noise_memory
        self.adaptive = adaptive
        self.normalization = normalization
        self.flag_linear = eom_param['EQUATION_OF_MOTION'] == 'LINEAR'
        self.mpo_cores = []
        self._list_cores_op = []
        self._list_cores_ham = []

        self._has_lt_corr = False
        self._C2_LT_corr_phys = None
        self._C2_LT_corr_hier = None

        # Side-channel diagnostics: set by derivative() / step functions.
        self.last_matvec_complexity = 0
        self.max_complexity_step = 0

        self.mpo_builder = MpoBuilder(
            k_max=self.wavefunction.k_max,
            n_state=self.system.size,
            modes_per_state=self.wavefunction.M1_modes_per_state,
            n_lop_full=self.mode.n_l2,
            ham=self.system.param['HAMILTONIAN'],
            state_list=self.system.state_list,
            mode=self.mode,
            normalization=self.normalization,
            n_states_full=self.system.param['NSTATES'],
            flag_nearest_neighbor_ham=self.system.flag_nearest_neighbor_ham,
            flag_gs_vacuum=self.wavefunction.flag_gs_vacuum,
            mpo_epsilon=self.wavefunction.mpo_epsilon,
        )

    def refresh_builder(self) -> None:
        """
        Refreshes the persistent MpoBuilder's state-dependent attributes
        after an adaptive basis change.

        Clears the MPO cores and the LTC corrections, both of which are
        rebuilt from the new active basis on the next build_generator call.
        refresh_state_data drops the builder's cached bond factors.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        state_list = self.system.state_list
        self.mpo_builder.refresh_state_data(len(state_list), state_list)
        self.mpo_cores = []
        self._list_cores_op = []
        self._list_cores_ham = []
        self._has_lt_corr = False
        self._C2_LT_corr_phys = None
        self._C2_LT_corr_hier = None

    def build_generator(
        self,
        z_mem: np.ndarray,
        z_rnd: np.ndarray,
        z_rnd2: np.ndarray,
    ) -> np.ndarray:
        """
        Build the time-evolution MPO into self.mpo_cores and return dz/dt.

        Reads config flags from self.wavefunction. When flag_norm is True,
        a nonlinear norm correction is included in the MPO.

        Parameters
        ----------
        1. z_mem: np.ndarray(complex)
                  Current memory term values.
        2. z_rnd: np.ndarray(complex)
                  Primary noise at this sub-step (absolute indices).
        3. z_rnd2: np.ndarray(complex)
                   Secondary noise at this sub-step (absolute indices).

        Returns
        -------
        1. dz_dt: np.ndarray(complex)
                  Time derivative of the memory terms.
        """
        # Local aliases for mode indexing arrays
        list_idx_L2_by_hmode = self.mode.list_index_L2_by_hmode
        list_L2_coo = self.mode.list_L2_coo
        list_absidx_mode = self.mode.list_modeidx_abs
        list_absidx_L2 = self.mode.list_l2idx_abs

        # Linear EOM: no zmem, no <L> feedback, no norm correction, dz/dt = 0
        if self.flag_linear:
            # LTC for LINEAR: -sum_n c_n L_n^2 applied to physical wf only
            list_lt_corr_param = self.system.list_lt_corr_param
            if any(list_lt_corr_param):
                psi = self.wavefunction.psi
                C2_phys = calc_LT_corr_linear(
                    list_lt_corr_param, self.mode.list_L2_sq_csr,
                )
                self._C2_LT_corr_phys = (
                    C2_phys.toarray() if sp.issparse(C2_phys)
                    else np.asarray(C2_phys)
                )
                self._C2_LT_corr_hier = None
                self._has_lt_corr = True
            else:
                self._has_lt_corr = False

            list_z_hat = (
                np.conj(z_rnd[list_absidx_L2])
                - 1j * z_rnd2[list_absidx_L2]
            )
            self._construct_MPO(list_z_hat, [0.0] * len(list_L2_coo), 0.0)
            return np.zeros_like(z_mem)

        # Compute <L2> expectation values for each L2 operator.
        #
        # In the ground-state-as-vacuum convention extract_psi only
        # returns the single-excitation amplitudes; the physical GS lives
        # at the |0,...,0> configuration and is invisible to traj.psi.
        # When wavefunction.flag_gs_vacuum is True we add |<0,...,0|psi>|^2
        # to <psi|psi> so the denominator matches the GS-as-state-core
        # trajectory's denominator under matching noise, keeping the two
        # conventions trajectory-equivalent under NL EOM.
        psi = self.wavefunction.psi
        if self.wavefunction.flag_gs_vacuum:
            norm_sq = self.wavefunction.manifold_norm_sq
            list_expect_L2 = [
                (np.conj(psi) @ (list_L2_coo[idx] @ psi)) / norm_sq
                for idx in range(len(list_L2_coo))
            ]
        else:
            list_expect_L2 = [
                operator_expectation(list_L2_coo[idx], psi)
                for idx in range(len(list_L2_coo))
            ]

        # Low-temperature correction
        list_lt_corr_param = self.system.list_lt_corr_param
        norm_corr_lt = 0.0
        if any(list_lt_corr_param):
            list_L2_sq_csr = self.mode.list_L2_sq_csr
            list_avg_L2_sq = [
                operator_expectation(list_L2_sq_csr[idx], psi)
                for idx in range(len(list_L2_sq_csr))
            ]
            C2_phys, C2_hier = calc_LT_corr(
                list_lt_corr_param,
                self.mode.list_L2_csr,
                list_expect_L2,
                list_L2_sq_csr,
            )
            self._C2_LT_corr_phys = (
                C2_phys.toarray() if sp.issparse(C2_phys)
                else np.asarray(C2_phys)
            )
            self._C2_LT_corr_hier = (
                C2_hier.toarray() if sp.issparse(C2_hier)
                else np.asarray(C2_hier)
            )
            self._has_lt_corr = True

            if self.wavefunction.flag_norm:
                norm_corr_lt = calc_LT_corr_to_norm_corr(
                    list_lt_corr_param, list_expect_L2, list_avg_L2_sq,
                )
        else:
            self._has_lt_corr = False

        # TODO: compress_zmem uses list_zmemactivemodeidx_rel which may
        # diverge from the mode basis in adaptive runs, producing a wrong
        # list_z_hat_rel for the norm correction.

        # First z_hat: uses active z_mem indices (relative) for norm correction.
        # This includes only the z_mem modes currently tracked in the basis.
        # The full noise coupling is z_hat = conj(z_rnd) + z_mem_compressed
        # minus the secondary noise contribution: -1j * z_rnd2
        # (see hops_eom.py line 580: z_hat[j] - 1j * z_rnd2[j])
        list_z_hat_rel = (
            np.conj(z_rnd[list_absidx_L2])
            - 1j * z_rnd2[list_absidx_L2]
            + compress_zmem(
                z_mem,
                list_idx_L2_by_hmode,
                self.noise_memory.list_zmemactivemodeidx_rel,
            )
        )

        # Compute the nonlinear norm correction (zero if unnormalized EOM)
        if self.wavefunction.flag_norm:
            norm_corr = calc_norm_corr_tensor(
                self.wavefunction,
                psi,
                list_z_hat_rel,
                list_expect_L2,
                self.mode,
                self.mode.list_index_L2_by_hmode,
            ) + norm_corr_lt
        else:
            norm_corr = norm_corr_lt

        # Second z_hat: uses absolute mode indices for the MPO construction.
        # This differs from the first z_hat because the MPO needs the full
        # mode indexing, while the norm correction uses the active subset.
        list_z_hat_abs = (
            np.conj(z_rnd[list_absidx_L2])
            - 1j * z_rnd2[list_absidx_L2]
            + compress_zmem(
                z_mem, list_idx_L2_by_hmode, list_absidx_mode,
            )
        )

        # Build the time-evolution MPO (method-dependent)
        if self.wavefunction.method not in (
            'fullstate', 'number',
        ):
            raise UnsupportedRequest(self.wavefunction.method, 'build_generator')
        self._construct_MPO(list_z_hat_abs, list_expect_L2, norm_corr)

        # Compute dz/dt for the memory term integration
        dz_dt = calc_delta_zmem(
            z_mem,
            list_expect_L2,
            self.noise_memory.list_zmemg_abs,
            self.noise_memory.list_zmemw_abs,
            list_idx_L2_by_hmode,
            list_absidx_mode,
            self.noise_memory.list_zmemmodeidx_abs,
            list_absidx_L2,
            self.system.list_activel2idx_abs,
        )

        return dz_dt

    def derivative(self) -> tuple[list[np.ndarray], int]:
        """
        Apply the last-built MPO to the wavefunction via tensor_matvec_prod.

        Uses wavefunction.list_cores_phi as the input MPS.
        Pure computation — does not mutate the wavefunction. The calling
        integrator is responsible for all wavefunction mutations.

        Returns
        -------
        1. cores: list(np.ndarray)
                  Derivative of the wavefunction in MPS form (-i H|phi>).

        Side effects
        ------------
        Sets `self.last_matvec_complexity` to the scalar complexity proxy
        of the uncompressed contracted MPS (i.e. its peak size before SVD
        truncation) from this call. The step function reads it across
        sub-steps rather than the derivative threading it through every
        return value to propagate().
        """
        dphi_cores, complexity = tensor_matvec_prod(
            self.wavefunction.list_cores_phi,
            self.mpo_cores,
            self.wavefunction.mps_epsilon,
            self.wavefunction.bond_dim_max,
        )
        self.last_matvec_complexity = complexity

        # Apply the -1j Schrödinger factor to the MPO-MPS product,
        # matching the vector convention where -1j is in the derivative.
        scale_mps(dphi_cores, -1j)

        return dphi_cores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _construct_MPO(
        self,
        list_z_hat: np.ndarray,
        list_expect_L2: list[complex],
        norm_corr: float | complex,
    ) -> None:
        """
        Assembles mpo_cores from hierarchy and Hamiltonian MPO parts.

        Calls _build_mpo_parts to populate _list_cores_op and _list_cores_ham, then
        combines them into self.mpo_cores. For number,
        the two parts are added via tensor_add; for fullstate,
        _list_cores_op is used directly. LTC corrections stored on self
        are folded into the final MPO when _has_lt_corr is True.

        Parameters
        ----------
        1. list_z_hat: np.ndarray(complex)
                       Conjugate noise, indexed by L2.
        2. list_expect_L2: list(complex)
                           <L2^dagger> expectation values, one per L2
                           operator.
        3. norm_corr: float | complex
                      Normalization correction term.

        Returns
        -------
        None
        """
        self._build_mpo_parts(list_z_hat, list_expect_L2, norm_corr)

        method = self.wavefunction.method
        mpo_epsilon = self.wavefunction.mpo_epsilon
        if self._list_cores_ham:
            # Separate Hamiltonian part: add it to the hierarchy MPO. MPO
            # compression uses mpo_epsilon (defaults to 0 — exact). The MPS
            # epsilon/bond_dim_max control wavefunction approximation and are
            # decoupled from operator truncation; setting mpo_epsilon > 0 lets
            # the user opt into operator compression independently.
            mpo_bond_dim_max = (
                max(c.shape[-1] for c in self._list_cores_op)
                + max(c.shape[-1] for c in self._list_cores_ham)
            )
            self.mpo_cores = tensor_add(
                self._list_cores_op,
                self._list_cores_ham,
                epsilon=mpo_epsilon,
                bond_dim_max=mpo_bond_dim_max,
            )
        else:
            self.mpo_cores = self._list_cores_op

        # Fold LTC corrections into the assembled MPO
        if self._has_lt_corr:
            k_max = self.wavefunction.k_max
            M1_modes_per_state = self.wavefunction.M1_modes_per_state
            n_state = self.mpo_builder.n_state

            # Hierarchy correction: system-space operator on all k levels.
            # For fullstate this is in the Hamiltonian channel;
            # for statenumber it is a separate operator MPO.
            if (self._C2_LT_corr_hier is not None
                    and method == 'number'):
                ltc_hier_mpo = build_statenumber_operator_mpo(
                    1j * self._C2_LT_corr_hier,
                    n_state, k_max, M1_modes_per_state,
                )
                max_bond = (
                    max(c.shape[-1] for c in self.mpo_cores)
                    + max(c.shape[-1] for c in ltc_hier_mpo)
                )
                self.mpo_cores = tensor_add(
                    self.mpo_cores, ltc_hier_mpo,
                    epsilon=mpo_epsilon, bond_dim_max=max_bond,
                )

            # Physical correction: system-space operator at k=0 only
            if self._C2_LT_corr_phys is not None:
                ltc_phys_mpo = build_physical_correction_mpo(
                    1j * self._C2_LT_corr_phys, method, k_max,
                    M1_modes_per_state,
                )
                max_bond = (
                    max(c.shape[-1] for c in self.mpo_cores)
                    + max(c.shape[-1] for c in ltc_phys_mpo)
                )
                self.mpo_cores = tensor_add(
                    self.mpo_cores, ltc_phys_mpo,
                    epsilon=mpo_epsilon, bond_dim_max=max_bond,
                )

    def _build_mpo_parts(
        self,
        list_z_hat: np.ndarray,
        list_expect_L2: list[complex],
        norm_corr: float | complex,
    ) -> None:
        """
        Build _list_cores_op and _list_cores_ham from noise and system inputs.

        Delegates to MpoBuilder and the appropriate MPO builder method for the
        current representation. The number method builds a single combined
        generator for any Hamiltonian, each bond carrying as many channels as
        the rank of the coupling block that crosses it; fullstate always
        builds one MPO, with the Hamiltonian embedded in it. _list_cores_ham
        is left empty in every case except the FLAG_MPO_OPTIMIZE=False
        reference path, which is how _construct_MPO knows whether there is
        anything to add.

        Parameters
        ----------
        1. list_z_hat: np.ndarray(complex)
                       Conjugate noise (z* + compressed z_mem),
                       indexed by L2.
        2. list_expect_L2: list(complex)
                           <L2^dagger> expectation values, one per L2
                           operator.
        3. norm_corr: float | complex
                      Nonlinear norm correction (zero for linear or
                      unnormalized EOM). Includes LTC when active.

        Returns
        -------
        None
        """
        if self.normalization != 'homps':
            raise UnsupportedRequest(
                self.normalization,
                '_build_mpo_parts normalization',
            )

        builder = self.mpo_builder

        self._list_cores_op = []
        self._list_cores_ham = []
        if self.wavefunction.method == 'number':
            if not self.wavefunction.flag_mpo_optimize:
                # Reference path: the hierarchy and the Hamiltonian are built
                # apart and summed by _construct_MPO. Kept as the independent
                # check the combined generators are tested against.
                self._list_cores_op = builder.build_statenumber_hierarchy_mpo(
                    list_z_hat,
                    list_expect_L2,
                    norm_corr,
                )
                self._list_cores_ham = builder.build_statenumber_ham_mpo()
                return
            # One combined Hamiltonian + hierarchy MPO, each bond built at the
            # rank of the coupling block that crosses it, leaving nothing to
            # add or compress. _list_cores_ham stays empty to signal that.
            self._list_cores_op = builder.build_general_generator_mpo(
                list_z_hat, list_expect_L2, norm_corr,
            )
        elif self.wavefunction.method == 'fullstate':
            # One MPO carries the hierarchy, the Hamiltonian and the LTC
            # hierarchy correction together.
            self._list_cores_op = builder.build_fullstate_mpo(
                list_z_hat,
                list_expect_L2,
                norm_corr,
                C2_lt_corr_hier=self._C2_LT_corr_hier,
            )
        else:
            raise UnsupportedRequest(self.wavefunction.method, '_build_mpo_parts')
