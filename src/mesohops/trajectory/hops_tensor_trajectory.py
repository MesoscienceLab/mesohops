from __future__ import annotations

import time as timer
import warnings
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sparse

from mesohops.integrator.tensor_integrator import (
    runge_kutta_step_tensor,
    runge_kutta_variables,
    single_point_variables,
    tdvp_step_tensor,
)
from mesohops.storage.storage_functions import (
    save_max_tensor_complexity,
    save_phi_traj_tensor,
    save_phi_norm_tensor,
)
from mesohops.tensor.hops_tensor_wavefunction import HopsTensorWavefunction
from mesohops.tensor.hops_tensor_basis import HopsTensorBasis
from mesohops.tensor.hops_tensor_eom import HopsTensorEOM
from mesohops.tensor.mpo_constructors import (
    build_statenumber_dipole_lower_plus_ident_mpo,
    build_statenumber_dipole_mpo,
    build_statenumber_dipole_raise_plus_ground_ident_mpo,
)
from mesohops.tensor.tensor_eom_functions import (
    apply_system_operator,
    tensor_matvec_prod,
)
from mesohops.trajectory.hops_trajectory import HopsTrajectory
from mesohops.util.dynamic_dict import Dict_wDefaults
from mesohops.util.exceptions import LockedException, TrajectoryError, UnsupportedRequest
from mesohops.util.physical_constants import precision

__title__ = 'Tensor HOPS Trajectory'
__author__ = 'B. Z. Citty'
__maintainer__ = 'B. Z. Citty'

# Default values for the tensor_param dictionary accepted by
# HopsTensorTrajectory: MPS representation, SVD truncation thresholds, MPO
# builder selection and IVP solver settings. FLAG_MPO_OPTIMIZE is read only
# on the number path. IVP_MAX_STEP of None leaves the step size unlimited.
TENSOR_DICT_DEFAULT = {
    'METHOD': 'fullstate',
    'MPS_EPSILON': 1e-2,
    'MPO_EPSILON': 0.0,
    'FLAG_MPO_OPTIMIZE': True,
    'BOND_DIM_MAX': 10,
    'TDVP_UPDATE_TYPE': 'krylov',
    'IVP_METHOD': 'BDF',
    'IVP_RTOL': 1e-7,
    'IVP_ATOL': 1e-9,
    'IVP_MAX_STEP': None,
}

# Allowed types for each tensor_param key, used by DynamicDict for validation.
TENSOR_DICT_TYPES = {
    'METHOD': [str],
    'MPS_EPSILON': [float],
    'MPO_EPSILON': [float],
    'FLAG_MPO_OPTIMIZE': [bool],
    'BOND_DIM_MAX': [int],
    'TDVP_UPDATE_TYPE': [str],
    'IVP_METHOD': [str],
    'IVP_RTOL': [float],
    'IVP_ATOL': [float],
    'IVP_MAX_STEP': [float, type(None)],
}


VALID_INTEGRATORS = {'RUNGE_KUTTA', 'TDVP1', 'TDVP2'}


class HopsTensorTrajectory(HopsTrajectory):
    """
    Subclass of HopsTrajectory for tensor-network (TadHOPS) calculations.
    The wavefunction is represented as a matrix product state (MPS) via
    HopsTensorWavefunction, enabling efficient simulation of large systems.

    Initialization, propagation, and inchworm integration are all overridden
    to operate on the tensor representation. The parent class provides shared
    infrastructure: noise, basis management, storage, and utility methods.
    """

    # NOTE: dsystem_dt is not set on tensor trajectories. The parent's dsystem_dt
    # is a derivative closure rebuilt when the basis changes. The tensor EOM
    # (self.tensor_basis.eom) is a two-phase object: build_generator constructs
    # an MPO each step, then derivative applies it. The MPO rebuilds on the fly,
    # so the EOM does not need to be rebuilt or passed through update_basis.
    #
    # NOTE: the parent's _phi slot is intentionally unused in the tensor path.
    # The tensor wavefunction lives on self.wavefunction, and the phi property
    # (below) returns self.wavefunction.list_cores_phi. Any inherited code path
    # that accesses self._phi directly would raise AttributeError.

    __slots__ = (
        # --- Tensor wavefunctions ---
        'tensor_param',  # Tensor configuration dict
        'wavefunction',  # Tensor wavefunction
        'tensor_basis',  # HopsTensorBasis (system/mode/noise_memory/eom)
        # --- Tensor integrator ---
        'is_tdvp',  # True when using TDVP integrator
        # step and integration_var are inherited from HopsTrajectory
        # and overridden during initialize to use tensor integrators
        '_tdvp_method',  # TDVP variant string ('1tdvp' or '2tdvp')
        # --- Output ---
        # psi_traj stored through self.storage (same as parent)
    )

    def __init__(
        self,
        system_param=None,
        eom_param=None,
        noise_param=None,
        noise2_param=None,
        hierarchy_param=None,
        storage_param=None,
        integration_param=None,
        tensor_param=None,
    ) -> None:
        """
        Initializes the tensor trajectory. Calls the parent constructor for all
        shared infrastructure, then sets up tensor-specific components.

        Parameters
        ----------
        1. system_param: dict
                          Dictionary of user-defined system parameters.
                          [see hops_system.py]

        2. eom_param: dict
                       Dictionary of user-defined equation-of-motion parameters.
                       [see hops_eom.py]

        3. noise_param: dict
                         Dictionary of user-defined noise parameters.
                         [see hops_noise.py]

        4. noise2_param: dict
                          Dictionary of user-defined secondary noise parameters.
                          [see hops_noise.py]

        5. hierarchy_param: dict
                             Dictionary of user-defined hierarchy parameters.
                             [see hops_hierarchy.py]

        6. storage_param: dict
                           Dictionary of user-defined storage parameters.
                           [see hops_storage.py]

        7. integration_param: dict
                               Dictionary of user-defined integration parameters.
                               [see integrator.py]

        8. tensor_param: dict
                          Dictionary of tensor configuration parameters.
            a. 'METHOD': str
                          Tensor representation type.
                          [options: 'fullstate',
                                    'number']
            b. 'MPS_EPSILON': float
                               SVD truncation threshold for MPS (wavefunction)
                               compression.
            c. 'MPO_EPSILON': float
                               SVD truncation threshold on MPO construction,
                               live in two places. Where MPO parts are
                               summed: the hierarchy plus Hamiltonian MPO on
                               the FLAG_MPO_OPTIMIZE=False path, and the
                               low-temperature correction folds. And on the
                               coupling-block factorization behind the
                               general generator, where raising it drops
                               weak couplings and narrows the MPO.
                               Default 0.0 (no compression).
            d. 'FLAG_MPO_OPTIMIZE': bool
                                     True (default) builds the number-method
                                     generator as one MPO, each bond carrying
                                     as many channels as the rank of the
                                     coupling block that crosses it. False
                                     takes the reference path: a hierarchy MPO
                                     plus a separate Hamiltonian MPO, added
                                     and compressed.
                                     Read only when METHOD is 'number'.
            e. 'BOND_DIM_MAX': int
                                Maximum MPS bond dimension.
            f. 'TDVP_UPDATE_TYPE': str
                                    TDVP update scheme. Default: 'krylov'.
            g. 'IVP_METHOD': str
                              IVP solver method. Default: 'BDF'.
            h. 'IVP_RTOL': float
                            IVP relative tolerance. Default: 1e-7.
            i. 'IVP_ATOL': float
                            IVP absolute tolerance. Default: 1e-9.
            j. 'IVP_MAX_STEP': float | None
                                IVP maximum step size. Default: None.

        Returns
        -------
        None
        """
        # TODO: enable sparse Hamiltonian use (not necessary for this paper)
        # Ensure dense Hamiltonian before parent constructor builds HopsSystem.
        # Copy the dict to avoid mutating the caller's input.
        if system_param is not None and 'HAMILTONIAN' in system_param:
            if sparse.issparse(system_param['HAMILTONIAN']):
                system_param = dict(system_param)
                system_param['HAMILTONIAN'] = system_param['HAMILTONIAN'].toarray()

        # Register the tensor-specific max_tensor_complexity key through the
        # base HopsStorage constructor rather than mutating storage_dic /
        # dic_save / data post-hoc. Passing a callable as the value lets the
        # adaptive.setter registration loop wire it into dic_save and data
        # in one pass.
        if storage_param is None:
            storage_param = {}
        else:
            storage_param = dict(storage_param)
        storage_param.setdefault(
            'max_tensor_complexity', save_max_tensor_complexity,
        )

        # Initialize all shared infrastructure via the parent constructor
        super().__init__(
            system_param=system_param,
            eom_param=eom_param,
            noise_param=noise_param,
            noise2_param=noise2_param,
            hierarchy_param=hierarchy_param,
            storage_param=storage_param,
            integration_param=integration_param,
        )

        # Read k_max from the authoritative source (parent already resolved
        # defaults via HIERARCHY_DICT_DEFAULT).
        k_max = self.basis.hierarchy.param['MAXHIER']

        # Tensor configuration: fill missing keys from defaults and
        # validate types, matching the parent's integration_param pattern
        if tensor_param is None:
            tensor_param = {}
        self.tensor_param = Dict_wDefaults._initialize_dictionary(
            tensor_param,
            TENSOR_DICT_DEFAULT,
            TENSOR_DICT_TYPES,
            'tensor_param in HopsTensorTrajectory',
        )

        # Construct HopsTensorBasis with shared references from self.basis.
        # IMPORTANT: tensor_basis.system, tensor_basis.mode, and
        # tensor_basis.noise_memory are the SAME objects as self.basis.system,
        # self.basis.mode, and self.basis.noise_memory — not copies.
        # Mutations through either path (self.basis.* or self.tensor_basis.*)
        # affect the same underlying objects. Convention: inherited
        # infrastructure accesses through self.basis, tensor-specific logic
        # accesses through self.tensor_basis.
        self.tensor_basis = HopsTensorBasis(
            self.basis.system, self.basis.mode, self.basis.noise_memory,
        )
        self.wavefunction = HopsTensorWavefunction(
            k_max, self.tensor_param, self.integration_param,
            self.basis.eom.param,
        )

        # Register tensor-aware storage functions that understand MPS data.
        # phi_traj: store full MPS cores instead of a flat vector.
        # phi_norm: compute hierarchy norm via MPS double-layer contraction.
        if 'phi_traj' in self.storage.dic_save:
            self.storage.dic_save['phi_traj'] = save_phi_traj_tensor
        if 'phi_norm' in self.storage.dic_save:
            self.storage.dic_save['phi_norm'] = save_phi_norm_tensor
        # list_aux_norm: vector-HOPS diagnostic (per-auxiliary norms via
        # reshape). Not meaningful for tensor HOPS where the hierarchy is
        # compressed into MPS bond dimensions.
        if 'list_aux_norm' in self.storage.dic_save:
            warnings.warn(
                'list_aux_norm is not supported for tensor trajectories '
                '(hierarchy is compressed into MPS bond dimensions, not '
                'enumerated as discrete auxiliary vectors). Disabling.',
                stacklevel=2,
            )
            del self.storage.dic_save['list_aux_norm']
            self.storage.data.pop('list_aux_norm', None)

        # max_tensor_complexity (per-timestep peak MPS complexity) was
        # registered via storage_param above, before super().__init__, so
        # the base HopsStorage adaptive.setter already populated
        # storage_dic, dic_save, and data for this key in one pass.

    def make_adaptive(self,
                      delta_a: float = 1e-4,
                      delta_s: float = 1e-4,
                      update_step: int = 1,
                      f_discard: float = 0.01,
                      list_permanent_sites: list[int] | None = None,
                      adaptive_noise: bool = True) -> None:
        """
        Configures this trajectory for adaptive HOPS. Overrides the parent
        to reject `list_permanent_sites`, which `HopsTensorBasis` does not
        honor — passing it to the parent would store it on
        `system.param["list_permanent_sites"]` but the tensor adaptive
        basis never reads that key, so the requested sites would silently
        not be preserved. Failing here gives the caller a clear signal at
        configuration time rather than wrong physics at run time.
        See `HopsTrajectory.make_adaptive` for the parameter semantics.
        """
        if list_permanent_sites is not None:
            raise NotImplementedError(
                'list_permanent_sites is not supported for tensor '
                'trajectories. HopsTensorBasis does not read '
                'system.param["list_permanent_sites"], so passing this '
                'argument would be silently ignored at runtime.'
            )
        super().make_adaptive(
            delta_a=delta_a,
            delta_s=delta_s,
            update_step=update_step,
            f_discard=f_discard,
            list_permanent_sites=list_permanent_sites,
            adaptive_noise=adaptive_noise,
        )

    def _setup_integrator(self) -> None:
        """
        Configures tensor-specific integration step function and variable gatherer.

        Overrides the parent to support TDVP integrators.
        """
        if self.integrator == 'RUNGE_KUTTA':
            self.step = runge_kutta_step_tensor
            self.integration_var = runge_kutta_variables
            self.integrator_step = 0.5
            self.is_tdvp = False
            self._tdvp_method = None
        elif self.integrator == 'TDVP1':
            self.step = tdvp_step_tensor
            self.integration_var = single_point_variables
            self.integrator_step = 1.0
            self.is_tdvp = True
            self._tdvp_method = '1tdvp'
        elif self.integrator == 'TDVP2':
            self.step = tdvp_step_tensor
            self.integration_var = single_point_variables
            self.integrator_step = 1.0
            self.is_tdvp = True
            self._tdvp_method = '2tdvp'
        else:
            raise UnsupportedRequest(
                f'Integrator {self.integrator!r}, expected one of '
                f'{sorted(VALID_INTEGRATORS)}',
                type(self).__name__,
            )

    def initialize(
        self,
        psi_0: Sequence[complex] | np.ndarray,
        timer_checkpoint: float | None = None,
    ) -> None:
        """
        Initializes the tensor trajectory. Sets up both the vector basis (shared
        infrastructure used by inchworm) and the tensor wavefunction representation.

        Parameters
        ----------
        1. psi_0: np.ndarray(complex)
                   Wave function at initial time.

        2. timer_checkpoint: float | None
                              Wall-clock time prior to initialization [units: s].
                              If None, uses current wall-clock time.

        Returns
        -------
        None
        """
        if timer_checkpoint is None:
            timer_checkpoint = timer.time()

        psi_0 = np.array(psi_0, dtype=np.complex128)

        if not self.__initialized__:
            # --- Step 1: Initialize basis and EOM ---
            # The parent calls self.basis.initialize(psi_0) which initializes
            # hierarchy, system, mode, noise_memory, and builds the vector EOM
            # derivative closure in one call. Tensor HOPS cherry-picks:
            #
            # - Hierarchy init is skipped because hierarchy depth is encoded in
            #   MPS core dimensions (k_max + 1 per mode), not through explicit
            #   auxiliary vector enumeration.
            # - Vector EOM is skipped because tensor HOPS uses HopsTensorEOM
            #   (MPO-based) instead of a derivative closure.
            # - The mode union with hierarchy modes is a no-op because the
            #   hierarchy object never populates its mode list in the tensor
            #   path.
            self.basis.system.initialize(self.basis.adaptive_s, psi_0)
            self.basis.mode.list_modeidx_abs = sorted(
                self.basis.system.list_statemodeidx_abs
            )
            self.basis.noise_memory.initialize()
            self.tensor_basis.initialize(self.basis.eom.param.get('DELTA_S', 0))
            self.wavefunction.initialize(
                psi_0,
                self.tensor_basis.system,
            )
            self.tensor_basis.eom = HopsTensorEOM(
                self.wavefunction,
                self.tensor_basis.system,
                self.tensor_basis.mode,
                self.tensor_basis.noise_memory,
                self.tensor_basis.adaptive,
                self.basis.eom.param,
            )

            # --- Step 2: z_mem ---
            self.z_mem = np.zeros(
                len(self.basis.noise_memory.list_zmemmodeidx_abs),
                dtype=np.complex128,
            )

            # --- Step 3: storage.n_dim ---
            self.storage.n_dim = self.basis.system.param['NSTATES']

            # --- Step 4: Adaptive basis setup ---
            if self.basis.adaptive:
                if self.static_basis is not None:
                    raise NotImplementedError(
                        'static_basis is not yet supported for tensor '
                        'trajectories.'
                    )
                # storage.adaptive is already set by make_adaptive();
                z_step = self._prepare_zstep(self.z_mem)
                list_states_old, list_states_new = (
                    self.tensor_basis.define_basis(self.wavefunction, z_step)
                )
                self.wavefunction, self.z_mem = self.tensor_basis.update_basis(
                    self.wavefunction, self.z_mem,
                    list_states_old, list_states_new,
                )

            # --- Step 5: Set time ---
            self.t = 0

            # --- Step 6: Store initial state ---
            # phi_new: the parent passes the full hierarchy vector phi.
            # We pass psi because save_psi_traj (always active) slices
            # phi_new[:len(state_list)], which is a no-op on psi.
            # save_phi_traj_tensor and save_phi_norm_tensor ignore
            # phi_new and read wavefunction directly.
            self.storage.store_step(
                phi_new=self.wavefunction.psi,
                wavefunction=self.wavefunction,
                state_list=list(self.tensor_basis.system.state_list),
                t_new=0,
                aux_list=self.auxiliary_list,
                z_mem_new=self.z_mem,
                list_zmemmodeidx_abs=(
                    self.basis.noise_memory.list_zmemmodeidx_abs
                ),
                max_tensor_complexity=0,
            )

            # --- Step 7: Metadata and lock ---
            self.storage.metadata['INITIALIZATION_TIME'] = (
                timer.time() - timer_checkpoint
            )
            self.__initialized__ = True
        else:
            raise LockedException('initialize', 'HopsTensorTrajectory')

    def propagate(
        self,
        t_advance: float,
        tau: float,
        timer_checkpoint: float | None = None,
    ) -> None:
        """
        Propagates the tensor wavefunction forward in time. At each step the MPS
        is advanced via the configured integrator.

        Parameters
        ----------
        1. t_advance: float
                       How far out in time the calculation will run [units: fs].

        2. tau: float
                 Time step [units: fs].

        3. timer_checkpoint: float | None
                              System time prior to propagation [units: s].
                              If None, uses current system time.

        Returns
        -------
        None
        """
        if timer_checkpoint is None:
            timer_checkpoint = timer.time()

        # Construct the time axis
        # NOTE: t_axis is only defined inside this branch. If TAU is None
        # and INTERPOLATE is True, t_axis will be undefined and line
        # `np.max(t_axis)` below will raise UnboundLocalError. Same issue
        # exists in the parent HopsTrajectory.propagate. Awaiting team
        # review before fixing.
        t0 = self.t
        if (self.noise1.param['TAU'] is not None) or not (
            self.noise1.param['INTERPOLATE']
        ):
            if self._check_tau_step(tau, precision):
                n_steps = int(np.ceil(t_advance / tau))
                t_axis = t0 + np.arange(1, 1 + n_steps) * tau
                print('Integration from ', t0, ' to ', np.max(t_axis))
            else:
                raise TrajectoryError(
                    'Timesteps ('
                    + str(tau * self.integrator_step)
                    + ") that do not match noise.param['TAU'] ("
                    + str(self.noise1.param['TAU'])
                    + ')'
                )

        if np.max(t_axis) > self.noise1.param['TLEN']:
            raise TrajectoryError(
                "Trajectory times longer than noise.param['TLEN'] ("
                + str(self.noise1.param['TLEN'])
                + ')'
            )

        # Tracks the system timescale so the timestep warning fires only once
        tau_sys = None

        store_step_timing = self.integration_param['STORE_STEP_TIMING']

        for idx_t, t in enumerate(t_axis):
            if store_step_timing:
                t_step_start = timer.time()
            # Check that timestep is resolved by system timescale
            if tau > self.basis.system.system_timescale and (
                tau_sys is None or tau_sys > self.basis.system.system_timescale
            ):
                tau_sys = self.basis.system.system_timescale

            # Tensor step
            dict_var = self.integration_var(
                self.z_mem,
                self.t,
                self.noise1,
                self.noise2,
                tau,
                self.basis.mode.list_l2idx_abs,
                self.effective_noise_integration,
            )
            # Parent does: phi, z_mem = self.step(self.dsystem_dt, **var_list).
            # Tensor _step returns just z_mem because the wavefunction is
            # mutated in place by the integrator. Peak uncompressed-MPS
            # size during the step is published on eom.max_complexity_step
            # (0 for TDVP — see step function docstrings) and read here.
            z_mem = self._step(dict_var)
            max_complexity_step = self.tensor_basis.eom.max_complexity_step
            # Parent does: phi = self.normalize(phi). Tensor version mutates
            # wavefunction class instance mutated in place by the integrator.
            self.normalize()

            # (C) Adaptive basis update
            if self.basis.adaptive:
                if self.use_early_integrator:
                    print(f'Early Integration: Using {self.early_integrator}')
                    # The parent checks both INCH_WORM and STATIC hierarchy
                    # options. Tensor HOPS has no hierarchy adaptivity, so
                    # only the INCH_WORM path applies.
                    if self.early_integrator == 'INCH_WORM':
                        z_step = self._prepare_zstep(z_mem)
                        # TODO: make output match parent class (state
                        # update abstraction)
                        list_states_old, list_states_new = (
                            self.tensor_basis.define_basis(self.wavefunction, z_step)
                        )
                        # Deep copy: statenumber has nested lists (groups
                        # of arrays), so a flat list comprehension would
                        # only shallow-copy the outer list.
                        if self.wavefunction.method == 'number':
                            list_cores_checkpoint = [
                                [arr.copy() for arr in g]
                                for g in self.wavefunction.list_cores_phi
                            ]
                        else:
                            list_cores_checkpoint = [
                                c.copy() for c in self.wavefunction.list_cores_phi
                            ]
                        # Iterate until basis converges (define_basis proposes
                        # no further changes) or the inchworm cap is reached.
                        step_num = 0
                        while list_states_old != [] or list_states_new != []:
                            (
                                z_mem,
                                max_complexity_inch,
                                list_states_old,
                                list_states_new,
                                list_cores_checkpoint,
                            ) = self.inchworm_integrate(
                                tau,
                                list_cores_checkpoint,
                                list_states_old,
                                list_states_new,
                            )
                            # Accumulate the per-timestep max across
                            # inchworm iterations.
                            if max_complexity_inch > max_complexity_step:
                                max_complexity_step = max_complexity_inch
                            step_num += 1
                            if step_num >= self.inchworm_cap:
                                break
                        # Parent returns (phi, z_mem, self.dsystem_dt). Tensor omits
                        # dsystem_dt — tensor EOM rebuilds its MPO on the fly.
                        # TODO: adaptive path needs dsystem_dt in form of
                        # calc_deriv_cores being passed in
                        self.wavefunction, z_mem = self.tensor_basis.update_basis(
                            self.wavefunction, z_mem, list_states_old, list_states_new
                        )
                    else:
                        raise UnsupportedRequest(
                            self.early_integrator,
                            'early time integrator clause of the tensor propagate',
                        )
                    self._early_step_counter += 1

                # Standard adaptive integration: check every update_step
                # steps whether states should be added or removed
                elif (idx_t + 1) % self.update_step == 0:
                    z_step = self._prepare_zstep(z_mem)
                    list_states_old, list_states_new = (
                        self.tensor_basis.define_basis(self.wavefunction, z_step)
                    )
                    # Parent returns (phi, z_mem, self.dsystem_dt). Tensor omits
                    # dsystem_dt — tensor EOM rebuilds its MPO on the fly.
                    # TODO: adaptive path needs dsystem_dt in form of
                    # calc_deriv_cores being passed in
                    self.wavefunction, z_mem = self.tensor_basis.update_basis(
                        self.wavefunction, z_mem, list_states_old, list_states_new
                    )

            # Parent also does: self.phi = phi. Tensor omits this because
            # wavefunction is already on self, mutated in place.
            self.z_mem = z_mem
            self.t = t

            if self.storage.check_storage_time(t):
                # phi_new receives psi, not the full hierarchy vector;
                # see the comment in initialize() step 6.
                self.storage.store_step(
                    phi_new=self.wavefunction.psi,
                    wavefunction=self.wavefunction,
                    state_list=list(self.tensor_basis.system.state_list),
                    t_new=t,
                    aux_list=self.auxiliary_list,
                    z_mem_new=self.z_mem,
                    list_zmemmodeidx_abs=self.basis.noise_memory.list_zmemmodeidx_abs,
                    max_tensor_complexity=max_complexity_step,
                )

            if store_step_timing:
                self.storage.metadata['LIST_PROPAGATION_TIME'].append(
                    (t, timer.time() - t_step_start)
                )

        # Store propagation time
        if not store_step_timing:
            self.storage.metadata['LIST_PROPAGATION_TIME'].append(
                timer.time() - timer_checkpoint
            )

        if tau_sys is not None:
            warnings.warn(
                f'At some point during propagation, the time step ({tau} fs)'
                f' was larger than the estimated timescale associated with '
                f'the system Hamiltonian ({tau_sys} fs). A smaller time step '
                f'may be necessary to correctly resolve dynamics.'
            )

    def _step(self, dict_var: dict) -> np.ndarray:
        """
        Dispatches a single tensor integration step (RK4 or TDVP).

        Mutates self.tensor_basis.eom.wavefunction in place.

        Parameters
        ----------
        1. dict_var : dict
                      Variables from integration_var.

        Returns
        -------
        1. z_mem : np.ndarray(complex)
                   Updated noise memory drift terms [units: cm^-1].

        Side effects
        ------------
        The underlying step function sets
        `self.tensor_basis.eom.max_complexity_step` to the peak
        uncompressed-MPS complexity observed (RK4 tracks across
        sub-stages; TDVP sets 0). propagate() reads that attribute
        for storage rather than threading the scalar through a tuple.
        """
        if self.is_tdvp:
            # TDVP requires additional solver configuration (Krylov
            # subspace size, solver type, IVP tolerances) beyond the
            # base noise/memory variables.
            tp = self.tensor_param
            return self.step(
                self.tensor_basis.eom,
                dict_var['z_mem'],
                dict_var['z_rnd'],
                dict_var['z_rnd2'],
                dict_var['tau'],
                method=self._tdvp_method,
                # TODO: give krylov_conv_tol its own tensor_param key instead
                # of coupling it to mps_epsilon (they control different things).
                krylov_conv_tol=tp['MPS_EPSILON'] / 10,
                update_type=tp['TDVP_UPDATE_TYPE'],
                ivp_method=tp['IVP_METHOD'],
                ivp_rtol=tp['IVP_RTOL'],
                ivp_atol=tp['IVP_ATOL'],
                ivp_max_step=tp['IVP_MAX_STEP'],
            )
        # RK4 path: only needs the EOM and noise/memory variables
        return self.step(
            self.tensor_basis.eom,
            dict_var['z_mem'],
            dict_var['z_rnd'],
            dict_var['z_rnd2'],
            dict_var['tau'],
        )

    def _operator(self, op: np.ndarray | sparse.spmatrix) -> None:
        """
        Applies an operator to the tensor wavefunction. Mirrors the parent
        HopsTrajectory._operator: expands the adaptive basis to include all
        states coupled by the operator, trims to the active basis, applies
        the operator, then cleans up the basis afterward.

        Parameters
        ----------
        1. op: np.ndarray | sparse.spmatrix
               The operator as a full system-space matrix,
               shape (n_state_full, n_state_full).

        Returns
        -------
        None
        """
        if sparse.issparse(op):
            op = op.tocsr()

        # Validate operator dimensions against full system size. Without
        # this guard, too-small operators leak an IndexError from the
        # np.ix_ trim below, and too-large operators are silently sliced
        # to the first n_state_full x n_state_full block — neither is
        # what a caller who passed a mismatched operator actually wants.
        n_state_full = self.basis.system.param['NSTATES']
        if op.shape != (n_state_full, n_state_full):
            raise ValueError(
                f'op must have shape ({n_state_full}, {n_state_full}) '
                f'(full system size); got {op.shape}'
            )

        # TODO: adaptive basis expansion/cleanup around operator application
        # is not fully tested and may not correctly handle all edge cases
        # (e.g., states that become depopulated after the operator).
        # Expand adaptive basis if operator couples to new states
        if self.tensor_basis.adaptive:
            list_states_operator = np.unique(
                np.nonzero(op[:, self.tensor_basis.system.state_list])[0]
            )
            list_states_new = sorted(
                set(list_states_operator) - set(self.tensor_basis.system.state_list)
            )
            self.wavefunction, self.z_mem = self.tensor_basis.update_basis(
                self.wavefunction,
                self.z_mem,
                [],
                list_states_new,
            )

        # Trim to active basis and apply
        state_list = self.tensor_basis.system.state_list
        if sparse.issparse(op):
            O2_trimmed = op[np.ix_(state_list, state_list)].toarray()
        else:
            O2_trimmed = np.asarray(op)[np.ix_(state_list, state_list)]

        self.wavefunction.list_cores_phi = apply_system_operator(
            self.wavefunction.list_cores_phi,
            O2_trimmed,
            self.wavefunction.method,
            self.wavefunction.k_max,
            self.wavefunction.M1_modes_per_state,
            self.wavefunction.mps_epsilon,
            self.wavefunction.bond_dim_max,
        )

        # Post-operator basis cleanup
        if self.tensor_basis.adaptive:
            z_step = self._prepare_zstep(self.z_mem)
            list_states_old, list_states_new = (
                self.tensor_basis.define_basis(self.wavefunction, z_step)
            )
            self.wavefunction, self.z_mem = self.tensor_basis.update_basis(
                self.wavefunction,
                self.z_mem,
                list_states_old,
                list_states_new,
            )
            self.reset_early_time_integrator()

    def apply_dipole_raise(self, list_mu: np.ndarray) -> None:
        """
        Applies the bond-dim-2 dipole-raise MPO
            sum_k list_mu[k] * a_k^dagger
        to the wavefunction in the vacuum convention.  Used by the
        fluorescence path, which matches gs_core fluorescence's raise
        (no |g> preservation).  For the absorption path, use
        apply_dipole_raise_plus_ground_ident instead.

        Parameters
        ----------
        1. list_mu: np.ndarray(complex)
                    Per-site dipole amplitudes, shape (n_state,).
                    Entries set to 0 mark sites excluded from the
                    dipole's site selection.

        Returns
        -------
        None
        """
        n_state = len(self.tensor_basis.system.state_list)
        list_cores_mpo = build_statenumber_dipole_mpo(
            list_mu, n_state, self.wavefunction.k_max,
            self.wavefunction.M1_modes_per_state, 'raise',
        )
        new_cores, _ = tensor_matvec_prod(
            self.wavefunction.list_cores_phi, list_cores_mpo,
            self.wavefunction.mps_epsilon, self.wavefunction.bond_dim_max,
        )
        self.wavefunction.list_cores_phi = new_cores

    def apply_dipole_raise_plus_ground_ident(
        self, list_mu: np.ndarray,
    ) -> None:
        """
        Applies the bond-dim-3 MPO
            sum_k list_mu[k] * a_k^dagger  +  I_g
        to the wavefunction in the vacuum convention.  The +I_g term
        (projector onto the all-zeros configuration) preserves the GS
        amplitude after the raise, matching the gs_core absorption
        raise's "(0,0)=1 keep |g>" behavior.  Used by the absorption
        path; the fluorescence raise (no |g> preservation) uses
        apply_dipole_raise instead.

        Parameters
        ----------
        1. list_mu: np.ndarray(complex)
                    Per-site dipole amplitudes, shape (n_state,).
                    Entries set to 0 mark sites excluded from the
                    dipole's site selection.

        Returns
        -------
        None
        """
        n_state = len(self.tensor_basis.system.state_list)
        list_cores_mpo = build_statenumber_dipole_raise_plus_ground_ident_mpo(
            list_mu, n_state, self.wavefunction.k_max,
            self.wavefunction.M1_modes_per_state,
        )
        new_cores, _ = tensor_matvec_prod(
            self.wavefunction.list_cores_phi, list_cores_mpo,
            self.wavefunction.mps_epsilon, self.wavefunction.bond_dim_max,
        )
        self.wavefunction.list_cores_phi = new_cores

    def apply_dipole_lower_plus_ident(self, list_mu: np.ndarray) -> None:
        """
        Applies the bond-dim-4 MPO
            sum_k list_mu[k] * a_k  +  I_excited
        to the wavefunction in the vacuum convention. The +I_excited
        term preserves the single-excitation manifold content; the
        sum_k a_k term creates the GS amplitude on the all-zeros
        configuration of the MPS.  Used by the fluorescence path as the
        de-excitation step: applied with E_sig after the t2 waiting
        period to inject the GS amplitude that forms the G/E coherence
        measured during the detection phase.

        Parameters
        ----------
        1. list_mu: np.ndarray(complex)
                    Per-site dipole amplitudes, shape (n_state,).
                    Entries set to 0 mark sites excluded from the
                    dipole's site selection.

        Returns
        -------
        None
        """
        n_state = len(self.tensor_basis.system.state_list)
        list_cores_mpo = build_statenumber_dipole_lower_plus_ident_mpo(
            list_mu, n_state, self.wavefunction.k_max,
            self.wavefunction.M1_modes_per_state,
        )
        new_cores, _ = tensor_matvec_prod(
            self.wavefunction.list_cores_phi, list_cores_mpo,
            self.wavefunction.mps_epsilon, self.wavefunction.bond_dim_max,
        )
        self.wavefunction.list_cores_phi = new_cores

    def normalize(self) -> None:
        """
        Normalizes the tensor wavefunction in place if the EOM requires it.

        The parent HopsTrajectory.normalize takes phi as an argument and returns
        the normalized phi — a value-passing pattern. The tensor version operates
        on self.wavefunction in place because the wavefunction is a class instance
        (HopsTensorWavefunction) rather than a bare numpy array. This difference
        will shrink when the parent gets a wavefunction class.

        The normalization policy (whether to normalize based on EOM type) lives
        here in the trajectory, not in the wavefunction class — matching the
        parent where HopsTrajectory.normalize checks self.basis.eom.normalized.
        """
        if self.basis.eom.normalized:
            self.wavefunction.normalize()

    def inchworm_integrate(
        self,
        tau: float,
        list_cores_checkpoint: list,
        list_states_old=None,
        list_states_new=None,
    ) -> tuple:
        """
        Performs one inchworm iteration for the tensor trajectory.

        The tensor basis is expanded by the proposed (list_states_old,
        list_states_new) update. wavefunction is restored to the checkpoint
        state in the new basis, then a full integration step is taken. The
        resulting wavefunction is used to propose the next basis update.

        Note: the signature and return tuple differ from the parent's
        inchworm_integrate because tensor HOPS has no auxiliary vectors
        (the hierarchy is compressed into MPS bond dimensions) and uses
        MPS cores instead of a flat phi vector.

        Parameters
        ----------
        1. tau: float
                 Time step [units: fs].

        2. list_cores_checkpoint: list(np.ndarray)
                              Saved list_cores_phi representing the wavefunction at
                              the start of the current timestep, expressed in
                              the basis prior to this iteration's update.

        3. list_states_old: list | None
                                      Tensor basis state indices to remove in
                                      this iteration's basis update.

        4. list_states_new: list | None
                               New tensor states to add in this iteration's
                               basis update.

        Returns
        -------
        1. z_mem: np.array(complex)
                   Noise memory drift after the tensor step [units: cm^-1].

        2. max_complexity: int
                            Peak uncompressed-MPS complexity across this
                            iteration's integrator call (0 for TDVP).
                            Read from eom.max_complexity_step which the
                            step function publishes.

        3. list_states_old: list
                               Tensor state indices proposed for removal in the
                               next basis update.

        4. list_states_new: list
                        New tensor states proposed for addition in the next
                        basis update.

        5. list_cores_checkpoint: list(np.ndarray)
                              Updated checkpoint cores in the new (expanded)
                              basis, for use in the next inchworm iteration.
        """
        # Restore wavefunction to the checkpoint, then expand the basis.
        # After update_basis, wavefunction.list_cores_phi holds the checkpoint state
        # expressed in the new (expanded) basis.
        self.wavefunction.restore_phi(list_cores_checkpoint)
        # TODO: for adaptive path, z_mem changes, so this logic should
        # update to give the updated z_mem
        self.wavefunction, _ = self.tensor_basis.update_basis(
            self.wavefunction, self.z_mem, list_states_old, list_states_new
        )
        # Deep copy: checkpoint must be independent of subsequent
        # integrator mutations to list_cores_phi. Statenumber has nested
        # lists (groups of arrays) requiring inner-level copy.
        if self.wavefunction.method == 'number':
            list_cores_checkpoint = [
                [arr.copy() for arr in g]
                for g in self.wavefunction.list_cores_phi
            ]
        else:
            list_cores_checkpoint = [
                c.copy() for c in self.wavefunction.list_cores_phi
            ]

        dict_var = self.integration_var(
            self.z_mem,
            self.t,
            self.noise1,
            self.noise2,
            tau,
            self.basis.mode.list_l2idx_abs,
            self.effective_noise_integration,
        )
        z_mem = self._step(dict_var)
        max_complexity = self.tensor_basis.eom.max_complexity_step
        self.normalize()

        # Define next basis proposal from the stepped state
        z_step = self._prepare_zstep(z_mem)
        list_states_old, list_states_new = self.tensor_basis.define_basis(
            self.wavefunction, z_step,
        )

        return (
            z_mem,
            max_complexity,
            list_states_old,
            list_states_new,
            list_cores_checkpoint,
        )

    # TODO: when implemented, match parent signatures and use cls(...)
    # for subclassability:
    #   save_checkpoint(filepath: str | os.PathLike, ...)
    #   load_checkpoint(
    #       filename: str | os.PathLike,
    #       add_seed1: int | str | os.PathLike | np.ndarray | None,
    #       add_seed2: int | str | os.PathLike | np.ndarray | None,
    #       add_system_param: str | os.PathLike | None,
    #   ) -> HopsTensorTrajectory  (via cls(...))
    def save_checkpoint(
        self,
        filepath: str,
        compress: bool = True,
        drop_seed: bool = False,
    ) -> None:
        """
        Not yet supported for tensor trajectories.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            'Tensor trajectory checkpointing is not yet supported.'
        )

    @classmethod
    def load_checkpoint(
        cls,
        filename: str,
        add_seed1: int | None = None,
        add_seed2: int | None = None,
        add_system_param: str | None = None,
    ) -> HopsTensorTrajectory:
        """
        Not yet supported for tensor trajectories.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            'Tensor trajectory checkpointing is not yet supported.'
        )

    @property
    def psi(self) -> np.ndarray:
        """Current physical wavefunction (compact, active states only).

        Matches the parent HopsTrajectory.psi which returns
        phi[:n_state]. Storage handles reconstruction to full-size
        arrays via state_list when the user accesses storage['psi_traj'].
        """
        return self.wavefunction.psi

    @property
    def phi(self) -> list:
        """Current hierarchy wavefunction as MPS cores."""
        return self.wavefunction.list_cores_phi

    @phi.setter
    def phi(self, value: list) -> None:
        self.wavefunction.list_cores_phi = value
