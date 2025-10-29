from __future__ import annotations

import copy
import os
import time as timer
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import scipy as sp
import scipy.sparse as sparse

from mesohops.basis.hops_aux import AuxiliaryVector as AuxVec
from mesohops.basis.hops_basis import HopsBasis
from mesohops.basis.hops_hierarchy import HopsHierarchy, HIERARCHY_DICT_DEFAULT
from mesohops.basis.hops_system import HopsSystem
from mesohops.eom.hops_eom import HopsEOM
from mesohops.noise.hops_noise import NOISE_DICT_DEFAULT
from mesohops.noise.prepare_functions import prepare_hops_noise
from mesohops.storage.hops_storage import HopsStorage
from mesohops.util.dynamic_dict import Dict_wDefaults
from mesohops.util.exceptions import (
    LockedException,
    TrajectoryError,
    UnsupportedRequest,
)
from mesohops.util.physical_constants import precision  # Constant

__title__ = "HOPS"
__author__ = "D. I. G. B. Raccah, L. Varvelo, J. K. Lynd"
__version__ = "1.6"

INTEGRATION_DICT_DEFAULT = {
    "INTEGRATOR": "RUNGE_KUTTA",
    "EARLY_ADAPTIVE_INTEGRATOR": "INCH_WORM",
    "EARLY_INTEGRATOR_STEPS": 5,
    "INCHWORM_CAP": 5,
    "STATIC_BASIS": None,
    "EFFECTIVE_NOISE_INTEGRATION": False,
}

INTEGRATION_DICT_TYPES = {
    "INTEGRATOR": [str],
    "EARLY_ADAPTIVE_INTEGRATOR": [str],
    "EARLY_INTEGRATOR_STEPS": [int],
    "INCHWORM_CAP": [int],
    "STATIC_BASIS": [type(None), list, np.ndarray],
    "EFFECTIVE_NOISE_INTEGRATION": [bool],
}


class HopsTrajectory:
    """
    Acts as an interface for users to run a single trajectory calculation.
    """

    __slots__ = (
        # --- Core basis components ---
        'basis',           # Basis management (HopsBasis)
        'storage',         # Storage management (HopsStorage)
        'dsystem_dt',      # System derivative function
        '_phi',            # Full hierarchy vector (current state)

        # --- Bookkeeping ---
        '__initialized__',    # Initialization status flag
        '_t',                 # Current simulation time

        # --- Integration control ---
        'step',                # Integration step function
        'integration_var',     # Integration variables function
        'integrator_step',     # Integrator step size
        '_early_step_counter', # Counter for early time integration steps
        'integration_param',   # Integration parameters (timestep, method, etc.)

        # --- Noise management ---
        'noise1',       # Primary noise trajectory (complex)
        'noise2',       # Secondary noise trajectory (real)
        '_z_mem',       # Noise memory drift
        'noise_param',  # Noise parameters (type, seed, etc.)
        'noise2_param', # Noise parameters for purely-real noise 2 (type, seed, etc.)
    )

    def __init__(
        self,
        system_param: dict[str, Any] | str | os.PathLike[str] | Path | None = None,
        eom_param: Dict[str, object] | None  = None,
        noise_param: Dict[str, object] | None = None,
        noise2_param: Dict[str, object] | None = None,
        hierarchy_param: Dict[str, object] | None = None,
        storage_param: Dict[str, object] | None = None,
        integration_param: Dict[str, object] | None = None,
    ) -> None:
        """
        This class manages four classes:
        1. Class: HopsNoise1 (Hierarchy Noise)
          * Class: NoiseTrajectory
        2. Class: HopsNoise2 (Optional Unitary Noise )
          * Class: NoiseTrajectory
        3. Class: HopsBasis
          * Class: HopsHierarchy
          * Class: HopsSystem
          * Class: HopsEOM
        4. Class: HopsStorage

        Inputs
        ------
        1. system_param: dict
                         Dictionary of user-defined system parameters.
                         [see hops_system.py (includes optional parameters)]
            a. HAMILTONIAN
            b. GW_SYSBATH
            c. L_HIER
            d. L_NOISE1
            e. ALPHA_NOISE1
            f. PARAM_NOISE1

        2. eom_param: dict
                      Dictionary of user-defined eom parameters.
                      [see hops_eom.py]
            a. EQUATION_OF_MOTION
            b. ADAPTIVE_H
            c. ADAPTIVE_S
            d. DELTA_A
            e. DELTA_S
            f. UPDATE_STEP

        3. noise_param: dict
                        Dictionary of user-defined noise parameters.
                        [see hops_noise.py]
            a. SEED
            b. MODEL
            c. TLEN
            d. TAU
            e. INTERP

        4. hierarchy_param: dict
                            Dictionary of user-defined hierarchy parameters.
                            [see hops_hierarchy.py]
            a. MAXHIER
            b. STATIC_FILTERS

        5. storage_param: dict
                          Dictionary of user-defined storage parameters.
                          [see hops_storage.py]
            a. phi_traj
            b. psi_traj
            c. t_axis
            d. aux_list
            e. state_list
            f. list_nhier
            g. list_nstate

        6. integration_param: dict
                              Dictionary of user-defined integration parameters.
                              [see integrator_rk.py]
            a. INTEGRATOR
            b. EARLY_ADAPTIVE_INTEGRATOR
            c. EARLY_INTEGRATOR_STEPS
            d. INCHWORM_CAP
            e. STATIC_BASIS

        """

        if system_param is None:
            system_param = {}
        if eom_param is None:
            eom_param = {}
        if noise_param is None:
            noise_param = {}
        if hierarchy_param is None:
            hierarchy_param = {}
        if storage_param is None:
            storage_param = {}

        # Variables for the current state of the system
        self._phi = []
        self._z_mem = []
        self._t = 0

        # Instantiates sub-classes
        # -----------------------
        eom = HopsEOM(eom_param)
        system = HopsSystem(system_param)
        hierarchy = HopsHierarchy(hierarchy_param, system.param)

        self.noise_param = noise_param

        # Defaults the second noise to zero
        if noise2_param is None:
           noise2_param = {
                "TLEN": noise_param["TLEN"],
                "TAU": noise_param["TAU"],
                "MODEL": "ZERO",
                "SEED": None,
            }
        # Warn if the user is employing risky practices
        else:
            if "SEED" in noise_param.keys() and "SEED" in noise2_param.keys():
                if (noise2_param['SEED'] == noise_param['SEED'] and noise_param['SEED'] is
                        not None):
                    warnings.warn("Using the same seed for both noise 1 and "
                             "noise 2 may introduce unphysical correlations between "
                                  "independent noise terms.")
            if (noise2_param["TAU"] != noise_param["TAU"] or noise2_param["TLEN"] !=
            noise_param["TLEN"]):
                if "INTERPOLATE" not in noise2_param.keys():
                    warnings.warn("Time axes of noise 1 and noise 2 are mismatched. "
                      "Be cautious when choosing propagation time step to avoid "
                      "exceptions.")
                elif not noise2_param["INTERPOLATE"]:
                    warnings.warn("Time axes of noise 1 and noise 2 are mismatched. "
                                  "Be cautious when choosing propagation time step to avoid "
                                  "exceptions.")
        self.noise2_param = noise2_param

        self.basis = HopsBasis(system, hierarchy, eom)
        self.storage = HopsStorage(self.basis.eom.param['ADAPTIVE'],storage_param)
        self.noise1 = prepare_hops_noise(self.noise_param, self.basis.system.param)
        self.noise2 = prepare_hops_noise(self.noise2_param, self.basis.system.param,
                                         noise_type=2)

        # Defines integration method
        # -------------------------
        if integration_param == None:
            integration_param = INTEGRATION_DICT_DEFAULT

        self.integration_param = Dict_wDefaults._initialize_dictionary(
            integration_param,
            INTEGRATION_DICT_DEFAULT,
            INTEGRATION_DICT_TYPES,
            "integration_param in the HopsTrajectory initialization",
        )
        self._early_step_counter = 0
        if self.integrator == "RUNGE_KUTTA":
            from mesohops.integrator.integrator_rk import (
                runge_kutta_step,
                runge_kutta_variables,
            )

            self.step = runge_kutta_step
            self.integration_var = runge_kutta_variables
            self.integrator_step = 0.5
        else:
            raise UnsupportedRequest(
                ("Integrator of type " + str(self.integrator)),
                type(self).__name__,
            )

        # LOCKING VARIABLE
        self.__initialized__ = False

    def initialize(
        self,
        psi_0: Sequence[complex] | np.ndarray,
        timer_checkpoint: float | None = None,
    ) -> None:
        """
        Initializes the trajectory module by ensuring that each sub-component is
        prepared to begin propagating a trajectory.

        Parameters
        ----------
        1. psi_0 : np.array(complex)
                   Wave function at initial time.

        2. timer_checkpoint : float
                              System time prior to initialization [units: s].
                              If None, uses current system time.

        Returns
        -------
        None
        """
        # Gets system time prior to initialization
        if timer_checkpoint is None:
            timer_checkpoint = timer.time()

        psi_0 = np.array(psi_0, dtype=np.complex128)

        if not self.__initialized__:
            # Prepares the derivative
            # ----------------------
            self.dsystem_dt = self.basis.initialize(psi_0)
            # Initializes System State
            # -----------------------
            self.storage.n_dim = self.basis.system.param["NSTATES"]
            phi_tmp = np.zeros(self.n_hier * self.n_state, dtype=np.complex128)
            phi_tmp[: self.n_state] = np.array(psi_0)[self.state_list]
            self.z_mem = sp.sparse.coo_array(
                                (len(self.basis.system.param["L_NOISE1"]),1)
                                ,dtype=np.complex128).tocsr()
            if self.basis.adaptive:
                if self.static_basis is None:
                    # Update Basis
                    z_step = self._prepare_zstep(self.z_mem)
                    (state_update, aux_update) = self.basis.define_basis(phi_tmp, 1,
                                                                         z_step)
                    (phi_tmp, dsystem_dt) = self.basis.update_basis(
                        phi_tmp, state_update, aux_update
                    )
                    self.dsystem_dt = dsystem_dt
                else:
                    # Construct initial basis
                    list_stable_state = self.state_list
                    list_state_new = list(
                        set(self.state_list).union(set(self.static_basis[0])))

                    list_stable_aux = self.auxiliary_list
                    list_aux_new = list(
                        set(self.auxiliary_list).union(set(self.static_basis[1])))

                    (phi_tmp, dsystem_dt) = self.basis.update_basis(
                        phi_tmp, list_state_new, list_aux_new
                    )

                    self.dsystem_dt = dsystem_dt

            # Stores System State
            # ------------------
            self.storage.store_step(
                phi_new=phi_tmp, aux_list=self.auxiliary_list, state_list=self.state_list,
                t_new=0, z_mem_new=self.z_mem
            )
            self.t = 0
            self.phi = phi_tmp

            # Stores initialization time
            # --------------------------
            self.storage.metadata["INITIALIZATION_TIME"] = timer.time() - timer_checkpoint

            # Locks Trajectory
            # ---------------
            self.__initialized__ = True
        else:
            raise LockedException("HopsTrajectory.initialize()")

    def make_adaptive(
        self,
        delta_a: float = 1e-4,
        delta_s: float = 1e-4,
        update_step: int = 1,
        f_discard: float = 0.01,
        adaptive_noise: bool = True,
    ) -> None:
        """
        Transforms a not-yet-initialized HOPS trajectory from a standard HOPS to an
        adaptive HOPS approach.

        Parameters
        ----------
        1. delta_a : float
                     Value of the adaptive grid for the hierarchy of auxiliary
                     nodes (options: < 1).

        2. delta_s : float
                     Value of the adaptive grid in the system basis (options: < 1).

        3. update_step : int
                         Number of time points between updates to the adaptive
                         basis.

        4. f_discard : float
                       Fraction of the boundary error devoted to removing error
                       terms from list_e2_kflux for memory conservation (recommended
                       value: 0.2).

        Returns
        -------
        None
        """
        if float(f_discard) < 0 or float(f_discard) > 1:
            raise UnsupportedRequest(self.make_adaptive(), "f_discard not in "
                                                           "acceptable range of [0,1]")

        if not self.__initialized__:
            if delta_a > 0:
                self.basis.eom.param["ADAPTIVE"] = True
                self.basis.eom.param["ADAPTIVE_H"] = True
                self.basis.eom.param["DELTA_A"] = delta_a
                self.basis.eom.param["UPDATE_STEP"] = update_step
                self.basis.eom.param["F_DISCARD"] = float(f_discard)
                self.storage.adaptive = True

            if delta_s > 0:
                self.basis.eom.param["ADAPTIVE"] = True
                self.basis.eom.param["ADAPTIVE_S"] = True
                self.basis.eom.param["DELTA_S"] = delta_s
                self.basis.eom.param["UPDATE_STEP"] = update_step
                self.basis.eom.param["F_DISCARD"] = float(f_discard)
                self.storage.adaptive = True

            if adaptive_noise:
                self.noise1.param["ADAPTIVE"] = True
                self.noise2.param["ADAPTIVE"] = True


        else:
            raise TrajectoryError("Calling make_adaptive on an initialized trajectory")

    def propagate(
        self,
        t_advance: float,
        tau: float,
        timer_checkpoint: Optional[float] = None,
    ) -> None:
        """
        Performs the integration along fixed time-points. The kind of integration
        that is performed is controlled by 'step' which was setup in the initialization.

        Parameters
        ----------
        1. t_advance : float
                       How far out in time the calculation will run [units: fs].

        2. tau : float
                 Time step [units: fs].

        3. timer_checkpoint : float
                              System time prior to initialization [units: s].
                              If None, uses current system time.

        Returns
        -------
        None
        """
        # Gets system time prior to propagation
        if timer_checkpoint is None:
            timer_checkpoint = timer.time()

        # Checks and prepares time axis
        # ---------------------------
        t0 = self.t
        if (self.noise1.param["TAU"] is not None) or not (self.noise1.param["INTERPOLATE"]):
            if self._check_tau_step(tau, precision):
                t_axis = t0 + np.arange(1, 1 + int(np.ceil(t_advance / tau))) * tau
                print("Integration from ", t0, " to ", np.max(t_axis))

            else:
                raise TrajectoryError(
                    "Timesteps("
                    + str(tau * self.integrator_step)
                    + ") that do not match noise.param['TAU'] ("
                    + str(self.noise1.param["TAU"])
                    + ")"
                )

        if (t0 + t_advance + tau) > self.noise1.param["TLEN"]:
            raise TrajectoryError(
                "Trajectory times longer than noise.param['TLEN'] ="
                + str(self.noise1.param["TLEN"])
            )

        # Performs integration
        # -------------------
        for (index_t, t) in enumerate(t_axis):
            var_list = self.integration_var(self.phi, self.z_mem, self.t, self.noise1,
                                            self.noise2, tau, self.storage,
                                            self.basis.mode.list_absindex_L2,
                                            self.effective_noise_integration)
            phi, z_mem = self.step(self.dsystem_dt, **var_list)
            phi = self.normalize(phi)

            if self.basis.adaptive:
                aux_update = [self.auxiliary_list, self.auxiliary_list, []]

                # Checks for Early Time Integration
                # ================================
                if self.use_early_integrator:
                    print(f'Early Integration: Using {self.early_integrator}')
                    # Early Integrator: Inch Worm
                    # ---------------------------
                    if self.early_integrator == 'INCH_WORM' or \
                            self.early_integrator == 'STATIC_STATE_INCHWORM_HIERARCHY':
                        # Define New Basis
                        z_step = self._prepare_zstep(z_mem)
                        (state_update, aux_update) = self.basis.define_basis(phi, tau,
                                                                             z_step)
                        if self.early_integrator == 'STATIC_STATE_INCHWORM_HIERARCHY':
                            state_update = self.static_basis[0]

                        # Update basis for new step
                        step_num = 0
                        while (set(state_update) != set(self.basis.system.state_list)
                               or set(aux_update) != set(self.basis.hierarchy.auxiliary_list)):
                            state_update, aux_update, phi, z_mem = self.inchworm_integrate(
                                state_update, aux_update, tau
                            )
                            if self.early_integrator == 'STATIC_STATE_INCHWORM_HIERARCHY':
                                state_update = self.static_basis[0]
                            step_num += 1
                            if step_num >= self.inchworm_cap:
                                break

                        # Update basis
                        (phi, self.dsystem_dt) = self.basis.update_basis(
                            phi, state_update, aux_update
                        )

                    # Early Integrator: Static Basis
                    # ------------------------------
                    elif self.early_integrator == 'STATIC':
                        pass

                    else:
                        raise UnsupportedRequest(self.early_integrator,
                                                 "early time integrator "
                                                 "clause of the propagate")

                    self._early_step_counter += 1
                # Standard Adaptive Integration
                # =============================
                elif (index_t + 1) % self.update_step == 0:

                    # Define New Basis
                    # ----------------
                    z_step = self._prepare_zstep(z_mem)
                    (state_update, aux_update) = self.basis.define_basis(phi, tau,
                                                                         z_step)

                    # Updates Basis
                    # ------------
                    (phi, self.dsystem_dt) = self.basis.update_basis(
                        phi, state_update, aux_update
                    )
            if self.storage.check_storage_time(t):
                self.storage.store_step(
                    phi_new=phi, aux_list=self.auxiliary_list, state_list=self.state_list, t_new=t,
                    z_mem_new=self.z_mem
                )
            self.phi = phi
            self.z_mem = z_mem
            self.t = t

        # Stores propagation time
        # --------------------------
        self.storage.metadata["LIST_PROPAGATION_TIME"].append(timer.time() -
                                                              timer_checkpoint)

    def _operator(self, op: np.ndarray | sparse.spmatrix) -> None:
        """
        Acts an operator on the full hierarchy. Automatically adds all states that
        become populated to the basis in the adaptive case to avoid uncontrolled error,
        then updates the full basis immediately thereafter. Finally, resets early time
        integration to ensure that the basis is updated aggressively.

        Parameters
        ----------
        1. op : np.array(float)
                The operator.

        Returns
        -------
        None
        """
        if (sp.sparse.issparse(op)):
            op = op.tocsr()
        # If state adaptivity in use, add the states needed for the operation and
        # update basis.
        if self.basis.eom.param["DELTA_S"] > 0:
            updated_state_list = list(self.state_list)
            updated_state_list += list(np.nonzero(op[:, self.state_list])[0])
            updated_state_list = list(set(updated_state_list))
            (self.phi, self.dsystem_dt) = self.basis.update_basis(
                self.phi,updated_state_list, self.auxiliary_list)
        # Trim the operator based on the state_list and perform the operation.
        op = op[np.ix_(self.state_list, self.state_list)]
        phi_mat = np.reshape(self.phi, [self.n_state, self.n_hier], order="F")
        self.phi = np.reshape(op@phi_mat, len(self.phi), order="F")
        # Operation may populate states not in the basis: therefore update the basis,
        # define a dummy integration time step such that none of the states are removed
        # and perform early time integration.
        if self.basis.eom.param["DELTA_S"] > 0:
            delta_t=np.min(np.abs(self.phi[np.nonzero(self.phi)]))
            self.basis.define_basis(self.phi, delta_t, self._prepare_zstep(self.z_mem))
            self.basis.update_basis(self.phi, self.state_list, self.auxiliary_list)
            self.reset_early_time_integrator()

    def _check_tau_step(self, tau: float, precision: float) -> bool:
        """
        Checks if tau_step is within precision of noise1.param['TAU'] and if tau is
        greater than or equal to noise1.param['TAU'].

        Parameters
        ----------
        1. tau : float
                 Time step [units: fs].

        2. precision : float
                       Constant that defines precision when comparing time points.

        Returns
        -------
        1. tau_step_precision_check : bool
                                      True indicates the time step is consistent with
                                      noise while False indicates otherwise.
        """
        # ERROR: We are assuming that if noise1 is numeric then noise2 is numeric
        tau_step = tau * self.integrator_step
        p_bool1 = (
                tau_step / self.noise1.param["TAU"]
                - int(round(tau_step / self.noise1.param["TAU"]))
                < precision
        )
        p_bool2 = (
                tau_step / self.noise2.param["TAU"]
                - int(round(tau_step / self.noise2.param["TAU"]))
                < precision
        )

        s_bool1 = int(round(tau / self.noise1.param["TAU"])) >= 1
        s_bool2 = int(round(tau / self.noise2.param["TAU"])) >= 1

        return p_bool1 and p_bool2 and s_bool1 and s_bool2

    def normalize(self, phi: np.ndarray) -> np.ndarray:
        """
        Re-normalizes the wave function at each step to correct for loss of norm due
        to finite numerical accuracy.

        Parameters
        ----------
        1. phi : np.array(complex)
                 Current full state of the hierarchy.

        Returns
        -------
        1. phi : np.array(complex)
                 Full state of the hierarchy, normalized if appropriate.
        """
        if self.basis.eom.normalized:
            return phi / np.sqrt(np.sum(np.abs(phi[: self.n_state]) ** 2))
        else:
            return phi

    def inchworm_integrate(
            self,
            list_state_new: Sequence[int],
            list_aux_new: Sequence[AuxVec],
            tau: float,
    ) -> Tuple[list[int], list[AuxVec], np.ndarray, sparse.sparray]:
        """
        Performs inchworm integration.

        Parameters
        ----------
        1. list_state_new : list(int)
                            List of states in the new basis (S_1).

        2. list_aux_new : list(instance(AuxiliaryVector))
                          List of auxiliaries in new basis (H_1).

        3. tau : float
                 Time step [units: fs].

        Returns
        -------
        1. list_state_new : list(int)
                            List of states in the new basis (S_1).

        2. list_aux_new : list(instance(AuxiliaryVector))
                          List of auxiliaries in new basis (H_1).

        3. phi : np.array(complex)
                 Full state of the hierarchy normalized if appropriate.

        4. z_mem : np.array(complex)
                   Noise memory drift terms for the bath [units: cm^-1].

        """
        # Incorporate the new states into the state basis
        state_update = list(set(list_state_new) | set(self.state_list))

        # Incorporate the new hierarchy into the hierarchy basis
        aux_update = list(set(list_aux_new) | set(self.auxiliary_list))

        # Update phi and derivative for new basis
        (phi, self.dsystem_dt) = self.basis.update_basis(
            self.phi, state_update, aux_update
        )
        self.phi = phi

        # Perform integration step with extended basis
        var_list = self.integration_var(self.phi, self.z_mem, self.t, self.noise1, self.noise2, tau, self.storage, self.basis.mode.list_absindex_L2)
        phi, z_mem = self.step(self.dsystem_dt, **var_list)
        phi = self.normalize(phi)

        # Define new basis for step in extended space
        z_step = self._prepare_zstep(z_mem)
        (state_update, aux_update) = self.basis.define_basis(phi, tau, z_step)
        return state_update, aux_update, phi, z_mem

    def _prepare_zstep(
        self, z_mem: sparse.spmatrix | np.ndarray
    ) -> List[Union[np.ndarray, sparse.spmatrix]]:
        """
        Constructs a compressed version of the noise terms to be used
        in the following time step

        Parameters
        ----------
        1. z_mem : list(complex)
                   Noise memory drift terms for the bath [units: cm^-1].

        Returns
        -------
        1. z_step : list(np.array(complex))
                    Noise terms (compressed) for the next timestep [units: cm^-1].
        """
        t = self.t
        list_absindex_L2 = self.basis.mode.list_absindex_L2
        z_rnd1 = self.noise1.get_noise([t], list_absindex_L2)[:, 0]
        z_rnd2 = self.noise2.get_noise([t], list_absindex_L2)[:, 0]
        return [z_rnd1, z_rnd2, z_mem]

    def construct_noise_correlation_function(
        self, n_l2: int, n_traj: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses correlated noise trajectories to reconstruct the full noise correlation
        function.
        ** Warning if SEED is not None function will not work properly **

        Parameters
        ----------
        1. n_l2 : int
                  Index of the site that the noise is being generated on.

        2. n_traj : int
                    Number of trajectories summed over.

        Returns
        -------
        1. list_corr_func_1 : list(np.array(complex))
                              Correlation function calculated for noise1 [units: cm^-2].

        2. list_corr_func_2 : list(np.array(complex))
                              Correlation function calculated for noise2 [units: cm^-2].
        """
        if self.noise_param['SEED'] is not None:
            warnings.warn('SEED is not None; Summing over identical noise trajectories'
                          'In order to reconstruct correlation function seed must be None')
        t_axis = np.arange(0, self.noise1.param['TLEN'], self.noise1.param['TAU'])
        list_ct1 = np.zeros(len(t_axis), dtype=np.complex128)
        list_ct2 = np.zeros(len(t_axis), dtype=np.complex128)
        for _ in np.arange(n_traj):
            noise1 = prepare_hops_noise(self.noise_param, self.basis.system.param)
            noise1._prepare_noise(self.basis.system.param["LIST_INDEX_L2_BY_HMODE"])
            result = np.correlate(noise1.get_noise(t_axis, self.basis.system.param["LIST_INDEX_L2_BY_HMODE"])[n_l2, :],
                                  noise1.get_noise(t_axis, self.basis.system.param["LIST_INDEX_L2_BY_HMODE"])[n_l2, :], mode='full')
            list_ct1 += result[result.size // 2:]
            if "L_NOISE2" in self.basis.system.param.keys():
                noise2 = prepare_hops_noise(self.noise2_param,
                                            self.basis.system.param, noise_type=2)
                noise2._prepare_noise()
                result = np.correlate(noise2.get_noise(t_axis, self.basis.system.param["LIST_INDEX_L2_BY_HMODE"])[n_l2, :],
                                      noise2.get_noise(t_axis, self.basis.system.param["LIST_INDEX_L2_BY_HMODE"])[n_l2, :], mode='full')
                list_ct2 += result[result.size // 2:]
        return list_ct1 / (n_traj * len(t_axis)), list_ct2 / (n_traj * len(t_axis))

    def reset_early_time_integrator(self) -> None:
        """
        Sets self._early_integrator_time to the current time so that the next use of
        propagate will make the first self.early_steps early time integrator
        propagation steps.
        """
        if self.early_integrator != "INCH_WORM":
            raise UnsupportedRequest("Early type integrators of type other than "
                                     "INCH_WORM", "reset_early_time_integrator",
                                     override=True)
        self._early_step_counter = 0

    def save_slices(
        self,
        dir_path_save: str | os.PathLike,
        list_key: Sequence[str] | None = None,
        file_header: str | None = None,
        seed: int | None = None,
        step: int = 1,
        compress: bool = False,
    ) -> None:
        """
        Saves data to disk. By default it will save all the default keys in hops storage
        dictionary. It can also save specific keys at user defined time intervals.

        Parameters
        ----------
        1. dir_path_save : str
                           Output directory path

        2. list_key : list of str, optional
                      List of keys that will be saved at user defined time intervals.
                      If None, saves all keys flagged as True in storage_dic.

        3. file_header : str, optional
                         Header for the output file name. Default is 'data'.

        4. seed : int, optional
                  Seed value for the filename. If None, uses the seed from noise_param.

        5. step : int, optional
                  Time interval for slicing data. Default is 1.

        6. compress : bool, optional
                      Whether to compress the saved files using gzip compression.
                      Default is False.

        Returns
        -------
        None

        """
        if file_header is None:
            file_header = 'data'
        if seed is None:
            seed=self.noise_param['SEED']
            if not isinstance(seed, int):
                 raise TypeError("Saving failed. Either set 'SEED' in noise parameters" 
                    "to an integer or provide a custom seed value for the file name.")

        if not isinstance(step, int):
            warnings.warn("Step should be an integer. Setting step to the default "
                          "value of 1.")
            step=1

        if list_key is None:
            list_key =[k for k, v in self.storage.storage_dic.items() if v is not False]

        os.makedirs(dir_path_save, exist_ok=True)
        dict_sliced_data = {}
        for key in list_key:
            try:
                data = self.storage[key]
                if isinstance(data, list):
                    data = np.array(data, dtype=object)
                length = np.shape(data)[0]
                idx = np.arange(0, length, step)
                slice_data = data[idx]
                dict_sliced_data[key] = np.array(slice_data, dtype=object)

            except Exception as e:
                warnings.warn(f"Skipping key [{key}]. Error processing: {e}")
                continue

        filepath = os.path.join(dir_path_save, file_header + "_seed_" + f'{seed}' + ".npz")
        if compress:
            np.savez_compressed(filepath, **dict_sliced_data, allow_pickle=True)
            print(f"Saved compressed sliced data to {filepath}")
        else:
            np.savez(filepath, **dict_sliced_data, allow_pickle=True)
            print(f"Saved sliced data to {filepath}")

    def save_checkpoint(self, filepath: str | os.PathLike,
                        compress: bool = True,
                        drop_seed: bool = False) -> None:
        """
        Save the current state of the trajectory to a file. This involves saving:
        a. the input parameters for HopsSystem, HopsHierarchy, HopsNoise1, HopsNoise2,
        HopsEOM, HopsIntegration, and HopsStorage objects.
        b. the current full wavefunction (phi),
        c. the current state basis (state_list),
        d. the current auxiliary basis (aux_list),
        e. noise memory (z_mem),
        f. the simulation time (t), and
        g. the early integration counter
        h. all data in hops_storage.data
        i. all metadata in hops_storage.metadata

        Parameters
        ----------
        1. filepath : str
                      Output filename for the checkpoint.
        2. compress : bool, optional
                      If True the file is gzip-compressed. Default is True.
        3. drop_seed : bool, optional
                       If True the noise seed will not be saved. Default is False.

        Returns
        -------
        None

        """
        # HopsEOM Parameter Dictionary: Only keep the default parameters
        eom_param = {k: v for k, v in self.basis.eom.param.items() if k != "ADAPTIVE"}
        for key in ["DELTA_A", "DELTA_S", "F_DISCARD"]:
            if key in eom_param:
                eom_param[key] = float(eom_param[key])
        if eom_param.get("UPDATE_STEP") is None:
            eom_param.pop("UPDATE_STEP")

        # HopsSystem Parameter Dictionary:
        list_hops_sys_param = ['HAMILTONIAN', 'GW_SYSBATH', 'L_HIER', 'L_NOISE1', 'ALPHA_NOISE1', 'PARAM_NOISE1',
                               'L_NOISE2', 'ALPHA_NOISE2', 'PARAM_NOISE2', 'L_LT_CORR', 'PARAM_LT_CORR']

        # Create a dictionary of HopsTrajectory parameters
        params = {
            'system_param': {k: self.basis.system.param[k] for k in list_hops_sys_param if k in self.basis.system.param},
            'hierarchy_param': {k: self.basis.hierarchy.param[k] for k in HIERARCHY_DICT_DEFAULT if k in self.basis.hierarchy.param},
            'eom_param': eom_param,
            'noise1_param': {k: self.noise1.param[k] for k in NOISE_DICT_DEFAULT if ((k in self.noise1.param)
                                                                                     and not ((k == 'SEED')
                                                                                              and drop_seed
                                                                                              )
                                                                                     )},
            'noise2_param': {k: self.noise2.param[k] for k in NOISE_DICT_DEFAULT if ((k in self.noise2.param)
                                                                                     and not ((k == 'SEED')
                                                                                              and drop_seed
                                                                                              )
                                                                                     )},
            'integration_param': self.integration_param,
            'storage_param': self.storage.storage_dic,
        }

        # Create a dictionary of indexing/bookkeeping variables
        checkpoint = {
            'phi': self.phi,
            'z_mem': self.z_mem,
            't': self.t,
            'state_list': np.array(self.state_list, dtype=int),
            'aux_list': np.array([aux.array_aux_vec for aux in self.auxiliary_list], dtype=object),
            'early_counter': self._early_step_counter,
            'storage_data': self.storage.data,
            'storage_meta': self.storage.metadata,
            'params': params,
        }
        if (not drop_seed) and (self.noise1.param['SEED'] is None):
            warnings.warn(
                "Noise1 seed is None but drop_seed is False; "
                "this may cause unphysical behavior.",
                UserWarning
            )

        if compress:
            np.savez_compressed(filepath, **checkpoint, allow_pickle=True)
        else:
            np.savez(filepath, **checkpoint, allow_pickle=True)

    @classmethod
    def load_checkpoint(cls,
                        filename: str | os.PathLike,
                        add_seed1: int | str | os.PathLike | np.ndarray | None = None,
                        add_seed2: int | str | os.PathLike | np.ndarray | None = None,
                        add_system_param: str | os.PathLike | None = None) -> HopsTrajectory:
        """
        Loads a trajectory object from a checkpoint file that has been generated using save_checkpoint().

        Parameters
        ----------
        1. filename : str or os.PathLike
                      Path to the ``.npz`` file generated using save_checkpoint().

        2. add_seed1 : int, str, os.PathLike, np.ndarray or None
                       Optional parameter to specify a seed value for Noise1,
                       this will override the seed value (if any) stored in the
                       checkpoint file.

        3. add_seed2 : int, str, os.PathLike, np.ndarray or None
                       Optional parameter to specify a seed value for Noise2,
                       this will override the seed value (if any) stored in the
                       checkpoint file.

        4. add_system_param : str or os.PathLike or None
                       Optional parameter to specify a path to a file containing a pre-saved
                       system parameter file. This will enable directly loading the system
                       parameter dictionary rather than having to pass it through the constructor.

        Returns
        -------
        1. traj : instance(HopsTrajectory)
                   Reconstructed trajectory with the state and parameters
                   restored from the checkpoint.
        """
        # Load checkpoint data from the .npz file
        data = np.load(filename, allow_pickle=True)
        params = data['params'].item()

        # Update noise if needed
        if add_seed1 is not None:
            params['noise1_param']['SEED'] = add_seed1

        # Update noise if needed
        if add_seed2 is not None:
            params['noise2_param']['SEED'] = add_seed2

        # Update system parameters if needed
        if add_system_param is not None:
            params['system_param'] = add_system_param

        # Instantiate a new trajectory object with the stored parameters
        traj = cls(
            params['system_param'],
            eom_param=params['eom_param'],
            noise_param=params['noise1_param'],
            noise2_param=params['noise2_param'],
            hierarchy_param=params['hierarchy_param'],
            storage_param=params['storage_param'],
            integration_param=params['integration_param'],
        )

        # Initialize the trajectory with the stored wave function
        psi_0 = np.zeros(traj.basis.system.param['NSTATES'], dtype=np.complex128)
        psi_0[data['state_list']] = data['phi'][:data['state_list'].size]
        traj.initialize(psi_0)

        # Set the auxiliary list based on the stored data
        list_aux = [AuxVec(aux, traj.basis.hierarchy.n_hmodes) for aux in data['aux_list']]
        # The next block ensures that if an aux is already in the basis it is re-used, rather than
        # being defined again. This is consistent with how aux are defined in the adaptive propagation.
        for aux_orig in traj.auxiliary_list:
            if aux_orig in list_aux:
                index_aux = list_aux.index(aux_orig)
                list_aux[index_aux] = aux_orig

        # Update the aux basis
        # Because of how the hierarchy class is updated, we need to update the basis one level of the
        # hierarchy at a time until the entire hierarchy is updated. (There is a requirement that the
        # auxiliaries that are added to the basis are within one step of a previously defined aux.)
        for depth in range(1,traj.basis.hierarchy.param["MAXHIER"]+1):
            list_aux_depth = [aux for aux in list_aux if aux._sum <= depth]
            (traj.phi, traj.dsystem_dt) = traj.basis.update_basis(traj.phi,
                                                        data['state_list'],
                                                        list_aux_depth)

        # The trajectory has the correct basis. Restore the: state vector,
        # noise memory and bookkeeping variables.
        traj.phi = data['phi']
        traj.z_mem = data['z_mem'].item()
        traj.t = float(data['t'])
        traj._early_step_counter = int(data['early_counter'])

        # Restore the storage data from the checkpoint file
        traj.storage.data = data['storage_data'].item()
        traj.storage.metadata = data['storage_meta'].item()
        return traj

    def save_system_parameters(self, filepath: str | os.PathLike) -> None:
        """
        Saves the system parameters to a file.

        Parameters
        ----------
        1. filepath : str or os.PathLike
                      Path to the output file where the system parameters will be saved.

        Returns
        -------
        None
        """
        self.basis.system.save_dict_param(filepath)

    @property
    def early_integrator(self) -> str:
        return self.integration_param["EARLY_ADAPTIVE_INTEGRATOR"]

    @property
    def integrator(self) -> str:
        return self.integration_param["INTEGRATOR"]

    @property
    def inchworm_cap(self) -> int:
        return self.integration_param["INCHWORM_CAP"]

    @property
    def effective_noise_integration(self) -> bool:
        return self.integration_param["EFFECTIVE_NOISE_INTEGRATION"]

    @property
    def static_basis(self) -> list | np.ndarray | None:
        return self.integration_param["STATIC_BASIS"]

    @property
    def psi(self) -> np.ndarray:
        return self.phi[: self.n_state]

    @property
    def n_hier(self) -> int:
        return self.basis.n_hier

    @property
    def n_state(self) -> int:
        return self.basis.n_state

    @property
    def state_list(self) -> list[int]:
        return self.basis.system.state_list

    @property
    def auxiliary_list(self) -> list[AuxVec]:
        return self.basis.hierarchy.auxiliary_list

    @property
    def update_step(self) -> int | bool | None:
        return self.basis.eom.param["UPDATE_STEP"]

    @property
    def phi(self) -> np.ndarray:
        return self._phi

    @phi.setter
    def phi(self, phi: np.ndarray) -> None:
        self._phi = phi

    @property
    def z_mem(self) -> sparse.spmatrix:
        return self._z_mem

    @z_mem.setter
    def z_mem(self, z_mem: sparse.spmatrix) -> None:
        self._z_mem = z_mem

    @property
    def t(self) -> float:
        return self._t

    @property
    def early_steps(self) -> int:
        return self.integration_param["EARLY_INTEGRATOR_STEPS"]

    @property
    def use_early_integrator(self) -> bool:
        return self._early_step_counter < self.early_steps

    @t.setter
    def t(self, t: float) -> None:
        self._t = t

