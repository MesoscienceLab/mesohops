import copy
import warnings
import scipy as sp
import numpy as np
from mesohops.dynamics.hops_basis import HopsBasis
from mesohops.dynamics.hops_eom import HopsEOM
from mesohops.dynamics.hops_hierarchy import HopsHierarchy
from mesohops.dynamics.hops_system import HopsSystem
from mesohops.dynamics.prepare_functions import prepare_hops_noise
from mesohops.dynamics.hops_storage import HopsStorage
from mesohops.util.dynamic_dict import Dict_wDefaults
from mesohops.util.exceptions import UnsupportedRequest, LockedException, TrajectoryError
from mesohops.util.physical_constants import precision  # Constant

__title__ = "HOPS"
__author__ = "D. I. G. Bennett, L. Varvelo"
__version__ = "1.2"

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

    def __init__(
        self,
        system_param,
        eom_param={},
        noise_param={},
        hierarchy_param={},
        storage_param={},
        integration_param={},
    ):
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
            d. DELTA_H
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
        self.basis = HopsBasis(system, hierarchy, eom)
        self.storage = HopsStorage(self.basis.eom.param['ADAPTIVE'],storage_param)
        self.noise1 = prepare_hops_noise(noise_param, self.basis.system.param)
        if "L_NOISE2" in system.param.keys():
            noise_param2 = copy.copy(noise_param)
            rand = np.random.RandomState(seed=noise_param['SEED'])
            noise_param2['SEED'] = int(rand.randint(0, 2 ** 20, 1)[0])
            self.noise2 = prepare_hops_noise(noise_param2, self.basis.system.param, flag=2)
        else:
            noise_param2 = {
                "TLEN": noise_param["TLEN"],
                "TAU": noise_param["TAU"],
                "MODEL": "ZERO",
            }
            if "NOISE_WINDOW" in noise_param.keys():
                noise_param2["NOISE_WINDOW"] = noise_param["NOISE_WINDOW"]
            self.noise2 = prepare_hops_noise(noise_param2, self.basis.system.param, flag=1)

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
            from mesohops.dynamics.integrator_rk import (
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

    def initialize(self, psi_0):
        """
        Initializes the trajectory module by ensuring that each sub-component is
        prepared to begin propagating a trajectory.

        Parameters
        ----------
        1. psi_0 : np.array(complex)
                   Wave function at initial time.

        Returns
        -------
        None
        """
        psi_0 = np.array(psi_0, dtype=np.complex128)

        if not self.__initialized__:
            # Initializes Noise
            # ----------------
            if not self.noise1.__locked__:
                self.noise1.prepare_noise()
            if not self.noise2.__locked__:
                self.noise2.prepare_noise()

            # Prepares the derivative
            # ----------------------
            self.dsystem_dt = self.basis.initialize(psi_0)
            # Initializes System State
            # -----------------------
            self.storage.n_dim = self.basis.system.param["NSTATES"]
            phi_tmp = np.zeros(self.n_hier * self.n_state, dtype=np.complex128)
            phi_tmp[: self.n_state] = np.array(psi_0)[self.state_list]
            self.z_mem = np.zeros(len(self.basis.system.param["L_NOISE1"]))
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
                    list_add_state = set(list_state_new) - set(list_stable_state)
                    state_update = (list_state_new, list_stable_state, list_add_state)

                    list_stable_aux = self.auxiliary_list
                    list_aux_new = list(
                        set(self.auxiliary_list).union(set(self.static_basis[1])))
                    list_add_aux = set(list_aux_new) - set(list_stable_aux)
                    aux_update = (list_aux_new, list_stable_aux, list_add_aux)

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

            # Locks Trajectory
            # ---------------
            self.__initialized__ = True
        else:
            raise LockedException("HopsTrajectory.initialize()")

    def make_adaptive(self, delta_h=1e-4, delta_s=1e-4, update_step=1, f_discard=
    0):
        """
        Transforms a not-yet-initialized HOPS trajectory from a standard HOPS to an
        adaptive HOPS approach.

        Parameters
        ----------
        1. delta_h : float
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
            if delta_h > 0:
                self.basis.eom.param["ADAPTIVE"] = True
                self.basis.eom.param["ADAPTIVE_H"] = True
                self.basis.eom.param["DELTA_H"] = delta_h
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


        else:
            raise TrajectoryError("Calling make_adaptive on an initialized trajectory")

    def propagate(self, t_advance, tau):
        """
        Performs the integration along fixed time-points. The kind of integration
        that is performed is controlled by 'step' which was setup in the initialization.

        Parameters
        ----------
        1. t_advance : float
                       How far out in time the calculation will run [units: fs].

        2. tau : float
                 Time step [units: fs].

        Returns
        -------
        None
        """

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
                            state_update, aux_update, phi = self.inchworm_integrate(
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

            self.storage.store_step(
                phi_new=phi, aux_list=self.auxiliary_list, state_list=self.state_list, t_new=t,
                z_mem_new=self.z_mem
            )
            self.phi = phi
            self.z_mem = z_mem
            self.t = t

    def _operator(self, op):
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

    def _check_tau_step(self, tau, precision):
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

    def normalize(self, phi):
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

    def inchworm_integrate(self, list_state_new, list_aux_new, tau):
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
        var_list = self.integration_var(self.phi, self.z_mem, self.t, self.noise1, self.noise2, tau, self.storage)
        phi, z_mem = self.step(self.dsystem_dt, **var_list)
        phi = self.normalize(phi)

        # Define new basis for step in extended space
        z_step = self._prepare_zstep(z_mem)
        (state_update, aux_update) = self.basis.define_basis(phi, tau, z_step)
        return state_update, aux_update, phi

    def _prepare_zstep(self, z_mem):
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
        z_rnd1 = self.noise1.get_noise([t])[:, 0]
        z_rnd2 = self.noise2.get_noise([t])[:, 0]
        return [z_rnd1, z_rnd2, z_mem]

    def construct_noise_correlation_function(self, n_l2, n_traj):
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
            noise1.prepare_noise()
            result = np.correlate(noise1.get_noise(t_axis)[n_l2, :],
                                  noise1.get_noise(t_axis)[n_l2, :], mode='full')
            list_ct1 += result[result.size // 2:]
            if "L_NOISE2" in self.basis.system.param.keys():
                noise2 = prepare_hops_noise(self.noise_param, self.basis.system.param,
                                       flag=2)
                noise2.prepare_noise()
                result = np.correlate(noise2.get_noise(t_axis)[n_l2, :],
                                      noise2.get_noise(t_axis)[n_l2, :], mode='full')
                list_ct2 += result[result.size // 2:]
        return list_ct1 / (n_traj * len(t_axis)), list_ct2 / (n_traj * len(t_axis))

    def reset_early_time_integrator(self):
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


    @property
    def early_integrator(self):
        return self.integration_param["EARLY_ADAPTIVE_INTEGRATOR"]

    @property
    def integrator(self):
        return self.integration_param["INTEGRATOR"]

    @property
    def inchworm_cap(self):
        return self.integration_param["INCHWORM_CAP"]

    @property
    def effective_noise_integration(self):
        return self.integration_param["EFFECTIVE_NOISE_INTEGRATION"]

    @property
    def static_basis(self):
        return self.integration_param["STATIC_BASIS"]

    @property
    def psi(self):
        return self.phi[: self.n_state]

    @property
    def n_hier(self):
        return self.basis.n_hier

    @property
    def n_state(self):
        return self.basis.n_state

    @property
    def state_list(self):
        return self.basis.system.state_list

    @property
    def auxiliary_list(self):
        return self.basis.hierarchy.auxiliary_list

    @property
    def update_step(self):
        return self.basis.eom.param["UPDATE_STEP"]

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = phi

    @property
    def z_mem(self):
        return self._z_mem

    @z_mem.setter
    def z_mem(self, z_mem):
        self._z_mem = z_mem

    @property
    def t(self):
        return self._t

    @property
    def early_steps(self):
        return self.integration_param["EARLY_INTEGRATOR_STEPS"]

    @property
    def use_early_integrator(self):
        return self._early_step_counter < self.early_steps

    @t.setter
    def t(self, t):
        self._t = t
