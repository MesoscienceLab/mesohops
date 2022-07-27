import copy
import warnings
import scipy as sp
import numpy as np
from pyhops.dynamics.hops_basis import HopsBasis
from pyhops.dynamics.hops_eom import HopsEOM
from pyhops.dynamics.hops_hierarchy import HopsHierarchy
from pyhops.dynamics.hops_system import HopsSystem
from pyhops.dynamics.prepare_functions import prepare_noise
from pyhops.dynamics.hops_storage import HopsStorage
from pyhops.util.exceptions import UnsupportedRequest, LockedException, TrajectoryError
from pyhops.util.physical_constants import precision  # Constant

__title__ = "HOPS"
__author__ = "D. I. G. Bennett, L. Varvelo"
__version__ = "1.2"


###########################################################
class HopsTrajectory:
    """
     HopsTrajectory is the class that a user should interface with to run a single 
     trajectory calculation.
    """

    def __init__(
        self,
        system_param,
        eom_param={},
        noise_param={},
        hierarchy_param={},
        storage_param={},
        integration_param=dict(
            INTEGRATOR="RUNGE_KUTTA", EARLY_ADAPTIVE_INTEGRATOR='INCH_WORM',EARLY_INTEGRATOR_STEPS=5, INCHWORM_CAP=5,
            STATIC_BASIS=None
        ),
    ):
        """
        This class contains four classes:
        1. Class: HopsNoise1 (Hierarchy Noise)
          * Class: NoiseTrajectory
        2. Class: HopsNoise2 (Optional Unitary Noise )
          * Class: NoiseTrajectory
        3. Class: HopsBasis
          * Class: HopsHierarchy
          * Class: HopsSystem
          * Class: HopsEOM
        4. Class: HopsStorage

        INPUTS:
        -------
        1. system_param: dictionary of user-defined system parameters
            [see hops_system.py]
            * HAMILTONIAN
            * GW_SYSBATH
            * CORRELATION_FUNCTION_TYPE
            * LOPERATORS
            * CORRELATION_FUNCTION
        2. eom_parameters: dictionary of user-defined eom parameters
            [see hops_eom.py]
            * EQUATION_OF_MOTION
            * ADAPTIVE_H
            * ADAPTIVE_S
            * DELTA_H
            * DELTA_S
            * UPDATE_STEP
        3. noise_parameters: dictionary of user-defined noise parameters
            [see hops_noise.py]
            * SEED
            * MODEL
            * TLEN
            * TAU
            * INTERP
        4. hierarchy_parameters: dictionary of user-defined hierarchy parameters
            [see hops_hierarchy.py]
            * MAXHIER
            * STATIC_FILTERS
        5. integration_parameters: dictionary of user-defined integration parameters
            [see integrator_rk.py]
            * INTEGRATOR
            * INCHWORM
            * INCHWORM_MIN


        PROPERTIES:
        -----------
        1. t         - the current time of the simulation
        2. psi       - the current system wave function
        3. psi_traj  - the full trajectory of the system wave function
        4. phi       - the current full state of the hierarchy
        5. t_axis    - the time points at which psi_traj has been stored
        """
        # Variables for the current state of the system
        self._phi = []
        self._z_mem = []
        self._t = 0
        # Instantiate sub-classes
        # -----------------------
        eom = HopsEOM(eom_param)
        system = HopsSystem(system_param)
        hierarchy = HopsHierarchy(hierarchy_param, system.param)
        self.noise_param = noise_param
        self.basis = HopsBasis(system, hierarchy, eom)
        self.storage = HopsStorage(self.basis.eom.param['ADAPTIVE'],storage_param)
        self.noise1 = prepare_noise(noise_param, self.basis.system.param)
        if "L_NOISE2" in system.param.keys():
            noise_param2 = copy.copy(noise_param)
            rand = np.random.RandomState(seed=noise_param['SEED'])
            noise_param2['SEED'] = int(rand.randint(0, 2 ** 20, 1)[0])
            self.noise2 = prepare_noise(noise_param2, self.basis.system.param, flag=2)
        else:
            noise_param2 = {
                "TLEN": noise_param["TLEN"],
                "TAU": noise_param["TAU"],
                "MODEL": "ZERO",
            }
            self.noise2 = prepare_noise(noise_param2, self.basis.system.param, flag=1)

        # Define integration method
        # -------------------------
        self._early_integrator = integration_param['EARLY_ADAPTIVE_INTEGRATOR']
        self._static_basis = integration_param["STATIC_BASIS"]
        self._early_steps = integration_param["EARLY_INTEGRATOR_STEPS"]
        if integration_param['EARLY_ADAPTIVE_INTEGRATOR'] == 'INCH_WORM':
            self._inchworm_cap = integration_param["INCHWORM_CAP"]

        if integration_param["INTEGRATOR"] == "RUNGE_KUTTA":
            from pyhops.dynamics.integrator_rk import (
                runge_kutta_step,
                runge_kutta_variables,
            )

            self.step = runge_kutta_step
            self.integration_var = runge_kutta_variables
            self.integrator_step = 0.5
        else:
            raise UnsupportedRequest(
                ("Integrator of type " + str(integration_param["INTEGRATOR"])),
                type(self).__name__,
            )

        # LOCKING VARIABLE
        self.__initialized__ = False

    def initialize(self, psi_0):
        """
        This function initializes the trajectory module by ensuring that
        each sub-component is prepared to begin propagating a trajectory.

        PARAMETERS
        ----------
        1. psi_0 : np.array
                   Wave function at initial time

        RETURNS
        -------
        None
        """
        psi_0 = np.array(psi_0, dtype=np.complex128)

        if not self.__initialized__:
            # Initialize Noise
            # ----------------
            if not self.noise1.__locked__:
                self.noise1.prepare_noise()
            if not self.noise2.__locked__:
                self.noise2.prepare_noise()

            # Prepare the derivative
            # ----------------------
            self.dsystem_dt = self.basis.initialize(psi_0)

            # Initialize System State
            # -----------------------
            self.storage.n_dim = self.basis.system.param["NSTATES"]
            phi_tmp = np.zeros(self.n_hier * self.n_state, dtype=np.complex128)
            phi_tmp[: self.n_state] = np.array(psi_0)[self.state_list]
            self.z_mem = np.zeros(len(self.basis.system.param["L_NOISE1"]))
            if self.basis.adaptive:
                if self._static_basis is None:
                    # Update Basis
                    z_step = self._prepare_zstep(self.noise1.param['T_AXIS'][2],
                                                 self.z_mem)
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
                        set(self.state_list).union(set(self._static_basis[0])))
                    list_add_state = set(list_state_new) - set(list_stable_state)
                    state_update = (list_state_new, list_stable_state, list_add_state)

                    list_stable_aux = self.auxiliary_list
                    list_aux_new = list(
                        set(self.auxiliary_list).union(set(self._static_basis[1])))
                    list_add_aux = set(list_aux_new) - set(list_stable_aux)
                    aux_update = (list_aux_new, list_stable_aux, list_add_aux)

                    (phi_tmp, dsystem_dt) = self.basis.update_basis(
                        phi_tmp, state_update, aux_update
                    )
                    self.dsystem_dt = dsystem_dt
            else:
                aux_update = [[],[],[]]

            # Store System State
            # ------------------
            self.storage.store_step(
                phi_new=phi_tmp, aux_new=aux_update, state_list=self.state_list, t_new=0, z_mem_new=self.z_mem
            )
            self.t = 0
            self.phi = phi_tmp

            # Lock Trajectory
            # ---------------
            self.__initialized__ = True
        else:
            raise LockedException("HopsTrajectory.initialize()")

    def make_adaptive(self, delta_h=1e-4, delta_s=1e-4, update_step=1):
        """
        This is a convenience function for transforming a not-yet-initialized
        HOPS trajectory from a standard hops to an adaptive HOPS approach.

        PARAMETERS
        ----------
        1. delta_h : float < 1
                     The value of the adaptive grid for the hierarchy of auxiliary nodes
        2. delta_s : float < 1
                     The value of the adaptive grid in the system basis
        3. update_step : int
                         The number of time points between updates to the adaptive basis

        RETURNS
        -------
        None
        """
        if not self.__initialized__:
            if delta_h > 0:
                self.basis.eom.param["ADAPTIVE"] = True
                self.basis.eom.param["ADAPTIVE_H"] = True
                self.basis.eom.param["DELTA_H"] = delta_h
                self.basis.eom.param["UPDATE_STEP"] = update_step
                self.storage.adaptive = True

            if delta_s > 0:
                self.basis.eom.param["ADAPTIVE"] = True
                self.basis.eom.param["ADAPTIVE_S"] = True
                self.basis.eom.param["DELTA_S"] = delta_s
                self.storage.adaptive = True
                self.basis.eom.param["UPDATE_STEP"] = update_step

        else:
            raise TrajectoryError("Calling make_adaptive on an initialized trajectory")

    def propagate(self, t_advance, tau):
        """
        This is the function that perform integration along fixed time-points.
        The kind of integration that is performed is controlled by 'step' which
        was setup in the initialization.

        PARAMETERS
        ----------
        1. t_advance : float
                       How far out in time the calculation will run
        2. tau : float
                 the time step

        RETURNS
        -------
        None
        """

        # check and prepare time axis
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

        # Perform integration
        # -------------------
        for (i_t, t) in enumerate(t_axis):
            var_list = self.integration_var(self.phi,self.z_mem,self.t, self.noise1,
                                            self.noise2, tau, self.storage)
            phi, z_mem = self.step(self.dsystem_dt,  **var_list)
            phi = self.normalize(phi)

            if self.basis.adaptive:
                aux_update = [self.auxiliary_list, self.auxiliary_list, []]
                # Check for early time integration
                if i_t <= self._early_steps:

                    # Early Integrator: Inch Worm
                    if self._early_integrator == 'INCH_WORM':
                        # Define New Basis
                        z_step = self._prepare_zstep(tau, z_mem)
                        (state_update, aux_update) = self.basis.define_basis(phi, tau,
                                                                             z_step)

                        # Update basis for new step
                        step_num = 0
                        while len(state_update[2]) > 0 or len(aux_update[2]) > 0:
                            state_update, aux_update, phi = self.inchworm_integrate(
                                state_update, aux_update, tau
                            )
                            step_num += 1
                            if step_num >= self._inchworm_cap:
                                break

                        # update basis
                        (phi, self.dsystem_dt) = self.basis.update_basis(
                            phi, state_update, aux_update
                        )

                    elif self._early_integrator == 'STATIC':
                        if i_t == self._early_steps:
                            z_step = self._prepare_zstep(tau, z_mem)
                            (state_update, aux_update) = self.basis.define_basis(phi,
                                                                                 tau,
                                                                                 z_step)
                            # update basis
                            (phi, self.dsystem_dt) = self.basis.update_basis(
                                phi, state_update, aux_update
                            )
                elif (i_t + 1) % self.update_step == 0:
                    # Define New Basis
                    z_step = self._prepare_zstep(tau, z_mem)
                    (state_update, aux_update) = self.basis.define_basis(phi, tau,
                                                                         z_step)

                    # update basis
                    (phi, self.dsystem_dt) = self.basis.update_basis(
                        phi, state_update, aux_update
                    )

            else:
                aux_update = [[],[],[]]

            self.storage.store_step(
                phi_new=phi, aux_new=aux_update, state_list=self.state_list, t_new=t,
                z_mem_new=self.z_mem
            )
            self.phi = phi
            self.z_mem = z_mem
            self.t = t

    def _operator(self, op):
        """
        This is the function that acts an operator on the full hierarchy.

        PARAMETERS
        ----------
        1. op : an operator

        RETURNS
        -------
        None
        """
        phi_mat = np.transpose(
            np.reshape(
                self.phi, [int(len(self.phi) / self.n_state), self.n_state], order="C"
            )
        )
        self.phi = np.reshape(
            np.transpose(sp.matmul(op, phi_mat)), len(self.phi)
        )

    def _check_tau_step(self, tau, precision):
        """
        This is the function that checks if tau_step is within precision of
        noise1.param['TAU'] and if tau is greater than or equal to noise1.param['TAU']

        PARAMETERS
        ----------
        1. tau : float
                 a time step
        2. precision : float
                       a constant that defines precision when comparing time points

        RETURNS
        -------
        1. (p_bool1 and p_bool2 and s_bool1 and s_bool2) : boolean
                                                           True if time step consistent
                                                           with noise
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
        This is the function that re-normalizes the wave function at each step
        to correct for loss of norm due to finite numerical accuracy

        PARAMETERS
        ----------
        1. phi : np.array
                 the current full state of the hierarchy

        RETURNS
        -------
        1. phi : np.array
                 the full state of the hierarchy normalized if appropriate
        """
        if self.basis.eom.normalized:
            return phi / np.sqrt(np.sum(np.abs(phi[: self.n_state]) ** 2))
        else:
            return phi

    def inchworm_integrate(self, state_update, aux_update, tau):
        """
        This function performs inchworm integration. 

        PARAMETERS
        ----------
        1. state_update
            a. list_state_new : list
                                list of new states
            b. state_stable : list
                              list of stable states in the current basis
            c. list_add_state : list
                                list of new states that were not in previous state list
        2. aux_update
            d. aux_new : list
                         list of new auxiliaries
            e. stable_aux : list
                            list of stable auxiliaries in the current basis
            f. add_aux : list
                         list of new auxiliaries that were not in the previous aux list
        3. tau : float
                 the time step

        RETURNS
        -------
        1. state_update
            a. list_state_new : list
                           list of new states
            b. state_stable : list
                              list of stable states in the current basis
            c. list_add_state : list
                           list of new states that were not in previous state list
        2. aux_update
            d. aux_new : list
                         list of new auxiliaries
            e. stable_aux : list
                            list of stable auxiliaries in the current basis
            f. add_aux : list
                         list of new auxiliaries that were not in the previous aux list
        3. phi : np.array
                 the full state of the hierarchy normalized if appropriate

        """
        # Unpack update tuples
        (list_state_new, list_stable_state, list_add_state) = state_update
        (aux_new, stable_aux, add_aux) = aux_update

        # Incorporate the new states into the state basis
        state_update = (
            list(set(list_state_new) | set(self.state_list)),
            self.state_list,
            list_add_state,
        )

        # Incorporate the new hierarchy into the hierarchy basis
        aux_update = (
            list(set(aux_new) | set(self.auxiliary_list)),
            self.auxiliary_list,
            add_aux,
        )

        # Update phi and derivative for new basis
        (phi, self.dsystem_dt) = self.basis.update_basis(
            self.phi, state_update, aux_update
        )
        self.phi = phi

        # Perform integration step with extended basis
        var_list = self.integration_var(self.phi,self.z_mem,self.t, self.noise1, self.noise2, tau, self.storage)
        phi, z_mem = self.step(self.dsystem_dt, **var_list)
        phi = self.normalize(phi)

        # define new basis for step in extended space
        z_step = self._prepare_zstep(tau, z_mem)
        (state_update, aux_update) = self.basis.define_basis(phi, tau, z_step)
        return state_update, aux_update, phi

    def _prepare_zstep(self, tau, z_mem):
        """
        This function constructs a compressed version of the noise terms to be used
        in the following time step

        PARAMETERS
        ----------
        1. tau : float
                 the time step
        2. z_mem : list
                   a list of the memory terms
        RETURNS
        -------
        1. z_step : list
                    the list of noise terms (compressed) for the next timestep
        """
        t = self.t
        z_rnd1 = np.max(
            np.abs(self.noise1.get_noise([t + tau, t + tau + tau * 0.5, t + 2 * tau])),
            axis=1,
        )
        z_rnd2 = np.max(
            self.noise2.get_noise([t + tau, t + tau + tau * 0.5, t + 2 * tau]), axis=1
        )

        return [z_rnd1, z_rnd2, z_mem]

    def construct_noise_correlation_function(self, n_l2, n_traj):
        """
        This function uses correlated noise trajectories to reconstruct the full
        noise correlation function
        ** Warning if SEED is not None function will not work properly **
        PARAMETERS
        ----------
        1. n_l2 : int
                  index of the site that the noise is being generated on
        2. n_traj : int
                    the number of trajectories summed over
        Returns
        -------
        1. list_corr_func_1 : list

        2. list_corr_func_2 : list
        """
        if self.noise_param['SEED'] is not None:
            warnings.warn('SEED is not None; Summing over identical noise trajectories'
                          'In order to reconstruct correlation function seed must be None')
        t_axis = np.arange(0, self.noise1.param['TLEN'], self.noise1.param['TAU'])
        list_ct1 = np.zeros(len(t_axis), dtype=np.complex)
        list_ct2 = np.zeros(len(t_axis), dtype=np.complex)
        for _ in np.arange(n_traj):
            noise1 = prepare_noise(self.noise_param, self.basis.system.param)
            noise1.prepare_noise()
            result = np.correlate(noise1.get_noise(t_axis)[n_l2, :],
                                  noise1.get_noise(t_axis)[n_l2, :], mode='full')
            list_ct1 += result[result.size // 2:]
            if "L_NOISE2" in self.basis.system.param.keys():
                noise2 = prepare_noise(self.noise_param, self.basis.system.param,
                                       flag=2)
                noise2.prepare_noise()
                result = np.correlate(noise2.get_noise(t_axis)[n_l2, :],
                                      noise2.get_noise(t_axis)[n_l2, :], mode='full')
                list_ct2 += result[result.size // 2:]
        return list_ct1 / (n_traj * len(t_axis)), list_ct2 / (n_traj * len(t_axis))

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

    @t.setter
    def t(self, t):
        self._t = t
