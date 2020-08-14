import scipy as sp
import numpy as np
from mesohops.dynamics.eom_functions import compress_zmem
from mesohops.dynamics.hops_basis import HopsBasis
from mesohops.dynamics.hops_eom import HopsEOM
from mesohops.dynamics.hops_hierarchy import HopsHierarchy
from mesohops.dynamics.hops_system import HopsSystem
from mesohops.dynamics.prepare_functions import prepare_noise
from mesohops.dynamics.hops_storage import TrajectoryStorage, AdaptiveTrajectoryStorage
from mesohops.util.exceptions import UnsupportedRequest, LockedException, TrajectoryError
from mesohops.util.physical_constants import precision  # Constant

__title__ = "HOPS"
__author__ = "D. I. G. Bennett, Leo Varvelo"
__version__ = "1.0"


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
        integration_param=dict(
            INTEGRATOR="RUNGE_KUTTA", INCHWORM=False, INCHWORM_MIN=5
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

        FUNCTIONS:
        ----------
        1. initialize(psi_0): this function prepares the class for actually running a
                              a calculation by instantiating a specific hierarchy, pre-
                              paring the EOM matrices, and defining the initial
                              condition.
        2. propagate(t,dt): this function calculates the time evolution of psi(t)
                            according to the EOM from the current time to the time
                            "t" in steps of "dt."

        PROPERTIES:
        -----------
        1. t         - the current time of the simulation
        2. psi       - the current system wave function
        3. psi_traj  - the full trajectory of the system wave function
        4. phi       - the current full state of the hierarchy
        5. t_axis    - the time points at which psi_traj has been stored
        """
        # Instantiate sub-classes
        # -----------------------
        eom = HopsEOM(eom_param)
        system = HopsSystem(system_param)
        hierarchy = HopsHierarchy(hierarchy_param, system.param)
        self.basis = HopsBasis(system, hierarchy, eom)
        self.noise1 = prepare_noise(noise_param, self.basis.system.param)
        if "L_NOISE2" in system.param.keys():
            self.noise2 = prepare_noise(noise_param, self.basis.system.param, flag=2)
        else:
            noise_param2 = {
                "TLEN": noise_param["TLEN"],
                "TAU": noise_param["TAU"],
                "MODEL": "ZERO",
            }
            self.noise2 = prepare_noise(noise_param2, self.basis.system.param, flag=1)

        # Define integration method
        # -------------------------
        self._inchworm = integration_param["INCHWORM"]
        self._inchworm_count = 0
        self._inchworm_min = integration_param["INCHWORM_MIN"]

        if integration_param["INTEGRATOR"] == "RUNGE_KUTTA":
            from mesohops.dynamics.integrator_rk import (
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

    def initialize(self, psi_0, store_aux=False):
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

            # Prepare Storage
            # ---------------
            if self.basis.adaptive:
                self.storage = AdaptiveTrajectoryStorage()
            else:
                self.storage = TrajectoryStorage()
            self.storage.store_aux = store_aux

            # Initialize System State
            # -----------------------
            self.storage.n_dim = self.basis.system.param["NSTATES"]
            phi_tmp = np.zeros(self.n_hier * self.n_state, dtype=np.complex128)
            phi_tmp[: self.n_state] = np.array(psi_0)[self.state_list]
            z_mem = np.zeros(len(self.basis.system.param["L_NOISE1"]))
            if self.basis.adaptive:
                # Update Basis
                z_step = self.noise1.get_noise([0])[:, 0]
                (state_update, aux_update) = self.basis.define_basis(phi_tmp, 1, z_step)
                (phi_tmp, dsystem_dt) = self.basis.update_basis(
                    phi_tmp, state_update, aux_update
                )
                self.dsystem_dt = dsystem_dt
            else:
                aux_update = [[],[],[]]

            # Store System State
            # ------------------
            self._store_step(
                self.storage, phi_tmp, z_mem, aux_update, self.state_list, 0
            )

            # Lock Trajectory
            # ---------------
            self.__initialized__ = True
        else:
            raise LockedException("HopsTrajectory.initialize()")

    def make_adaptive(self, delta_h=1e-4, delta_s=1e-4):
        """
        This is a convenience function for transforming a not-yet-initialized
        HOPS trajectory from a standard hops to an adaptive HOPS approach.

        PARAMETERS
        ----------
        1. delta_h : float < 1
                     The value of the adaptive grid for the hierarchy of auxiliary nodes
        2. delta_s : float < 1
                     The value of the adaptive grid in the system basis

        RETURNS
        -------
        None
        """
        if not self.__initialized__:
            if delta_h > 0:
                self.basis.eom.param["ADAPTIVE"] = True
                self.basis.eom.param["ADAPTIVE_H"] = True
                self.basis.eom.param["DELTA_H"] = delta_h

            if delta_s > 0:
                self.basis.eom.param["ADAPTIVE"] = True
                self.basis.eom.param["ADAPTIVE_S"] = True
                self.basis.eom.param["DELTA_S"] = delta_s

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
        t0 = self.storage.t
        if self.noise1.param["TAU"] is not None:
            if self._check_tau_step(tau, precision):
                t_axis = t0 + np.arange(0, int(np.ceil(t_advance / tau))) * tau
                print("Integration from ", t0, " to ", np.max(t_axis) + tau)

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
        for t in t_axis:
            var_list = self.integration_var(self.storage, self.noise1, self.noise2, tau)
            phi, z_mem = self.step(self.dsystem_dt, **var_list)
            phi = self.normalize(phi)

            if self.basis.adaptive:
                # Define New Basis
                z_step = self._prepare_zstep(tau, z_mem)
                (state_update, aux_update) = self.basis.define_basis(phi, tau, z_step)

                # Extend basis until self-consistent during time-step
                if self.inchworm:
                    self._inchworm_count += 1

                    # Update basis for new step
                    while len(state_update[2]) > 0 or len(aux_update[2]) > 0:
                        state_update, aux_update, phi = self.inchworm_integrate(
                            state_update, aux_update, tau
                        )

                # update basis
                (phi, self.dsystem_dt) = self.basis.update_basis(
                    phi, state_update, aux_update
                )

            else:
                aux_update = [[],[],[]]

            self._store_step(
                self.storage,
                phi,
                z_mem,
                aux_update,
                self.basis.system.state_list,
                t + tau,
            )

    @staticmethod
    def _store_step(storage, phi_new, zmem_new, aux_new, state_list, t_new):
        """
        This is the function that inserts data into the HopsStorage class at each
        time point of the simulation.

        PARAMETERS
        ----------
        1. storage : HopsStorage Class
                     instance of HopsStorage
        2. phi_new : np.array
                     the updated full hierarchy
        3. zmem_new : list
                      the new list of zmem
        4. aux_new : list
                     a list of list defining the auxiliaries in the hierarchy basis
                     [aux_crit, aux_bound]
        5. state_list : list
                        the list of current states in the system basis
        6. t_new : float
                   the new time point (t+tau)

        RETURNS
        -------
        None
        """

        if storage.adaptive:
            storage.aux = aux_new
            storage.state_list = state_list

        storage.psi_traj = phi_new[: len(state_list)]
        storage.phi_traj = phi_new
        storage.z_mem = zmem_new
        storage.t_axis = t_new
        storage.phi = phi_new
        storage.t = t_new

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
        self.storage.phi = np.reshape(
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
            self.storage.phi, state_update, aux_update
        )
        self.storage.phi = phi

        # Perform integration step with extended basis
        var_list = self.integration_var(self.storage, self.noise1, self.noise2, tau)
        phi, z_mem = self.step(self.dsystem_dt, **var_list)
        phi = self.normalize(phi)

        # define new basis for step in extended space
        z_step = self._prepare_zstep(tau, z_mem)
        (state_update, aux_update) = self.basis.define_basis(phi, tau, z_step)
        return state_update, aux_update, phi

    def _prepare_zstep(self, tau, z_mem):
        """

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
        z_hat1 = np.conj(z_rnd1[self.basis.system.list_absindex_L2]) + compress_zmem(
            z_mem,
            self.basis.system.list_index_L2_by_hmode,
            self.basis.system.list_absindex_mode,
        )
        z_tmp2 = z_rnd2[self.basis.system.list_absindex_L2]

        return z_hat1 + z_tmp2

    @property
    def t(self):
        return self.storage.t

    @property
    def psi(self):
        return self.storage.phi[: self.n_state]

    @property
    def psi_traj(self):
        return self.storage.psi_traj

    @property
    def phi(self):
        return self.storage.phi

    @property
    def t_axis(self):
        return self.storage.t_axis

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
    def inchworm(self):
        if not self._inchworm and self._inchworm_count > self._inchworm_min:
            return False
        else:
            return True

    @inchworm.setter
    def inchworm(self, inchworm):
        if not self.__initialized__:
            self._inchworm = inchworm
        else:
            raise LockedException("HopsTrajectory.initialize()")
