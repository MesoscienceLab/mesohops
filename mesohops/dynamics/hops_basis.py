import copy
import numpy as np
from scipy import sparse
from pyhops.dynamics.basis_functions_adaptive import (error_sflux_hier,
                                                      error_deriv,
                                                      error_flux_up,
                                                      error_flux_down,
                                                      error_deletion,
                                                      error_sflux_state)
from pyhops.dynamics.basis_functions import determine_error_thresh
from pyhops.util.physical_constants import hbar


__title__ = "Basis Class"
__author__ = "D. I. G. Bennett"
__version__ = "1.2"


class HopsBasis:
    """
    Every HOPS calculation is defined by the HopsSystem, HopsHierarchy, and HopsEOM
    classes (and their associated parameters). These form the basis set for the
    calculation. HopsBasis is the class that contains all of these sub-classes and
    mediates the way the HopsTrajectory interacts with them.
    """

    def __init__(self, system, hierarchy, eom):
        """
        Sets the dictionaries of user-defined parameters that will describe the HOPS
        simulation.

        INPUTS:
        -------
        1. system: dictionary of user inputs
            [see hops_system.py]
            a. HAMILTONIAN
            b. GW_SYSBATH
            c. CORRELATION_FUNCTION_TYPE
            d. LOPERATORS
            e. CORRELATION_FUNCTION
        2. hierarchy: dictionary of user inputs
            [see hops_hierarchy.py]
            f. MAXHIER
            g. TERMINATOR
            h. STATIC_FILTERS
        3. eom: dictionary of user inputs
            [see hops_eom.py]
            i. TIME_DEPENDENCE
            j. EQUATION_OF_MOTION
            k. ADAPTIVE_H
            l. ADAPTIVE_S
            m. DELTA_H
            n. DELTA_S

        RETURNS
        -------
        None
        """
        self.system = system
        self.hierarchy = hierarchy
        self.eom = eom

    def initialize(self, psi_0):
        """
        This function initializes the hierarchy and equations of motion classes
        so that everything is prepared for integration. It returns the
        dsystem_dt function to be used in the integrator.

        PARAMETERS
        ----------
        1. psi_0 : np.array
                  the initial wave function

        RETURNS
        -------
        1. dsystem_dt : function
                       this is the core function for calculating the time-evolution of
                       the wave function
        """
        self.hierarchy.initialize(self.adaptive_h)
        self.system.initialize(self.adaptive_s, psi_0)
        dsystem_dt = self.eom._prepare_derivative(self.system, self.hierarchy)

        return dsystem_dt

    def define_basis(self, Φ, delta_t, z_step):
        """
        This is the function that determines the basis that is needed for a given
        full hierarchy (Φ) in order to construct an approximate derivative with
        error below the specified threshold.

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy
        2. delta_t : float
                     the timestep for the calculation
        3. z_step : list
                    the list of noise terms (compressed) for the next timestep

        RETURNS
        -------
        1. state_update :
            a. list_state_new : list
                                list of states in the new basis (S_1)
            b. list_state_stable : list
                                   list of stable states in the new basis (S_S)
            c. list_state_bound : list
                                  list of boundary states in the new basis (S_B)
        2. hierarchy_update :
            d. list_aux_new : list
                              list of auxiliaries in new basis (H_1)
            e. list_stable_aux : list
                                 list of stable auxiliaries in the new basis (H_S)
            f. list_aux_bound : list
                                list of boundary auxiliaries in the new basis (H_B)

        """
        # ==========================================
        # =======      Calculate Updates      ======
        # ==========================================

        # Calculate New Hierarchy List
        # ----------------------------
        if self.adaptive_h:
            list_aux_stable, list_aux_bound = self._check_hierarchy_list(
                Φ/np.linalg.norm(Φ[:self.n_state]), delta_t, z_step
            )
            list_aux_new = list(set(list_aux_stable) | set(list_aux_bound))
            list_index_stable_aux = [
                self.hierarchy._aux_index(aux) for aux in list_aux_stable
            ]
            list_index_stable_aux.sort()
        else:
            list_aux_new = self.hierarchy.auxiliary_list
            list_aux_stable = self.hierarchy.auxiliary_list
            list_aux_bound = []
            list_index_stable_aux = np.arange(len(self.hierarchy.auxiliary_list))
            E2_flux = None

        # Calculate New State List
        # ------------------------
        if self.adaptive_s:
            list_state_stable, list_state_bound = self._check_state_list(
                Φ/np.linalg.norm(Φ[:self.n_state]), delta_t, z_step, list_index_stable_aux
            )
            list_state_new = list(set(list_state_stable) | set(list_state_bound))
            list_state_stable.sort()
        else:
            list_state_new = list(set(self.system.state_list))
            list_state_bound = []
            list_state_stable = list_state_new

        return [
            (list_state_new, list_state_stable, list_state_bound),
            (list_aux_new, list_aux_stable, list_aux_bound),
        ]

    def update_basis(self, Φ, state_update, aux_update):
        """
        This function updates the derivative function and full hierarchy vector (Φ) for the
        new basis (hierarchy and/or system).

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy
        2. state_update : list
                          list of list containing list_state_new, list_stable_state, and list_add_state
        3. aux_update : list
                        list of list containing list_aux_new, list_stable_aux, and list_add_aux

        RETURNS
        -------
        1. Φ_new : np.array
                   the updated full hierarchy
        2. dsystem_dt : function
                        the updated derivative function
        """
        # Unpack input values
        # ===================
        (list_state_new, list_stable_state, list_add_state) = state_update
        (list_aux_new, list_stable_aux, list_add_aux) = aux_update

        # Update State List
        # =================
        flag_update = False
        if set(list_state_new) != set(self.system.state_list):
            flag_update = True
            list_state_previous = copy.deepcopy(self.system.state_list)
            list_absindex_l2_old = copy.deepcopy(self.system.list_absindex_L2)
            self.system.state_list = np.array(list_state_new)
        else:
            list_state_previous = self.system.state_list
            list_absindex_l2_old = self.system.list_absindex_L2

        # Update Hierarchy List
        # =====================
        if set(list_aux_new) != set(self.hierarchy.auxiliary_list):
            flag_update = True
            # Update Auxiliary List
            list_old_aux = self.hierarchy.auxiliary_list
            list_stable_aux_old_index = [aux._index for aux in list_stable_aux]
            self.hierarchy.auxiliary_list = list_aux_new
        else:
            list_old_aux = self.hierarchy.auxiliary_list
            list_stable_aux_old_index = [aux._index for aux in list_stable_aux]

        # Update state of calculation for new basis
        # =========================================
        if flag_update:

            # Define permutation matrix from old basis --> new basis
            # ------------------------------------------------------
            permute_aux_row = []
            permute_aux_col = []
            nstate_old = len(list_state_previous)
            list_index_old_stable_state = np.array(
                [
                    i_rel
                    for (i_rel, i_abs) in enumerate(list_state_previous)
                    if i_abs in list_stable_state
                ]
            )
            list_index_new_stable_state = np.array(
                [
                    i_rel
                    for (i_rel, i_abs) in enumerate(self.system.state_list)
                    if i_abs in list_stable_state
                ]
            )

            for (index_stable, aux) in enumerate(list_stable_aux):
                permute_aux_row.extend(
                    self.hierarchy._aux_index(aux) * self.n_state
                    + list_index_new_stable_state
                )
                permute_aux_col.extend(
                    list_stable_aux_old_index[index_stable] * nstate_old + list_index_old_stable_state
                )
            list_stable_aux_new_index = [self.hierarchy._aux_index(aux) for aux in list_stable_aux]

            # Update phi
            # ----------
            norm_old = np.linalg.norm(Φ[:len(list_state_previous)])
            Φ_new = np.zeros(self.n_hier * self.n_state, dtype=np.complex128)
            Φ_new[permute_aux_row] = Φ[permute_aux_col]
            Φ_new = norm_old * Φ_new / np.linalg.norm(Φ_new[:self.n_state])

            # Update dsystem_dt
            # -----------------
            dsystem_dt = self.eom._prepare_derivative(
                self.system,
                self.hierarchy,
                list_stable_aux,
                list_add_aux,
                list_stable_state,
                list_absindex_l2_old,
                len(list_old_aux) * nstate_old,
                [permute_aux_row, permute_aux_col, list_stable_aux_old_index, list_stable_aux_new_index, len(list_old_aux)],
                update=True,
            )

            return (Φ_new, dsystem_dt)
        else:
            return (Φ, self.eom.dsystem_dt)

    def _check_error(self, Φ):
        """
        Returns an upper-bound on the adaptive error using the current basis.

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy

        RETURNS
        -------
        1. error_bound : list
                         list of errors
        """
        # Construct error from missing state terms
        e_state = np.sum(np.abs(error_sflux_hier(Φ, self.system.state_list,
                                         self.n_state, self.n_hier,
                                         self.system.param["SPARSE_HAMILTONIAN"][:,
                                         self.system.state_list]))**2)

        # Construct error from missing flux to boundary hierarchy (+1)
        basis_filter = np.ones([self.n_hmodes, self.n_hier])
        up_filter = np.ones([self.n_hmodes, self.n_hier])
        for aux in self.hierarchy.auxiliary_list:
            array_index = np.array([list(self.system.list_absindex_mode).index(mode)
                                    for mode in
                                    aux.dict_aux_p1.keys() if
                                    mode in self.system.list_absindex_mode],
                                   dtype=int)

            basis_filter[array_index, aux._index] = 0

            if aux._sum >= self.hierarchy.param["MAXHIER"]:
                up_filter[:, aux._index] = 0
        filter_up = basis_filter * up_filter

        e_flux_up = np.sum(np.abs(filter_up*error_flux_up(Φ, self.n_state,
                                                                  self.n_hier,
                                             self.n_hmodes,
                                      self.system.w,
                                      self.system.list_state_indices_by_hmode,
                                      self.system.list_absindex_mode,
                                      self.hierarchy.auxiliary_list,
                                      self.hierarchy.param["MAXHIER"],
                                      self.hierarchy.param["STATIC_FILTERS"]))**2)

        # Construct error from missing flux to boundary hierarchy (-1)
        basis_filter_d = np.ones([self.n_hmodes, self.n_hier])
        down_filter = np.zeros([self.n_hmodes, self.n_hier])
        for aux in self.hierarchy.auxiliary_list:
            array_index = np.array([list(self.system.list_absindex_mode).index(mode)
                                    for mode in
                                    aux.dict_aux_m1.keys() if
                                    mode in self.system.list_absindex_mode],
                                   dtype=int)

            basis_filter_d[array_index, aux._index] = 0

            array_index = np.array([list(self.system.list_absindex_mode).index(mode)
                                    for mode in
                                    aux.keys() if
                                    mode in self.system.list_absindex_mode],
                                   dtype=int)

            down_filter[array_index, aux._index] = 1

        filter_down = basis_filter_d*down_filter

        e_flux_down = np.sum(filter_down*np.abs(error_flux_down(Φ,self.n_state,
                                                                self.n_hier,
                                       self.n_hmodes,
                                       self.system.list_state_indices_by_hmode,
                                       self.system.list_absindex_mode,
                                       self.hierarchy.auxiliary_list,
                                       self.system.g, self.system.w, "H"))**2)

        return [e_state, e_flux_down, e_flux_up, np.sqrt(e_flux_up+
        e_flux_down+e_state)]

    def _check_state_list(self, Φ, delta_t, z_step, list_index_aux_stable):
        """
        This is a function that determines the states which should be
        included in the adaptive integration for the next time point.

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy
        2. delta_t : float
                     the timestep for the calculation
        3. z_step : list
                    the list of noise terms (compressed) for the next timestep
        4. list_index_aux_stable : list
                                   a list of relative indices for the stable auxiliaries

        RETURNS
        -------
        1. list_state_stable : list
                               list of stable states (absolute state index, S_S)
        2. list_state_boundary : list
                                 a list of the boundary states (absolute state index, S_B)
        """
        # Define Constants
        # ----------------
        delta_state = self.delta_s

        # CONSTRUCT STABLE STATE (S_S)
        # ============================

        # Construct Error For Excluding Member of S0
        # ------------------------------------------
        error_by_state = self.error_stable_state(
            Φ, delta_t, z_step, list_index_aux_stable
        )

        # Determine the Stable States (S_S = S_0 & S_1)
        # ---------------------------------------------
        list_index_stable, list_state_stable = self._determine_basis_from_list(
            error_by_state, delta_state / 2, self.system.state_list
        )

        # CONSTRUCT BOUNDARY STATE (S_B)
        # ==============================

        # Establish the error available for the boundary states
        # -----------------------------------------------------
        stable_error = np.sqrt(
            np.max([
                np.sum(error_by_state ** 2) - np.sum(error_by_state[list_index_stable] ** 2),
                0])
        )
        bound_error = delta_state - stable_error

        # Construct Error for Excluding Member of S0^C
        # --------------------------------------------
        list_index_nonzero, list_error_nonzero = self.error_boundary_state(
            Φ, list_index_stable, list_index_aux_stable
        )

        # Determine Boundary States
        # -------------------------
        if len(list_error_nonzero) > 0:
            _, list_state_boundary = self._determine_basis_from_list(
                list_error_nonzero, bound_error, list_index_nonzero
            )
        else:
            list_state_boundary = []

        # Check for overlap with populated states
        # ---------------------------------------
        list_state_boundary = list(
            set(list_state_boundary) - set(self.system.state_list)
        )
        list_state_boundary.sort()

        return (
            np.array(list_state_stable, dtype=np.int),
            np.array(list_state_boundary, dtype=np.int),
        )

    def _check_hierarchy_list(self, Φ, delta_t, z_step):
        """
        This is a function that determines the auxiliaries which should be
        included in the adaptive integration for the next time point.

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy
        2. delta_t : float
                     the timestep for the calculation
        3. z_step : list
                    the list of noise terms (compressed) for the next timestep

        RETURNS
        -------
        1. list_aux_stable : list
                             a list of the stable auxiliaries (H_S)
        2. list_aux_boundary : list
                               a list of auxiliaries that share a boundary with stable
                               auxiliaries (H_B)
        """
        # Define Constants
        # ----------------
        delta_hier = self.delta_h


        # CONSTRUCT STABLE HIERARCHY
        # ==========================

        # Construct Error For Excluding Member of H0
        # ------------------------------------------
        error_by_aux, list_e2_kflux = self.hier_stable_error(Φ, delta_t, z_step)

        # Determine the Stable Auxiliaries (H_S = H_0 & H_1)
        # --------------------------------------------------
        list_index_stable, list_aux_stable = self._determine_basis_from_list(
            error_by_aux, delta_hier / 2, self.hierarchy.auxiliary_list
        )

        # CONSTRUCT BOUNDARY HIERARCHY
        # ============================

        # Establish the error available for the boundary auxiliaries
        # ----------------------------------------------------------
        stable_error = np.sqrt(
            np.max([
                np.sum(error_by_aux ** 2) - np.sum(error_by_aux[list_index_stable] ** 2),
                0])
        )
        bound_error = delta_hier - stable_error

        # Construct Error For Excluding Members of H0^C
        # ---------------------------------------------
        E2_flux_up = list_e2_kflux[0][:, list_index_stable]
        E2_flux_down = list_e2_kflux[1][:, list_index_stable]

        # Determine the Boundary Auxiliaries (H_B = H0^C & H_1)
        # -----------------------------------------------------
        list_aux_up, list_aux_down = self._determine_boundary_hier(
            [E2_flux_up, E2_flux_down], list_index_stable, bound_error
        )

        # Check Boundary Set For Duplication
        # ----------------------------------
        # NOTE: Theoretically, the boundary should not contain any members of H0, but
        #       when we implemented that in practice it resulted in a huge explosion
        #       of the number of inch wormsteps that were required.
        list_aux_boundary = list(
            (set(list_aux_up) | set(list_aux_down)) - set(self.hierarchy.auxiliary_list)
        )

        # Filter Boundary Set for Auxiliaries That Are Not Part of H_T
        # ------------------------------------------------------------
        if len(list_aux_boundary) > 0:
            list_aux_boundary = self.hierarchy.filter_aux_list(list_aux_boundary)
        return list_aux_stable, list_aux_boundary

    def _determine_boundary_hier(
        self, list_e2_kflux, list_index_aux_stable, bound_error
    ):
        """
        This function determines the set of boundary auxiliaries for the next time step.

        PARAMETERS
        ----------
        1. list_e2_kflux : list
                           a list of list containing the error values for the flux up
                           and flux down terms
        2. list_index_aux_stable : list
                                   a list of the indices for stable auxiliaries
        3. bound_error : float
                         the boundary error value

        RETURNS
        -------
        1. list_aux_up : list
                         a list of the flux up auxiliaries
        2. list_aux_down : list
                           a list of the flux down auxiliaries
        """
        # Construct constants
        # -------------------
        E2_flux_up = list_e2_kflux[0]
        E2_flux_down = list_e2_kflux[1]

        # Find the error threshold for edge auxiliaries
        # ---------------------------------------------
        E1_nonzero_flux = E2_flux_up[E2_flux_up != 0]
        sorted_error = np.sort(
            np.append(E1_nonzero_flux, E2_flux_down[E2_flux_down != 0])
        )
        error_thresh = determine_error_thresh(sorted_error, bound_error)

        # Loop over residual fluxes to identify boundary auxiliaries
        # ----------------------------------------------------------
        list_aux_up = [
            self.hierarchy.auxiliary_list[list_index_aux_stable[i_aux]].e_step(
                self.system.list_absindex_mode[i_mode_rel], 1
            )
            for (i_mode_rel, i_aux) in zip(*np.where(E2_flux_up > error_thresh))
        ]

        list_aux_down = [
            self.hierarchy.auxiliary_list[list_index_aux_stable[i_aux]].e_step(
                self.system.list_absindex_mode[i_mode_rel], -1
            )
            for (i_mode_rel, i_aux) in zip(*np.where(E2_flux_down > error_thresh))
        ]
        return (list_aux_up, list_aux_down)

    def _determine_basis_from_list(self, error_by_member, max_error, list_member):  # ask about being a static method
        """
        This function determines the members of a list that must be kept in order
        for the total error (terms that are dropped) to be below the max_error value.

        PARAMETERS
        ----------
        1. error_by_member : np.array
                             a list of error values
        2. max_error : float
                       the maximum error value
        3. list_member : np.array
                         a list of members

        RETURNS
        -------
        1. list_index : np.array
                        a list of indices for the members
        2. list_new_member : list
                             a list of the members
        """
        error_thresh = determine_error_thresh(np.sort(error_by_member), max_error)
        list_index = np.where(error_by_member > error_thresh)[0]
        list_new_member = [list_member[i_aux] for i_aux in list_index]
        return (list_index, list_new_member)

    def error_boundary_state(self, Φ, list_index_stable, list_index_aux_stable):
        """
        This function determines the error associated with neglecting flux into n not
        a member of S_t. This corresponds to equations 43-45 in arXiv:2008.06496.

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy vector
        2. list_index_stable : list
                               a list of the stable states
        3. list_index_aux_stable : list
                                   a list relative indices for the stable auxiliaries

        RETURNS
        -------
        1. list_index_nonzero : list
                                a list of the nonzero state boundary indices
        2. list_error_nonzero : list
                                a list of the nonzero state boundary error values
        """
        if len(list_index_stable) < self.system.param["NSTATES"]:
            # Remove aux components from H0\H1
            # -------------------------------------
            C2_phi = np.zeros([self.n_state, self.n_hier], dtype=np.complex128)
            C2_phi[np.ix_(list_index_stable, list_index_aux_stable)] = np.array(
                Φ
            ).reshape([self.n_state, self.n_hier], order="F")[
                np.ix_(list_index_stable, list_index_aux_stable)
            ]

            # Construct Hamiltonian
            # ---------------------------------------------------------
            list_s0 = np.array(self.system.state_list)

            # First construct ST<--S0 Hamiltonian
            H2_sparse_hamiltonian = self.system.param["SPARSE_HAMILTONIAN"][:, list_s0]

            # Remove components that map S0<--S0
            H2_sparse_subset = sparse.coo_matrix(
                H2_sparse_hamiltonian[np.ix_(list_s0, range(len(list_s0)))]
            )
            H2_removal = sparse.csc_matrix(
                (
                    H2_sparse_subset.data,
                    (
                        list_s0[H2_sparse_subset.row],
                        np.arange(len(list_s0))[H2_sparse_subset.col],
                    ),
                ),
                shape=H2_sparse_hamiltonian.shape,
            )
            H2_sparse_hamiltonian = H2_sparse_hamiltonian - H2_removal

            # Determine Boundary States
            # -------------------------
            C2_phi_deriv = abs(
                H2_sparse_hamiltonian @ sparse.csc_matrix(C2_phi / hbar)
            ).power(2)
            C1_sum_deriv = np.sum(C2_phi_deriv, axis=1)
            return (
                C1_sum_deriv.nonzero()[0],
                np.sqrt(np.array(C1_sum_deriv[C1_sum_deriv.nonzero()])[0]),
            )

        else:
            return [], []

    def error_stable_state(self, Φ, delta_t, z_step, list_index_aux_stable):
        """
        This function finds the total error associated with neglecting n in the state
        basis S_t. This includes the error associated with deleting a state from the basis,
        the error associated with losing all flux into a state, and the error associated
        with losing all flux out of a state. This corresponds to equations 37-42 in
        arXiv:2008.06496.


        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy vector
        2. delta_t : float
                     the timestep for the calculation
        3. z_step : list
                    the list of noise terms (compressed) for the next timestep
        4. list_index_aux_stable : list
                                   a list of relative indices for the stable
                                   auxiliaries (A_p)

        RETURNS
        -------
        1. error : np.array
                   list of error associated with removing each state
        """
        M2_state_from_mode = np.zeros([self.n_state, self.system.n_hmodes])
        M2_state_from_mode[
            self.system.list_state_indices_by_hmode[:, 0],
            np.arange(np.shape(self.system.list_state_indices_by_hmode)[0]),
        ] = 1

        # Construct the error terms
        # -------------------------
        E2_deriv_state = error_deriv(self.eom.dsystem_dt, Φ, z_step,
                                     self.n_state, self.n_hier,
                                     list_index_aux_stable)[:, list_index_aux_stable]
        E2_deletion = error_deletion(Φ, delta_t, self.n_state, self.n_hier)[:,
                      list_index_aux_stable]
        E1_state_flux = error_sflux_state(Φ, self.n_state, self.n_hier,
                                          self.system.param["SPARSE_HAMILTONIAN"],
                                          list_index_aux_stable,
                                          self.system.state_list)
        E2_flux_down = error_flux_down(Φ, self.n_state, self.n_hier, self.n_hmodes,
                                       self.system.list_state_indices_by_hmode,
                                       self.system.list_absindex_mode,
                                       self.hierarchy.auxiliary_list,
                                       self.system.g, self.system.w, "S")[:, list_index_aux_stable]
        E2_flux_up = error_flux_up(Φ, self.n_state, self.n_hier, self.n_hmodes,
                                      self.system.w,
                                      self.system.list_state_indices_by_hmode,
                                      self.system.list_absindex_mode,
                                      self.hierarchy.auxiliary_list,
                                      self.hierarchy.param["MAXHIER"],
                                      self.hierarchy.param["STATIC_FILTERS"])[:, list_index_aux_stable]
        # Map flux_up error from mode to state space
        E2_flux_up = np.sqrt(M2_state_from_mode @ E2_flux_up ** 2)

        # Compress the error onto the state/mode axis
        # -------------------------------------------
        return np.sqrt(
            np.sum(E2_deriv_state ** 2, axis=1)
            + np.sum(E2_deletion ** 2, axis=1)
            + np.sum(E2_flux_up ** 2, axis=1)
            + np.sum(E2_flux_down ** 2, axis=1)
            + E1_state_flux ** 2
        )

    def hier_stable_error(self, Φ, delta_t, z_step):
        """
        This function finds the total error associated with removing k in A_t.
        This corresponds to the sum of equations 29,30,31,33, and 34 in
        arXiv:2008.06496.

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy vector
        2. delta_t : float
                     the timestep for the calculation
        3. z_step : list
                    the list of noise terms (compressed) for the next timestep

        RETURNS
        -------
        1. error : np.array
                   list of error associated with removing each auxiliary in A_t
        2. E2_flux_up : np.array
                        the error induced by neglecting flux from A_t (or A_p)
                        to auxiliaries with lower summed index in A_t^C.
        3. E2_flux_down : np.array
                          the error induced by neglecting flux from A_t (or A_p)
                          to auxiliaries with higher summed index in A_t^C.
        """
        # Construct the error terms
        # -------------------------
        E2_deletion = error_deletion(Φ, delta_t, self.n_state, self.n_hier)
        E2_deriv_self = error_deriv(self.eom.dsystem_dt, Φ, z_step,
                                          self.n_state, self.n_hier)
        E1_flux_state = error_sflux_hier(Φ, self.system.state_list,
                                         self.n_state, self.n_hier,
                                         self.system.param["SPARSE_HAMILTONIAN"])
        E2_flux_up = error_flux_up(Φ, self.n_state, self.n_hier, self.n_hmodes,
                                      self.system.w,
                                      self.system.list_state_indices_by_hmode,
                                      self.system.list_absindex_mode,
                                      self.hierarchy.auxiliary_list,
                                      self.hierarchy.param["MAXHIER"],
                                      self.hierarchy.param["STATIC_FILTERS"])
        E2_flux_down = error_flux_down(Φ, self.n_state, self.n_hier, self.n_hmodes,
                                       self.system.list_state_indices_by_hmode,
                                       self.system.list_absindex_mode,
                                       self.hierarchy.auxiliary_list,
                                       self.system.g, self.system.w, "H")

        # Compress the error onto the aux axis
        # ------------------------------------
        return (
            np.sqrt(
                np.sum(E2_deriv_self ** 2, axis=0)
                + np.sum(E2_deletion ** 2, axis=0)
                + np.sum(E2_flux_down ** 2, axis=0)
                + np.sum(E2_flux_up ** 2, axis=0)
                + E1_flux_state
            ),
            [E2_flux_up, E2_flux_down],
        )

    @property
    def n_hmodes(self):
        return self.system.n_hmodes

    @property
    def n_state(self):
        return self.system.size

    @property
    def n_hier(self):
        return self.hierarchy.size

    @property
    def adaptive(self):
        return self.eom.param["ADAPTIVE"]

    @property
    def adaptive_h(self):
        return self.eom.param["ADAPTIVE_H"]

    @property
    def adaptive_s(self):
        return self.eom.param["ADAPTIVE_S"]

    @property
    def delta_h(self):
        return self.eom.param["DELTA_H"]

    @property
    def delta_s(self):
        return self.eom.param["DELTA_S"]
