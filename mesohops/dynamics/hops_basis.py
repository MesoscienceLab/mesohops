import copy
import numpy as np
import scipy as sp
from scipy import sparse
from mesohops.dynamics.eom_functions import operator_expectation
from mesohops.util.physical_constants import hbar
from mesohops.util.exceptions import UnsupportedRequest


__title__ = "Basis Class"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"


class HopsBasis(object):
    """
    Every HOPS calculation is defines by the HopsSystem, HopsHierarchy, and HopsEOM
    classes (and their associated parameters). These form the basis set for the
    calculation. HopsBasis is the class that contains all of these sub-classes and
    mediates the way the HopsTrajectory will interact with them.
    """

    def __init__(self, system, hierarchy, eom):
        """
        INPUTS:
        -------
        1. system: dictionary of user inputs
            [see hops_system.py]
            a. HAMILTONIAN
            b. GW_SYSBATH
            c. CORRELATION_FUNCTION_TYPE
            d. LOPERATORS
            e. CORRELATION_FUNCTION
        2. hierarchy_parameters: dictionary of user inputs
            [see hops_hierarchy.py]
            f. MAXHIER
            g. TERMINATOR
            h. STATIC_FILTERS
        3. eom_parameters: dictionary of user inputs
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
        This is the function that determines what basis is needed for a given
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
                Φ, delta_t, z_step
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
                Φ, delta_t, z_step, list_index_stable_aux
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
            self.hierarchy.auxiliary_list = list_aux_new
        else:
            list_old_aux = self.hierarchy.auxiliary_list

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

            for aux in list_stable_aux:
                permute_aux_row.extend(
                    self.hierarchy._aux_index(aux) * self.n_state
                    + list_index_new_stable_state
                )
                permute_aux_col.extend(
                    list_old_aux.index(aux) * nstate_old + list_index_old_stable_state
                )

            # Update phi
            # ----------
            Φ_new = np.zeros(self.n_hier * self.n_state, dtype=np.complex128)
            Φ_new[permute_aux_row] = Φ[permute_aux_col]

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
                [permute_aux_row, permute_aux_col],
                update=True,
            )

            return (Φ_new, dsystem_dt)
        else:
            return (Φ, self.eom.dsystem_dt)

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
        delta_state = self.delta_s  # * np.linalg.norm(Φ)

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
        delta_hier = self.delta_h  # * np.linalg.norm(Φ)

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
        This function determines the set of boundary auxiliaries for the next time step

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
        error_thresh = self._determine_error_thresh(sorted_error, bound_error)

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

    def _determine_basis_from_list(self, error_by_member, max_error, list_member):
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
        error_thresh = self._determine_error_thresh(np.sort(error_by_member), max_error)
        list_index = np.where(error_by_member > error_thresh)[0]
        list_new_member = [list_member[i_aux] for i_aux in list_index]
        return (list_index, list_new_member)

    @staticmethod
    def _determine_error_thresh(sorted_error, max_error):
        """
        This function determines which error value becomes the error threshold such
        that the sum of all errors below the threshold remains less then max_error.

        PARAMETERS
        ----------
        1. sorted_error : np.array
                          a list of error values
        2. max_error : float
                       the maximum error value

        RETURNS
        -------
        3. error_thresh : float
                          the error value at which the threshold is established

        """
        index_thresh = np.argmax(np.sqrt(np.cumsum(sorted_error ** 2)) > max_error)

        if index_thresh > 0:
            error_thresh = sorted_error[index_thresh - 1]
        else:
            error_thresh = 0.0

        return error_thresh

    def error_boundary_state(self, Φ, list_index_stable, list_index_aux_stable):
        """
        This function determines the error associated with neglecting flux into n' not
        a member of S_t.

        .. math::
            \sum_{\\vec{k} \in \mathcal{H}_{t}} \\left \\vert (\\hat{H} \psi^{(\\vec{k})}_t)_n \\right\\vert^2

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
        This function finds the total error associated with neglecting n' in the state
        basis S_t. This includes the error associated with deleting a state from the basis,
        the error associated with losing all flux into a state, and the error associated
        with losing all flux out of a state.

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy vector
        2. delta_t : float
                     the timestep for the calculation
        3. z_step : list
                    the list of noise terms (compressed) for the next timestep
        4. list_index_aux_stable : list
                                   a list relative indices for the stable auxiliaries (H_S)

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
        E2_deriv_state = self.error_deriv(Φ, z_step, list_index_aux_stable)[
            :, list_index_aux_stable
        ]
        E2_deletion = self.error_deletion(Φ, delta_t)[:, list_index_aux_stable]
        E1_state_flux = self.error_sflux_state(
            Φ, list_index_aux_stable, self.system.state_list
        )
        E2_flux_down = self.error_flux_down(Φ, "S")[:, list_index_aux_stable]
        E2_flux_up = self.error_flux_up(Φ)[:, list_index_aux_stable]
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
        This function finds the total error associated with removing k' in H_t.
        This includes the error associated with deleting a auxiliary from the basis,
        the error associated with losing all flux into a k' auxiliary, and the error
        associated with losing all flux out of a k' auxiliary.

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
                   list of error associated with removing each auxiliary in H_t
        2. E2_flux_up : np.array
                        the error induced by neglecting flux from H_t (or H_S)
                                to auxiliaries with lower summed index in H_t^C.
        3. E2_flux_down : np.array
                          the error induced by neglecting flux from H_t (or H_S)
                                to auxiliaries with higher summed index in H_t^C.
        """
        # Construct the error terms
        # -------------------------
        E2_deletion = self.error_deletion(Φ, delta_t)
        E2_deriv_self = self.error_deriv(Φ, z_step)
        E1_flux_state = self.error_sflux_hier(Φ, self.system.state_list)
        E2_flux_up = self.error_flux_up(Φ)
        E2_flux_down = self.error_flux_down(Φ, "H")

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

    def error_sflux_state(self, Φ, list_index_aux_stable, list_states):
        """
        The error associated with losing all flux out of n’ in S_t. This flux always involves
        changing the state index and, as a result, can be rewritten in terms of the -iH
        component of the self-interaction.

        .. math::
            \sum_{\\vec{k} \in \mathcal{H}_{t}} \sum_{n' \\notin \mathcal{S}_t} \\vert \\hat{H}_{n',n} \\vert^2 \\vert \psi^{(\\vec{k})}_{t,n} \\vert^2

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy vector
        2. list_index_aux_stable : list
                                   a list relative indices for the stable auxiliaries (H_S)
        3. list_states : list
                         the list of current states (absolute index)

        RETURNS
        -------
        1. E1_state_flux : array
                           the error associated with flux out of each state in S_t
        """
        C2_phi = np.asarray(Φ).reshape([self.n_state, self.n_hier], order="F")[
            :, list_index_aux_stable
        ]
        H2_sparse_hamiltonian = self.system.param["SPARSE_HAMILTONIAN"][:, list_states]
        H2_sparse_hamiltonian = H2_sparse_hamiltonian - sp.sparse.diags(
            H2_sparse_hamiltonian.diagonal(0),
            format="csc",
            shape=H2_sparse_hamiltonian.shape,
        )
        return (
            np.array(np.sum(np.abs(H2_sparse_hamiltonian), axis=0))[:, 0]
            * np.sum(np.abs(C2_phi), axis=1)
            / hbar
        )

    def error_sflux_hier(self, Φ, list_s0):
        """
        The error associated with losing all flux terms inside the kth auxiliary to
        states not contained in S_t.

        .. math::
            \sum_{n \\notin \mathcal{S}_{t}} \\left \\vert (\\hat{H} \psi^{(\\vec{k})}_t)_n \\right\\vert^2

        PARAMETERS
        ----------
        1. Φ : np.array
               the current full hierarchy vector
        2. list_s0 : list
                     a list of the current states (absolute index)

        RETURNS
        -------
        1. E2_flux_state : array
                           the error introduced by losing flux within k from S_t to S_t^C
                           for each k in H_t

        """
        # Construct the 2D phi and sparse Hamiltonian
        # -------------------------------------------
        list_s0 = np.array(list_s0)
        C2_phi = np.asarray(Φ).reshape([self.n_state, self.n_hier], order="F")

        H2_sparse_hamiltonian = self.system.param["SPARSE_HAMILTONIAN"][:, list_s0]

        # Remove the components of the Hamiltonian that map S0-->S0
        # ---------------------------------------------------------
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

        return np.array(
            np.sum(
                np.abs(H2_sparse_hamiltonian @ sparse.csc_matrix(C2_phi) / hbar).power(
                    2
                ),
                axis=0,
            )
        )[0]

    def error_deriv(self, Φ, z_step, list_index_aux_stable=None):
        """
        The error associated with losing all flux terms into the k auxiliary and n’ state.
        Where k is in H_t and n' is in S_t.

        .. math::
            \sum_{n \in \mathcal{S}_{t}} \\left\\vert \\frac{\delta_{a} \psi^{(\\vec{k})}_{t,n}}{\delta t_a} \\right\\vert^2

        .. math::
            \sum_{\\vec{k} \in \mathcal{H}_{t}} \\left\\vert \\frac{\delta_{a} \psi^{(\\vec{k})}_{t,n}}{\delta t_a} \\right\\vert^2

        PARAMETERS
        ----------
        1. Φ : np.array
               The current full hierarchy vector
        2. z_step : list
                    the list of noise terms (compressed) for the next timestep
        3. list_index_aux_stable : list
                                   a list relative indices for the stable auxiliaries

        RETURNS
        -------
        1. E2_del_phi : np.array
                        the error associated with losing flux to a component (either
                        hierarchy or state basis element) in H_t direct sum S_t
        """
        if list_index_aux_stable is not None:
            # Error arises for flux only out of the stable auxiliaries
            # --------------------------------------------------------
            Φ_stab = np.zeros(self.n_state * self.n_hier, dtype=np.complex128)
            Φ_stab_v = Φ_stab.view().reshape([self.n_state, self.n_hier], order="F")
            Φ_stab_v[:, list_index_aux_stable] = Φ.view().reshape(
                [self.n_state, self.n_hier], order="F"
            )[:, list_index_aux_stable]
        else:
            Φ_stab = Φ

        list_avg_L2 = [
            operator_expectation(L, Φ[: self.n_state]) for L in self.system.list_L2_coo
        ]

        P1_del_phi = (
            self.eom.K2_k @ Φ_stab + self.eom.K2_kp1 @ Φ_stab + self.eom.K2_km1 @ Φ_stab
        )

        for j in range(len(self.system.list_absindex_L2)):
            P1_del_phi += z_step[j] * self.eom.Z2_k[j] @ Φ_stab
            P1_del_phi += np.conj(list_avg_L2[j]) * self.eom.Z2_kp1[j] @ Φ_stab

        E2_del_phi = np.abs(
            P1_del_phi.reshape([self.n_state, self.n_hier], order="F") / hbar
        )

        return E2_del_phi

    def error_deletion(self, Φ, delta_t):
        """
        The error associated with setting the corresponding component of Phi to 0. This
        is equivalent to adding a term to our approximate derivative that is absent in
        the full derivative, corresponding to

        .. math::
            \sum_{n \in \mathcal{S}_{t}} \\frac{\\vert \psi^{(\\vec{k})}_{t,n}\\vert^2}{dt}

        .. math::
            \sum_{\\vec{k} \in \mathcal{H}_{t}} \\frac{\\vert \psi^{(\\vec{k})}_{t,n}\\vert^2}{dt}

        PARAMETERS
        ----------
        1. Φ : np.array
               The current position of the full hierarchy vector
        2. delta_t : float
                     the timestep for the calculation

        RETURNS
        -------
        1. E2_site_aux : np.array
                         the error induced by removing components of Φ in H_t+S_t
        """

        # Error arising from removing the auxiliary directly
        # --------------------------------------------------
        E2_site_aux = np.abs(
            np.asarray(Φ).reshape([self.n_state, self.n_hier], order="F") / delta_t
        )

        return E2_site_aux

    def error_flux_down(self, Φ, type):
        """
        A function that returns the error associated with neglecting flux from members of
        H_t to auxiliaries in H_t^C that arise due to flux from higher auxiliaries to
        lower auxiliaries.

        .. math::
            \sum_{n \in \mathcal{S}_{t}} \\left \\vert F[\\vec{k}-\\vec{e}_n] \\frac{g_n}{\gamma_n} N^{(\\vec{k})}_t \psi_{t,n}^{(\\vec{k})}\\right \\vert^2

        .. math::
            \sum_{\\vec{k} \in \mathcal{H}_{t}} \\left \\vert F[\\vec{k}-\\vec{e}_n] \\frac{g_n}{\gamma_n} N^{(\\vec{k})}_t \psi_{t,n}^{(\\vec{k})}\\right \\vert^2

        PARAMETERS
        ----------
        1. Φ : np.array
               The current state of the hierarchy
        2. type: string
                 'H' - Hierarchy type calculation
                 'S' - State type calculation

        RETURNS
        -------
        1. E2_flux_down_error : np.array
                                the error induced by neglecting flux from H_t (or H_S)
                                to auxiliaries with higher summed index in H_t^C.
        """
        # Constants
        # ---------
        list_new_states = [
            self.system.list_state_indices_by_hmode[:, 0] + i * self.n_state
            for i in range(self.n_hier)
        ]
        list_modes_from_site_index = [
            item for sublist in list_new_states for item in sublist
        ]

        # Reshape hierarchy (to matrix)
        # ------------------------------
        P2_pop_site = (
            np.abs(np.asarray(Φ).reshape([self.n_state, self.n_hier], order="F")) ** 2
        )
        P1_aux_norm = np.sqrt(np.sum(P2_pop_site, axis=0))
        P2_modes_from0 = np.asarray(Φ)[
            np.tile(self.system.list_state_indices_by_hmode[:, 0], self.n_hier)
        ]
        P2_pop_modes_down_1 = (np.abs(P2_modes_from0) ** 2).reshape(
            [self.n_hmodes, self.n_hier], order="F"
        )
        P1_modes = np.asarray(Φ)[list_modes_from_site_index]
        P2_pop_modes = np.abs(P1_modes).reshape([self.n_hmodes, self.n_hier], order="F")

        # Get flux factors
        # ----------------
        G2_bymode = np.array(self.system.g)[
            np.tile(list(range(self.n_hmodes)), self.n_hier)
        ].reshape([self.n_hmodes, self.n_hier], order="F")
        W2_bymode = np.array(self.system.w)[
            np.tile(list(range(self.n_hmodes)), self.n_hier)
        ].reshape([self.n_hmodes, self.n_hier], order="F")

        F2_filter_aux = np.array(
            [
                [
                    1 if aux[self.system.list_absindex_mode[i_mode_rel]] - 1 >= 0 else 0
                    for aux in self.hierarchy.auxiliary_list
                ]
                for i_mode_rel in range(self.n_hmodes)
            ]
        )

        if type == "H":
            # Hierarchy Type Downward Flux
            # ============================
            E2_flux_down_error = (
                np.real(
                    F2_filter_aux
                    * np.abs(G2_bymode / W2_bymode)
                    * (P2_pop_modes_down_1 * P1_aux_norm[None, :] + P2_pop_modes)
                )
                / hbar
            )
        elif type == "S":
            # State Type Downward Flux
            # ========================
            # Construct <L_m> term
            # --------------------
            E2_lm = np.tile(
                np.sum(
                    F2_filter_aux * np.abs(G2_bymode / W2_bymode) * P2_pop_modes_down_1,
                    axis=0,
                ),
                [self.n_state, 1],
            )

            # Map Error to States
            # -------------------
            M2_state_from_mode = np.zeros([self.n_state, self.system.n_hmodes])
            M2_state_from_mode[
                self.system.list_state_indices_by_hmode[:, 0],
                np.arange(np.shape(self.system.list_state_indices_by_hmode)[0]),
            ] = 1
            E2_flux_down_error = (
                M2_state_from_mode
                @ np.real(F2_filter_aux * np.abs(G2_bymode / W2_bymode) * P2_pop_modes)
                / hbar
            )
            E2_flux_down_error += E2_lm * P2_pop_site / hbar
        else:
            E2_flux_down_error = 0
            raise UnsupportedRequest(type, "error_flux_down")

        return E2_flux_down_error

    def error_flux_up(self, Φ):
        """
        A function that returns the error associated with neglecting flux from members of
        H_t to auxiliaries in H_t^C that arise due to flux from lower auxiliaries to
        higher auxiliaries.

        .. math::
            \sum_{n \in \mathcal{S}_{t}} \\left \\vert F[\\vec{k}+\\vec{e}_n] \gamma_n (1+\\vec{k}[n]) \psi_{t,n}^{(\\vec{k})}   \\right \\vert^2

        .. math::
            \sum_{\\vec{k} \in \mathcal{H}_{t}}\\left \\vert F[\\vec{k}+\\vec{e}_n] \gamma_n (1+\\vec{k}[n]) \psi_{t,n}^{(\\vec{k})}   \\right \\vert^2

        PARAMETERS
        ----------
        1. Φ : np.array
               The current state of the hierarchy

        RETURNS
        -------
        1. E2_flux_up_error : np. array
                              the error induced by neglecting flux from H_t (or H_S)
                              to auxiliaries with lower summed index in H_t^C.
        """
        # Constants
        # ---------
        list_new_states = [
            self.system.list_state_indices_by_hmode[:, 0] + i * self.n_state
            for i in range(self.n_hier)
        ]
        list_modes_from_site_index = [
            item for sublist in list_new_states for item in sublist
        ]

        # Reshape hierarchy (to matrix)
        # ------------------------------
        P1_modes = np.asarray(Φ)[list_modes_from_site_index]
        P2_pop_modes = np.sqrt(np.abs(P1_modes) ** 2).reshape(
            [self.n_hmodes, self.n_hier], order="F"
        )

        # Get flux factors
        # ----------------
        W2_bymode = np.array(self.system.w)[
            np.tile(list(range(self.n_hmodes)), self.n_hier)
        ].reshape([self.n_hmodes, self.n_hier], order="F")
        K2aux_bymode = np.transpose(
            np.array(
                [
                    aux.get_values(self.system.list_absindex_mode)
                    for aux in self.hierarchy.auxiliary_list
                ]
            )
        )

        # Filter out fluxes beyond the hierarchy depth
        # --------------------------------------------
        filter_aux = np.array(
            [
                i_aux
                for (i_aux, aux) in enumerate(self.hierarchy.auxiliary_list)
                if np.sum(aux) + 1 > self.hierarchy.param["MAXHIER"]
            ]
        )
        F2_filter = np.ones([self.n_hmodes, self.n_hier])
        if filter_aux.size > 0:
            F2_filter[:, filter_aux] = 0

        # Filter out Markovian Modes
        # --------------------------
        array2D_mark_param = np.array(
            [
                np.array(param)[self.system.list_absindex_mode]
                for (name, param) in self.hierarchy.param["STATIC_FILTERS"]
                if name == "Markovian"
            ]
        )
        if len(array2D_mark_param) > 0:
            array_mark_param = np.any(array2D_mark_param, axis=0)
            F2_filter_markov = np.ones([self.n_hmodes, self.n_hier])
            F2_filter_markov[array_mark_param, 1:] = 0
            F2_filter = F2_filter * F2_filter_markov

        # Test upward fluxes
        # ------------------
        return F2_filter * np.abs(W2_bymode) * (1 + K2aux_bymode) * P2_pop_modes / hbar

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
