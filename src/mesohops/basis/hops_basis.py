from scipy import sparse

from mesohops.basis.basis_functions import calculate_delta_bound, determine_error_thresh
import numpy as np
from mesohops.basis.basis_functions_adaptive import *
from mesohops.basis.basis_functions import determine_error_thresh, calculate_delta_bound
from mesohops.basis.hops_fluxfilters import HopsFluxFilters
from mesohops.basis.hops_modes import HopsModes
from mesohops.eom.eom_functions import compress_zmem, operator_expectation
from mesohops.util.exceptions import UnsupportedRequest
from scipy import sparse

__title__ = "Basis Class"
__author__ = "D. I. G. Bennett, Brian Citty, J. K. Lynd"
__version__ = "1.4"

class HopsBasis:
    """
    Every HOPS calculation is defined by the HopsSystem, HopsHierarchy, and HopsEOM
    classes (and their associated parameters). These form the basis set for the
    calculation. HopsBasis is the class that contains all of these sub-classes and
    mediates the way the HopsTrajectory interacts with them.
    """

    __slots__ = (
        # --- Core basis components  ---
        'system',        # System parameters and operators (HopsSystem)
        'hierarchy',     # Hierarchy management (HopsHierarchy)
        'mode',          # Mode management (HopsModes)
        'eom',           # Equation of motion (HopsEOM)

        # --- Filters ---
        'flux_filters',  # Flux filtering object (HopsFluxFilters)

        # --- Cached sparse operators & LTC matrices ---
        '_Z2_noise_sparse',  # Noise sparse matrix (for noise contributions)
        '_T2_ltc_phys',      # Physical LTC matrix (low-temperature correction)
        '_T2_ltc_hier',      # Hierarchy LTC matrix (low-temperature correction)

        # --- L-operator expectation cache ---
        '_psi',              # Cached wavefunction (for expectation values)
        '__list_avg_L2',     # Cached L2 averages (for expectation values)
    )

    def __init__(self, system, hierarchy, eom):
        """
        Inputs
        ------
        1. system: dict
                   dictionary of user inputs.
                   [see hops_system.py]
            a. HAMILTONIAN
            b. GW_SYSBATH
            c. CORRELATION_FUNCTION_TYPE
            d. LOPERATORS
            e. CORRELATION_FUNCTION

        2. hierarchy: dict
                      dictionary of user inputs.
                      [see hops_hierarchy.py]
            a. MAXHIER
            b. TERMINATOR
            c. STATIC_FILTERS

        3. eom: dict
                dictionary of user inputs.
                [see hops_eom.py]
            a. TIME_DEPENDENCE
            b. EQUATION_OF_MOTION
            c. ADAPTIVE_H
            d. ADAPTIVE_S
            e. DELTA_A
            f. DELTA_S

        Returns
        -------
        None
        """
        self.system = system
        self.hierarchy = hierarchy
        self.mode = HopsModes(system, hierarchy)
        self.eom = eom
        self._Z2_noise_sparse = sparse.csr_array((self.system.param[
                                                    "SPARSE_HAMILTONIAN"].shape[0],
                          self.system.param["SPARSE_HAMILTONIAN"].shape[1]),
                         dtype=np.complex64)
        self._T2_ltc_phys, self._T2_ltc_hier = None, None
        self.flux_filters = HopsFluxFilters(self.system, self.hierarchy, self.mode)

    def initialize(self, psi_0):
        """
        Initializes the hierarchy and equations of motion classes
        so that everything is prepared for integration. It returns the
        dsystem_dt function to be used in the integrator.

        Parameters
        ----------
        1. psi_0 : np.array
                   Initial wave function.

        Returns
        -------
        1. dsystem_dt : function
                        Core function for calculating the time-evolution of the wave
                        function.
        """
        self.hierarchy.initialize(self.adaptive_h)
        self.system.initialize(self.adaptive_s, psi_0)
        self.mode.list_absindex_mode = list(set(self.hierarchy.list_absindex_hierarchy_modes)
                                       | set(self.system.list_absindex_state_modes))

        dsystem_dt = self.eom._prepare_derivative(self.system,
                                                  self.hierarchy,
                                                  self.mode)
        return dsystem_dt

    def define_basis(self, Φ, delta_t, z_step):
        """
        Determines the basis that is needed for a given full hierarchy (Φ) in order
        to construct an approximate derivative with error below the specified threshold.

        Parameters
        ----------
        1. Φ : np.array
               Current full hierarchy.

        2. delta_t : float
                     Timestep for the calculation.

        3. z_step : list
                    List of noise terms (compressed) for the next timestep.

        Returns
        -------
        1. list_state_new : list
                            List of states in the new basis (S_1).

        2. list_aux_new : list
                          List of auxiliaries in new basis (H_1).

        """
        # Manages generation of the L-operator expecation values.
        self.psi = Φ[:self.n_state]

        # Get the off-diagonal contributions to the system Hamiltonian from the noise
        # and low-temperature corrections
        self._Z2_noise_sparse = self.get_Z2_noise_sparse(z_step)
        self._T2_ltc_phys, self._T2_ltc_hier = self.get_T2_ltc()

        # Calculate New Bases
        # ===================

        # Calculate New Hierarchy List
        # ----------------------------
        if self.adaptive_h:
            list_aux_stable, list_aux_bound = self._define_hierarchy_basis(
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

        # Calculate New State List
        # ------------------------
        if self.adaptive_s:
            list_state_stable, list_state_bound = self._define_state_basis(
                Φ/np.linalg.norm(Φ[:self.n_state]), delta_t, z_step,
                list_index_stable_aux, list_aux_bound, list_aux_new=list_aux_new
            )
            list_state_new = list(set(list_state_stable) | set(list_state_bound))
            list_state_new.sort()
        else:
            list_state_new = list(self.system.state_list)

        return [list_state_new, list_aux_new]

    def update_basis(self, Φ, list_state_new, list_aux_new):
        """
        Updates the derivative function and full hierarchy vector (Φ) for the
        new basis (hierarchy and/or system).

        Parameters
        ----------
        1. Φ : np.array
               Current full hierarchy.

        2. list_state_new: list
                           List of states in the new basis (S_1).

        3. list_aux_new : list
                          List of auxiliaries in new basis (H_1).

        Returns
        -------
        1. Φ_new : np.array
                   Updated full hierarchy.

        2. dsystem_dt : function
                        Updated derivative function.
        """
        # Update State List
        # =================
        flag_update_state = False
        if set(list_state_new) != set(self.system.state_list): flag_update_state = True
        # Setter manages many other updates
        self.system.state_list = np.array(list_state_new, dtype=int)

        # Update Hierarchy List
        # =====================
        flag_update_hierarchy = False
        if set(list_aux_new) != set(self.hierarchy.auxiliary_list): flag_update_hierarchy = True
        # Setter manages many other updates
        self.hierarchy.auxiliary_list = list_aux_new

        # Update Mode List
        # ================
        if flag_update_state or flag_update_hierarchy:
            # Setter manages many other updates
            self.mode.list_absindex_mode = list(set(self.hierarchy.list_absindex_hierarchy_modes)
                                           | set(self.system.list_absindex_state_modes))
        
        # Update state of calculation for new basis
        # =========================================
        if (flag_update_state or flag_update_hierarchy):

            # Define permutation matrix from old basis --> new basis
            # ------------------------------------------------------
            permute_aux_row = []
            permute_aux_col = []
            nstate_old = len(self.system.previous_state_list)
            list_index_old_stable_state = np.array(
                [
                    i_rel
                    for (i_rel, i_abs) in enumerate(self.system.previous_state_list)
                    if i_abs in self.system.list_stable_state
                ]
            )
            list_index_new_stable_state = np.array(
                [
                    i_rel
                    for (i_rel, i_abs) in enumerate(self.system.state_list)
                    if i_abs in self.system.list_stable_state
                ]
            )

            # Columns are the indices of wave function entries in the previous basis,
            # rows are the indices of those same entries in the updated basis.
            for (index_stable, aux) in enumerate(self.hierarchy.list_aux_stable):
                permute_aux_row.extend(
                    aux._index * self.n_state
                    + list_index_new_stable_state
                )
                permute_aux_col.extend(
                    self.hierarchy.previous_list_auxstable_index[index_stable] * nstate_old
                    + list_index_old_stable_state
                )

            n_hier_old = len(self.hierarchy.previous_auxiliary_list)
            list_stable_aux_new_index = [aux._index for aux in self.hierarchy.list_aux_stable]
            list_stable_aux_old_index = self.hierarchy.previous_list_auxstable_index

            # Update phi
            # ----------
            norm_old = np.linalg.norm(Φ[:len(self.system.previous_state_list)])
            Φ_new = np.zeros(self.n_hier * self.n_state, dtype=np.complex128)
            Φ_new[permute_aux_row] = Φ[permute_aux_col]
            Φ_new = norm_old * Φ_new / np.linalg.norm(Φ_new[:self.n_state])

            # Update dsystem_dt
            # -----------------
            dsystem_dt = self.eom._prepare_derivative(
                self.system,
                self.hierarchy,
                self.mode,
                [permute_aux_row,
                 permute_aux_col,
                 list_stable_aux_old_index,
                 list_stable_aux_new_index,
                 n_hier_old],
                update=True,
            )

            return (Φ_new, dsystem_dt)
        else:
            return (Φ, self.eom.dsystem_dt)

    def _define_state_basis(self, Φ, delta_t, z_step, list_index_aux_stable,
                            list_aux_bound, list_aux_new=None):
        """
        Determines the states which should be included in the adaptive integration
        for the next time point. This function corresponds to section S2.B
        from the SI of "Characterizing the Role of Peierls Vibrations in Singlet
        Fission with the Adaptive Hierarchy of Pure States," available at
        https://arxiv.org/abs/2505.02292.

        Parameters
        ----------
        1. Φ : np.array
               Current full hierarchy.

        2. delta_t : float
                     Timestep for the calculation.

        3. z_step : list
                    List of noise terms (compressed) for the next timestep.

        4. list_index_aux_stable : list(int)
                                   List of relative indices for the stable auxiliaries.

        5. list_aux_bound : list(HopsAux)
                            List of the auxiliary objects in the Boundary Auxiliary
                            Basis.

        6. list_aux_new : list(HopsAux)
                          List of the auxiliary objects in the newly-generated
                          Auxiliary Basis for the next time step.

        Returns
        -------
        1. list_state_stable : np.array
                               Array of stable states (absolute state index, S_S).

        2. list_state_boundary : np.array
                                 Array of the boundary states (absolute state index, S_B).
        """
        # Test that the physical wave function is normalized.
        if not np.allclose(np.linalg.norm(Φ[:self.n_state]), 1):
            raise UnsupportedRequest("non-normalized Φ", "_define_state_basis")
        # Ensure L-operator expectation values are current.
        if not np.allclose(self.psi, Φ[:self.n_state]):
            self.psi = Φ[:self.n_state]

        # Define Constants
        # ----------------
        delta_state_sq = self.delta_s ** 2

        # CONSTRUCT STABLE STATE (S_S) - section S2.B.1 of the SI
        # =======================================================

        # Construct Error For Excluding Member of S_t
        # -------------------------------------------

        error_by_state_sq = self.state_stable_error(
            Φ, delta_t, z_step, list_index_aux_stable, list_aux_bound,
            list_aux_new=list_aux_new
        )

        # Determine the Stable States (S_S)
        # ---------------------------------
        list_relindex_state_stable, list_state_stable = self._determine_basis_from_list(
            error_by_state_sq, delta_state_sq / 2.0, self.system.state_list
        )

        # CONSTRUCT BOUNDARY STATE (S_B) - section S2.B.2 of the SI
        # =========================================================

        # Establish the error available for the boundary states
        # -----------------------------------------------------
        list_index_state_unstable = np.setdiff1d(np.arange(self.n_state), list_relindex_state_stable)
        stable_error_sq = np.sum(error_by_state_sq[list_index_state_unstable])
        
        delta_bound_sq = calculate_delta_bound(delta_state_sq, stable_error_sq)
        
        # Construct Error for Excluding Member of S_t^C
        # ---------------------------------------------
        list_sc = self.system.list_sc
        if not self.off_diagonal_couplings:
            # Fluxes to states not in the current basis stem purely from the system
            # Hamiltonian in this case.
            list_index_nonzero, list_error_nonzero = (
                error_sflux_boundary_state(Φ,
                                           list_state_stable,
                                           list_sc,
                                           self.n_state,
                                           self.n_hier,
                                           -1j * self.system.param["SPARSE_HAMILTONIAN"]
                                           + self.Z2_noise_sparse,
                                           list_relindex_state_stable,
                                           list_index_aux_stable,
                                           [],
                                           [],
                                           self.T2_ltc_phys,
                                           self.T2_ltc_hier)
            )

        else:
            # Generate the list of the indices of the states not in the basis (in
            # list_sc) that are also destination states for flux, and a list of the
            # associated off-diagonal mode-from-state matrices for those destination
            # states.

            list_sc_dest = []
            list_M2_sc_dest = []
            list_M2_mode_from_state_off = self.list_M2_by_dest_off_diag
            for d_ind in range(len(self.system.list_destination_state)):
                d = self.system.list_destination_state[d_ind]
                if d in list_sc:
                    list_sc_dest.append(np.where(list_sc == d)[0][0])
                    list_M2_sc_dest.append(list_M2_mode_from_state_off[d_ind])

            # Generate the flux down into each destination state not in the current
            # basis.
            F2_filter_off = None
            if list_aux_new is not None:
                F2_filter_off = self.flux_filters.construct_filter_state_stable_down(
                    list_aux_new)
            list_E_down = error_flux_down_by_dest_state(Φ,
                                                        self.n_state,
                                                        self.n_hier,
                                                        self.n_hmodes,
                                                        self.list_g,
                                                        self.list_w,
                                                        list_M2_sc_dest,
                                                        list_index_aux_stable,
                                                        F2_filter_off,
                                                        list_relindex_state_stable)

            # Generate the corresponding flux up into each destination state not in the
            # current basis. We do not worry about static filters here, because the
            # filter constructed below already limits flux to only auxiliaries that
            # were added to the basis.
            F2_filter_off = None
            if list_aux_new is not None:
                F2_filter_off = self.flux_filters.construct_filter_state_stable_up(
                    list_aux_new)
            list_E_up = error_flux_up_by_dest_state(Φ,
                                                    self.n_state,
                                                    self.n_hier,
                                                    self.n_hmodes,
                                                    self.list_w,
                                                    self.K2_aux_by_mode,
                                                    list_M2_sc_dest,
                                                    list_index_aux_stable,
                                                    F2_filter_off,
                                                    list_relindex_state_stable)

            # Calculate the total state flux into all boundary states, including any
            # fluxes up or down calculated above.
            list_flux_updown = list_E_up + list_E_down
            list_index_nonzero, list_error_nonzero = (
                error_sflux_boundary_state(Φ,
                                           list_state_stable,
                                           list_sc,
                                           self.n_state,
                                           self.n_hier,
                                           -1j*self.system.param["SPARSE_HAMILTONIAN"]
                                           + self.Z2_noise_sparse,
                                           list_relindex_state_stable,
                                           list_index_aux_stable,
                                           list_sc_dest,
                                           list_flux_updown,
                                           self.T2_ltc_phys,
                                           self.T2_ltc_hier)
            )

        # Determine Boundary States
        # -------------------------
        if len(list_error_nonzero) > 0:
            _, list_state_boundary = self._determine_basis_from_list(
                list_error_nonzero, delta_bound_sq, list_index_nonzero
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
            np.array(list_state_stable, dtype=int),
            np.array(list_state_boundary, dtype=int),
        )

    def _define_hierarchy_basis(self, Φ, delta_t, z_step):
        """
        Determines the auxiliaries which should be included in the adaptive
        integration for the next time point. This function corresponds to section S2.A
        from the SI of "Characterizing the Role of Peierls Vibrations in Singlet
        Fission with the Adaptive Hierarchy of Pure States," available at
        https://arxiv.org/abs/2505.02292.

        Parameters
        ----------
        1. Φ : np.array
               Current full hierarchy.

        2. delta_t : float
                     Timestep for the calculation.

        3. z_step : list
                    List of noise terms (compressed) for the next timestep.

        Returns
        -------
        1. list_aux_stable : list
                             List of the stable auxiliaries (H_S).

        2. list_aux_boundary : list
                               List of auxiliaries that share a boundary with stable
                               auxiliaries (H_B).
        """
        # Test that the physical wave function is normalized.
        if not np.allclose(np.linalg.norm(Φ[:self.n_state]), 1):
            raise UnsupportedRequest("non-normalized Φ", "_define_hierarchy_basis")
        # Ensure L-operator expectation values are current.
        if not np.allclose(self.psi, Φ[:self.n_state]):
            self.psi = Φ[:self.n_state]

        # Define constants
        delta_aux_sq = self.delta_a ** 2

        # CONSTRUCT STABLE HIERARCHY (A_S) - section S2.A.1 of the SI
        # ===========================================================

        # Construct Error For Excluding Member of A_t
        # --------------------------------------------
        error_by_aux_sq, list_e2_kflux = self.hier_stable_error(
            Φ, delta_t, z_step
        )

        # Determine the Stable Auxiliaries (A_S)
        # --------------------------------------
        list_index_aux_stable, list_aux_stable = self._determine_basis_from_list(
            error_by_aux_sq, delta_aux_sq/2.0, self.hierarchy.auxiliary_list
        )

        # CONSTRUCT BOUNDARY HIERARCHY (A_B) - section S2.A.2 of the SI
        # =============================================================

        # Establish the error available for the boundary auxiliaries
        # ----------------------------------------------------------
        list_index_aux_unstable = np.setdiff1d(np.arange(self.n_hier), list_index_aux_stable)
        stable_error_sq = np.sum(error_by_aux_sq[list_index_aux_unstable])

        delta_bound_sq = calculate_delta_bound(delta_aux_sq, stable_error_sq)

        # Determine the Boundary Auxiliaries (A_B)
        # ----------------------------------------
        list_aux_boundary = self._determine_boundary_hier(
            list_e2_kflux, list_index_aux_stable, delta_bound_sq, self.f_discard
        )

        # Filter Boundary Set for Auxiliaries That Are Not Part of A_T
        # ------------------------------------------------------------
        if len(list_aux_boundary) > 0 and not self.hierarchy.only_markovian_filter:
            list_aux_boundary = self.hierarchy.filter_aux_list(list_aux_boundary)

        return list_aux_stable, list_aux_boundary

    def hier_stable_error(self, Φ, delta_t, z_step):
        """
        Finds the total error associated with removing k in A_t. This function
        corresponds to section S2.A.1 from the SI of "Characterizing the Role of
        Peierls Vibrations in Singlet Fission with the Adaptive Hierarchy of Pure
        States," available at https://arxiv.org/abs/2505.02292.

        Parameters
        ----------
        1. Φ : np.array
               Current full hierarchy vector.

        2. delta_t : float
                     Timestep for the calculation.

        3. z_step : list
                    List of noise terms (compressed) for the next timestep.

        Returns
        -------
        1. error : np.array
                   List of error associated with removing each auxiliary in A_t.

        2. E2_flux_up : np.array
                        Error induced by neglecting flux from A_t
                        to auxiliaries with lower summed index in A_t^C.

        3. E2_flux_down : np.array
                          Error induced by neglecting flux from A_t
                          to auxiliaries with higher summed index in A_t^C.
        """
        # Ensure L-operator expectation values are current.
        if not np.allclose(self.psi, Φ[:self.n_state]):
            self.psi = Φ[:self.n_state]

        # Construct arrays in the space of [mode m, state s] corresponding to L_m[s,s],
        # L_m[d,s] for each d, and <L_m>*I[s,s].
        list_M2_mode_from_state_off = self.list_M2_by_dest_off_diag
        M2_mode_from_state_diag = self.M2_mode_from_state_diag
        X2_expectation = self.X2_exp_lop_mode_state

        # Calculate the error term by term
        # ================================
        # Derivative flux
        # ---------------
        E1_error = np.sum(error_deriv(self.eom.dsystem_dt,
                                      Φ,
                                      z_step,
                                      self.n_state,
                                      self.n_hier,
                                      delta_t),
                          axis=0)

        # State flux
        # ----------
        E1_error += error_sflux_hier(Φ,
                                     self.system.state_list,
                                     self.system.list_sc,
                                     self.n_state,
                                     self.n_hier,
                                     -1j*self.system.param["SPARSE_HAMILTONIAN"] +
                                     self.Z2_noise_sparse,
                                     self.T2_ltc_phys,
                                     self.T2_ltc_hier)

        # Calculate flux up
        # -----------------
        E2_flux_up_nofilter = error_flux_up_hier_stable(Φ,
                                                        self.n_state,
                                                        self.n_hier,
                                                        self.n_hmodes,
                                                        self.list_w,
                                                        self.K2_aux_by_mode,
                                                        M2_mode_from_state_diag,
                                                        list_M2_mode_from_state_off)

        # Filter the stable hierarchy flux up - no flux into auxiliaries that violate
        # triangular truncation or Markovian filter allowed.
        F2_filter = self.flux_filters.construct_filter_auxiliary_stable_up()
        F2_filter *= self.flux_filters.construct_filter_markov_up()
        F2_filter *= self.flux_filters.construct_filter_triangular_up()
        F2_filter *= self.flux_filters.construct_filter_longedge_up()

        E1_error += np.sum(F2_filter * E2_flux_up_nofilter,axis=0)

        # Flux down
        # ---------
        E2_flux_down_nofilter = error_flux_down_hier_stable(Φ,
                                                            self.n_state,
                                                            self.n_hier,
                                                            self.n_hmodes,
                                                            self.list_g,
                                                            self.list_w,
                                                            M2_mode_from_state_diag,
                                                            list_M2_mode_from_state_off,
                                                            X2_expectation)

        # Filter the stable hierarchy flux down - no flux into auxiliaries with
        # negative indices allowed.
        F2_filter = self.flux_filters.construct_filter_auxiliary_stable_down()
        E1_error += np.sum(F2_filter * E2_flux_down_nofilter,axis=0)

        return (
            E1_error,
            [E2_flux_up_nofilter, E2_flux_down_nofilter],
        )

    def _determine_boundary_hier( self, list_e2_kflux_up_down, list_index_aux_stable,
                                  delta_bound_sq, f_discard):
        """
        Determines the set of boundary auxiliaries for the next time step. This function
        corresponds to section S2.A.2 from the SI of "Characterizing the Role of
        Peierls Vibrations in Singlet Fission with the Adaptive Hierarchy of Pure
        States," available at https://arxiv.org/abs/2505.02292.

        Parameters
        ----------
        1. list_e2_kflux : list
                           List of list containing the squared error values for the
                           flux up and flux down terms.

        2. list_index_aux_stable : list
                                   List of the indices for stable auxiliaries.

        3. delta_bound_sq : float
                            Boundary error tolerance value.

        4. f_discard : float
                       Fraction of the boundary error devoted to removing error
                       terms from list_e2_kflux for memory conservation.


        Returns
        -------
        1. list_aux_boundary : list
                               List of the flux up and flux down auxiliaries.
        """
        # Construct filters
        # -----------------
        # Generate filter for boundary auxiliary flux up - no fluxes into auxiliaries
        # that violate the triangular truncation condition or Markovian filter
        # allowed and all fluxes into auxiliaries in stable basis ignored.
        F2_filter = self.flux_filters.construct_filter_auxiliary_boundary_up()
        F2_filter *= self.flux_filters.construct_filter_markov_up()
        F2_filter *= self.flux_filters.construct_filter_triangular_up()
        F2_filter *= self.flux_filters.construct_filter_longedge_up()
        list_e2_kflux_up_down[0] = list_e2_kflux_up_down[0] * F2_filter

        # Generate filter for boundary auxiliary flux down - no fluxes into auxiliaries
        # with negative indices allowed and all fluxes into auxiliaries in stable basis
        # ignored.
        F2_filter = self.flux_filters.construct_filter_auxiliary_boundary_down()
        list_e2_kflux_up_down[1] = list_e2_kflux_up_down[1] * F2_filter


        # Apply filters to fluxes from stable auxiliaries (A_s) to get flux into
        # boundary auxiliaries.
        list_e2_kflux_up_down[0] = list_e2_kflux_up_down[0][:,list_index_aux_stable]
        list_e2_kflux_up_down[1] = list_e2_kflux_up_down[1][:, list_index_aux_stable]

        # Remove small fluxes with f_discard
        # ----------------------------------
        # Sort nonzero fluxes by magnitude and filter out small errors without
        # allocating to boundary auxiliaries to save time.
        E1_nonzero_flux = list_e2_kflux_up_down[0][list_e2_kflux_up_down[0] != 0]
        sorted_error = np.sort(
            np.append(E1_nonzero_flux, list_e2_kflux_up_down[1][list_e2_kflux_up_down[1] != 0])
        )
        error_thresh = determine_error_thresh(sorted_error, delta_bound_sq*f_discard*f_discard)

        # Sum up the discarded small errors to ensure calculations don't violate
        # error bound.
        discarded_error = np.sum(list_e2_kflux_up_down[0][list_e2_kflux_up_down[0]
                                                          <= error_thresh])
        discarded_error += np.sum(list_e2_kflux_up_down[1][list_e2_kflux_up_down[1]
                                                           <= error_thresh])

        # Remove the discarded errors.
        list_e2_kflux_up_down[0][list_e2_kflux_up_down[0] <= error_thresh] = 0
        list_e2_kflux_up_down[1][list_e2_kflux_up_down[1] <= error_thresh] = 0

        # Find the error threshold for edge auxiliaries.
        # ----------------------------------------------
        boundary_aux_dict = {}
        boundary_connect_dict = {}
        list_stable_aux = [self.hierarchy.auxiliary_list[i] for i in range(self.n_hier)
                           if i in list_index_aux_stable]

        # Bin error flux by boundary aux.
        for (i_aux,aux) in enumerate(list_stable_aux):
            # Flux up error
            nonzero_modes_up = list_e2_kflux_up_down[0][:,i_aux].nonzero()[0]
            if(len(nonzero_modes_up) > 0):
                # Get the id values for boundary auxiliaries up along modes with
                # nonzero flux.
                list_id_up, list_value_connect,list_mode_connect = (
                     aux.get_list_id_up(self.list_absindex_mode[nonzero_modes_up]))

                #For each id up, add the flux error to its entry in the
                # boundary_aux_dict dictionary. We assume that the filter is
                # constructed correctly and that these are all indeed boundary
                # auxiliaries. If it is the first flux for a boundary auxiliary, we keep
                # track of the connection in the boundary_connect_dict, so we can use
                # the e_step method to construct the new Auxiliary as before.
                for (id_ind,my_id) in enumerate(list_id_up):
                    try:
                        boundary_aux_dict[my_id] += list_e2_kflux_up_down[0][nonzero_modes_up[id_ind],i_aux]
                    except:
                        boundary_aux_dict[my_id] = list_e2_kflux_up_down[0][nonzero_modes_up[id_ind],i_aux]
                        boundary_connect_dict[my_id] = [aux, list_mode_connect[id_ind], 1]

            # Flux down error
            nonzero_modes_down = self.list_absindex_mode[list_e2_kflux_up_down[1][:,i_aux].nonzero()[0]]
            if(len(nonzero_modes_down) > 0):
                list_id_down, list_value_connects, list_mode_connects = aux.get_list_id_down()
                for (id_ind,my_id) in enumerate(list_id_down):
                    if(list_mode_connects[id_ind] in nonzero_modes_down):
                        try:
                            boundary_aux_dict[my_id] += list_e2_kflux_up_down[1][list(self.list_absindex_mode).index(list_mode_connects[id_ind]),i_aux]
                        except:
                            boundary_aux_dict[my_id] = list_e2_kflux_up_down[1][list(self.list_absindex_mode).index(list_mode_connects[id_ind]),i_aux]
                            boundary_connect_dict[my_id] = [aux,list_mode_connects[id_ind],-1]

        # Sort the errors and find the error threshold
        sorted_error = np.sort(np.array(list(boundary_aux_dict.values())))
        error_thresh = determine_error_thresh(sorted_error, delta_bound_sq, offset=discarded_error)

        # Identify and construct boundary auxiliaries
        list_aux_updown = [boundary_connect_dict[aux_id][0].e_step(
            boundary_connect_dict[aux_id][1], boundary_connect_dict[aux_id][2])
            for aux_id in boundary_aux_dict.keys()
            if boundary_aux_dict[aux_id] > error_thresh]

        return list_aux_updown

    def state_stable_error(self, Φ, delta_t, z_step, list_index_aux_stable,
                           list_aux_bound, list_aux_new=None):
        """
        Finds the total error associated with neglecting n in the state basis S_t.
        This includes the error associated with deleting a state from the basis,
        the error associated with losing all flux into a state, and the error associated
        with losing all flux out of a state. This function corresponds to section S2.B.1
        from the SI of "Characterizing the Role of Peierls Vibrations in Singlet
        Fission with the Adaptive Hierarchy of Pure States," available at
        https://arxiv.org/abs/2505.02292.

        Parameters
        ----------
        1. Φ : np.array
               Current full hierarchy vector.

        2. delta_t : float
                     Timestep for the calculation.

        3. z_step : list
                    List of noise terms (compressed) for the next timestep.

        4. list_index_aux_stable : list(int)
                                   List of relative indices for the stable auxiliaries (A_s).

        5. list_aux_bound : list(HopsAux)
                            List of Auxiliary Objects in the Boundary Auxiliary Basis.

        6. list_aux_new : list(HopsAux)
                          List of auxiliary objects in the new auxiliary basis.

        Returns
        -------
        1. error : np.array
                   List of error associated with removing each state.
        """
        # Ensure L-operator expectation values are current.
        if not np.allclose(self.psi, Φ[:self.n_state]):
            self.psi = Φ[:self.n_state]

        # Construct arrays in the space of [mode m, state s] corresponding to L_m[s,s],
        # L_m[d,s] for each d, and <L_m>*I[s,s].
        list_M2_mode_from_state_off = self.list_M2_by_dest_off_diag
        M2_mode_from_state_diag = self.M2_mode_from_state_diag
        X2_expectation = self.X2_exp_lop_mode_state

        # Calculate the error term by term
        # ================================
        # Derivative Flux
        # ---------------
        E1_error = np.sum(error_deriv(self.eom.dsystem_dt,
                                      Φ,
                                      z_step,
                                      self.n_state,
                                      self.n_hier,
                                      delta_t,
                                      list_index_aux_stable),
                          axis=1)

        # State Flux
        # ----------
        E1_error += error_sflux_stable_state(Φ,
                                             self.n_state,
                                             self.n_hier,
                                            -1j * self.system.param[
                                                 "SPARSE_HAMILTONIAN"]
                                             + self.Z2_noise_sparse,
                                             list_index_aux_stable,
                                             self.system.state_list,
                                             self.T2_ltc_phys,
                                             self.T2_ltc_hier)

        # Flux Down
        # ---------
        # Generate filter for stable state flux down
        F2_filter_boundary = self.flux_filters.construct_filter_state_stable_down(
            list_aux_bound)
        F2_filter_off = None
        if list_aux_new is not None and self.off_diagonal_couplings:
            F2_filter_off = self.flux_filters.construct_filter_state_stable_down(
            list_aux_new)
        E2_down = error_flux_down_state_stable(Φ,
                                               self.n_state,
                                               self.n_hier,
                                               self.n_hmodes,
                                               self.list_g,
                                               self.list_w,
                                               M2_mode_from_state_diag,
                                               list_M2_mode_from_state_off,
                                               X2_expectation,
                                               F2_filter_boundary,
                                               F2_filter_off)
        E1_error += np.sum(E2_down[:,list_index_aux_stable],axis=1)

        # Flux Up
        # -------
        # Generate filter for stable state flux up. We do not worry about static
        # filters here, because the filters constructed below already limit flux to
        # only auxiliaries that were removed from or added to the basis.
        F2_filter_boundary = self.flux_filters.construct_filter_state_stable_up(
            list_aux_bound)
        F2_filter_off = None
        if list_aux_new is not None and self.off_diagonal_couplings:
            F2_filter_off = self.flux_filters.construct_filter_state_stable_up(
                list_aux_new)
        E2_up = error_flux_up_state_stable(Φ,
                                           self.n_state,
                                           self.n_hier,
                                           self.n_hmodes,
                                           self.list_w,
                                           self.K2_aux_by_mode,
                                           M2_mode_from_state_diag,
                                           list_M2_mode_from_state_off,
                                           F2_filter_boundary,
                                           F2_filter_off)
        E1_error += np.sum(E2_up[:,list_index_aux_stable],axis=1)

        return E1_error

    def get_Z2_noise_sparse(self, z_step):
        """
        Get the matrix of noise in the shape of the full system Hamiltonian at the
        current time.

        Parameters
        ----------
        1. z_step : list(list(complex))
                    List of noise terms for the next timestep. First and second terms
                    are noise1 and noise2, and the third term is noise memory drift.
                    The first two are indexed by the list of unique L-operators in
                    HopsNoise (which may be a subset of the unique L-operators in
                    the full basis), and the third is indexed by the full list of modes.

        Returns
        -------
        1. Z2_noise_sparse : sp.sparse.csr_array(complex)
                             Operator that projects the noise and noise memory drift
                             projected onto the full system Hamiltonian [units :
                             cm^-1]. Defaults to an empty sparse array.
        """
        if self.off_diagonal_couplings:
            # Get the noise associated with system-bath projection operators that
            # couple states in the current basis to a different state.
            noise_t = (np.conj(z_step[0]) - 1j * z_step[1])[
                self.mode.list_rel_ind_off_diag_L2]
            # Get the noise memory drift associated with system-bath projection
            # operators that couple states in the current basis to a different state.
            noise_mem = np.array(
                compress_zmem(z_step[2], self.mode.list_index_L2_by_hmode,
                              self.mode.list_absindex_mode)
            )[self.mode.list_rel_ind_off_diag_L2]
            # Broadcast noise and noise memory drift onto the appropriate system-bath
            # projection operator.
            return np.sum((noise_t + noise_mem) * self.list_L2_csr[
                self.mode.list_off_diag_active_mask])
        else:
            return sparse.csr_array((self.system.param["SPARSE_HAMILTONIAN"].shape[0],
                                     self.system.param["SPARSE_HAMILTONIAN"].shape[1]),
                                    dtype=np.complex64)

    def get_T2_ltc(self):
        """
        Get the matrix form of the low-temperature correction at the current time
        in the shape of the full system Hamiltonian. Diagonal terms are ignored,
        as they have no bearing on the state flux that this matrix is used to calculate.

        Parameters
        ----------
        None

        Returns
        -------
        1. T2_phys : sp.sparse.csr_array(complex)
                     Diagonal portion of the low-temperature correction self-derivative
                     term applied to the physical wave function only to account for
                     both the terminator correction and the approximated noise memory
                     drift [units: cm^-1]. If none of the L-operators in the current
                     basis have associated LTC parameters, or none of them have
                     off-diagonal entries, this defaults to None.

        2. T2_hier : sp.sparse.csr_array(complex)
                     Diagonal portion of the low-temperature correction
                     self-derivative term applied to all auxiliary wave functions to
                     account for the approximated noise memory drift [units: cm^-1].
                     If none of the L-operators in the current basis have associated
                     LTC parameters, or none of them have off-diagonal entries,
                     this defaults to None.
        """
        if (not self.ltc_active) or (not self.off_diagonal_couplings):
            return None, None
        X1 = self.list_avg_L2[self.mode.list_off_diag_active_mask]
        G1 = self.lt_corr_param[self.mode.list_off_diag_active_mask]
        list_L2 = self.list_L2_csr[self.mode.list_off_diag_active_mask]
        list_L2_sq = np.array([L2@L2 for L2 in list_L2])
        # For each bath n, L_n is the Hermitian system-bath projection operator,
        # and G_n is the LTC parameter. ^H indicates a Hermitian conjugate.
        # T_h = \sum_n {G^*_n<L_n>L_n}
        T2_hier = np.sum(np.conj(G1)*X1*list_L2,axis=0)
        # T_p = \sum_n {G^*_n<L_n>L_n + G_n<L_n>L_n - G_nL_nL_n}
        # = \sum_n {G^*_n<L_n>L_n + (G^*^*_n<L_n>^*L_n^H) - G_nL_nL_n}
        # = \sum_n {G^*_n<L_n>L_n + (G^*_n<L_n>L_n)^H - G_nL_nL_n}
        # = \sum_n {T_h + T_h^H - G_nL_nL_n}
        T2_phys = T2_hier + T2_hier.T.conjugate() - np.sum(G1*list_L2_sq,axis=0)
        return T2_phys, T2_hier

    @staticmethod
    def _determine_basis_from_list(error_by_member, max_error, list_member):
        """
        Determines the members of a list that must be kept in order for the total
        error (terms that are dropped) to be below the max_error value.

        Parameters
        ----------
        1. error_by_member : np.array
                             Array of error values.

        2. max_error : float
                       Maximum error value.

        3. list_member : np.array
                         Array of members.

        Returns
        -------
        1. list_index : np.array
                        Array of indices for the members.

        2. list_new_member : list
                             List of the members.
        """
        error_thresh = determine_error_thresh(np.sort(error_by_member), max_error)
        list_index = np.where(error_by_member > error_thresh)[0]
        list_new_member = [list_member[i_aux] for i_aux in list_index]
        return (list_index, list_new_member)

    @property
    def M2_mode_from_state_diag(self):
        # See Eq S26 of the SI of "Characterizing the Role of Peierls Vibrations in
        # Singlet Fission with the Adaptive Hierarchy of Pure States," available at
        # https://arxiv.org/abs/2505.02292.
        Row = []
        Col = []
        Data = []
        for mode in range(self.n_hmodes):
            L2 = self.mode.list_L2_coo[self.mode.list_index_L2_by_hmode[mode]]
            # Find COO format entries associated with portions of the L-operators.
            list_index_diag_mask = np.where(L2.row == L2.col)[0]
            # Row of the M2 matrix is the relative mode index, column is the absolute
            # state index.
            Row += [mode] * len(list_index_diag_mask)
            Col += list(L2.col[list_index_diag_mask])
            Data += list(L2.data[list_index_diag_mask])
        return sparse.csr_array(
            (Data, (Row, Col,),), shape=(self.n_hmodes, self.n_state),
            dtype=np.complex128
        )

    @property
    def list_M2_by_dest_off_diag(self):
        # See Eq S27 of the SI of "Characterizing the Role of Peierls Vibrations in
        # Singlet Fission with the Adaptive Hierarchy of Pure States," available at
        # https://arxiv.org/abs/2505.02292.
        if not self.off_diagonal_couplings:
            return []
        list_M2 = []
        list_L2_csr = self.list_L2_csr
        for dest in self.system.list_destination_state:
            Row = []
            Col = []
            Data = []
            state_list_off_diag = [s for s in self.system.state_list if s != dest]
            for m, mode in enumerate(self.mode.list_absindex_mode):
                # Gets the index of the unique L-operator associated with the mode in
                # list_L2_csr.
                lind = self.mode.list_index_L2_by_hmode[m]

                # Gets the indices of the sparse data points in the mode's L-operator
                # corresponding to the proper destination state: for L[d,s]psi[s] -->
                # psi[d], d would be the destination state. This excludes data points
                # for which the column index m is not in the current state basis,
                # as there can be no flux to destination states from unoccupied
                # states. We also exclude the diagonal portion of each L-operator,
                # L[d,d].
                L_reduced = sparse.coo_array((list_L2_csr[lind])[[dest],
                state_list_off_diag])

                # Row is given by the relative index of the mode in question.
                Row += [self.mode.dict_relative_index_by_mode[mode]] * len(L_reduced.col)

                # Column and data are given by the sparse data points of the L-operator
                # corresponding to the proper destination state: that is, fluxes into
                # the destination state via the L-operator of the mode in question.
                # Note that the absolute state index from the full L-operator must be
                # converted to a relative state index.
                Col += [self.system.dict_relative_index_by_state[state_list_off_diag[c]]
                        for c in L_reduced.col]
                Data += list(L_reduced.data)

            # For each destination state, we generate an M2_mode_from_state sparse
            # matrix.
            list_M2.append(sparse.csr_array(
                (Data, (Row, Col)), shape=(self.n_hmodes, self.n_state),
                dtype=np.complex128
            ))
        return list_M2

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        # Manages generation of the list of <L> expectation values. This setter is
        # convenient as it a) avoids calculating expectation values repeatedly and b)
        # makes testing the list of <L> easy.
        self._psi = psi
        self.__list_avg_L2 = np.array([operator_expectation(L2, self.psi)
                                       for L2 in self.mode.list_L2_csr])  # <L>

    @property
    def list_avg_L2(self):
        return self.__list_avg_L2

    @property
    def X2_exp_lop_mode_state(self):
        # See Eq S28 of the SI of "Characterizing the Role of Peierls Vibrations in
        # Singlet Fission with the Adaptive Hierarchy of Pure States," available at
        # https://arxiv.org/abs/2505.02292.

        # The L_m[d,s] values in list_M2_mode_from_state are stored in the matrix
        # entry Md[m,s]. In flux down, we modulate these L_m matrices by <L_m>I,
        # which is to say that Md[m,d] = <L_m>. In other words, if destination state
        # d is a state in the state basis, then the row corresponding to the d in that
        # destination state's mode_from_state matrix will be a list of the
        # expectation values of the L-operators for all modes in the current basis.
        Row = list(range(self.mode.n_hmodes))*len(self.system.state_list)
        Col = list(np.repeat([self.system.dict_relative_index_by_state[state]
                             for state in self.system.state_list],
                   self.mode.n_hmodes))
        Data = [self.list_avg_L2[index] for index in
                self.mode.list_index_L2_by_hmode]*len(self.system.state_list)
        return sparse.csr_array( (Data, (Row, Col)), shape=(self.n_hmodes,
                                                             self.n_state),
                                  dtype=np.complex128)
    @property
    def K2_aux_by_mode(self):
        # Construct the array values of k[n] in the space of [mode, aux]
        K2_aux_by_mode = np.zeros([self.n_hmodes, self.n_hier], dtype=np.uint8)
        for aux in self.hierarchy.auxiliary_list:
            array_index = np.array([list(self.list_absindex_mode).index(mode)
                                    for (mode, value) in aux.tuple_aux_vec
                                    if mode in self.list_absindex_mode],
                                   dtype=int)
            array_values = [np.uint8(value) for (mode, value) in aux.tuple_aux_vec
                            if mode in self.list_absindex_mode]
            K2_aux_by_mode[array_index, aux._index] = array_values
        return K2_aux_by_mode

    @property
    def n_hmodes(self):
        return np.size(self.mode.list_absindex_mode)

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
    def delta_a(self):
        return self.eom.param["DELTA_A"]

    @property
    def delta_s(self):
        return self.eom.param["DELTA_S"]

    @property
    def eom_type(self):
        return self.eom.param["EQUATION_OF_MOTION"]
    
    @property
    def f_discard(self):
        return self.eom.param["F_DISCARD"]

    @property
    def list_absindex_mode(self):
        return self.mode.list_absindex_mode

    @property
    def list_w(self):
        return self.mode.list_w

    @property
    def list_g(self):
        return self.mode.list_g

    @property
    def list_L2_csr(self):
        # Unlike the version in HopsModes, these are not reduced to the current state
        # basis.
        return np.array([sparse.csr_array(self.system.param["LIST_L2_COO"][l]) for
                         l in self.mode.list_absindex_L2])

    @property
    def Z2_noise_sparse(self):
        return self._Z2_noise_sparse

    @property
    def T2_ltc_phys(self):
        return self._T2_ltc_phys

    @property
    def T2_ltc_hier(self):
        return self._T2_ltc_hier

    @property
    def off_diagonal_couplings(self):
        return any(self.mode.list_off_diag_active_mask)

    @property
    def lt_corr_param(self):
        # Unlike the list of lt_corr_param in HopsSystem, this uses the L-operator
        # indices from all modes in the basis for consistency's sake.
        return np.array(self.mode.list_lt_corr_param_mode_indexing)

    @property
    def ltc_active(self):
        return any(self.lt_corr_param)
