import copy
import numpy as np
from mesohops.dynamics.basis_functions_adaptive import (error_sflux_hier,
                                                      error_deriv,
                                                      error_flux_up,
                                                      error_flux_down,
                                                      error_sflux_stable_state,
                                                      error_sflux_boundary_state)
from mesohops.dynamics.basis_functions import determine_error_thresh, calculate_delta_bound
from mesohops.dynamics.hops_fluxfilters import HopsFluxFilters
from mesohops.dynamics.hops_modes import HopsModes
from scipy import sparse
__title__ = "Basis Class"
__author__ = "D. I. G. Bennett, Brian Citty"
__version__ = "1.4"


class HopsBasis:
    """
    Every HOPS calculation is defined by the HopsSystem, HopsHierarchy, and HopsEOM
    classes (and their associated parameters). These form the basis set for the
    calculation. HopsBasis is the class that contains all of these sub-classes and
    mediates the way the HopsTrajectory interacts with them.
    """

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
            e. DELTA_H
            f. DELTA_S

        Returns
        -------
        None
        """
        self.system = system
        self.hierarchy = hierarchy
        self.mode = HopsModes(system)
        self.eom = eom
        self.flag_gcorr = False
        if self.eom.param["EQUATION_OF_MOTION"] == "NONLINEAR ABSORPTION":
            self.flag_gcorr = True
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
        # ==========================================
        # =======      Calculate Updates      ======
        # ==========================================

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
                Φ/np.linalg.norm(Φ[:self.n_state]), delta_t, z_step, list_index_stable_aux, list_aux_bound
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
        self.system.state_list = np.array(list_state_new, dtype=int)

        # Update Hierarchy List
        # =====================
        flag_update_hierarchy = False
        if set(list_aux_new) != set(self.hierarchy.auxiliary_list): flag_update_hierarchy = True
        self.hierarchy.auxiliary_list = list_aux_new

        # Update Mode List
        # ================
        if flag_update_state or flag_update_hierarchy:
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

    def _define_state_basis(self, Φ, delta_t, z_step, list_index_aux_stable, list_aux_bound):
        """
        Determines the states which should be included in the adaptive integration
        for the next time point.

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

        5. list_aux_bound: list(HopsAux)
                           List of the auxiliary objects in the Boundary Auxiliary Basis.

        Returns
        -------
        1. list_state_stable : np.array
                               Array of stable states (absolute state index, S_S).

        2. list_state_boundary : np.array
                                 Array of the boundary states (absolute state index, S_B).
        """
        # Define Constants
        # ----------------
        delta_state_sq = self.delta_s ** 2

        # CONSTRUCT STABLE STATE (S_S)
        # ============================

        # Construct Error For Excluding Member of S_t
        # -------------------------------------------
        error_by_state_sq = self.state_stable_error(
            Φ, delta_t, z_step, list_index_aux_stable, list_aux_bound
        )

        # Determine the Stable States (S_S)
        # ---------------------------------
        list_relindex_state_stable, list_state_stable = self._determine_basis_from_list(
            error_by_state_sq, delta_state_sq / 2.0, self.system.state_list
        )

        # CONSTRUCT BOUNDARY STATE (S_B)
        # ==============================

        # Establish the error available for the boundary states
        # -----------------------------------------------------
        list_index_state_unstable = np.setdiff1d(np.arange(self.n_state), list_relindex_state_stable)
        stable_error_sq = np.sum(error_by_state_sq[list_index_state_unstable])
        
        delta_bound_sq = calculate_delta_bound(delta_state_sq, stable_error_sq)
        
        # Construct Error for Excluding Member of S_t^C
        # ---------------------------------------------
        list_sc = np.setdiff1d(np.arange(self.system.param["NSTATES"]),
                               self.system.state_list)
        
        list_index_nonzero, list_error_nonzero = error_sflux_boundary_state(
            Φ, list_state_stable, list_sc, self.n_state, self.n_hier,
            self.system.param["SPARSE_HAMILTONIAN"],
            list_relindex_state_stable,
            list_index_aux_stable
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
        integration for the next time point.

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
        # Define Constants
        # ----------------
        delta_hier_sq = self.delta_h ** 2

        # CONSTRUCT STABLE HIERARCHY
        # ==========================

        # Construct Error For Excluding Member of A_t
        # --------------------------------------------
        error_by_aux_sq, list_e2_kflux = self.hier_stable_error(Φ, delta_t, z_step)
        
        
        
        # Determine the Stable Auxiliaries (A_S)
        # --------------------------------------------------
        list_index_aux_stable, list_aux_stable = self._determine_basis_from_list(
            error_by_aux_sq, delta_hier_sq/2.0, self.hierarchy.auxiliary_list
        )
        
        # CONSTRUCT BOUNDARY HIERARCHY
        # ============================

        # Establish the error available for the boundary auxiliaries
        # ----------------------------------------------------------
        list_index_aux_unstable = np.setdiff1d(np.arange(self.n_hier), list_index_aux_stable)
        stable_error_sq = np.sum(error_by_aux_sq[list_index_aux_unstable])
        
        delta_bound_sq = calculate_delta_bound(delta_hier_sq, stable_error_sq)
        
        # Determine the Boundary Auxiliaries (A_B)
        # -----------------------------------------------------
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
        Finds the total error associated with removing k in A_t.

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
        # =======================================
        # ===    Build Array Constructions    ===
        # =======================================

       
        # Construct the L_m[s,s] values in the space of [mode, states]
        # ------------------------------------------------------------
        M2_mode_from_state = self.M2_mode_from_state

        # ===================================
        # ===    Calculate Error Terms    ===
        # ===================================

        # Calculate the error term by term
        # --------------------------------
        E1_error = np.sum(error_deriv(self.eom.dsystem_dt, Φ, z_step,
                                    self.n_state, self.n_hier, delta_t), axis=0)

        E1_error += error_sflux_hier(Φ, self.system.state_list,
                                         self.n_state, self.n_hier,
                                         self.system.param["SPARSE_HAMILTONIAN"])

        E2_flux_up_nofilter = error_flux_up(Φ, self.n_state, self.n_hier,
                                            self.n_hmodes, self.w,
                                            self.K2_aux_by_mode,
                                            M2_mode_from_state,
                                            "H")
        # Filter for Stable Hierarchy, Flux Up
        # -------------------------------------
        F2_filter = self.flux_filters.construct_filter_auxiliary_stable_up()
        F2_filter *= self.flux_filters.construct_filter_markov_up()
        
        E1_error += np.sum(F2_filter * E2_flux_up_nofilter,axis=0)

        E2_flux_down_nofilter = error_flux_down(Φ, self.n_state, self.n_hier, self.n_hmodes,
                                                self.g, self.w,
                                                M2_mode_from_state, "H",
                                                flag_gcorr=self.flag_gcorr)
                                                
        # Filter for Stable Hierarchy, Flux Down
        # --------------------------------------
        F2_filter = self.flux_filters.construct_filter_auxiliary_stable_down()                                        
        E1_error += np.sum(F2_filter * E2_flux_down_nofilter,axis=0)
        

        return (
            E1_error,
            [E2_flux_up_nofilter, E2_flux_down_nofilter],
        )

    def _determine_boundary_hier( self, list_e2_kflux_up_down, list_index_aux_stable,
                                  delta_bound_sq, f_discard):
        """
        Determines the set of boundary auxiliaries for the next time step.

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
        # Filter for Boundary Auxiliary, Flux Up
        # --------------------------------------
        F2_filter = self.flux_filters.construct_filter_auxiliary_boundary_up()
        F2_filter *= self.flux_filters.construct_filter_markov_up()
        list_e2_kflux_up_down[0] = list_e2_kflux_up_down[0] * F2_filter
        
        # Filter for Boundary Auxiliary, Flux Up
        # --------------------------------------
        F2_filter = self.flux_filters.construct_filter_auxiliary_boundary_down()
        list_e2_kflux_up_down[1] = list_e2_kflux_up_down[1] * F2_filter
        
        
        # Apply filters: flux from stable auxiliaries (A_s)
        # -------------------------------------------------
        list_e2_kflux_up_down[0] = list_e2_kflux_up_down[0][:,list_index_aux_stable]
        list_e2_kflux_up_down[1] = list_e2_kflux_up_down[1][:, list_index_aux_stable]

        # Filter out small errors
        # -----------------------
        E1_nonzero_flux = list_e2_kflux_up_down[0][list_e2_kflux_up_down[0] != 0]
        
        sorted_error = np.sort(
            np.append(E1_nonzero_flux, list_e2_kflux_up_down[1][list_e2_kflux_up_down[1] != 0])
        )
        
        error_thresh = determine_error_thresh(sorted_error, delta_bound_sq*f_discard*f_discard)
        
        
        discarded_error = np.sum(list_e2_kflux_up_down[0][list_e2_kflux_up_down[0] <= error_thresh])
        discarded_error += np.sum(list_e2_kflux_up_down[1][list_e2_kflux_up_down[1] <= error_thresh])

        list_e2_kflux_up_down[0][list_e2_kflux_up_down[0] <= error_thresh] = 0
        list_e2_kflux_up_down[1][list_e2_kflux_up_down[1] <= error_thresh] = 0
        
        # Find the error threshold for edge auxiliaries
        # ---------------------------------------------
        boundary_aux_dict = {}
        boundary_connect_dict = {}

        # Bin error flux by boundary aux
        # -------------------------------
        list_stable_aux = [self.hierarchy.auxiliary_list[i] for i in range(self.n_hier)
                           if i in list_index_aux_stable]
        for (i_aux,aux) in enumerate(list_stable_aux):

            # Flux Up Error
            nonzero_modes_up = list_e2_kflux_up_down[0][:,i_aux].nonzero()[0]
            if(len(nonzero_modes_up) > 0):
                #Get the id values for boundary auxiliaries up along modes with nonzero flux
                list_id_up, list_value_connect,list_mode_connect = aux.get_list_id_up(self.list_absindex_mode[nonzero_modes_up])
                #For each id up, add the flux error to its entry in the boundary_aux_dict dictionary.
                #We assume that the filter is constructed correctly and that these are all indeed boundary auxiliaries.
                #If it is the first flux for a boundary auxiliary, we keep track of the connection in the boundary_connect_dict, so we can use the 
                #e_step method to construct the new Auxiliary as before.
                for (id_ind,my_id) in enumerate(list_id_up):
                    try: 
                        boundary_aux_dict[my_id] += list_e2_kflux_up_down[0][nonzero_modes_up[id_ind],i_aux]
                    except:
                        boundary_aux_dict[my_id] = list_e2_kflux_up_down[0][nonzero_modes_up[id_ind],i_aux]
                        boundary_connect_dict[my_id] = [aux, list_mode_connect[id_ind], 1]

            # Flux Down Error
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
        # --------------------------------------------
                        
        sorted_error = np.sort(np.array(list(boundary_aux_dict.values())))
        error_thresh = determine_error_thresh(sorted_error, delta_bound_sq, offset=discarded_error)
        
        # Identify and construct boundary auxiliaries
        # -------------------------------------------
        list_aux_updown = [boundary_connect_dict[aux_id][0].e_step(boundary_connect_dict[aux_id][1],
                                                                     boundary_connect_dict[aux_id][2])
                              for aux_id in boundary_aux_dict.keys() if boundary_aux_dict[aux_id] > error_thresh]

        return list_aux_updown

    def state_stable_error(self, Φ, delta_t, z_step, list_index_aux_stable, list_aux_bound):
        """
        Finds the total error associated with neglecting n in the state basis S_t.
        This includes the error associated with deleting a state from the basis,
        the error associated with losing all flux into a state, and the error associated
        with losing all flux out of a state.

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

        Returns
        -------
        1. error : np.array
                   List of error associated with removing each state.
        """
 

        # Construct Array Inputs
        # ======================
        # Construct the L_m[s,s] values in the space of [mode, states]
        # ------------------------------------------------------------
        M2_mode_from_state = self.M2_mode_from_state
        
        # Construct the Error Terms
        # =========================
        E1_error = np.sum(error_deriv(self.eom.dsystem_dt, Φ, z_step,
                                     self.n_state, self.n_hier, delta_t,
                                     list_index_aux_stable),axis=1)
        E1_error += error_sflux_stable_state(Φ, self.n_state, self.n_hier,
                                          self.system.param["SPARSE_HAMILTONIAN"],
                                          list_index_aux_stable,
                                          self.system.state_list)
                                          
        # Filter for Stable State, Flux Up
        # --------------------------------
        F2_filter_boundary = self.flux_filters.construct_filter_state_stable_down(
            list_aux_bound)   
        E1_error += np.sum(error_flux_down(Φ, self.n_state, self.n_hier, self.n_hmodes, self.g, self.w,
                                       M2_mode_from_state, "S", flag_gcorr=self.flag_gcorr,
                                       F2_filter=F2_filter_boundary)[:, list_index_aux_stable],axis=1)
                                     
        # Filter for Stable State, Flux Down
        # ----------------------------------
        F2_filter_boundary = self.flux_filters.construct_filter_state_stable_up(
            list_aux_bound)                                 
        E1_error += np.sum(error_flux_up(Φ, self.n_state, self.n_hier, self.n_hmodes,
                                         self.w, self.K2_aux_by_mode,
                                         M2_mode_from_state, "S",
                                         F2_filter=F2_filter_boundary)[:, list_index_aux_stable],axis=1)

        # Compress the error onto the state/mode axis
        # -------------------------------------------
        return E1_error

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
    def M2_mode_from_state(self):
        Row = []
        Col = []
        Data = []
        for mode in range(self.n_hmodes):
            Row += [mode] * len(self.mode.list_L2_coo[self.mode.list_index_L2_by_hmode[mode]].row)
            Col += list(self.mode.list_L2_coo[self.mode.list_index_L2_by_hmode[mode]].col)
            Data += list(self.mode.list_L2_coo[self.mode.list_index_L2_by_hmode[mode]].data)
        return sparse.csc_matrix(
	          (Data, (Row, Col,),),shape=(self.n_hmodes,self.n_state),
        )

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
    def delta_h(self):
        return self.eom.param["DELTA_H"]

    @property
    def delta_s(self):
        return self.eom.param["DELTA_S"]
    
    @property
    def f_discard(self):
        return self.eom.param["F_DISCARD"]

    @property
    def list_absindex_mode(self):
        return self.mode.list_absindex_mode

    @property
    def w(self):
        return self.mode.w

    @property
    def g(self):
        return self.mode.g

