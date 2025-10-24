import numpy as np


class HopsFluxFilters:
    """
    Adaptive HOPS calculations use a variety of filtering arrays to ensure that only the
    appropriate flux terms are associated with each error calculation. This is a class
    that will build and manage these filters.

    In the current version of the code (1.2x) we expect the filters to be built each
    time they are required. In the future, though, we will want to implement an
    iterative construction of the filters that uses the previous information.
    """

    __slots__ = (
        # --- Core basis components ---
        'system',        # System parameters and operators (HopsSystem)
        'hierarchy',     # Hierarchy management (HopsHierarchy)
        'mode',          # Mode management (HopsModes)

        # --- State-basis flux filters ---
        # These filters manage flux between states in the hierarchy
        '_list_F2_filter_state_stable_up',      # Filter for upward flux between stable states
        '_list_F2_filter_state_stable_down',    # Filter for downward flux between stable states
        '_list_F2_filter_state_boundary_up',    # Filter for upward flux to boundary states
        '_list_F2_filter_state_boundary_down',  # Filter for downward flux to boundary states

        # --- Auxiliary-basis flux filters ---
        # These filters manage flux between auxiliary wavefunctions
        '_list_F2_filter_aux_stable_up',        # Filter for upward flux between stable auxiliaries
        '_list_F2_filter_aux_stable_down',      # Filter for downward flux between stable auxiliaries
        '_list_F2_filter_aux_boundary_up',      # Filter for upward flux to boundary auxiliaries
        '_list_F2_filter_aux_boundary_down'     # Filter for downward flux to boundary auxiliaries
    )

    def __init__(self, system, hierarchy, mode):
        """
        Parameters
        ----------
        1. system : instance(HopsSystem)

        2. hierarchy : instance(HopsHierarchy)

        3. mode : instance(HopsModes)

        Returns
        -------
        None
        """
        self.system = system
        self.hierarchy = hierarchy
        self.mode = mode

    def construct_filter_auxiliary_stable_up(self):
        """
        Constructs a filter array that ensures that flux up from members of the
        current auxiliary basis is only counted if flux goes to an auxiliary wave
        function allowed by the triangular truncation condition. This function
        corresponds to the filter detailed between Eqs S37 and S38 from the SI of
        "Characterizing the Role of Peierls Vibrations in Singlet Fission with the
        Adaptive Hierarchy of Pure States," available at
        https://arxiv.org/abs/2505.02292.

        Parameters
        ----------
        None

        Returns
        -------
        1. F2_filter : np.array(bool)
                       True indicates the associated flux is allowed (not
                       filtered out) while False indicates otherwise
                       (positioning is (mode, aux)).
        """
        # Filter out flux to aux with depth>k_max
        # ---------------------------------------
        # Start by assuming that all flux is allowed
        F2_filter = np.ones([self.n_hmodes, len(self.hierarchy.auxiliary_list)],dtype=bool)
        list_index_terminator_aux = [aux._index for aux in self.hierarchy.auxiliary_list
                                     if aux._sum == self.hierarchy.param['MAXHIER']]

        if len(list_index_terminator_aux) > 0:
            F2_filter[:, np.array(list_index_terminator_aux)] = False

        return F2_filter

    def construct_filter_auxiliary_stable_down(self):
        """
        Constructs a filter array that ensures that flux down from members of the
        current auxiliary basis is only counted if flux goes an auxiliary wave
        function that has no negative indices.  This function corresponds to the filter
        detailed between Eqs S39 and S40 from the SI of "Characterizing the Role of
        Peierls Vibrations in Singlet Fission with the Adaptive Hierarchy of Pure
        States," available at https://arxiv.org/abs/2505.02292.

        Parameters
        ----------
        None

        Returns
        -------
        1. F2_filter_any_m1 : np.array(bool)
                              True indicates the associated flux is allowed (not
                              filtered out) while False indicates otherwise
                              (positioning is (mode, aux)).
        """
        # Filter for Stable Hierarchy, Flux Down
        # --------------------------------------
        # Start by assuming that all flux is not allowed
        F2_filter_any_m1 = np.zeros([self.n_hmodes, len(self.hierarchy.auxiliary_list)],dtype=bool)

        # Now find the allowed flux down along modes that have non-zero indices
        list_absindex_mode = list(self.mode.list_absindex_mode)
        for aux in self.hierarchy.auxiliary_list:
            array_index2 = np.array(
                [list_absindex_mode.index(mode) for mode in aux.keys()
                 if mode in list_absindex_mode], dtype=int)

            F2_filter_any_m1[array_index2, aux._index] = True

        return F2_filter_any_m1

    def construct_filter_auxiliary_boundary_up(self):
        """
        Constructs a filter array that ensures that only flux up to boundary
        auxiliaries (not present in the current basis) is considered, and then only
        when the boundary auxiliary is allowed by the triangular truncation condition.
        This function corresponds to the first filter detailed after Eq 43 from the
        SI of "Characterizing the Role of Peierls Vibrations in Singlet Fission with the
        Adaptive Hierarchy of Pure States," available at
        https://arxiv.org/abs/2505.02292.

        NOTE: Additional filtering to subset to flux originating from stable
        auxiliaries is done in _determine_boundary_hier.

        Parameters
        ----------
        None

        Returns
        -------
        1. F2_filter_p1 : np.array(bool)
                          True indicates the associated flux is allowed (not
                          filtered out) while False indicates otherwise
                          (positioning is (mode, aux)).
        """
        # Filter for Boundary Auxiliary, Flux Up
        # --------------------------------------
        F2_filter_p1 = np.ones([self.n_hmodes, len(self.hierarchy.auxiliary_list)],dtype=bool)
        list_absindex_mode = list(self.mode.list_absindex_mode)
        for aux in self.hierarchy.auxiliary_list:
            if aux._sum < self.hierarchy.param['MAXHIER']:
                # Remove flux that contributes to an aux in A_t
                # ---------------------------------------------
                array_index = np.array([list_absindex_mode.index(mode)
                                        for mode in aux.dict_aux_p1.keys()
                                        if mode in list_absindex_mode],
                                       dtype=int)
                F2_filter_p1[array_index, aux._index] = False
            else:
                # Remove flux that would contribute to an aux beyond maximum hier depth
                F2_filter_p1[:, aux._index] = False

        return F2_filter_p1

    def construct_filter_auxiliary_boundary_down(self):
        """
        Constructs a filter array that ensures that only flux down to boundary
        auxiliaries (not present in the current basis) is considered, and then only
        when the boundary auxiliary has no negative indices in its indexing vector.
        This function corresponds to the second filter detailed after Eq 43 from the
        SI of "Characterizing the Role of Peierls Vibrations in Singlet Fission with the
        Adaptive Hierarchy of Pure States," available at
        https://arxiv.org/abs/2505.02292.

        NOTE: Additional filtering to subset to flux originating from stable
        auxiliaries is done in _determine_boundary_hier.

        Parameters
        ----------
        None

        Returns
        -------
        1. F2_filter_m1 : np.array(bool)
                          True indicates the associated flux is allowed (not
                          filtered out) while False indicates otherwise
                          (positioning is (mode, aux)).
        """
        # Filter for Boundary Auxiliary, Flux Down
        # ----------------------------------------
        list_absindex_mode = list(self.mode.list_absindex_mode)

        # Assume all fluxes are allowed
        F2_filter_m1 = np.ones([self.n_hmodes, len(self.hierarchy.auxiliary_list)],dtype=bool)

        for aux in self.hierarchy.auxiliary_list:
            if list(aux.dict_aux_m1.keys()) != list(aux.keys()):
                # Filter out flux to auxiliaries present in the previous basis
                array_index = np.array([list_absindex_mode.index(mode) for mode in
                                        aux.dict_aux_m1.keys() if mode in list_absindex_mode],
                                       dtype=int)
                F2_filter_m1[array_index, aux._index] = False

                # Also filter out connections from modes that are not ever in auxiliary
                # basis for example, all modes in the main auxiliary will be filtered
                # here, all but one mode in first-order auxiliaries, two modes in
                # second-order auxiliaries, etc.
                array_index2 = np.array([list_absindex_mode.index(mode) for mode in
                                         aux.keys() if mode in list_absindex_mode], dtype=int)
                array_index2 = np.setdiff1d(np.arange(self.n_hmodes), array_index2)

                F2_filter_m1[array_index2, aux._index] = False
            else:
                F2_filter_m1[:, aux._index] = False

        return F2_filter_m1

    def construct_filter_state_stable_down(self, list_aux_bound):
        """
        Constructs a filter array that ensures that only flux down to auxiliaries in
        list_aux_bound from higher-lying auxiliaries in the current basis is
        considered, to avoid double-counting flux down inside of a state (which is
        calculated in error_deriv).

        For the off-diagonal portion of state flux down, the filter ensures that only
        flux down to auxiliaries in the full newly-generated auxiliary basis from
        higher-lying auxiliaries in the current basis is considered, as these flux
        terms go from one state to another. This is accomplished by inputting the
        full list of auxiliaries in the new basis rather than the list of only
        boundary auxiliaries included in the new basis.

        This function corresponds to the second filter detailed after Eq 49 and
        the second filter detailed after Eq 51, depending on the list_aux_bound
        parameter input, from the SI of "Characterizing the Role of  Peierls
        Vibrations in Singlet Fission with the Adaptive Hierarchy of Pure States,"
        available at https://arxiv.org/abs/2505.02292.

        NOTE: Additional filtering to subset to flux originating from stable
        auxiliaries is done in state_stable_error.

        Parameters
        ----------
        1. list_aux_bound : list(HopsAux)
                            List of boundary auxiliaries (or all auxiliaries in the
                            newly-generated auxiliary basis for the off-diagonal
                            portion of flux down).

        Returns
        -------
        1. F2_filter : np.array(bool)
                       True indicates the associated flux is allowed (not
                       filtered out) while False indicates otherwise
                       (positioning is (mode, aux)).
        """
        F2_filter = np.zeros([self.n_hmodes, self.n_hier],dtype=bool)
        for aux in list_aux_bound:
            list_id_up, list_value_connects, list_mode_connect = \
                aux.get_list_id_up(self.mode.list_absindex_mode)
            for (rel_ind,my_id) in enumerate(list_id_up):
                if (my_id in self.hierarchy.dict_aux_by_id.keys()):
                    aux_up = self.hierarchy.dict_aux_by_id[my_id]
                    F2_filter[rel_ind, aux_up._index] = True
        return F2_filter

    def construct_filter_state_stable_up(self, list_aux_bound):
        """
        Constructs a filter array that ensures that only flux up to auxiliaries in
        list_aux_bound from lower-lying auxiliaries in the current basis is
        considered, to avoid double-counting flux up inside of a state (which is
        calculated in error_deriv).

        For the off-diagonal portion of state flux up, the filter ensures that
        only flux up to auxiliaries in the full newly-generated auxiliary basis from
        lower-lying auxiliaries in the current basis is considered, as these flux
        terms go from one state to another. This is accomplished by inputting the
        full list of auxiliaries in the new basis rather than the list of only
        boundary auxiliaries included in the new basis.

        This function corresponds to the first filter detailed after Eq 49 and
        the first filter detailed after Eq 51, depending on the list_aux_bound
        parameter input, from the SI of "Characterizing the Role of  Peierls
        Vibrations in Singlet Fission with the Adaptive Hierarchy of Pure States,"
        available at https://arxiv.org/abs/2505.02292.

        NOTE: Additional filtering to subset to flux originating from stable
        auxiliaries is done in state_stable_error.

        Parameters
        ----------
        1. list_aux_bound : list(HopsAux)
                            List of boundary auxiliaries (or all auxiliaries in the
                            newly-generated auxiliary basis for the off-diagonal
                            portion of flux up).

        Returns
        -------
        1. F2_filter : np.array(bool)
                       True indicates the associated flux is allowed (not
                       filtered out) while False indicates otherwise
                       (positioning is (mode, aux)).
        """
        F2_filter = np.zeros([self.n_hmodes, self.n_hier],dtype=bool)
        for aux in list_aux_bound:
            list_ids_down, list_id_values, list_mode_connects = aux.get_list_id_down()
            for (rel_ind, my_id) in enumerate(list_ids_down):
                if (my_id in self.hierarchy.dict_aux_by_id.keys()):
                    aux_down = self.hierarchy.dict_aux_by_id[my_id]
                    F2_filter[list(self.mode.list_absindex_mode).index(list_mode_connects[
                                                                  rel_ind]), aux_down._index] = True
        return F2_filter

    def construct_filter_markov_up(self):
        """
        Constructs a filter array that accounts for the fact that Markovian-filtered
        modes only allow flux between first-order auxiliaries and the physical wave
        function. The filter generation supports multiple instances of the same
        filter: the array begins with F[m,k] = 1 everywhere, and each Markovian filter
        sets the appropriate elements to 0.

        NOTE: Assumes that list_aux only contains legal auxiliaries with
        respect to the Markovian filter.

        Parameters
        ----------
        None

        Returns
        -------
        1. F2_filter : np.array(bool)
                       True indicates the associated flux is allowed (not
                       filtered out) while False indicates otherwise
                       (positioning is (mode, aux)).
        """
        if len(self.mode.list_absindex_mode) == 0:
            return True

        M2_mark_filtered_modes = np.array(
            [
                np.array([param[m] for m in self.mode.list_absindex_mode])
                for (name, param) in self.hierarchy.param["STATIC_FILTERS"]
                if name == "Markovian"
            ]
        )
        # If there is no filter, return a scalar for lightweight multiplication
        if len(M2_mark_filtered_modes) == 0:
            return True

        F2_filter = np.ones([self.n_hmodes, self.n_hier], dtype=bool)
        if len(M2_mark_filtered_modes) > 0:
            # Determine which modes are Markovian
            # -----------------------------------
            M1_filtered_mode_mask = np.any(M2_mark_filtered_modes, axis=0)

            # Remove flux along Markovian Modes except
            # from the physical wave function
            # -----------------------------------------
            F2_filter[M1_filtered_mode_mask, 1:] = False

            # Remove flux from any mode for the first order
            # wave functions along Markovian modes
            # ---------------------------------------------
            aux0 = self.hierarchy.auxiliary_list[0]
            mark_aux1 = np.array([aux0.dict_aux_p1[mode]._index for mode in
                                  aux0.dict_aux_p1.keys() if mode in
                                  self.mode.list_absindex_mode[M1_filtered_mode_mask]])
            if len(mark_aux1) > 0:
                F2_filter[:, mark_aux1] = False

        return F2_filter

    def construct_filter_triangular_up(self):
        """
        Constructs a filter array that accounts for the fact that a secondary
        triangular filter only allows flux up to auxiliary wave functions that do not
        have a total depth greater than some specified depth in the filtered subset
        of modes. The filter generation supports multiple instances of the same
        filter: the array begins with F[m,k] = 1 everywhere, and each triangular filter
        sets the appropriate elements to 0.

        NOTE: Assumes that list_aux only contains legal auxiliaries with
        respect to the triangular filter.

        Parameters
        ----------
        None

        Returns
        -------
        1. F2_filter : np.array(bool)
                       True indicates the associated flux is allowed (not
                       filtered out) while False indicates otherwise
                       (positioning is (mode, aux)).
        """
        if len(self.mode.list_absindex_mode) == 0:
            return True

        M2_tri_filtered_modes = np.array(
            [
                np.array([param[0][m] for m in self.mode.list_absindex_mode])
                for (name, param) in self.hierarchy.param["STATIC_FILTERS"]
                if name == "Triangular"
            ]
        )
        # If there is no filter, return a scalar for lightweight multiplication
        if len(M2_tri_filtered_modes) == 0:
            return True

        F2_filter = np.ones([self.n_hmodes, self.n_hier],dtype=bool)

        list_kmax_2 = [param[1] for (name, param) in self.hierarchy.param[
            "STATIC_FILTERS"] if name == "Triangular"]

        for i in range(len(M2_tri_filtered_modes)):
            M1_filtered_mode_mask = M2_tri_filtered_modes[i]
            # Determine which modes are filtered
            # -----------------------------------
            list_modes_filtered = self.mode.list_absindex_mode[M1_filtered_mode_mask]
            kmax_2 = list_kmax_2[i]
            for aux in self.hierarchy.auxiliary_list:
                # If the sum of the depth in the filtered modes would be greater than
                # kmax_2 if we took a step up, that flux up is filtered out.
                if np.sum(aux.get_values_nonzero(list_modes_filtered)) >= kmax_2:
                    F2_filter[M1_filtered_mode_mask, aux._index] = False

        return F2_filter

    def construct_filter_longedge_up(self):
        """
        Constructs a filter array that accounts for the fact that a longedge filter
        only allows flux up to auxiliary wave functions below a specified depth,
        unless they are edge terms (have depth in only one mode) or have no depth in the
        filtered subset of modes. The filter generation supports multiple instances of
        the same filter: the array begins with F[m,k] = 1 everywhere, and each
        longedge filter sets the appropriate elements to 0.

        NOTE: Assumes that list_aux only contains legal auxiliaries with
        respect to the longedge filter.

        Parameters
        ----------
        None

        Returns
        -------
        1. F2_filter : np.array(bool)
                       True indicates the associated flux is allowed (not
                       filtered out) while False indicates otherwise
                       (positioning is (mode, aux)).
        """
        if len(self.mode.list_absindex_mode) == 0:
            return True

        M2_le_filtered_modes = np.array(
            [
                np.array([param[0][m] for m in self.mode.list_absindex_mode])
                for (name, param) in self.hierarchy.param["STATIC_FILTERS"]
                if name == "LongEdge"
            ]
        )
        # If there is no filter, return a scalar for lightweight multiplication
        if len(M2_le_filtered_modes) == 0:
            return True

        F2_filter = np.ones([self.n_hmodes, self.n_hier], dtype=bool)

        list_kmax_2 = [param[1] for (name, param) in self.hierarchy.param[
            "STATIC_FILTERS"] if name == "LongEdge"]
        for i in range(len(M2_le_filtered_modes)):
            M1_filtered_mode_mask = M2_le_filtered_modes[i]
            # Determine which modes are filtered
            # -----------------------------------
            list_modes_filtered = self.mode.list_absindex_mode[M1_filtered_mode_mask]
            kmax_2 = list_kmax_2[i]
            for aux in self.hierarchy.auxiliary_list:
                depth = aux.sum()
                # No need to filter auxes that have upward connections only to auxes at
                # kmax2 or lower.
                if depth >= kmax_2:
                    # If we have any filtered modes in the aux, it should only
                    # connect upward if it lies on an edge.
                    if any([key in list_modes_filtered for key in aux.keys()]):
                        F2_filter[:, aux._index] = False
                        # Edge auxes connect to the edge aux one step upward.
                        if len(aux.keys()) == 1:
                            mode_index = np.where(self.mode.list_absindex_mode ==
                                                  aux.keys()[0])[0][0]
                            F2_filter[mode_index, aux._index] = True
                        # Non-edge auxes don't connect upwards to anything.
                        else:
                            pass
                    # Other auxes this deep that have no depth in a filtered mode will
                    # still break kmax2 if we take one step up along a filtered mode.
                    else:
                        F2_filter[M1_filtered_mode_mask, aux._index] = False
                # Auxes that remain are necessarily connected only to unfiltered auxes.
                else:
                    pass
            # Auxes with depth below kmax2 have no filtered connections.
            else:
                pass

        return F2_filter

    @property
    def n_state(self):
        return self.system.size

    @property
    def n_hier(self):
        return self.hierarchy.size

    @property
    def n_hmodes(self):
        return self.mode.n_hmodes
