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
        function allowed by the triangular truncation condition.

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
        function that has no negative indices.

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

                # Also filter out connections from modes that are not ever in auxiliary basis
                # for example, all modes in the main auxiliary will be filtered here,
                # all but one mode in first-order auxiliaries, two modes in second-order auxiliaries, etc.
                array_index2 = np.array([list_absindex_mode.index(mode) for mode in
                                         aux.keys() if mode in list_absindex_mode], dtype=int)
                array_index2 = np.setdiff1d(np.arange(self.n_hmodes), array_index2)

                F2_filter_m1[array_index2, aux._index] = False
            else:
                F2_filter_m1[:, aux._index] = False

        return F2_filter_m1

    def construct_filter_state_stable_down(self, list_aux_bound):
        """
        Constructs a filter array that ensures that only flux up to auxiliaries in
        list_aux_bound is considered.

        Parameters
        ----------
        1. list_aux_bound : list(HopsAux)
                            List of boundary auxiliaries.

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
        Constructs a filter array that ensures that only flux down to auxiliaries in
        list_aux_bound is considered.

        Parameters
        ----------
        1. list_aux_bound : list(HopsAux)
                            List of boundary auxiliaries.

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
        function.

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
        F2_filter = np.ones([self.n_hmodes, self.n_hier],dtype=bool)

        array2D_mark_param = np.array(
            [
                np.array(param)[self.mode.list_absindex_mode]
                for (name, param) in self.hierarchy.param["STATIC_FILTERS"]
                if name == "Markovian"
            ]
        )
        if len(array2D_mark_param) > 0:
            # Determine which modes are Markovian
            # -----------------------------------
            array_mark_mode = np.any(array2D_mark_param, axis=0)

            # Remove flux along Markovian Modes except
            # from the physical wave function
            # -----------------------------------------
            F2_filter[array_mark_mode, 1:] = False

            # Remove flux from any mode for the fist order
            # wave functions along Markovian modes
            # ---------------------------------------------
            aux0 = self.hierarchy.auxiliary_list[0]
            mark_aux1 = np.array([aux0.dict_aux_p1[mode]._index for mode in aux0.dict_aux_p1.keys()
                                  if mode in self.mode.list_absindex_mode[np.where(
                    array_mark_mode)[0]]])
            if len(mark_aux1) > 0:
                F2_filter[:, mark_aux1] = False

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
