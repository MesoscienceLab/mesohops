from collections import Counter
import itertools as it
import numpy as np
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxVec
from mesohops.util.dynamic_dict import Dict_wDefaults
from mesohops.dynamics.hierarchy_functions import (
    filter_aux_triangular,
    filter_aux_longedge,
    filter_markovian,
)
from mesohops.util.exceptions import AuxError, UnsupportedRequest

__title__ = "Hierarchy Class"
__author__ = "D. I. G. Bennett, L. Varvelo, J. K. Lynd"
__version__ = "1.2"


HIERARCHY_DICT_DEFAULT = {"MAXHIER": int(3), "TERMINATOR": False, "STATIC_FILTERS": []}

HIERARCHY_DICT_TYPES = dict(
    MAXHIER=[type(int())],
    TERMINATOR=[type(False), type(str())],
    STATIC_FILTERS=[type([])],
)


class HopsHierarchy(Dict_wDefaults):
    """
    HopsHierarchy defines the representation of the hierarchy in the HOPS
    calculation. It contains the user parameters, helper functions, and
    the list of all nodes contained in the hierarchy.
    """

    def __init__(self, hierarchy_param, system_param):
        """
        Initializes the HopsHierarchy object with the parameters that it will use to
        construct the hierarchy.

        Inputs
        ------
        1. hierarchy_param :
            [see hops_basis]
            a. MAXHIER : int                    [Allowed: >0]
                         The maximum depth in the hierarchy that will be kept
                         in the calculation
            b. TERMINATOR : boolean             [Allowed: False]
                            The name of the terminator condition to be used
            c. STATIC_FILTERS :                 [Allowed: Triangular, Markovian,
                                                LongEdge, Domain]
                              The total set of nodes defined by MAXHIER can
                              be further filtered. This is a list of filters
                              [(filter_name1, [filter_param_1]), ...]

        2. system_param :
            [see hops_system]
            a. N_HMODES : int
                          number of modes that appear in hierarchy

        Returns
        -------
        None

        NOTE: This class is only well defined within the context of a specific
        calculation where the system parameters and hierarchy parameters have
        been set. YOU SHOULDN'T INSTANTIATE ONE OF THESE CLASSES YOURSELF.
        IF YOU FEEL THE NEED TO, YOU ARE PROBABLY DOING SOMETHING STRANGE.
        """
        self._default_param = HIERARCHY_DICT_DEFAULT
        self._param_types = HIERARCHY_DICT_TYPES

        self.param = hierarchy_param
        self.n_hmodes = system_param["N_HMODES"]
        self._auxiliary_list = []
        self._auxiliary_by_depth = {depth: [] for depth in range(self.param['MAXHIER']+1)}
        self._hash_by_depth = {depth : [] for depth in range(self.param['MAXHIER']+1)}
        self._new_connect_p1 = {mode: {} for mode in range(system_param["N_HMODES"])}
        self._new_connect_m1 = {mode: {} for mode in range(system_param["N_HMODES"])}
        self._new_mode_values_m1 = {mode: {} for mode in range(system_param["N_HMODES"])}
        self._aux_by_hash = {}
        self._list_modes_in_use = []
        self._count_by_modes = {}

    def initialize(self, flag_adaptive):
        """
        This function will initialize the hierarchy.

        Parameters
        ----------
        1. flag_adaptive : boolean
                           Boolean describing whether the calculation is adaptive or not.

        Returns
        -------
        None
        """
        # Prepare the hierarchy
        # ---------------------
        # The hierarchy is only predefined if the basis is not adaptive
        if not flag_adaptive:
            # If there are no static filters, use the standard triangular hierarchy
            # generator
            if len(self.param["STATIC_FILTERS"]) == 0:
                self.auxiliary_list = self.filter_aux_list(
                    self.define_triangular_hierarchy(self.n_hmodes,
                                                     self.param["MAXHIER"]
                                                     )
                )
            # If the first static filter is not Markovian, use the standard
            # triangular hierarchy generator and then apply filters
            elif not "Markovian" in self.param["STATIC_FILTERS"][0]:
                self.auxiliary_list = self.filter_aux_list(
                    self.define_triangular_hierarchy(self.n_hmodes,
                                                     self.param["MAXHIER"]
                                                     )
                )
            # If the first static filter is Markovian, then use either the Markovian
            # triangular hierarchy generator or the Markovian-LongEdge triangular
            # hierarchy
            # generator special cases
            else:
                list_mark = self.param["STATIC_FILTERS"][0][1]
                # Check if there are additional static filters beyond Markovian. If
                # not, use the Markovian triangular hierarchy generator special case
                if len(self.param["STATIC_FILTERS"]) < 2:
                    self.auxiliary_list = self.filter_aux_list(
                        self.define_markovian_filtered_triangular_hierarchy(
                            self.n_hmodes, self.param["MAXHIER"], list_mark)
                    )
                # If there are additional static filters, but the second is not
                # LongEdge, use the Markovian triangular hierarchy generator special
                # case
                elif not "LongEdge" in self.param["STATIC_FILTERS"][1]:
                    self.auxiliary_list = self.filter_aux_list(
                        self.define_markovian_filtered_triangular_hierarchy(
                            self.n_hmodes, self.param["MAXHIER"], list_mark)
                    )
                # Finally, if the second filter is LongEdge, use the
                # Markovian-LongEdge triangular hierarchy generator special case
                else:
                    list_LE = self.param["STATIC_FILTERS"][1][1][0]
                    LE_depth, LE_edge = self.param["STATIC_FILTERS"][1][1][1]
                    self.auxiliary_list = self.filter_aux_list(
                        self.define_markovian_and_LE_filtered_triangular_hierarchy(
                            self.n_hmodes, self.param["MAXHIER"], list_mark, list_LE,
                            LE_edge)
                    )


        else:
            # Initialize Guess for the hierarchy
            # NOTE: The first thing a hierarchy does is start expanding from the
            #       zero index. It might, at some point, be more efficient to
            #       initialize the hierarchy with an explicit guess (in combination
            #       with a restriction that no auxiliary nodes are removed until after
            #       Nsteps). At the moment, though, I'm skipping this and allowing
            #       the hierarchy to control its own growth.
            self.auxiliary_list = [AuxVec([], self.n_hmodes)]
            if np.any([name != "Markovian" for (name, param) in self.param["STATIC_FILTERS"]]):
                self.only_markovian_filter = False
            else:
                self.only_markovian_filter = True

    def filter_aux_list(self, list_aux):
        """
        Applies all of the filters defined for the hierarchy
        in "STATIC_FILTERS".

        Parameters
        ----------
        1. list_aux : list
                      List of auxiliaries to be filtered.

        Returns
        -------
        1. list_aux : list
                      Filtered list of auxiliaries.

        """
        for (filter_i, param_i) in self.param["STATIC_FILTERS"]:
            list_aux = self.apply_filter(list_aux, filter_i, param_i)

        return list_aux

    def apply_filter(self, list_aux, filter_name, params):
        """
        Implements a variety of different hierarchy filtering methods. In all cases,
        this filtering begins with the current list of auxiliaries and then prunes
        down from that list according to a set of rules.

        Parameters
        ----------
        1. list_aux : list
                      List of auxiliaries that needs to be filtered.

        2. filter_name : str
                         Name of filter.

        3. params : list
                    List of parameters for the filter.

        Returns
        -------
        1. list_aux : list
                      List of filtered auxiliaries.


        ALLOWED LIST OF FILTERS:
        - Triangular: boolean_by_mode, [kmax, kdepth]
             If the mode has a non-zero value, then it is kept only if
             the value is less than kmax and the total auxillary is less
             than kdepth. This essentially truncates some modes at a lower
             order than the other modes.
        - LongEdge: boolean_by_mode, [kmax, kdepth]
             Beyond kdepth, only keep the edge terms upto kmax.
        - Markovian: boolean_by_mode
             All vectors with depth in the specified modes are filtered out unless
             they are the "unit auxiliary:" that is, its indexing vector is 1 in the
             Markovian-filtered mode. Equivalent to a Triangular filter with kmax =
             kdepth = 1 in the filtered modes.
        """
        if filter_name == "Triangular":
            list_aux = filter_aux_triangular(
                list_aux=list_aux,
                list_boolean_by_mode=params[0],
                kmax=params[1][0],
                kdepth=params[1][1],
            )
        elif filter_name == "LongEdge":
            list_aux = filter_aux_longedge(
                list_aux=list_aux,
                list_boolean_by_mode=params[0],
                kmax=params[1][0],
                kdepth=params[1][1],
            )
        elif filter_name == "Markovian":
            list_aux = filter_markovian(list_aux=list_aux, list_boolean=params)
        else:
            raise UnsupportedRequest(filter_name, "hierarchy_static_filter")

        # Update STATIC_FILTERS parameters if needed
        # ------------------------------------------
        if not (
            [filter_name, params] in self.param["STATIC_FILTERS"]
            or (filter_name, params) in self.param["STATIC_FILTERS"]
        ):
            self.param["STATIC_FILTERS"].append((filter_name, params))

        return list_aux

    def _aux_index(self, aux):
        """
        Returns the index value for a given auxiliary. The important thing is that this
        function is aware of whether the calculation should be using an 'absolute'
        index or a 'relative' index.
        
        Absolute index: no matter which auxiliaries are in the hierarchy, the index
                        value does not change. This is a useful approach when trying
                        to do things like dynamic filtering. 
        
        Relative index: This is the more intuitive indexing scheme which only keeps
                        track of the auxiliary vectors that are actually in the
                        hierarchy.

        Parameters
        ----------
        1. aux : Auxiliary object
                 Hops_aux auxiliary object.

        2. absolute : boolean
                      Describes whether the indexing should be relative or absolute
                      the default value is False.

        Returns
        -------
        1. aux_index : int
                       Relative or absolute index of a single auxiliary
        """
        if aux._index is None:
            return self.auxiliary_list.index(aux)
        else:
            return aux._index

    @staticmethod
    def _const_aux_edge(absindex_mode, depth, n_hmodes):
        """
        Creates an auxiliary object for an edge node at
        a particular depth along a given mode. 

        Parameters
        ----------
        1. absindex_mode : int
                           Absolute index of the edge mode.

        2. depth : int
                   Depth of the edge auxiliary.

        3. n_hmodes : int
                      Number of modes that appear in the hierarchy.

        Returns
        -------
        1. aux : Auxiliary object
                 Auxiliary at the edge node.
        """
        return AuxVec([(absindex_mode, depth)], n_hmodes)

    @staticmethod
    def define_triangular_hierarchy(n_hmodes, maxhier):
        """
        Creates a triangular hierarchy for a given number of modes at a
        given depth.

        Parameters
        ----------
        1. n_hmodes : int
                      Number of modes that appear in the hierarchy.

        2. maxhier : int
                     Max single value of the hierarchy.

        Returns
        -------
        1. list_aux : list
                      List of auxiliaries in the new triangular hierarchy.
        """
        list_aux = []
        # first loop over hierarchy depth
        for k in range(maxhier + 1):
            # Second loop over
            for aux_raw in it.combinations_with_replacement(range(n_hmodes), k):
                count = Counter(aux_raw)
                list_aux.append(
                    AuxVec([(key, count[key]) for key in count.keys()], n_hmodes)
                )
        return list_aux

    @staticmethod
    def define_markovian_filtered_triangular_hierarchy(n_hmodes, maxhier,
                                                       list_boolean_mark):
        """
        Creates a triangular hierarchy when the Markovian filter is in
        use, applying the Markovian filter as the hierarchy is constructed to reduce
        transient memory burdens.

        Parameters
        ----------
        1. n_hmodes : int
                      Number of modes that appear in the hierarchy.

        2. maxhier : int
                     Max single value of the hierarchy

        3. list_boolean_mark : list(boolean)
                               List by mode of whether the Markovian filter will
                               be applied

        Returns
        -------
        1. list_aux : list
                      List of auxiliaries in the new triangular hierarchy.
        """
        list_not_boolean_mark = [not bool_mark for bool_mark in list_boolean_mark]
        list_aux = []

        # Loop over hierarchy depths at which the Markovian filter does not apply
        for k in [0,1]:
            for aux_raw in it.combinations_with_replacement(np.arange(n_hmodes), k):
                count = Counter(aux_raw)
                list_aux.append(
                    AuxVec([(key, count[key]) for key in count.keys()], n_hmodes)
                )

        # Generate an array of exclusively the non-Markovian-filtered modes
        M1_modes_filtered = np.arange(n_hmodes)[list_not_boolean_mark]
        # Loop over hierarchy depths at which the Markovian filter applies
        for k in np.arange(2, maxhier+1):
            # At each depth, add to list_aux all possible combinations of only the
            # non-Markovian-filtered modes
            for aux_raw in it.combinations_with_replacement(M1_modes_filtered, k):
                count = Counter(aux_raw)
                list_aux.append(
                    AuxVec([(key, count[key]) for key in count.keys()], n_hmodes)
                )
        return list_aux

    @staticmethod
    def define_markovian_and_LE_filtered_triangular_hierarchy(n_hmodes, maxhier,
                                                              list_boolean_mark,
                                                              list_boolean_le, edge_le):
        """
        Creates a triangular hierarchy when the Markovian and LongEdge filters are in
        use, applying the filters as the hierarchy is constructed to reduce
        transient memory burdens.

        IMPORTANT: This function relies on being filtered after it runs to get rid of
        the terms with a depth greater than LE_depth in the modes filtered by
        list_boolean_LE. THIS FUNCTION SHOULD NEVER BE RUN WITHOUT BEING WRAPPED BY
        SELF.FILTER_AUX_LIST()!

        Parameters
        ----------
        1. n_hmodes : int
                      Number of modes that appear in the hierarchy.

        2. maxhier : int
                     Max single value of the hierarchy.

        3. list_boolean_mark : list(boolean)
                               List by mode of whether the Markovian filter will
                               be applied.

        4. list_boolean_LE: list(boolean)
                            List by mode of whether the LongEdge filter will be applied.

        5. depth_le: int
                     Allowed hierarchy depth of the LongEdge-filtered modes. This
                     parameter is unused in the code: we include it due to the
                     dictionary structure associated with the LongEdge filter.

        6. edge_le: int
                    Depth beyond which LongEdge-filtered modes retain only edge terms.

        Returns
        -------
        1. list_aux : list
                      List of auxiliaries in the new triangular hierarchy.
        """
        list_aux = []
        list_not_boolean_mark = [not bool_mark for bool_mark in list_boolean_mark]
        list_not_boolean_le = [not bool_le for bool_le in list_boolean_le]

        # Loops over hierarchy depths at which the Markovian filter does not apply
        for k in [0, 1]:
            for aux_raw in it.combinations_with_replacement(np.arange(n_hmodes), k):
                count = Counter(aux_raw)
                list_aux.append(
                    AuxVec([(key, count[key]) for key in count.keys()], n_hmodes)
                )

        # Create a list of booleans that identifies entirely unfiltered modes
        list_modes_unfiltered = np.array(list_not_boolean_mark)*np.array(
            list_not_boolean_le)
        # Create a list of booleans that identifies modes with only the LongEdge filter
        list_modes_only_le_filtered = np.array(list_boolean_mark)*np.array(1-np.array(
            list_not_boolean_le), dtype=bool)
        # Generate an array of all non-Markovian-filtered modes
        M1_modes_filtered_mark = np.arange(n_hmodes)[list_not_boolean_mark]
        # Generate an array of all modes that are both non-Markovian-filtered and
        # non-LongEdge-filtered
        M1_modes_not_longedge = np.arange(n_hmodes)[list_modes_unfiltered]
        # Generate an array of all modes that are only LongEdge-filtered
        M1_modes_longedge_only = np.arange(n_hmodes)[list_modes_only_le_filtered]
        
        # Loop over hierarchy depths at which the LongEdge filter does not yet apply
        for k in np.arange(2, edge_le+1):
            # At each depth, add to list_aux all possible combinations of only the
            # non-Markovian-filtered modes
            for aux_raw in it.combinations_with_replacement(M1_modes_filtered_mark, k):
                count = Counter(aux_raw)
                list_aux.append(
                    AuxVec([(key, count[key]) for key in count.keys()], n_hmodes)
                )

        # Loop over hierarchy depths at which the LongEdge filter allows only edge terms
        for k in np.arange(edge_le+1, maxhier+1):
            # At each depth, add to list_aux all possible combinations of only the
            # modes that are both non-Markovian-filtered and non-LongEdge-filtered
            for aux_raw in it.combinations_with_replacement(M1_modes_not_longedge, k):
                count = Counter(aux_raw)
                list_aux.append(
                    AuxVec([(key, count[key]) for key in count.keys()], n_hmodes)
                )
            # In addition, add to list_aux all the edge terms associated with the
            # LongEdge-filtered modes
            for mode in M1_modes_longedge_only:
                list_aux.append(AuxVec([(mode, k)], n_hmodes))

        return list_aux

    def __update_count(self, aux, type):
        """
        Updates the dictionary of number of auxiliaries possessing non-zero depth in
        each mode, called when a new auxiliary is added to the basis.

        Parameters
        ----------
        1. aux : instance(HopsAuxiliary)

        2. type : str
                  Two options add or remove

        Returns
        -------
        None
        """
        if type == 'add':
            self._count_by_modes |= {mode:(self._count_by_modes[mode] + 1)
            if mode in self._count_by_modes.keys() else 1 for mode in aux.keys()}
        elif type == 'remove':
            self._count_by_modes |= {mode: (self._count_by_modes[mode] - 1)
                                     for mode in aux.keys()}
        else:
            raise UnsupportedRequest(type, "__update_count")

    def __update_modes_in_use(self):
        """
        Extracts the list of modes for which any auxiliaries in the current auxiliary
        basis has non-zero depth

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        list_modes_in_use = []
        list_to_remove = []
        for (mode, value) in self._count_by_modes.items():
            if value == 0:
                list_to_remove.append(mode)
            elif value>0:
                list_modes_in_use.append(mode)
            else:
                print(f'ERROR: _count_by_modes is negative for mode {mode}')

        [self._count_by_modes.pop(mode) for mode in list_to_remove]
        self._list_modes_in_use = list_modes_in_use

    def add_connections(self):
        """
        The method responsible for adding the connections between HopsAux objects
        composing an auxiliary list.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._new_connect_p1 = {mode: {} for mode in range(self.n_hmodes)}
        self._new_connect_m1 = {mode: {} for mode in range(self.n_hmodes)}
        self._new_mode_values_m1 = {mode: {} for mode in range(self.n_hmodes)}
        for aux in self.list_aux_add:
            sum_aux = np.sum(aux)
            # add connections to k+1
            
            if sum_aux < self.param['MAXHIER']:
                list_identity_str_p1, mode_values, mode_connects = aux.get_list_identity_string_up(self._list_modes_in_use)
                #we need to return mode_connects, so simple modification of identity_string_up was made.
                for (index,my_ident) in enumerate(list_identity_str_p1):
                    try:
                        aux_p1 = self._aux_by_hash[hash(my_ident)]
                        aux.add_aux_connect(mode_connects[index],aux_p1,1)
                        #We simply keep track of the index connections of new auxiliaries
                        #Note: It does not matter that the indices will change because we only use this dictionary 
                        #once, immediately after it is created in eom.ksuper
                        self._new_connect_p1[mode_connects[index]][aux._index] = aux_p1._index 
                        self._new_connect_m1[mode_connects[index]][aux_p1._index] = aux._index 
                        self._new_mode_values_m1[mode_connects[index]][aux_p1._index] = mode_values[index] + 1
                    except:
                        pass

            # Add connections to k-1
            if sum_aux > 0:
            
                list_identity_str_m1, list_value_connects, mode_connects = aux.get_list_identity_string_down()

                for (index,my_ident) in enumerate(list_identity_str_m1):
                    try:
                        aux_m1 = self._aux_by_hash[hash(my_ident)]
                        aux.add_aux_connect(mode_connects[index], aux_m1, -1)
                        if aux_m1 not in self.list_aux_add:
                            self._new_connect_m1[mode_connects[index]][aux._index] = aux_m1._index
                            self._new_connect_p1[mode_connects[index]][aux_m1._index] = aux._index
                            self._new_mode_values_m1[mode_connects[index]][aux._index] = list_value_connects[index]
                    except:
                        pass

    @property
    def new_connect_p1(self):
        return self._new_connect_p1

    @property
    def new_connect_m1(self):
        return self._new_connect_m1

    @property
    def new_mode_values_m1(self):
        return self._new_mode_values_m1

    @property
    def size(self):
        return len(self.auxiliary_list)

    @property
    def auxiliary_list(self):
        return self._auxiliary_list

    @property
    def aux_by_hash(self):
        return self._aux_by_hash

    @property
    def list_absindex_hierarchy_modes(self):
        return list(self._count_by_modes.keys())

    @auxiliary_list.setter
    def auxiliary_list(self, aux_list):
        # Construct information about pevious timestep
        # --------------------------------------------
        self.__previous_auxiliary_list = self.auxiliary_list

        set_aux_add = set(aux_list) - set(self.auxiliary_list)
        set_aux_remove = set(self.auxiliary_list) - set(aux_list)
        list_aux_remove = [self.auxiliary_list[self.auxiliary_list.index(aux)]
                           for aux in set_aux_remove]
        self.__list_aux_remove = list_aux_remove
        self.__list_aux_add = list(set_aux_add)
        self.__list_aux_add.sort()

        set_aux_stable = set(self.auxiliary_list).intersection(set(aux_list))
        self.__list_aux_stable = [aux for aux in self.auxiliary_list if aux in set_aux_stable]
        self.__previous_list_auxstable_index = [aux._index for aux in self.__list_aux_stable]

        if set(aux_list) != set(self.__previous_auxiliary_list):
            # Prepare New Auxiliary List
            # --------------------------
            # Update auxiliary_by_depth
            for aux in set_aux_add:
                self._auxiliary_by_depth[np.sum(aux)].append(aux)
                self._hash_by_depth[np.sum(aux)].append(aux.hash)
                self._aux_by_hash[aux.hash] = aux
                self.__update_count(aux, 'add')

            for aux in list_aux_remove:
                self._auxiliary_by_depth[np.sum(aux)].remove(aux)
                self._hash_by_depth[np.sum(aux)].remove(aux.hash)
                self._aux_by_hash.pop(aux.hash)
                self.__update_count(aux, 'remove')

            # Update auxiliary_list
            aux_list.sort()
            self._auxiliary_list = aux_list
            if not aux_list[0].absolute_index == 0:
                raise AuxError("Zero Vector not contained in list_aux!")

            # Update modes_in_use
            self.__update_modes_in_use()

            # Update indexing for aux
            for (index, aux) in enumerate(aux_list):
                aux._index = index

            # Remove auxiliary connections
            for aux in set_aux_remove:
                aux.remove_pointers()

        # Add auxiliary connections
        self.add_connections()
    
    @property
    def list_aux_stable(self):
        return self.__list_aux_stable

    @property
    def list_aux_add(self):
        return self.__list_aux_add

    @property
    def list_aux_remove(self):
        return self.__list_aux_remove

    @property
    def previous_auxiliary_list(self):
        return self.__previous_auxiliary_list

    @property
    def previous_list_auxstable_index(self):
        return self.__previous_list_auxstable_index
