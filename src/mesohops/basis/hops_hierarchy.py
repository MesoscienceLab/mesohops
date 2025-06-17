import itertools as it
from collections import Counter

import numpy as np

from mesohops.basis.hierarchy_functions import (filter_aux_longedge, filter_aux_triangular, filter_markovian)
from mesohops.basis.hops_aux import AuxiliaryVector as AuxVec
from mesohops.util.dynamic_dict import Dict_wDefaults
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
    Defines the representation of the hierarchy in the HOPS
    calculation. It contains the user parameters, helper functions, and
    the list of all nodes contained in the hierarchy.
    """

    __slots__ = (
        # --- Core basis components  ---
        'system',                    # System parameters and operators (HopsSystem)
        'n_hmodes',                  # Number of hierarchy modes
        '__ndim',                    # System dimension (for internal use)

        # --- Auxiliary list management ---
        '_auxiliary_list',           # List of current auxiliary vectors (main storage)
        '__previous_auxiliary_list', # Auxiliary list from previous step
        '__list_aux_stable',         # Auxiliaries stable between steps
        '__list_aux_remove',         # Auxiliaries to remove in update
        '__list_aux_add',            # Auxiliaries to add in update
        '__previous_list_auxstable_index', # Indices of stable auxiliaries from previous step
        '_previous_list_modes_in_use',     # Modes in use from previous step

        # --- Filter management flags ---
        'only_markovian_filter',     # True if only Markovian filter is used, else False

        # --- Parameter management ---
        '_default_param',            # Default parameter dictionary
        '_param_types',              # Parameter type dictionary

        # --- Connection and indexing dictionaries ---
        '_new_aux_index_conn_by_mode',   # New auxiliary index connections by mode
        '_new_aux_id_conn_by_mode',      # New auxiliary ID connections by mode
        '_stable_aux_id_conn_by_mode',   # Stable auxiliary ID connections by mode
        '_dict_aux_by_id',               # Dictionary mapping auxiliary IDs to objects
        '_list_modes_in_use',            # List of modes currently in use
        '_count_by_modes',               # Count of auxiliaries by mode
    )

    def __init__(self, hierarchy_param, system_param):
        """
        Initializes the HopsHierarchy object with the parameters that it will use to
        construct the hierarchy.

        Inputs
        ------
        1. hierarchy_param :
            [see hops_basis.py]
            a. MAXHIER : int
                         Maximum depth in the hierarchy that will be kept in the
                         calculation (options: >= 0).
            b. TERMINATOR : bool
                            True indicates the terminator condition is used while False
                            indicates otherwise (options: False).
            c. STATIC_FILTERS : str
                                Total set of nodes defined by MAXHIER can be further
                                filtered. This is a list of filters [(filter_name1,
                                [filter_param_1]), ...] (options: Triangular,
                                Markovian, LongEdge, Domain).

        2. system_param :
            [see hops_system.py]
            a. N_HMODES : int
                          Number of modes that appear in hierarchy.

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
        if self.param["MAXHIER"] > 255:
            print("Warning: using a hierarchy depth greater than 255 can cause "
                  "integer overflow issues when calculating derivative error. "
                  "Resetting hierarchy depth to 255.")
            self.param["MAXHIER"] = 255
        self.n_hmodes = system_param["N_HMODES"]
        self._auxiliary_list = []
        self._new_aux_index_conn_by_mode = {mode: {} for mode in range(system_param["N_HMODES"])}
        self._new_aux_id_conn_by_mode = {mode: {} for mode in range(system_param["N_HMODES"])}
        self._stable_aux_id_conn_by_mode = {mode: {} for mode in range(system_param["N_HMODES"])}
        self._dict_aux_by_id = {}
        self._list_modes_in_use = []
        self._count_by_modes = {}

    def initialize(self, flag_adaptive):
        """
        Initializes the hierarchy.

        Parameters
        ----------
        1. flag_adaptive : bool
                           True indicates an adaptive calculation while False indicates
                           otherwise.  

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
            # If the first static filter is Markovian, then use the Markovian
            # triangular hierarchy generator
            else:
                list_mark = self.param["STATIC_FILTERS"][0][1]
                self.auxiliary_list = self.filter_aux_list(
                    self.define_markovian_filtered_triangular_hierarchy(
                        self.n_hmodes, self.param["MAXHIER"], list_mark)
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
        1. list_aux : list(instance(AuxVec))
                      List of auxiliaries to be filtered.

        Returns
        -------
        1. list_aux : list(instance(AuxVec))
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
        1. list_aux : list(instance(AuxVec))
                      List of auxiliaries that needs to be filtered.

        2. filter_name : str
                         Name of filter.

        3. params : list
                    List of parameters for the filter.

        Returns
        -------
        1. list_aux : list(instance(AuxVec))
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
                kmax_2=params[1]
            )
        elif filter_name == "LongEdge":
            list_aux = filter_aux_longedge(
                list_aux=list_aux,
                list_boolean_by_mode=params[0],
                kdepth=params[1],
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
        1. aux : instance(AuxVec)

        Returns
        -------
        1. aux_index : int
                       Relative or absolute index of a single auxiliary.
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
        1. aux : instance(AuxVec)
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
        1. list_aux : list(instance(AuxVec))
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
                     Max single value of the hierarchy.

        3. list_boolean_mark : list(bool)
                               List by modes. True indicates that the Markovian filter
                               will be applied while False indicates otherwise.

        Returns
        -------
        1. list_aux : list(instance(AuxVec))
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

    def __update_count(self, aux, type):
        """
        Updates the dictionary of number of auxiliaries possessing non-zero depth in
        each mode, called when a new auxiliary is added to the basis.

        Parameters
        ----------
        1. aux : instance(AuxVec)

        2. type : str
                  Determines whether to add or remove (options: add, remove).

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
        self._list_modes_in_use = sorted(list_modes_in_use)

    def add_connections(self):
        """
        Adds the connections between HopsAux objects composing an auxiliary list.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for mode in self._previous_list_modes_in_use:
            self._stable_aux_id_conn_by_mode[mode].update(self._new_aux_id_conn_by_mode[mode])
        self._new_aux_index_conn_by_mode = {mode: {} for mode in self._list_modes_in_use}
        self._new_aux_id_conn_by_mode = {mode: {} for mode in self._list_modes_in_use}
        for aux in self.list_aux_add:
            sum_aux = np.sum(aux)
            # add connections to k+1
            if sum_aux < self.param['MAXHIER']:
                list_id_p1, list_value_connects_p1, list_mode_connects_p1 = aux.get_list_id_up(self._list_modes_in_use)
                for (rel_ind,my_id) in enumerate(list_id_p1):
                    try:
                        aux_p1 = self._dict_aux_by_id[my_id]
                        aux.add_aux_connect(list_mode_connects_p1[rel_ind],aux_p1,1)
                        
                        #We simply keep track of the index connections of new auxiliaries
                        #Note: It does not matter that the indices will change because we only use this dictionary 
                        #once, immediately after it is created in eom.ksuper
                        self._new_aux_id_conn_by_mode[list_mode_connects_p1[rel_ind]][aux.id] = [aux_p1.id,list_value_connects_p1[rel_ind] + 1]
                        self._new_aux_index_conn_by_mode[list_mode_connects_p1[rel_ind]][aux._index] = [aux_p1._index,list_value_connects_p1[rel_ind] + 1]  
                    except:
                        pass

            # Add connections to k-1
            if sum_aux > 0:
                list_id_m1, list_value_connects_m1, list_mode_connects_m1 = aux.get_list_id_down()

                for (index,id_m1) in enumerate(list_id_m1):
                    try:
                        aux_m1 = self._dict_aux_by_id[id_m1]
                        aux.add_aux_connect(list_mode_connects_m1[index], aux_m1, -1)
                        self._new_aux_id_conn_by_mode[list_mode_connects_m1[index]][aux_m1.id] = [aux.id,list_value_connects_m1[index]]
                        self._new_aux_index_conn_by_mode[list_mode_connects_m1[index]][aux_m1._index] = [aux._index,list_value_connects_m1[index]]
                    except:
                        pass
         
    @property
    def new_aux_index_conn_by_mode(self):
        return self._new_aux_index_conn_by_mode
        
    @property
    def stable_aux_id_conn_by_mode(self):
        return self._stable_aux_id_conn_by_mode  
         
    @property
    def size(self):
        return len(self.auxiliary_list)

    @property
    def auxiliary_list(self):
        return self._auxiliary_list

    @property
    def dict_aux_by_id(self):
        return self._dict_aux_by_id

    @property
    def list_absindex_hierarchy_modes(self):
        return self._list_modes_in_use

    @auxiliary_list.setter
    def auxiliary_list(self, aux_list):
        # Construct information about pevious timestep
        # --------------------------------------------
        self.__previous_auxiliary_list = self.auxiliary_list

        set_aux_add = set(aux_list) - set(self.auxiliary_list)
        set_aux_remove = set(self.auxiliary_list) - set(aux_list)
        list_aux_remove = [self.auxiliary_list[aux._index]
                           for aux in set_aux_remove]
        self.__list_aux_remove = list_aux_remove
        self.__list_aux_add = list(set_aux_add)
        self.__list_aux_add.sort()
 
        set_aux_stable = set(self.auxiliary_list).intersection(set(aux_list))
        self.__list_aux_stable = [aux for aux in self.auxiliary_list if aux in set_aux_stable]
        self.__previous_list_auxstable_index = [aux._index for aux in self.__list_aux_stable]
        
        self._previous_list_modes_in_use = self._list_modes_in_use.copy()
        
        
        if set(aux_list) != set(self.__previous_auxiliary_list):
            # Prepare New Auxiliary List
            # --------------------------
            for aux in set_aux_add:
                self._dict_aux_by_id[aux.id] = aux
                self.__update_count(aux, 'add')

            for aux in list_aux_remove:
                self._dict_aux_by_id.pop(aux.id)
                self.__update_count(aux, 'remove')
            # Update auxiliary_list
            aux_list.sort()
            self._auxiliary_list = aux_list
            if not aux_list[0].id == '':
                raise AuxError("Zero Vector not contained in list_aux!")
            
            # Update modes_in_use
            self.__update_modes_in_use()

            # Update indexing for aux
            for (index, aux) in enumerate(aux_list):
                aux._index = index

            

        # Add auxiliary connections
        self.add_connections()
        
        # Delete stable connections of deleted auxiliaries
        
        for aux in list_aux_remove:
            for (mode,aux_p1) in aux.dict_aux_p1.items():
                try:
                    self._stable_aux_id_conn_by_mode[mode].pop(aux.id)
                except:
                    pass
            for (mode,aux_m1) in aux.dict_aux_m1.items():
                try:
                    self._stable_aux_id_conn_by_mode[mode].pop(aux_m1.id)
                except:
                    pass
                    
        # Remove auxiliary connections
        for aux in set_aux_remove:
            aux.remove_pointers()
            
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
