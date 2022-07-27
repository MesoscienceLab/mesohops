from collections import Counter
import itertools as it
import numpy as np
from pyhops.dynamics.hops_aux import AuxiliaryVector as AuxVec
from pyhops.util.dynamic_dict import Dict_wDefaults
from pyhops.dynamics.hierarchy_functions import (
    filter_aux_triangular,
    filter_aux_longedge,
    filter_markovian,
)
from pyhops.util.exceptions import AuxError, UnsupportedRequest

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

        INPUTS
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

        RETURNS
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

    def initialize(self, flag_adaptive):
        """
        This function will initialize the hierarchy.

        PARAMETERS
        ----------
        1. flag_adaptive : boolean
                           boolean describing whether the calculation is adaptive or not

        RETURNS
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
                            LE_depth, LE_edge)
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

    def filter_aux_list(self, list_aux):
        """
        A function that applies all of the filters defined for the hierarchy
        in "STATIC_FILTERS"

        PARAMETERS
        ----------
        1. list_aux : list
                      the list of auxiliaries to be filtered

        RETURNS
        -------
        1. list_aux : list
                      the filtered list of auxiliaries

        """
        for (filter_i, param_i) in self.param["STATIC_FILTERS"]:
            list_aux = self.apply_filter(list_aux, filter_i, param_i)

        return list_aux

    def apply_filter(self, list_aux, filter_name, params):
        """
        This is a function that implements a variety of different hierarchy filtering
        methods. In all cases, this filtering begins with the current list of auxiliaries
        and then prunes down from that list according to a set of rules.

        PARAMETERS
        ----------
        1. list_aux : list
                      the list of auxiliaries that needs to be filtered
        2. filter_name : str
                         name of filter
        3. params : list
                    the list of parameters for the filter

        RETURNS
        -------
        1. list_aux : list
                      a list of filtered auxiliaries


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

    def _aux_index(self, aux, absolute=False):
        """
        This function will return the index value for a given auxiliary. The
        important thing is that this function is aware of whether or not the
        calculation should be using an 'absolute' index or a 'relative' index. 
        
        Absolute index: no matter which auxiliaries are in the hierarchy, the index
                        value does not change. This is a useful approach when trying
                        to do things like dynamic filtering. 
        
        Relative index: This is the more intuitive indexing scheme which only keeps
                        track of the auxiliary vectors that are actually in the
                        hierarchy.

        PARAMETERS
        ----------
        1. aux : Auxiliary object
                 a hops_aux auxiliary object
        2. absolute : boolean
                      describes whether the indexing should be relative or absolute
                      the default value is False

        RETURNS
        -------
        1. aux_index : int
                       the relative or absolute index of a single
                       auxiliary
        """
        if absolute:
            return aux.absolute_index
        elif aux._index is None:
            return self.auxiliary_list.index(aux)
        else:
            return aux._index

    @staticmethod
    def _const_aux_edge(absindex_mode, depth, n_hmodes):
        """
        This function creates an auxiliary object for an edge node at
        a particular depth along a given mode. 

        PARAMETERS
        ----------
        1. absindex_mode : int
                           absolute index of the edge mode
        2. depth : int
                   the depth of the edge auxiliary
        3. n_hmodes : int
                      number of modes that appear in the hierarchy

        RETURNS
        -------
        1. aux : Auxiliary object
                 the auxiliary at the the edge node

        """
        return AuxVec([(absindex_mode, depth)], n_hmodes)

    @staticmethod
    def define_triangular_hierarchy(n_hmodes, maxhier):
        """
        This function creates a triangular hierarchy for a given number of modes at a
        given depth.

        PARAMETERS
        ----------
        1. n_hmodes : int
                      number of modes that appear in the hierarchy
        2. maxhier : int
                     the max single value of the hierarchy

        RETURNS
        -------
        1. list_aux : list
                      list of auxiliaries in the new triangular hierarchy
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

        PARAMETERS
        ----------
        1. n_hmodes : int
                      number of modes that appear in the hierarchy
        2. maxhier : int
                     the max single value of the hierarchy
        3. list_boolean_mark : list(boolean)
                               the list by mode of whether the Markovian filter will be
                               applied

        RETURNS
        -------
        1. list_aux : list
                      list of auxiliaries in the new triangular hierarchy
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
                list_boolean_mark, list_boolean_le, depth_le, edge_le):
        """
        Creates a triangular hierarchy when the Markovian and LongEdge filters are in
        use, applying the filters as the hierarchy is constructed to reduce
        transient memory burdens.

        IMPORTANT: This function relies on being filtered after it runs to get rid of
        the terms with a depth greater than LE_depth in the modes filtered by
        list_boolean_LE. THIS FUNCTION SHOULD NEVER BE RUN WITHOUT BEING WRAPPED BY
        SELF.FILTER_AUX_LIST()!

        PARAMETERS
        ----------
        1. n_hmodes : int
                      number of modes that appear in the hierarchy
        2. maxhier : int
                     the max single value of the hierarchy
        3. list_boolean_mark : list(boolean)
                               the list by mode of whether the Markovian filter will be
                               applied
        4. list_boolean_LE: list(boolean)
                            the list by mode of whether the LongEdge filter will be
                            applied
        5. depth_le: int
                     The allowed hierarchy depth of the LongEdge-filtered modes. This
                     parameter is unused in the code: we include it due to the
                     dictionary structure associated with the LongEdge filter
        6. edge_le: int
                    The depth beyond which LongEdge-filtered modes retain only edge
                    terms

        RETURNS
        -------
        1. list_aux : list
                      list of auxiliaries in the new triangular hierarchy
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

    @property
    def size(self):
        return len(self.auxiliary_list)

    @property
    def auxiliary_list(self):
        return self._auxiliary_list

    @auxiliary_list.setter
    def auxiliary_list(self, aux_list):
        aux_list.sort()
        if not aux_list[0].absolute_index == 0:
            raise AuxError("Zero Vector not contained in list_aux!")

        # Determine the added and removed aux
        set_aux_add = set(aux_list) - set(self.auxiliary_list)
        set_aux_remove = set(self.auxiliary_list) - set(aux_list)

        # Update auxiliary_by_depth
        for aux in set_aux_add:
            self._auxiliary_by_depth[np.sum(aux)].append(aux)
            self._hash_by_depth[np.sum(aux)].append(aux.hash)

        for aux in set_aux_remove:
            self._auxiliary_by_depth[np.sum(aux)].remove(aux)
            self._hash_by_depth[np.sum(aux)].remove(aux.hash)

        # Update auxiliary_list
        self._auxiliary_list = aux_list
        for (index, aux) in enumerate(aux_list):
            aux._index = index

        # Add auxiliary connections
        self.add_connections(set_aux_add)

        # Remove auxiliary connections
        for aux in set_aux_remove:
            aux.remove_pointers()

    def add_connections(self, list_aux_add):
        """
        The method responsible for adding the connections between HopsAux objects
        composing an auxiliary list.

        PARAMETERS
        ----------
        1. list_aux_add : list
                          a list of HopsAux objects being added to the auxiliary list

        RETURNS
        -------
        None
        """
        for aux in list_aux_add:
            sum_aux = np.sum(aux)
            # add connections to k+1
            if sum_aux < self.param['MAXHIER']:
                # if there are fewer possible k+1 auxiliaries than the number of modes
                # in the hierarchy then loop over list of auxiliaries
                if len(self._auxiliary_by_depth[sum_aux + 1]) <= self.n_hmodes:
                    for aux_p1 in self._auxiliary_by_depth[sum_aux + 1]:
                        index_mode = aux.difference_by_mode(aux_p1)
                        if index_mode is not False:
                            aux.add_aux_connect(index_mode, aux_p1, 1)
                else:
                    for index_mode in range(self.n_hmodes):
                        aux_p1_hash = aux.hash_from_e_step(index_mode, 1)
                        aux_p1 = listobj_or_none(aux_p1_hash,
                                                 self._auxiliary_by_depth[
                                                     sum_aux + 1],
                                                 self._hash_by_depth[sum_aux + 1])
                        if aux_p1 is not None:
                            aux.add_aux_connect(index_mode, aux_p1, 1)

            # add connections to k-1
            if sum_aux > 0:
                # if there are fewer possible k-1 auxiliaries than the number of modes
                # in the hierarchy then loop over list of auxiliaries
                if len(self._auxiliary_by_depth[sum_aux - 1]) <= self.n_hmodes:
                    for aux_m1 in self._auxiliary_by_depth[sum_aux - 1]:
                        index_mode = aux.difference_by_mode(aux_m1)
                        if index_mode is not False:
                            aux.add_aux_connect(index_mode, aux_m1, -1)
                else:
                    for index_mode in range(self.n_hmodes):
                        aux_m1_hash = aux.hash_from_e_step(index_mode, -1)
                        aux_m1 = listobj_or_none(aux_m1_hash,
                                                 self._auxiliary_by_depth[
                                                     sum_aux - 1],
                                                 self._hash_by_depth[sum_aux - 1])
                        if aux_m1 is not None:
                            aux.add_aux_connect(index_mode, aux_m1, -1)

def listobj_or_none(hash, list_obj, list_hash):
    """
    A convenience function for determining which object in a list to return
    based on a list of the hash values for the objects.

    Parameters
    ----------
    1. hash : int
              The hash value for the desired object
    2. list_obj : list
                  A list of objects
    3. list_hash : list
                   The hash values for the objects in list_obj

    Returns
    -------
    1. obj : object or None
             The object from list_obj with matching hash or None (if there was no
             such object)
    """
    try:
        index = list_hash.index(hash)
        obj_new = list_obj[index]
    except:
        obj_new = None

    return obj_new
