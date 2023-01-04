import numpy as np
import scipy as sp
from scipy import sparse
import copy
from collections import Counter

__title__ = "System Class"
__author__ = "D. I. G. Bennett, L. Varvelo"
__version__ = "1.2"


class HopsSystem(object):
    """
    HopsSystem is a class that stores the basic information about the system and
    system-bath coupling.
    """

    def __init__(self, system_param):
        """
        Inputs
        ------
        1. system_param : dict
                          A dictionary with the system and system-bath coupling
                          parameters defined.

            =======================  HAMILTONIAN PARAMETERS  =======================
            a. HAMILTONIAN : np.array
                             an np.array() that contains the system Hamiltonian

            =======================  HIERARCHY PARAMETERS  =======================
            b. GW_SYSBATH : list
                            a list of parameters (g,w) that define the exponential
                            decomposition underlying the hierarchy
            c. L_HIER : list
                        a list of system-bath coupling operators in the same order
                        as GW_SYSBATH

            =======================  NOISE PARAMETERS  =======================
            d. L_NOISE1 : list
                          A list of system-bath coupling operators in the same order
                          as PARAM_NOISE_1
            e. ALPHA_NOISE1 : function
                              A function that calculates the
                              correlation function given (t_axis, *PARAM_NOISE_1)
            f. PARAM_NOISE1 : list
                              A list of parameters defining the decomposition of Noise1

        OPTIONAL PARAMETERS :
            g. L_NOISE2 : list
                          A list of system-bath coupling operators in the same order
                          as PARAM_NOISE_2
            h. ALPHA_NOISE2 : function
                              A pointer to the function that calculates the
                              correlation function given (t_axis, *PARAM_NOISE_2)
            i. PARAM_NOISE2 : list
                              A list of parameters defining the decomposition of Noise2

         ======================= DERIVED PARAMETERS ===========================
        The result is a dictionary HopsSystem.param which contains all the above plus
        additional parameters that are useful for indexing the simulation:
            j. 'NSTATES' : int
                        The dimension of the system Hilbert Space
            k. 'N_HMODES' : int
                            The number of modes that will appear in the hierarchy
            l. 'N_L2' : int
                        The number of unique system-bath coupling operators
            m. 'LIST_INDEX_L2_BY_NMODE1' : np.array
                                           An array (list_absindex_noise1 --> index_L2)
            n. 'LIST_INDEX_L2_BY_NMODE2' : np.array
                                           An array (list_absindex_noise2 --> index_L2)
            o. 'LIST_INDEX_L2_BY_HMODE' : np.array
                                          An array (list_absindex_by_hmode  --> index_L2)
            p. 'LIST_STATE_INDICES_BY_HMODE' : np.array
                                               An array (list_absindex_by_hmode   -->
                                               list_absindex_states)
            q. 'LIST_L2_COO' : np.array
                               An array (list_absindex_L2 --> coo_sparse L2)
            r. 'LIST_STATE_INDICES_BY_INDEX_L2 ' : np.array
                                                   (list_absindex_L2 -->
                                                   list_absindex_states)
            s. 'SPARSE_HAMILTONIAN' : sp.sparse.csc_martix
                                      the sparse representation of the Hamiltonian

        Returns
        -------
        None

        NOTE: L_HIER is required to contain all L-operators that are defined anywhere.
            This can be removed as a requirement by defining a third noise parameter
            that will get its own super-operators, but since we have no use-case yet
            this has not been implemented.
        """
        self.param = self._initialize_system_dict(system_param)
        self.__ndim = self.param["NSTATES"]
        self.__previous_state_list = None
        self.__state_list = []

    def _initialize_system_dict(self, system_param):
        """
        This function is responsible for extending the user input to the
        complete set of parameters defined above.

        Parameters
        ----------
        1. system_param : dict
                          Dictionary with the system and system-bath coupling
                          parameters defined.

        Returns
        -------
        1. param_dict : dict
                        Dictionary containing the user input and the derived parameters
        """
        param_dict = copy.deepcopy(system_param)
        param_dict["NSTATES"] = len(system_param["HAMILTONIAN"][0])
        param_dict["N_HMODES"] = len(system_param["GW_SYSBATH"])
        param_dict["G"] = np.array([g for (g, w) in system_param["GW_SYSBATH"]])
        param_dict["W"] = np.array([w for (g, w) in system_param["GW_SYSBATH"]])
        param_dict["LIST_STATE_INDICES_BY_HMODE"] = [
            self._get_states_from_L2(L2) for L2 in param_dict["L_HIER"]
        ]
        param_dict["SPARSE_HAMILTONIAN"] = sparse.csc_matrix(param_dict["HAMILTONIAN"])
        param_dict["SPARSE_HAMILTONIAN"].eliminate_zeros()

        # Define the Hierarchy Operator Values
        # ------------------------------------
        # Since arrays and lists are not hashable, we will turn our operators
        # into tuples in order to conveniently define a number of indexing
        # parameters.

        # Create list of unique l2 tuples in order they appear in "L_HIER"
        l2_as_tuples = [self._array_to_tuple(L2) for L2 in param_dict["L_HIER"]]
        list_unique_l2_as_tuples = list(Counter(l2_as_tuples))
        param_dict["N_L2"] = len(set(list_unique_l2_as_tuples))

        # Create L2 indexing parameters
        param_dict["LIST_INDEX_L2_BY_HMODE"] = [
            None for i in range(param_dict["N_HMODES"])
        ]
        flag_l2_list = [False for i in range(param_dict["N_L2"])]
        param_dict["LIST_L2_COO"] = [0 for i in range(param_dict["N_L2"])]
        param_dict["LIST_STATE_INDICES_BY_INDEX_L2"] = []
        list_unique_L2 = []
        for (i, l) in enumerate(list_unique_l2_as_tuples):
            # i is the index of operator l in the unique list of operators
            list_unique_L2.append(l)
            for j in range(param_dict["N_HMODES"]):
                if l2_as_tuples[j] == l:
                    param_dict["LIST_INDEX_L2_BY_HMODE"][j] = i
                    if not flag_l2_list[i]:
                        flag_l2_list[i] = True
                        tmp = sp.sparse.coo_matrix(param_dict["L_HIER"][j])
                        tmp.eliminate_zeros()
                        param_dict["LIST_L2_COO"][i] = tmp
                        param_dict["LIST_STATE_INDICES_BY_INDEX_L2"].append(
                            param_dict["LIST_STATE_INDICES_BY_HMODE"][j]
                        )

        # Define the Noise1 Operator Values
        # ---------------------------------
        param_dict["LIST_INDEX_L2_BY_NMODE1"] = [
            None for i in range(len(param_dict["PARAM_NOISE1"]))
        ]
        l2_as_tuples = [self._array_to_tuple(l) for l in param_dict["L_NOISE1"]]
        for (i, l) in enumerate(list_unique_L2):
            # i is the index of operator l in the unique list of operators
            for j in range(len(l2_as_tuples)):
                if l2_as_tuples[j] == l:
                    param_dict["LIST_INDEX_L2_BY_NMODE1"][j] = i

        # Define the Noise2 Operator Values
        # ---------------------------------
        if "L_NOISE2" in param_dict.keys():
            param_dict["LIST_INDEX_L2_BY_NMODE2"] = [
                None for i in range(len(param_dict["PARAM_NOISE2"]))
            ]
            l2_as_tuples = [self._array_to_tuple(l) for l in param_dict["L_NOISE2"]]
            for (i, l) in enumerate(list_unique_L2):
                # i is the index of operator l in the unique list of operators
                for j in range(len(l2_as_tuples)):
                    if l2_as_tuples[j] == l:
                        param_dict["LIST_INDEX_L2_BY_NMODE2"][j] = i

        return param_dict

    def initialize(self, flag_adaptive, psi_0):
        """
        Creates a state list depending on whether the calculation is adaptive or not.

        Parameters
        ----------
        1. flag_adaptive : boolean
                           Boolean that defines the adaptivity
                           True: Adaptive, False: Static.

        2. psi_0 : np.array
                   Initial user inputted wave function.

        Returns
        -------
        None
        """
        self.adaptive = flag_adaptive

        if flag_adaptive:
            self.state_list = np.where(np.abs(psi_0) > 0)[0]
        else:
            self.state_list = np.arange(self.__ndim)

    @staticmethod
    def _array_to_tuple(array):
        """
        Converts an inputted array to a tuple.

        Parameters
        ----------
        1. array : np.array
                   Numpy array.

        Returns
        -------
        1. tuple : tuple
                   Array in tuple form.
        """
        if sp.sparse.issparse(array):
            if array.getnnz() > 0:
                return tuple([tuple(l) for l in np.nonzero(array)])
            else:
                return tuple([])
        else:
            if len(array) > 0:
                return tuple([tuple(l) for l in np.nonzero(array)])
            else:
                return tuple([])

    @staticmethod
    def _get_states_from_L2(lop):
        """
        Fetches the states that the L operators interacts with.

        Parameters
        ----------
        1. lop : array
                 L2 operator.

        Returns
        -------
        1. tuple : tuple
                   Tuple of states that correspond to the specific L operator.
        """

        i_x, i_y = np.nonzero(lop)
        return tuple(set(i_x) | set(i_y))

    @property
    def size(self):
        return len(self.__state_list)

    @property
    def state_list(self):
        return self.__state_list

    @state_list.setter
    def state_list(self, new_state_list):
        # Construct information about pevious timestep
        # --------------------------------------------
        self.__previous_state_list = self.__state_list
        self.__list_add_state = list(set(new_state_list) - set(self.__previous_state_list ))
        self.__list_add_state.sort()
        self.__list_stable_state = list(
            set(self.__previous_state_list ).intersection(set(new_state_list))
        )
        self.__list_stable_state.sort()

        if set(new_state_list) != set(self.__previous_state_list):
            # Prepare New State List
            # ----------------------
            new_state_list.sort()
            self.__state_list = np.array(new_state_list)

            # Update Local Indexing
            # ----------------------
            # state_list is the indexing system for states (takes i_rel --> i_abs)
            # list_absindex_L2_active is the indexing system for L2 (takes i_rel --> i_abs)
            # list_absindex_state_modes is the indexing system for hierarchy modes (takes i_rel --> i_abs)
            self.__list_absindex_state_modes = np.array(
                [
                    i_mod
                    for i_mod in range(self.param["N_HMODES"])
                    if np.size(np.intersect1d(self.param["LIST_STATE_INDICES_BY_HMODE"][i_mod],
                                  self.state_list)) != 0
                ]
            )

            self.__list_absindex_L2_active = np.array(
                [
                    i_lop
                    for i_lop in range(self.param["N_L2"])
                    if np.size(np.intersect1d(self.param["LIST_STATE_INDICES_BY_INDEX_L2"][i_lop],
                                  self.state_list)) != 0
                ]
            )

            # Update Local Properties
            # -----------------------
            self._hamiltonian = self.param["HAMILTONIAN"][
                np.ix_(self.__state_list, self.__state_list)
            ]

    @property
    def previous_state_list(self):
        return self.__previous_state_list

    @property
    def list_stable_state(self):
        return self.__list_stable_state

    @property
    def list_add_state(self):
        return self.__list_add_state

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @property
    def list_absindex_state_modes(self):
        return self.__list_absindex_state_modes

    @property
    def list_absindex_L2_active(self):
        return self.__list_absindex_L2_active

    @staticmethod
    def reduce_sparse_matrix(coo_mat, state_list):
        """
        Takes in a sparse matrix and list which represents the absolute
        state to a new relative state represented in a sparse matrix

        Parameters
        ----------
        1. coo_mat : scipy sparse matrix
                     Sparse matrix.

        2. state_list : list
                       List of relative index.

        Returns
        -------
        1. sparse : np.array
                    Sparse matrix in relative basis.
        """
        coo_tuple = np.array([(i, j, data) for (i,j,data) in zip(coo_mat.row, coo_mat.col, coo_mat.data)
                               if ((i in state_list) and (j in state_list))])
        if len(coo_tuple) == 0:
            return sp.sparse.coo_matrix((len(state_list), len(state_list)))
        else:
            coo_tuple = np.atleast_2d(coo_tuple)
            coo_row = [list(state_list).index(i) for i in coo_tuple[:,0]]
            coo_col = [list(state_list).index(i) for i in coo_tuple[:,1]]
            coo_data = coo_tuple[:,2]

            return sp.sparse.coo_matrix(
                (coo_data, (coo_row, coo_col)), shape=(len(state_list), len(state_list))
            )
