import numpy as np
import scipy as sp
from scipy import sparse
import copy
from collections import Counter

__title__ = "System Class"
__author__ = "D. I. G. Bennett, Leo Varvelo"
__version__ = "1.0"


class HopsSystem(object):
    """
    HopsSystem is a class that stores the basic information about the system and
    system-bath coupling.
    """

    def __init__(self, system_param):
        """
        INPUTS
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

        RETURNS
        -------
        None

        NOTE: L_HIER is required to contain all L-operators that are defined anywhere.
            This can be removed as a requirement by defining a third noise parameter
            that will get its own super-operators, but since we have no use-case yet
            this has not been implemented.
        """
        self.param = self._initialize_system_dict(system_param)
        self.__ndim = self.param["NSTATES"]

    def _initialize_system_dict(self, system_param):
        """
        This function is responsible for extending the user input to the
        complete set of parameters defined above.

        PARAMETERS
        ----------
        1. system_param : dict
                          A dictionary with the system and system-bath coupling
                          parameters defined.

        RETURNS
        -------
        1. param_dict : dict
                        a dictionary containing the user input and the derived parameters
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
            0 for i in range(param_dict["N_HMODES"])
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
            0 for i in range(len(param_dict["PARAM_NOISE1"]))
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
                0 for i in range(len(param_dict["PARAM_NOISE2"]))
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
        This function creates a state list depending on whether the calculation
        is adaptive or not

        PARAMETERS
        ----------
        1. flag_adaptive : boolean
                           a boolean that defines the adaptivity
                           True: Adaptive, False: Static
        2. psi_0 : np.array
                   the initial user inputted wave function

        RETURNS
        -------
        None
        """
        self.adaptive = flag_adaptive

        if flag_adaptive:
            self.state_list = np.where(psi_0 > 0)[0]
        else:
            self.state_list = np.arange(self.__ndim)

    @staticmethod
    def _array_to_tuple(array):
        """
        This function converts an inputted array to a tuple

        PARAMETERS
        ----------
        1. array : np.array
                   a numpy array

        RETURNS
        -------
        1. tuple : tuple
                   array in tuple form
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
        This function fetches the states that the L operators interacts with

        PARAMETERS
        ----------
        1. lop : array
                 an L2 operator

        RETURNS
        -------
        1. tuple : tuple
                   a tuple of states that correspond the the specific L operator
        """

        try:
            tmp = np.abs(lop) > 0
            i_x, i_y = np.where(tmp)
            i_test = []
            i_test.extend(i_x)
            i_test.extend(i_y)
        except:
            sparse_lop = sp.sparse.find(np.abs(lop))
            i_test = []
            i_test.append(sparse_lop[0][0])
            i_test.append(sparse_lop[1][0])

        return tuple(set(i_test))

    @property
    def size(self):
        return len(self.__state_list)

    @property
    def state_list(self):
        return self.__state_list

    @state_list.setter
    def state_list(self, state_list):
        # Update Local Indexing
        # ------------------------
        # state_list is the indexing system for states (takes i_rel --> i_abs)
        # list_absindex_L2 is the indexing system for L2 (takes i_rel --> i_abs)
        # list_absindex_mode is the indexing system for hierarchy modes (takes i_rel --> i_abs)
        state_list.sort()
        self.__state_list = np.array(state_list)
        self._list_absindex_L2 = np.array(
            [
                i_lop
                for i_lop in range(self.param["N_L2"])
                if self.param["LIST_STATE_INDICES_BY_INDEX_L2"][i_lop][0]
                in self.state_list
            ]
        )
        self._list_absindex_mode = np.array(
            [
                i_mod
                for i_mod in range(self.param["N_HMODES"])
                if self.param["LIST_STATE_INDICES_BY_HMODE"][i_mod][0]
                in self.state_list
            ]
        )

        # Update Local Properties
        # -----------------------
        # The following variables are in the relative basis
        self._hamiltonian = self.param["HAMILTONIAN"][
            np.ix_(self.state_list, self.state_list)
        ]
        self._n_hmodes = len(self._list_absindex_mode)
        self._g = np.array(self.param["G"])[self._list_absindex_mode]
        self._w = np.array(self.param["W"])[self._list_absindex_mode]
        self._n_l2 = len(self._list_absindex_L2)
        self._list_L2_coo = np.array(
            [
                self.reduce_sparse_matrix(self.param["LIST_L2_COO"][k], self.state_list)
                for k in self._list_absindex_L2
            ]
        )
        self._list_index_L2_by_hmode = [
            list(self._list_absindex_L2).index(
                self.param["LIST_INDEX_L2_BY_HMODE"][imod]
            )
            for imod in self._list_absindex_mode
        ]
        self._list_state_indices_by_hmode = np.array(
            [
                self._get_states_from_L2(
                    self._list_L2_coo[self._list_index_L2_by_hmode[i_mode]]
                )
                for i_mode in range(self.n_hmodes)
            ]
        )
        self._list_state_indices_by_index_L2 = np.array(
            [
                self._get_states_from_L2(self._list_L2_coo[i_lop])
                for i_lop in range(self.n_l2)
            ]
        )

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @property
    def n_hmodes(self):
        return self._n_hmodes

    @property
    def g(self):
        return self._g

    @property
    def w(self):
        return self._w

    @property
    def n_l2(self):
        return self._n_l2

    @property
    def list_state_indices_by_hmode(self):
        return self._list_state_indices_by_hmode

    @property
    def list_state_indices_by_index_L2(self):
        return self._list_state_indices_by_index_L2

    @property
    def list_L2_coo(self):
        return self._list_L2_coo

    @property
    def list_index_L2_by_hmode(self):
        return self._list_index_L2_by_hmode

    @property
    def list_absindex_L2(self):
        return self._list_absindex_L2

    @property
    def list_absindex_mode(self):
        return self._list_absindex_mode

    @staticmethod
    def reduce_sparse_matrix(coo_mat, state_list):
        """
        This function takes in a sparse matrix and list which represents the absolute
        state to a new relative state represented in a sparse matrix

        NOTE: This function assumes that all states associated with the non-zero
        elements of the sparse matrix are present in the state list!

        PARAMETERS
        ----------
        1. coo_mat : scipy sparse matrix
                     a sparse matrix
        2. state_list : list
                       a list of relative index

        RETURNS
        -------
        1. sparse : np.array
                    sparse matrix in relative basis
        """
        coo_row = [list(state_list).index(i) for i in coo_mat.row]
        coo_col = [list(state_list).index(i) for i in coo_mat.col]
        coo_data = coo_mat.data
        return sp.sparse.coo_matrix(
            (coo_data, (coo_row, coo_col)), shape=(len(state_list), len(state_list))
        )
