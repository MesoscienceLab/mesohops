import numpy as np
import scipy as sp
from scipy import sparse
import copy
from collections import Counter
from mesohops.util.helper_functions import array_to_tuple

__title__ = "System Class"
__author__ = "D. I. G. Bennett, L. Varvelo, J. K. Lynd"
__version__ = "1.2"


class HopsSystem(object):
    """
    Stores the basic information about the system and system-bath coupling.
    """

    def __init__(self, system_param):
        """
        Inputs
        ------
        1. system_param : dict
                          Dictionary with the system and system-bath coupling
                          parameters defined.
            a. HAMILTONIAN : np.array
                             Array that contains the system Hamiltonian.
            b. GW_SYSBATH : list(complex)
                            List of parameters (g,w) that define the exponential
                            decomposition underlying the hierarchy.
            c. L_HIER : list(sparse matrix)
                        List of system-bath coupling operators associated with each
                        hierarchy bath mode in the same order as GW_SYSBATH.
            d. ALPHA_NOISE1 : function
                              Calculates the correlation function given (t_axis,
                              *PARAM_NOISE_1).
            e. PARAM_NOISE1 : list
                              List of parameters defining the decomposition of Noise1.
            f. L_NOISE1 : list(sparse matrix)
                          List of system-bath coupling operators associated with each
                          hierarchy bath mode in the same order as
                          PARAM_NOISE_1.

        Optional Parameters
            a. ALPHA_NOISE2 : function
                              Calculates the correlation function given (t_axis,
                              *PARAM_NOISE_2).
            b. PARAM_NOISE2 : list
                              List of parameters defining the decomposition of Noise2.
            c. L_NOISE2 : list(sparse matrix)
                          List of system-bath coupling operators in the same order as
                          PARAM_NOISE_2.
            d. PARAM_LT_CORR : list(complex)
                               List of low-temperature correction coefficients for each
                               independent thermal environment [units: cm^-1].
            e. L_LT_CORR : list(sparse matrix)
                           System-bath coupling operators associated with each
                           low-temperature correction coefficient in the same order as
                           PARAM_LT_CORR.

        Derived Parameters
        The result is a dictionary HopsSystem.param which contains all the above plus
        additional parameters that are useful for indexing the simulation:
            a. NSTATES : int
                         Dimension of the system Hilbert Space.
            b. N_HMODES : int
                          Number of modes that will appear in the hierarchy.
            c. N_L2 : int
                      Number of unique system-bath coupling operators.
            d. LIST_INDEX_L2_BY_NMODE1 : np.array(int)
                                         Maps list_absindex_noise1 to index_L2.
            e. LIST_INDEX_L2_BY_NMODE2 : np.array(int)
                                         Maps list_absindex_noise2 to index_L2.
            f. LIST_INDEX_L2_BY_LT_CORR : np.array(int)
                                          Maps list_absindex_LT_CORR to index_L2.
            g. LIST_INDEX_L2_BY_HMODE : np.array(int)
                                        Maps list_absindex_by_hmode to index_L2.
            h. LIST_STATE_INDICES_BY_HMODE : np.array(int)
                                             Maps list_absindex_by_hmode to
                                             list_absindex_states.
            i. LIST_L2_COO : np.array(sparse matrix)
                             Maps list_absindex_L2 to coo_sparse.
            j. LIST_STATE_INDICES_BY_INDEX_L2 : np.array(int)
                                                Maps list_absindex_L2 to
                                                list_absindex_states.
            k. SPARSE_HAMILTONIAN : sp.sparse.csc_matrix(complex)
                                    Sparse representation of the Hamiltonian.


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
        Extends the user input to the complete set of parameters defined above.

        Parameters
        ----------
        1. system_param : dict
                          Dictionary with the system and system-bath coupling
                          parameters defined.

        Returns
        -------
        1. param_dict : dict
                        Dictionary containing the user input and the derived parameters.
        """
        param_dict = copy.deepcopy(system_param)
        if(sparse.issparse(system_param["HAMILTONIAN"])):
            param_dict["NSTATES"] = sparse.coo_matrix.get_shape(system_param["HAMILTONIAN"])[0]
        else:
            param_dict["NSTATES"] = len(system_param["HAMILTONIAN"][0])
        param_dict["N_HMODES"] = len(system_param["GW_SYSBATH"])
        param_dict["G"] = np.array([g for (g, w) in system_param["GW_SYSBATH"]])
        param_dict["W"] = np.array([w for (g, w) in system_param["GW_SYSBATH"]])
        param_dict["LIST_STATE_INDICES_BY_HMODE"] = [
            self._get_states_from_L2(L2) for L2 in param_dict["L_HIER"]
        ]
        param_dict["LIST_HMODE_INDICES_BY_STATE"] = [[] for i in range(param_dict["NSTATES"])]
        for (hmode,state_indices) in enumerate(param_dict["LIST_STATE_INDICES_BY_HMODE"]):
            for state in state_indices:
                param_dict["LIST_HMODE_INDICES_BY_STATE"][state].append(hmode)
        
        
        param_dict["SPARSE_HAMILTONIAN"] = sparse.csc_matrix(param_dict["HAMILTONIAN"])
        param_dict["SPARSE_HAMILTONIAN"].eliminate_zeros()

        # Checks for low-temperature correction terms - if there are none, initialize
        # empty lists as placeholders:
        if not "L_LT_CORR" in param_dict.keys():
            param_dict["L_LT_CORR"] = []
            param_dict["PARAM_LT_CORR"] = []

        # Define the Hierarchy Operator Values
        # ------------------------------------
        # Since arrays and lists are not hashable, we will turn our operators
        # into tuples in order to conveniently define a number of indexing
        # parameters.

        # Creates list of unique l2 tuples in order they appear in "L_HIER"
        l2_as_tuples = [array_to_tuple(L2) for L2 in param_dict["L_HIER"]]
        list_unique_l2_as_tuples = list(Counter(l2_as_tuples))
        param_dict["N_L2"] = len(set(list_unique_l2_as_tuples))

        # Creates L2 indexing parameters
        param_dict["LIST_INDEX_L2_BY_HMODE"] = [
            None for i in range(param_dict["N_HMODES"])
        ]
        flag_l2_list = [False for i in range(param_dict["N_L2"])]
        param_dict["LIST_L2_COO"] = [0 for i in range(param_dict["N_L2"])]
        param_dict["LIST_LT_PARAM"] = [0 for i in range(param_dict["N_L2"])]


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

        l2_LT_CORR_as_tuples = [array_to_tuple(l) for l in
                                param_dict["L_LT_CORR"]]
                                
        param_dict["LIST_INDEX_L2_BY_STATE_INDICES"] = [[] for i in range(param_dict["NSTATES"])]
        for (index_L2,state_indices) in enumerate(param_dict["LIST_STATE_INDICES_BY_INDEX_L2"]):
            for state in state_indices:
                param_dict["LIST_INDEX_L2_BY_STATE_INDICES"][state].append(index_L2)
        # Build a list of low-temperature coefficients guaranteed to be in the same
        # order as the associated unique sparse L2 operators.
        for j in range(len(param_dict["L_LT_CORR"])):
            l_op_check = 0
            for (i, l) in enumerate(list_unique_l2_as_tuples):
                # i is the index of operator l in the unique list of operators
                if l2_LT_CORR_as_tuples[j] == l:
                    param_dict["LIST_LT_PARAM"][i] = param_dict["PARAM_LT_CORR"][j]
                    l_op_check += 1
            if not l_op_check:
                print("WARNING: the list of low-temperature correction "
                      "L-operators contains an L-operator not associated with any "
                      "existing thermal environment. This low-temperature "
                      "correction factor will be discarded!")

        # Define the Noise1 Operator Values
        # ---------------------------------
        param_dict["LIST_INDEX_L2_BY_NMODE1"] = [
            None for i in range(len(param_dict["PARAM_NOISE1"]))
        ]
        l2_as_tuples = [array_to_tuple(l) for l in param_dict["L_NOISE1"]]
        for (i, l) in enumerate(list_unique_L2):
            # i is the index of operator l in the unique list of operators
            for j in range(len(l2_as_tuples)):
                if l2_as_tuples[j] == l:
                    param_dict["LIST_INDEX_L2_BY_NMODE1"][j] = i
        if None in param_dict["LIST_INDEX_L2_BY_NMODE1"]:
            print("WARNING: the list of noise 1 L-operators contains an L-operator "
                  "not associated with any existing thermal environment. The noise "
                  "associated with this L-operator will be discarded!")

        # Define the Noise2 Operator Values
        # ---------------------------------
        if "L_NOISE2" in param_dict.keys():
            param_dict["LIST_INDEX_L2_BY_NMODE2"] = [
                None for i in range(len(param_dict["PARAM_NOISE2"]))
            ]
            l2_as_tuples = [array_to_tuple(l) for l in param_dict["L_NOISE2"]]
            for (i, l) in enumerate(list_unique_L2):
                # i is the index of operator l in the unique list of operators
                for j in range(len(l2_as_tuples)):
                    if l2_as_tuples[j] == l:
                        param_dict["LIST_INDEX_L2_BY_NMODE2"][j] = i
            if None in param_dict["LIST_INDEX_L2_BY_NMODE2"]:
                print("WARNING: the list of noise 2 L-operators contains an L-operator "
                      "not associated with any existing thermal environment. The noise "
                      "associated with this L-operator will be discarded!")

        return param_dict

    def initialize(self, flag_adaptive, psi_0):
        """
        Creates a state list depending on whether the calculation is adaptive or not.

        Parameters
        ----------
        1. flag_adaptive : bool
                           True indicates an adaptive basis while False indicates a static
                           basis.

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
    def _get_states_from_L2(lop):
        """
        Fetches the states that the L operators interacts with.

        Parameters
        ----------
        1. lop : np.array(complex)
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
        # Construct information about previous timestep
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
                    self.param["LIST_HMODE_INDICES_BY_STATE"][state][mode]
                    for state in self.state_list
                    for mode in range(len(self.param["LIST_HMODE_INDICES_BY_STATE"][state]))
                ], dtype=int
            )
            self.__list_absindex_state_modes = np.sort(np.array(list(set(self.__list_absindex_state_modes))))
            self.__list_absindex_new_state_modes = np.array(
                [   
                    self.param["LIST_HMODE_INDICES_BY_STATE"][new_state][mode]
                    for new_state in self.__list_add_state
                    for mode in range(len(self.param["LIST_HMODE_INDICES_BY_STATE"][new_state]))
                ], dtype=int
            )
            self.__list_absindex_new_state_modes = np.sort(np.array(list(set(self.__list_absindex_new_state_modes))))
            self.__list_absindex_L2_active = np.array(
                [   
                    self.param["LIST_INDEX_L2_BY_STATE_INDICES"][state][L2]
                    for state in self.state_list
                    for L2 in range(len(self.param["LIST_INDEX_L2_BY_STATE_INDICES"][state]))
                ], dtype=int
            )
            self.__list_absindex_L2_active = np.sort(np.array(list(set(self.__list_absindex_L2_active)),dtype=int))
            self._lt_corr_param = np.array(self.param["LIST_LT_PARAM"])[
                 self.__list_absindex_L2_active]

            # Update Local Properties
            # -----------------------
            if(sparse.issparse(self.param["HAMILTONIAN"])):
                self._hamiltonian = self.param["SPARSE_HAMILTONIAN"][
                    np.ix_(self.state_list, self.state_list)
                ]
            else:
                self._hamiltonian = self.param["HAMILTONIAN"][
                    np.ix_(self.state_list, self.state_list)
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
    def list_absindex_new_state_modes(self):
        return self.__list_absindex_new_state_modes
    @property
    def list_absindex_L2_active(self):
        return self.__list_absindex_L2_active

    @staticmethod
    def reduce_sparse_matrix(coo_mat, state_list):
        """
        Takes in a sparse matrix and list which represents the absolute
        state to a new relative state represented in a sparse matrix.

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
