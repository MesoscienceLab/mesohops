from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy as sp
from scipy import sparse

from mesohops.basis.system_functions import initialize_system_dict

__title__ = "System Class"
__author__ = "D. I. G. Bennett, L. Varvelo, J. K. Lynd, B. Z. Citty"
__version__ = "1.6"


class HopsSystem:
    """
    Stores the basic information about the system and system-bath coupling.
    """

    __slots__ = (
        # --- Core basis components ---
        'param',            # System parameters (main dictionary)
        '__ndim',           # System dimension (number of states)
        '_list_lt_corr_param',   # Low-temperature correction parameters
        '_hamiltonian',     # System Hamiltonian (sparse or dense)

        # --- State list bookkeeping (for adaptive basis) ---
        '__previous_state_list',  # Previous state list (for adaptive updates)
        '__state_list',           # Current state list
        'adaptive',               # Adaptive flag (True if adaptive basis is used)
        '__list_add_state',       # States to add in update
        '__list_stable_state',    # States stable between updates
        '_list_boundary_state',   # States coupled to basis by Hamiltonian

        # --- Indexing of modes & L-operators in the current basis ---
        '__list_absindex_state_modes',     # State mode indices (absolute)
        '__list_absindex_new_state_modes', # New state mode indices (absolute)
        '__list_absindex_L2_active',       # Active L2 indices (absolute)
        '__list_destination_state',        # Destination states for each state
        '__dict_relindex_states',          # Relative state indices
    )

    def __init__(self, system_param: dict[str, Any] | str | os.PathLike[str] | Path) -> None:
        """
        Inputs
        ------
        1. system_param : dict | str | Path
                          Either a dictionary with the system and system-bath coupling
                          parameters defined or a str/pathlike that points to a file
                          that has been generated with the method save_dict_param().
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
            k. SPARSE_HAMILTONIAN : sp.sparse.csc_array(complex)
                                    Sparse representation of the Hamiltonian.


        Returns
        -------
        None

        NOTE: L_HIER is required to contain all L-operators that are defined anywhere.
            This can be removed as a requirement by defining a third noise parameter
            that will get its own super-operators, but since we have no use-case yet
            this has not been implemented.
        """
        if isinstance(system_param, (str, os.PathLike)):
            self.param = pickle.load(open(system_param, "rb"))
        elif isinstance(system_param, dict):
            self.param = initialize_system_dict(system_param)
        else:
            raise TypeError("system_param must be a dictionary or a file path.")
        self.__ndim = self.param["NSTATES"]
        self.__previous_state_list = None
        self.__state_list = []

    def initialize(self, flag_adaptive: bool, psi_0: np.ndarray) -> None:
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

    def save_dict_param(self, filepath: str | os.PathLike[str] | Path) -> None:
        """
        Serialize the system parameters to a file.

        This method saves the current system parameters stored in `self.param` to the specified
        file path using the pickle serialization format. This allows for easy storage and retrieval
        of the system's configuration.

        Parameters
        ----------
        1. filepath : str or os.PathLike
                      The path to the file where the system parameters will be saved.

        Returns
        -------
        None
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.param, f)

    @property
    def size(self) -> int:
        return len(self.__state_list)

    @property
    def state_list(self) -> np.ndarray | list:
        return self.__state_list

    @property
    def list_destination_state(self) -> np.ndarray:
        return self.__list_destination_state
        
    @property
    def list_boundary_state(self) -> list[int]:
        return self._list_boundary_state

    @property 
    def list_sc(self) -> list[int]:
        list_boundary_lop = list(set(self.list_destination_state) - set(self.state_list))
        return list(set(list_boundary_lop) | set(self.list_boundary_state))
    @property
    def dict_relative_index_by_state(self) -> dict[int, int]:
        return self.__dict_relindex_states

    @state_list.setter
    def state_list(self, new_state_list: Sequence[int] | np.ndarray) -> None:
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


            # Get the list of destination states linked to the current state basis by
            # the full set of L-operators, under the assumption that an L-operator
            # must be active if a state associated with it is in the basis.
            self.__list_destination_state = np.array(
                list(
                    set(
                        list(
                            np.concatenate([self.param[
                                        "LIST_DESTINATION_STATES_BY_STATE_INDEX"][state]
                                                 for state in self.state_list]))
                    )
                ), dtype=int
            )

            self.__dict_relindex_states = {self.state_list[s]: s for s in range(len(
                self.state_list))}

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
            self._list_lt_corr_param = np.array(self.param["LIST_LT_PARAM"])[
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
            self._list_boundary_state = [self.param["COUPLED_STATES"][state] for state in self.state_list]
            self._list_boundary_state = list(set([state_conn for conn_list in self._list_boundary_state for state_conn in conn_list ]) - set(self.state_list))
            
    @property
    def previous_state_list(self) -> np.ndarray | None:
        return self.__previous_state_list

    @property
    def list_stable_state(self) -> np.ndarray | list:
        return self.__list_stable_state

    @property
    def list_add_state(self) -> np.ndarray | list:
        return self.__list_add_state

    @property
    def hamiltonian(self) -> sp.sparse.spmatrix | np.ndarray:
        return self._hamiltonian

    @property
    def list_absindex_state_modes(self) -> np.ndarray:
        return self.__list_absindex_state_modes
    @property
    def list_absindex_new_state_modes(self) -> np.ndarray:
        return self.__list_absindex_new_state_modes
    @property
    def list_absindex_L2_active(self) -> np.ndarray:
        return self.__list_absindex_L2_active
    @property
    def list_lt_corr_param(self) -> np.ndarray:
        return self._list_lt_corr_param

    @property
    def list_off_diag(self) -> np.ndarray:
        return self.param["list_L2_off_diag"]

    @staticmethod
    def reduce_sparse_matrix(
        coo_mat: sp.sparse.spmatrix, state_list: Sequence[int]
    ) -> sp.sparse.coo_matrix:
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
