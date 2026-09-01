from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy as sp
from scipy import sparse

from mesohops.basis.system_functions import initialize_system_dict
from mesohops.util.physical_constants import hbar

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
        '_hamiltonian',     # System Hamiltonian, basis states (sparse or dense)
        '_H2_hamiltonian_extd', # System Hamiltonian, basis + boundary states (sparse or dense)
        '_dict_nzhamiltonian_abs', # System Hamiltonian nonzero dictionary keyed by (row, col)

        # --- State list bookkeeping (for adaptive basis) ---
        '__previous_state_list',  # Previous state list (for adaptive updates)
        '_list_stateidx_abs',     # Current state list
        'adaptive',               # Adaptive flag (True if adaptive basis is used)
        '_list_newstateidx_abs',  # States to add in update
        '_list_stblstateidx_abs', # States stable between updates
        '_list_bndstateidx_abs',  # States coupled to basis by Hamiltonian
        '_system_timescale',      # The estimated fastest timescale of H2
        '_list_stateidx_extd',          # Indices of basis elements in system + boundary state list
        '_list_bndstateidx_extd',       # Indices of boundary elements in system + boundary state list

        # --- Indexing of modes & L-operators in the current basis ---
        '_list_statemodeidx_abs',          # State mode indices (absolute)
        '_list_newstatemodeidx_abs',       # New state mode indices (absolute)
        '_list_activel2idx_abs',           # Active L2 indices (absolute)
        '__list_destination_state',        # Destination states for each state
        '__dict_relindex_states',          # Relative state indices
        'flag_nearest_neighbor_ham',         # True if Hamiltonian has only nearest-neighbor coupling
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
                                         Maps noise1 mode indices to index_L2.
            e. LIST_INDEX_L2_BY_NMODE2 : np.array(int)
                                         Maps noise2 mode indices to index_L2.
            f. LIST_INDEX_L2_BY_LT_CORR : np.array(int)
                                          Maps low-temperature correction indices
                                          to index_L2.
            g. LIST_INDEX_L2_BY_HMODE : np.array(int)
                                        Maps hierarchy mode index to index_L2.
            h. LIST_STATE_INDICES_BY_HMODE : np.array(int)
                                             Maps hierarchy mode index to
                                             state indices.
            i. LIST_L2_COO : np.array(sparse matrix)
                             Maps list_l2idx_abs to coo_sparse.
            j. LIST_STATE_INDICES_BY_INDEX_L2 : np.array(int)
                                                Maps list_l2idx_abs to
                                                state indices.
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
        self._list_stateidx_abs = []
        H2_hamiltonian_abs_coo = self.param["SPARSE_HAMILTONIAN"].tocoo()
        self._dict_nzhamiltonian_abs = {}
        for row, col, data in zip(
            H2_hamiltonian_abs_coo.row, H2_hamiltonian_abs_coo.col, H2_hamiltonian_abs_coo.data
        ):
            key = (row, col)
            if key in self._dict_nzhamiltonian_abs:
                self._dict_nzhamiltonian_abs[key] += data
            else:
                self._dict_nzhamiltonian_abs[key] = data
        self.flag_nearest_neighbor_ham = self._is_nearest_neighbor(
            self.param['SPARSE_HAMILTONIAN']
        )

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
        return len(self._list_stateidx_abs)

    @property
    def state_list(self) -> np.ndarray | list:
        return self._list_stateidx_abs

    @property
    def list_destination_state(self) -> np.ndarray:
        return self.__list_destination_state
        
    @property
    def list_bndstateidx_abs(self) -> list[int]:
        return self._list_bndstateidx_abs

    @property
    def list_fullbndidx_abs(self) -> list[int]:
        list_bndl2 = sorted(set(self.list_destination_state) - set(self.state_list))
        return sorted(set(list_bndl2) | set(self.list_bndstateidx_abs))
    @property
    def dict_relative_index_by_state(self) -> dict[int, int]:
        return self.__dict_relindex_states

    @state_list.setter
    def state_list(self, new_state_list: Sequence[int] | np.ndarray) -> None:
        # Construct information about previous timestep
        # --------------------------------------------
        self.__previous_state_list = self._list_stateidx_abs
        self._list_newstateidx_abs = sorted(set(new_state_list) - set(self.__previous_state_list ))
        self._list_newstateidx_abs.sort()
        self._list_stblstateidx_abs = sorted(
            set(self.__previous_state_list ).intersection(set(new_state_list))
        )
        self._list_stblstateidx_abs.sort()

        if set(new_state_list) != set(self.__previous_state_list):
            # Prepare New State List
            # ----------------------
            new_state_list.sort()
            self._list_stateidx_abs = np.array(new_state_list)

            # Update Local Indexing
            # ----------------------
            # state_list is the indexing system for states (takes i_rel --> i_abs)
            # list_activel2idx_abs is the indexing system for L2 (takes i_rel --> i_abs)
            # list_statemodeidx_abs is the indexing system for hierarchy modes (takes i_rel --> i_abs)
            self._list_statemodeidx_abs = np.array(
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

            self._list_statemodeidx_abs = np.sort(np.array(sorted(set(self._list_statemodeidx_abs))))
            self._list_newstatemodeidx_abs = np.array(
                [
                    self.param["LIST_HMODE_INDICES_BY_STATE"][new_state][mode]
                    for new_state in self._list_newstateidx_abs
                    for mode in range(len(self.param["LIST_HMODE_INDICES_BY_STATE"][new_state]))
                ], dtype=int
            )
            self._list_newstatemodeidx_abs = np.sort(np.array(sorted(set(self._list_newstatemodeidx_abs))))
            self._list_activel2idx_abs = np.array(
                [
                    self.param["LIST_INDEX_L2_BY_STATE_INDICES"][state][L2]
                    for state in self.state_list
                    for L2 in range(len(self.param["LIST_INDEX_L2_BY_STATE_INDICES"][state]))
                ], dtype=int
            )
            self._list_activel2idx_abs = np.sort(np.array(sorted(set(self._list_activel2idx_abs)),dtype=int))
            self._list_lt_corr_param = np.array(self.param["LIST_LT_PARAM"])[
                 self._list_activel2idx_abs]

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

            self._list_bndstateidx_abs = [self.param["COUPLED_STATES"][state] for state in self.state_list]
            self._list_bndstateidx_abs = sorted(set([state_conn for conn_list in self._list_bndstateidx_abs for state_conn in conn_list ]) - set(self.state_list))
            energy_spread = np.max(self._hamiltonian) - np.min(self._hamiltonian)
            if energy_spread == 0:
                self._system_timescale = np.inf
            else:
                self._system_timescale = np.abs(hbar / energy_spread)

            list_state_extd = sorted(
                set(self.state_list)
                | set(self.list_destination_state)
                | set(self.list_bndstateidx_abs)
            )
            self._list_stateidx_extd = [list_state_extd.index(state) for state in self.state_list]
            self._list_bndstateidx_extd = [list_state_extd.index(state) for state in self.list_fullbndidx_abs]
            H2_hamiltonian_extd_coo = self.reduce_sparse_matrix(
                self._dict_nzhamiltonian_abs, list_state_extd, True
            )
            self._H2_hamiltonian_extd = sparse.csr_array(
                (H2_hamiltonian_extd_coo.data, (H2_hamiltonian_extd_coo.row, H2_hamiltonian_extd_coo.col)),
                shape=H2_hamiltonian_extd_coo.shape,
            )
    @property
    def previous_state_list(self) -> np.ndarray | None:
        return self.__previous_state_list

    @property
    def list_stblstateidx_abs(self) -> np.ndarray | list:
        return self._list_stblstateidx_abs

    @property
    def list_newstateidx_abs(self) -> np.ndarray | list:
        return self._list_newstateidx_abs

    @property
    def hamiltonian(self) -> sp.sparse.spmatrix | np.ndarray:
        return self._hamiltonian

    @property
    def H2_hamiltonian_extd(self) -> np.ndarray:
        return self._H2_hamiltonian_extd

    @property
    def list_statemodeidx_abs(self) -> np.ndarray:
        return self._list_statemodeidx_abs

    @property
    def list_newstatemodeidx_abs(self) -> np.ndarray:
        return self._list_newstatemodeidx_abs

    @property
    def list_activel2idx_abs(self) -> np.ndarray:
        return self._list_activel2idx_abs

    @property
    def list_lt_corr_param(self) -> np.ndarray:
        return self._list_lt_corr_param

    @property
    def list_off_diag(self) -> np.ndarray:
        return self.param["list_L2_off_diag"]

    @property
    def system_timescale(self) -> float:
        return self._system_timescale

    @property
    def list_stateidx_extd(self) -> list:
        return self._list_stateidx_extd

    @property
    def list_bndstateidx_extd(self) -> list:
        return self._list_bndstateidx_extd

    @staticmethod
    def reduce_sparse_matrix(
        dict_l2_nnz: dict,
        state_list: Sequence[int],
        off_diag: bool,
        filter_nz: bool = False,
    ) -> sp.sparse.coo_matrix:
        """
        Takes in a sparse matrix and list which represents the absolute
        state to a new relative state represented in a sparse matrix.

        This version is size invariant with respect to global operator size.
        Naive global slicing or filtering carries scaling with the full-system
        nonzero structure, which can grow as O(N) in the total number of
        system states.

        Parameters
        ----------
        1. dict_l2_nnz: dict
                        Sparse nonzero entries keyed by (state_i, state_j).

        2. state_list: list(int)
                        List of relative index.

        3. off_diag: bool
                     True if off-diagonal couplings are included.

        4. filter_nz: bool
                     If True, filter state_list to only states that participate
                     in nonzero entries before building the matrix.

        Returns
        -------
        1. sparse: scipy sparse matrix
                   Sparse matrix in relative basis.
        """
        state_list = list(state_list)

        # Determine which states to iterate over
        if filter_nz:
            if not off_diag:
                # Filter to diag states present in dict
                iter_states = [
                    s for s in state_list if (s, s) in dict_l2_nnz
                ]
            else:
                # Collect all states involved in any nonzero entry
                nonzero_states = []
                for s1 in state_list:
                    for s2 in state_list:
                        if (s1, s2) in dict_l2_nnz:
                            nonzero_states.append(s1)
                            nonzero_states.append(s2)
                iter_states = sorted(set(nonzero_states))
        else:
            iter_states = state_list

        # Build sparse matrix from iter_states
        row = []
        col = []
        data = []
        if not off_diag:
            for (i, state) in enumerate(iter_states):
                try:
                    value = dict_l2_nnz[(state, state)]
                    row.append(i)
                    col.append(i)
                    data.append(value)
                except KeyError:
                    pass
        else:
            for (i, state1) in enumerate(iter_states):
                for (j, state2) in enumerate(iter_states):
                    try:
                        value = dict_l2_nnz[(state1, state2)]
                        row.append(i)
                        col.append(j)
                        data.append(value)
                    except KeyError:
                        pass
        return sp.sparse.coo_matrix(
            (data, (row, col)), shape=(len(iter_states), len(iter_states))
        )

    @staticmethod
    def _is_nearest_neighbor(H2_ham_sparse: sp.sparse.spmatrix) -> bool:
        """
        Check if a Hamiltonian has only nearest-neighbor coupling.

        Returns True if all non-zero elements satisfy |row - col| <= 1.

        Parameters
        ----------
        1. H2_ham_sparse: sparse matrix
                          System Hamiltonian in any scipy sparse format.

        Returns
        -------
        1. flag_nearest_neighbor_ham: bool
                                     True if nearest-neighbor, False otherwise.
        """
        H2_ham_coo = H2_ham_sparse.tocoo()
        # Eliminate explicit zeros so that a user-supplied sparse matrix
        # with stored zeros on far off-diagonals is not misclassified.
        H2_ham_coo.eliminate_zeros()
        return bool(np.all(np.abs(H2_ham_coo.row - H2_ham_coo.col) <= 1))
