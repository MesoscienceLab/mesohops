import numpy as np
from scipy import sparse

class HopsModes:
    """
    Manages the mode basis in an adaptive HOPS calculation, facilitating communication
    between the state and auxiliary wave function bases.
    """

    __slots__ = (
        # --- Core basis components  ---
        'system',        # System parameters and operators (HopsSystem)
        'hierarchy',     # Hierarchy management (HopsHierarchy)

        # --- Current mode-basis indexing ---
        '__list_absindex_mode',           # Absolute mode indices
        '__previous_list_absindex_L2',    # Previous L2 indices

        # --- L2-indexing & mode-indexing lookups ---
        '_list_absindex_L2',              # L2 absolute indices
        '_list_index_mode_active',        # Active mode indices
        '_list_index_L2_by_hmode',        # L2 indices by mode
        '_list_index_L2_active',          # Active L2 indices
        '__dict_relindex_modes',          # Relative mode indices
        '_list_off_diag',                 # Indices of off-diagonal L2 operators

        # --- Mode parameters ---
        '_list_g',                      # Coupling strength
        '_list_w',                      # Frequency
        '_list_lt_corr_param',          # Coupling strength / frequency for LTC

        # --- Low-temperature correction operators & masks ---
        '_list_L2_coo',      # L2 coordinate matrices
        '_list_L2_masks',    # L2 masks
        '_n_l2',             # Number of L2 operators
        '_list_L2_csr',      # L2 CSR matrices
        '_list_L2_sq_csr',   # L2 squared CSR matrices
    )

    def __init__(self, system, hierarchy):
        self.system = system
        self.hierarchy = hierarchy
        self.__list_absindex_mode = []
        self._list_absindex_L2 = []


    @property
    def list_index_L2_by_hmode(self):
        return self._list_index_L2_by_hmode

    @property
    def dict_relative_index_by_mode(self):
        return self.__dict_relindex_modes

    @property
    def n_hmodes(self):
        return np.size(self.list_absindex_mode)

    @property
    def list_L2_coo(self):
        return self._list_L2_coo

    @property
    def list_L2_csr(self):
        return self._list_L2_csr

    @property
    def list_L2_sq_csr(self):
        return self._list_L2_sq_csr

    @property
    def n_l2(self):
        return self._n_l2

    @property
    def list_absindex_L2(self):
        return self._list_absindex_L2

    @property
    def previous_list_absindex_L2(self):
        return self.__previous_list_absindex_L2

    @property
    def list_index_L2_active(self):
        return self._list_index_L2_active

    @property
    def list_index_mode_active(self):
        return self._list_index_mode_active

    @property
    def list_g(self):
        return self._list_g

    @property
    def list_w(self):
        return self._list_w

    @property
    def list_lt_corr_param_mode_indexing(self):
        return self._list_lt_corr_param

    @property
    def list_absindex_mode(self):
        return self.__list_absindex_mode

    @property
    def list_L2_masks(self):
        return self._list_L2_masks

    @list_absindex_mode.setter
    def list_absindex_mode(self, list_absindex_mode):
        # Prepare Indexing For Modes
        # --------------------------
        list_absindex_mode.sort()
        self.__previous_list_absindex_L2 = self._list_absindex_L2
        self._list_index_mode_active = [list_absindex_mode.index(mode_from_states)
                                        for mode_from_states in self.system.list_absindex_state_modes]
        self.__list_absindex_mode = np.array(list_absindex_mode, dtype=int)

        # Prepare Indexing for L2
        # -----------------------
        self._list_absindex_L2 = list(set(
            [
                self.system.param["LIST_INDEX_L2_BY_HMODE"][hmode]
                for hmode in self.__list_absindex_mode
            ]
        ))
        self._list_absindex_L2.sort()

        self._list_index_L2_by_hmode = [
            list(self._list_absindex_L2).index(
                self.system.param["LIST_INDEX_L2_BY_HMODE"][imod]
            )
            for imod in self.__list_absindex_mode
        ]

        self._list_index_L2_active = [self._list_absindex_L2.index(absindex)
                                      for absindex in self.system.list_absindex_L2_active]

        self._list_absindex_L2 = np.array(self._list_absindex_L2, dtype=int)

        self.__dict_relindex_modes = {self.list_absindex_mode[m]:m for m in range(
            len(self.list_absindex_mode))}
        self._list_g = np.array([self.system.param["G"][m] for m in
                            self.__list_absindex_mode])
        self._list_w = np.array([self.system.param["W"][m] for m in
                            self.__list_absindex_mode])
        
        self._list_L2_coo = np.array(
            [
                self.system.reduce_sparse_matrix(self.system.param["LIST_L2_COO"][k],
                                                 self.system.state_list)
                for k in self._list_absindex_L2
            ]
        )

        self._list_L2_masks = [
            [list(set(self._list_L2_coo[i].row)),list(set(self._list_L2_coo[i].col)), np.ix_(list(set(self._list_L2_coo[i].row)),list(set(self._list_L2_coo[i].col)))]
            for i in range(len(self._list_L2_coo))
        ]

        self._n_l2 = len(self._list_absindex_L2)
        self._list_L2_csr = np.array([sparse.csr_array(L2_coo) for L2_coo in
                                      self._list_L2_coo])
        self._list_L2_sq_csr = np.array([L2@L2 for L2 in self._list_L2_csr])
 
        self._list_off_diag = self.system.list_off_diag[self._list_absindex_L2]
        
        self._list_lt_corr_param = np.array([self.system.param["LIST_LT_PARAM"][m]
                                             for m in self._list_absindex_L2])
        
    @property
    def list_off_diag_active_mask(self):
        return self._list_off_diag

    @property
    def list_rel_ind_off_diag_L2(self):
        return np.arange(len(self._list_absindex_L2))[
            self.list_off_diag_active_mask]
