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
        '_list_modeidx_abs',              # Absolute mode indices
        '_list_prevl2idx_abs',            # Previous L2 indices

        # --- L2-indexing & mode-indexing lookups ---
        '_list_l2idx_abs',                # L2 absolute indices
        '_list_activemodeidx_rel',        # Active mode indices
        '_list_index_L2_by_hmode',        # L2 indices by mode
        '_list_activel2idx_rel',          # Active L2 indices
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
        '_list_l2_extd_csr',  # L2 CSR matrices truncated to state + boundary basis
        '_list_l2_nz_csr',   # L2 CSR matrices reduced to nonzero entries
        '_list_L2_sq_csr',   # L2 squared CSR matrices
        '_list_state_extd',         # Sorted list of states in the extended basis
        '_dict_stateidx_extd',      # Maps absolute state index to ext-basis position
    )

    def __init__(self, system, hierarchy):
        self.system = system
        self.hierarchy = hierarchy
        self._list_modeidx_abs = []
        self._list_l2idx_abs = []

    @property
    def list_index_L2_by_hmode(self):
        return self._list_index_L2_by_hmode

    @property
    def dict_relative_index_by_mode(self):
        return self.__dict_relindex_modes

    @property
    def n_hmodes(self):
        return np.size(self.list_modeidx_abs)

    @property
    def list_L2_coo(self):
        return self._list_L2_coo

    @property
    def list_l2_extd_csr(self):
        return self._list_l2_extd_csr

    @property
    def list_state_extd(self):
        return self._list_state_extd

    @property
    def dict_stateidx_extd(self):
        return self._dict_stateidx_extd

    @property
    def list_L2_csr(self):
        return self._list_L2_csr
        
    @property
    def list_l2_nz_csr(self):
        return self._list_l2_nz_csr

    @property
    def list_L2_sq_csr(self):
        return self._list_L2_sq_csr

    @property
    def n_l2(self):
        return self._n_l2

    @property
    def list_l2idx_abs(self):
        return self._list_l2idx_abs

    @property
    def list_prevl2idx_abs(self):
        return self._list_prevl2idx_abs

    @property
    def list_activel2idx_rel(self):
        return self._list_activel2idx_rel

    @property
    def list_activemodeidx_rel(self):
        return self._list_activemodeidx_rel

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
    def list_modeidx_abs(self):
        return self._list_modeidx_abs

    @property
    def list_L2_masks(self):
        return self._list_L2_masks

    @list_modeidx_abs.setter
    def list_modeidx_abs(self, list_modeidx_abs):
        # Prepare Indexing For Modes
        # --------------------------
        list_modeidx_abs.sort()
        self._list_prevl2idx_abs = self._list_l2idx_abs
        self._list_activemodeidx_rel = [
            list_modeidx_abs.index(mode_from_states)
            for mode_from_states in self.system.list_statemodeidx_abs
        ]
        self._list_modeidx_abs = np.array(list_modeidx_abs, dtype=int)

        # Prepare Indexing for L2
        # -----------------------
        self._list_l2idx_abs = sorted(set(
            [
                self.system.param["LIST_INDEX_L2_BY_HMODE"][hmode]
                for hmode in self._list_modeidx_abs
            ]
        ))
        self._list_index_L2_by_hmode = [
            list(self._list_l2idx_abs).index(
                self.system.param["LIST_INDEX_L2_BY_HMODE"][imod]
            )
            for imod in self._list_modeidx_abs
        ]
        self._list_activel2idx_rel = [
            self._list_l2idx_abs.index(absindex)
            for absindex in self.system.list_activel2idx_abs
        ]

        self._list_l2idx_abs = np.array(self._list_l2idx_abs, dtype=int)

        self.__dict_relindex_modes = {self.list_modeidx_abs[m]:m for m in range(
            len(self.list_modeidx_abs))}
        self._list_g = np.array([self.system.param["G"][m] for m in
                            self._list_modeidx_abs])
        self._list_w = np.array([self.system.param["W"][m] for m in
                            self._list_modeidx_abs])
        self._list_L2_coo = np.array(
            [
                self.system.reduce_sparse_matrix(
                    self.system.param["list_dict_L2_nnz"][k],
                    self.system.state_list,
                    self.system.list_off_diag[k],
                )
                for k in self._list_l2idx_abs
            ]
        )
        self._list_l2_nz_csr = (
            [
                self.system.reduce_sparse_matrix(
                    self.system.param["list_dict_L2_nnz"][k],
                    self.system.state_list,
                    self.system.list_off_diag[k],
                    filter_nz=True,
                ).tocsr()
                for k in self._list_l2idx_abs
            ]
        )
        self._list_state_extd = sorted(
            set(self.system.state_list) | set(self.system.list_destination_state) |
            set(self.system.list_bndstateidx_abs)
        )
        self._dict_stateidx_extd = {
            state: i for i, state in enumerate(self._list_state_extd)
        }
        self._list_l2_extd_csr = np.array(
            [
                self.system.reduce_sparse_matrix(
                    self.system.param["list_dict_L2_nnz"][k],
                    self._list_state_extd,
                    self.system.list_off_diag[k],
                ).tocsr()
                for k in self._list_l2idx_abs
            ]
        )

        self._list_L2_masks = [
            [
                sorted(set(self._list_L2_coo[i].row)),
                sorted(set(self._list_L2_coo[i].col)),
                np.ix_(
                    sorted(set(self._list_L2_coo[i].row)),
                    sorted(set(self._list_L2_coo[i].col)),
                ),
            ]
            for i in range(len(self._list_L2_coo))
        ]

        self._n_l2 = len(self._list_l2idx_abs)
        self._list_L2_csr = np.array([sparse.csr_array(L2_coo) for L2_coo in
                                      self._list_L2_coo])
        self._list_L2_sq_csr = np.array([L2@L2 for L2 in self._list_L2_csr])

        self._list_off_diag = self.system.list_off_diag[self._list_l2idx_abs]

        self._list_lt_corr_param = np.array([self.system.param["LIST_LT_PARAM"][m]
                                             for m in self._list_l2idx_abs])

    @property
    def list_off_diag_active_mask(self):
        return self._list_off_diag

    @property
    def list_offdiagl2idx_rel(self):
        return np.arange(len(self._list_l2idx_abs))[
            self.list_off_diag_active_mask]
