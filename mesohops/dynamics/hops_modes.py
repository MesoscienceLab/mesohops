import numpy as np


class HopsModes:
    """
    This is a class that will be responsible for managing the modes basis
    in a adaptive HOPS calculation. At the moment, the logic here is fairly
    minimal, however, it will act as a compartment for the logic which will
    make it easier to thread information through the adaptive calculations.
    """

    def __init__(self, system, hierarchy):
        self.system = system
        self.hierarchy = hierarchy
        self.__list_absindex_mode = []
        self._list_absindex_L2 = []


    @property
    def list_index_L2_by_hmode(self):
        return self._list_index_L2_by_hmode

    @property
    def n_hmodes(self):
        return np.size(self.list_absindex_mode)

    @property
    def list_L2_coo(self):
        return self._list_L2_coo

    @property
    def list_L2_csc(self):
        return self._list_L2_csc

    @property
    def list_L2_diag(self):
        return self._list_L2_diag
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
    def g(self):
        return self._g

    @property
    def w(self):
        return self._w

    @property
    def lt_corr_param(self):
        return self.system._lt_corr_param

    @property
    def list_absindex_mode(self):
        return self.__list_absindex_mode

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

        self._g = np.array(self.system.param["G"])[self.__list_absindex_mode]
        self._w = np.array(self.system.param["W"])[self.__list_absindex_mode]
        
        self._list_L2_coo = np.array(
            [
                self.system.reduce_sparse_matrix(self.system.param["LIST_L2_COO"][k],
                                                 self.system.state_list)
                for k in self._list_absindex_L2
            ]
        )

        self._n_l2 = len(self._list_absindex_L2)
        self._list_L2_csc = [self._list_L2_coo[i].tocsr() for i in range(self._n_l2)]
        self._list_L2_diag = [self._list_L2_coo[i].diagonal() for i in range(self._n_l2)] #only works for diagonal L operators
        
        
