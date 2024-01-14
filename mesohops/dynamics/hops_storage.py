import copy
import numpy as np
from mesohops.dynamics.storage_functions import storage_default_func as default_func
from mesohops.util.exceptions import UnsupportedRequest

__title__ = "Storage Class"
__author__ = "D. I. G. Bennett, L. Varvelo"
__version__ = "1.4"


class HopsStorage:
    """
    This is an object that manages storing information for a HOPS trajectory.
    """
    def __init__(self, adaptive, storage_dic):
        """
        Inputs
        ------
        1. adaptive : bool
                      True indicates an adaptive calculation while False indicates
                      otherwise.

        2. storage_dic : dict
                         Dictionary of storage parameters.
        """
        self._adaptive = False
        self._n_dim = 0
        self.storage_dic = storage_dic
        self.cp_store_dic = copy.copy(storage_dic)
        self.dic_save = {}
        self.data = {}
        self.adaptive = adaptive

    def __repr__(self):
        key_dict = []
        for (key,value) in self.storage_dic.items():
            if value:
                key_dict.append(key)
        return 'Storage currently holds: {}'.format(key_dict)

    def __getitem__(self, item):
        if item in self.data.keys():
            if self._adaptive and item == 'psi_traj':
                psi_traj_full = np.zeros([len(self['t_axis']), self._n_dim],
                                         dtype=np.complex128)
                for (t_index, psi) in enumerate(self.data['psi_traj']):
                    psi_traj_full[
                        t_index, np.array(self.data['state_list'][t_index])] = psi
                return psi_traj_full
            else:
                return self.data[item]
        else:
            print('{} not a key of storage.data'.format(item))

    @property
    def adaptive(self):
        return self._adaptive

    @adaptive.setter
    def adaptive(self, new):
        self._adaptive = new
        self.storage_dic = copy.copy(self.cp_store_dic)
        self.dic_save = {}
        self.data = {}

        # Set default dictionary values
        # -----------------------------
        self.storage_dic.setdefault('phi_traj', False)
        self.storage_dic.setdefault('psi_traj', True)
        self.storage_dic.setdefault('t_axis', True)

        if self._adaptive:
            self.storage_dic.setdefault('aux_list', True)
            self.storage_dic.setdefault('state_list', True)
            self.storage_dic.setdefault('list_nhier', True)
            self.storage_dic.setdefault('list_nstate', True)


        for (key, value) in self.storage_dic.items():
            if value is not False:
                if isinstance(value, bool):
                    self.dic_save[key] = default_func[key]
                elif callable(value):
                    self.dic_save[key] = value
                else:
                    raise UnsupportedRequest('this value',
                                             'Storage.adaptive')
        for (key, value) in self.storage_dic.items():
            if value:
                self.data[key] = []

    def store_step(self, **kwargs):
        """
        Inserts data into the HopsStorage class at each time point of the simulation.

        Parameters
        ----------
        1. kwargs : any
                    The following parameters are the default key word arguments that
                    are currently being passed
                    1. phi_new : np.array(complex)
                                 Updated full hierarchy.
                    2. aux_list : list(AuxiliaryVector)
                                  List of the current auxiliaries in the hierarchy
                                  basis.
                    3. state_list : list(int)
                                    List of current states in the system basis.
                    4. t_new : float
                               New time point (t+tau).
                    5. z_mem_new : list(complex)
                                   List of memory values.

         Returns
         -------
         None
         """
        for (key,value) in self.dic_save.items():
            self.data[key].append(self.dic_save[key](**kwargs))

    @property
    def n_dim(self):
        return self._n_dim

    @n_dim.setter
    def n_dim(self, N_dim):
        self._n_dim = N_dim





