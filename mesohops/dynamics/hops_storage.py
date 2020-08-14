import numpy as np

__title__ = "Storage Class"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"


class TrajectoryStorage(object):
    """
    This is an object that manages storing information for a
    HOPS trajectory.
    """

    def __init__(self):
        self.store_aux = False
        self.adaptive = False
        self._aux_list = []
        self._phi = []
        self._t = []
        self._z_mem = []
        self._psi_traj = []
        self._phi_traj = []
        self._t_axis = []
        self._n_dim = 0

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        self._t = t

    @property
    def z_mem(self):
        return self._z_mem

    @z_mem.setter
    def z_mem(self, z_mem):
        self._z_mem = z_mem

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = phi

    @property
    def psi_traj(self):
        return self._psi_traj

    @psi_traj.setter
    def psi_traj(self, psi_traj_new):
        self._psi_traj.append(psi_traj_new)

    @property
    def phi_traj(self):
        return self._phi_traj

    @phi_traj.setter
    def phi_traj(self, phi_traj_new):
        if self.store_aux:
            self._phi_traj.append(phi_traj_new)

    @property
    def t_axis(self):
        return self._t_axis

    @t_axis.setter
    def t_axis(self, t_axis):
        self._t_axis.append(t_axis)

    @property
    def aux_list(self):
        return self._aux_list

    @aux_list.setter
    def aux_list(self, aux_list):
        if self.store_aux:
            self._aux_list.append(aux_list)

    @property
    def n_dim(self):
        return self._n_dim

    @n_dim.setter
    def n_dim(self, N_dim):
        self._n_dim = N_dim


class AdaptiveTrajectoryStorage(TrajectoryStorage):
    """
    This is an object that manages storing information for a
    HOPS trajectory.
    """

    def __init__(self):
        super().__init__()
        self.store_aux = False
        self.adaptive = True
        self._list_aux_pop = []
        self._list_aux_boundary = []
        self._list_aux_new = []
        self._state_list = []
        self._aux_list = []
        self._list_nhier = []
        self._list_nstate = []

    @property
    def psi_traj(self):
        return self._psi_traj

    @psi_traj.setter
    def psi_traj(self, psi_traj_new):
        psi_new = np.zeros(self.n_dim, dtype=np.complex128)
        psi_new[np.array(self.state_list[-1])] = psi_traj_new
        self._psi_traj.append(psi_new)

    @property
    def list_aux_pop(self):
        return self._list_aux_pop

    @property
    def list_aux_new(self):
        return self._list_aux_new

    @property
    def aux(self):
        return [self.list_aux_new[-1], self.list_aux_pop[-1], self.list_aux_boundary[-1]]

    @aux.setter
    def aux(self, aux):
        self._list_aux_new.append(aux[0])
        self._list_aux_pop.append(aux[1])
        self._list_aux_boundary.append(aux[2])
        self.list_nhier = len(aux[0])

    @property
    def list_aux_boundary(self):
        return self._list_aux_boundary

    @property
    def state_list(self):
        return self._state_list

    @state_list.setter
    def state_list(self, s_list):
        self._state_list.append(s_list)
        self._list_nstate.append(len(s_list))

    @property
    def list_nhier(self):
        return self._list_nhier

    @list_nhier.setter
    def list_nhier(self, N_hier):
        self._list_nhier.append(N_hier)

    @property
    def list_nstate(self):
        return self._list_nstate

    @list_nstate.setter
    def list_nstate(self, s_list):
        self._list_nstate.append(s_list)
