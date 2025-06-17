import time as timer

import numpy as np
from scipy import sparse

from mesohops.trajectory.hops_trajectory import HopsTrajectory
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.spectroscopy_analysis import _response_function_calc

__title__ = "hops_dyadic"
__author__ = "D. I. G. B. Raccah, T. Gera"
__version__ = "1.5"


class DyadicTrajectory(HopsTrajectory):
    """
    Acts as an interface for users to run a HOPS trajectory in a double Hilbert space.
    """

    __slots__ = (
        '__list_response_norm_sq',  # List of normalization factors
    )

    def __init__(self, system_param, eom_param=None, noise_param=None, hierarchy_param=None,
                 storage_param=None, integration_param=None):
        """
        Inputs
        ------
        1. system_param: dict
                         Dictionary of user-defined system parameters.
                         [see hops_system.py and hops_trajectory.py]

        2. eom_param: dict
                      Dictionary of user-defined eom parameters.
                      [see hops_eom.py and hops_trajectory.py]

        3. noise_param: dict
                        Dictionary of user-defined noise parameters.
                        [see hops_noise.py and hops_trajectory.py]

        4. hierarchy_param: dict
                            Dictionary of user-defined hierarchy parameters.
                            [see hops_hierarchy.py and hops_trajectory.py]

        5. storage_param: dict
                          Dictionary of user-defined storage parameters.
                          [see hops_storage.py and hops_trajectory.py]

        6. integration_param: dict
                              Dictionary of user-defined integration parameters.
                              [see integrator_rk.py and hops_trajectory.py]

        """

        if system_param is None:
            system_param = {}
        if eom_param is None:
            eom_param = {}
        if noise_param is None:
            noise_param = {}
        if hierarchy_param is None:
            hierarchy_param = {}
        if storage_param is None:
            storage_param = {}

        # Update the Hamiltonian
        system_param.update({'HAMILTONIAN':
                                 self._M2_dyad_conversion(system_param['HAMILTONIAN'])})

        # Update L-Operators
        for param in ['L_HIER', 'L_NOISE1', 'L_NOISE2','L_LT_CORR']:
            if param in system_param.keys():
                dyad_lop_list = []
                for i in range(len(system_param[param])):
                    dyad_lop_list.append(
                        self._M2_dyad_conversion(system_param[param][i]))
                system_param.update({param: dyad_lop_list})

        super().__init__(system_param, eom_param, noise_param,
                         hierarchy_param, storage_param, integration_param)

    def initialize(self, psi_ket, psi_bra, timer_checkpoint=None):
        """
        Prepares the initial dyadic wave function and passes it to
        HopsTrajectory.initialize.

        Parameters
        ----------
        1. psi_ket : np.array(complex)
                     Ket wave function at initial time.

        2. psi_bra : np.array(complex)
                     Bra wave function at initial time.

        3. timer_checkpoint : float
                              System time prior to initialization [units: s].
                              If None, uses current system time.

        Returns
        -------
        None

        """
        if timer_checkpoint is None:
            timer_checkpoint = timer.time()

        psi_0 = np.concatenate((psi_ket, psi_bra))
        self.__list_response_norm_sq = [np.linalg.norm(psi_0) ** 2]
        psi_0 = psi_0/np.sqrt(self.__list_response_norm_sq[0])
        super().initialize(psi_0, timer_checkpoint=timer_checkpoint)

    def _M2_dyad_conversion(self, M2_hilbert):
        """
        Converts a matrix of shape (N,N) to a matrix in dyadic space with shape (2N,2N).

        Parameters
        ----------
        1. M2_hilbert : np.array(complex)
                        Matrix of shape (N,N).

        Returns
        -------
        1. M2_dyad : np.array(complex)
                     Matrix of shape (2N,2N).

        """
        M2_dim = np.shape(M2_hilbert)[0]
        if (sparse.issparse(M2_hilbert)):
            M2_hilbert_coo=M2_hilbert.tocoo()
            col, row, data = M2_hilbert_coo.col, M2_hilbert_coo.row, M2_hilbert_coo.data
            col_dyad = np.concatenate((col, col + M2_dim))
            row_dyad = np.concatenate((row, row + M2_dim))
            data_dyad = np.concatenate((data, data))
            M2_dyad = sparse.coo_matrix((data_dyad, (row_dyad, col_dyad)),
                                           shape=(2*M2_dim, 2*M2_dim))

        else:
            M2_dyad = np.zeros((2 * M2_dim, 2 * M2_dim))
            M2_dyad[M2_dim:, M2_dim:] = M2_hilbert
            M2_dyad[:M2_dim, :M2_dim] = M2_hilbert

        return M2_dyad

    def _dyad_operator(self, op_hilbert, side):
        """
        Prepares a dyadic operator and passes it to HopsTrajectory._operator to be
        applied to either the ket or the bra side of the wave function. It transforms
        an operator for the ket side (K) in Hilbert space to [[K, 0], [0, I]] for the
        double Hilbert space, and similarly, it transforms an operator for the bra side
        (B) to [[I, 0], [0, B]]. Here, 'I' is the identity matrix with the same shape
        as the provided operator.

        Parameters
        ----------
        1. op_hilbert : np.array(complex)
                        Operator in Hilbert space.

        2. side : str
                  Operation side (Options: 'ket', 'bra').

        Returns
        -------
        None

        """
        op_dim = np.shape(op_hilbert)[0]
        # Denominator contributing to the Response Function norm
        response_norm_deno = np.linalg.norm(self.phi[:len(self.state_list)]) ** 2
        if (sparse.issparse(op_hilbert)):
            op_hilbert_coo=op_hilbert.tocoo()
            col, row, data = op_hilbert_coo.col, op_hilbert_coo.row, op_hilbert_coo.data
            # Creating a sparse dyadic operator for bra site with 1 in diagonals
            # for each ket site in state_list.
            if side == 'bra':
                ket_states = [state for state in self.state_list if state < op_dim]
                data_ket=[1]*len(ket_states)
                data_combined=np.concatenate((data_ket, data))
                col_combined=np.concatenate((ket_states, col+op_dim))
                row_combined=np.concatenate((ket_states, row+op_dim))
                op=sparse.coo_matrix((data_combined,(row_combined, col_combined)),
                                     shape=(2*op_dim,2*op_dim), dtype=np.complex128)
            elif side == 'ket':
                bra_states = [state for state in self.state_list if state >= op_dim]
                data_bra = [1] * len(bra_states)
                data_combined = np.concatenate((data,data_bra))
                col_combined = np.concatenate((col,bra_states))
                row_combined = np.concatenate((row,bra_states))
                op=sparse.coo_matrix((data_combined,(row_combined, col_combined)),
                                     shape=(2*op_dim,2*op_dim),dtype=np.complex128)
            else:
                raise UnsupportedRequest('sides other than "ket" or "bra"',
                                         '_dyad_operator')
        else:
            op = np.zeros((2 * op_dim, 2 * op_dim))
            op[self.state_list,self.state_list]=1
            if side == 'bra':
                op[op_dim:, op_dim:] = op_hilbert
            elif side == 'ket':
                op[:op_dim, :op_dim] = op_hilbert
            else:
                raise UnsupportedRequest('sides other than "ket" or "bra"',
                                         '_dyad_operator')
        super()._operator(op)
        # Contribution to the Response Function norm

        if self.basis.eom.normalized:
            self.__list_response_norm_sq.append(
                (np.linalg.norm(self.phi[:len(self.state_list)]) ** 2))
            self.phi = self.phi / np.sqrt(np.sum(np.abs(self.phi[: self.n_state]) ** 2))
        else:
            self.__list_response_norm_sq.append(
                (np.linalg.norm(self.phi[:len(self.state_list)]) ** 2) /
                response_norm_deno)

    def _response_function_comp(self, F_op, index_t):
        '''
        Calculates the normalized response function component using a dyadic operator
        at each time point beyond a user-defined index, managing the sparsity of the
        wave function passed to _response_function_calc from spectroscopy_analysis.py.

        Parameters
        ----------
        1. F_op : np.array(complex)
                  Dyadic operator to calculate the response function component.

        2. index_t : int
                     Time index after which the response function component is
                     calculated.

        Returns
        -------
        1. list_response_func_comp : list(complex)
                                     Calculated response function component for each
                                     time point beyond index_t.
        '''
        if self.basis.eom.param['ADAPTIVE']:
            traj = self.storage['psi_traj_sparse']
        else:
            traj = self.storage['psi_traj']
        return _response_function_calc(F_op, index_t, traj, self.list_response_norm_sq)

    @property
    def list_response_norm_sq(self):
        return self.__list_response_norm_sq

