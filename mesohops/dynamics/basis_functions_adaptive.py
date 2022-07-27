import numpy as np
import scipy.sparse as sparse
from pyhops.util.exceptions import UnsupportedRequest
from pyhops.util.physical_constants import hbar

__title__ = "Adaptive Basis Functions"
__author__ = "D. I. G. Bennett, B. Citty"
__version__ = "1.2"

def error_sflux_hier(Φ, list_s0, n_state, n_hier, H2_sparse_hamiltonian):
    """
    The error associated with losing all flux terms inside the kth auxiliary to
    states not contained in S_t. This corresponds to equation 30 in arXiv:2008.06496

    PARAMETERS
    ----------
    1. Φ : np.array
           the current full hierarchy vector
    2. list_s0 : list
                 a list of the current states (absolute index)
    3. n_state : int
                 the number of states in the system
    4. n_hier : int
                the number of auxiliary wavefunctions needed
    5. H2_sparse_hamiltonian : sparse array
                               self.system.param["SPARSE_HAMILTONIAN"][:, list_s0]

    RETURNS
    -------
    1. E2_flux_state : array
                       the error introduced by losing flux within k from S_t to S_t^C
                       for each k in A_t

    """
    # Construct the 2D phi and sparse Hamiltonian
    # -------------------------------------------
    list_s0 = np.array(list_s0)
    H2_sparse_hamiltonian = H2_sparse_hamiltonian[:,list_s0]
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")

    # Remove the components of the Hamiltonian that map S0-->S0
    # ---------------------------------------------------------
    H2_sparse_subset = sparse.coo_matrix(
        H2_sparse_hamiltonian[np.ix_(list_s0, range(len(list_s0)))]
    )
    H2_removal = sparse.csc_matrix(
        (
            H2_sparse_subset.data,
            (
                list_s0[H2_sparse_subset.row],
                np.arange(len(list_s0))[H2_sparse_subset.col],
            ),
        ),
        shape=H2_sparse_hamiltonian.shape,
    )
    H2_sparse_hamiltonian = H2_sparse_hamiltonian - H2_removal
    D2_derivative_abs_sq = np.abs(H2_sparse_hamiltonian @ sparse.csc_matrix(C2_phi) / hbar).power(2)
    return np.sqrt(np.array(np.sum(D2_derivative_abs_sq, axis=0))[0])


def error_deriv(dsystem_dt, Φ, z_step, n_state, n_hier,
                list_index_aux_stable=None):
    """
    The error associated with losing all flux terms into the k auxiliary and n state,
    where k is in A_t and n is in S_t. This corresponds to equation 29 in arXiv:2008.06496

    PARAMETERS
    ----------
    1. dsystem_dt : function
                    the derivative function
    2. Φ : np.array
           The current full hierarchy vector
    3. z_step : list
                the list of noise terms (compressed) for the next timestep
    4. n_state : int
                 the number of states in the system
    5. n_hier : int
                the number of auxiliary wavefunctions needed
    6. list_index_aux_stable : list
                               a list relative indices for the stable auxiliaries

    RETURNS
    -------
    1. E2_del_phi : np.array
                    the error associated with losing flux to a component (either
                    hierarchy or state basis element) in H_t direct sum S_t
    """
    if list_index_aux_stable is not None:
        # Error arises for flux only out of the stable auxiliaries
        # --------------------------------------------------------
        Φ_stab = np.zeros(n_state * n_hier, dtype=np.complex128)
        Φ_stab_v = Φ_stab.view().reshape([n_state, n_hier], order="F")
        Φ_stab_v[:, list_index_aux_stable] = Φ.view().reshape(
            [n_state, n_hier], order="F"
        )[:, list_index_aux_stable]
    else:
        Φ_stab = Φ

    return np.abs(dsystem_dt(Φ_stab, z_step[2], z_step[0], z_step[1])[0]
                  / hbar).reshape([n_state, n_hier], order="F")


def error_flux_up(Φ, n_state, n_hier, n_hmodes, list_w, list_state_indices_by_hmode,
                  list_absindex_mode, auxiliary_list, k_maxhier, static_filter_param):
    """
    A function that returns the error associated with neglecting flux from members of
    A_t to auxiliaries in A_t^C that arise due to flux from lower auxiliaries to
    higher auxiliaries. This corresponds to equation 31 and 40 in arXiv:2008.06496

    .. math::
        \sum_{n \in \mathcal{S}_{t}} \\left \\vert F[\\vec{k}+\\vec{e}_n] \gamma_n (1+\\vec{k}[n]) \psi_{t,n}^{(\\vec{k})}   \\right \\vert^2

    .. math::
        \sum_{\\vec{k} \in \mathcal{H}_{t}}\\left \\vert F[\\vec{k}+\\vec{e}_n] \gamma_n (1+\\vec{k}[n]) \psi_{t,n}^{(\\vec{k})}   \\right \\vert^2

    PARAMETERS
    ----------
    1. Φ : np.array
           The current state of the hierarchy
    2. n_state : int
                 the number of states in the system
    3. n_hier : int
                the number of auxiliary wavefunctions needed
    4. n_hmodes : int
                  the number of hierarchy modes
    5. list_w : list
                list of exponents for bath correlation functions
    6. list_state_indices_by_hmode : list
                                     list of the absolute state indices ordered by the
                                     hierarchy modes
    7. list_absindex_mode : list
                            list of the absolute indices  of the modes in current basis
    8. auxiliary_list : list
                        list of the auxiliaries in the current basis
    9. k_maxhier : int
                   the maximum depth in the hierarchy that will be kept in the
                   calculation
    10. static_filter_param : dict
                              a dictionary of parameters for the static filter

    RETURNS
    -------
    1. E2_flux_up_error : np. array
                          the error induced by neglecting flux from A_t (or A_S)
                          to auxiliaries with lower summed index in A_t^C.
    """
    # Constants
    # ---------
    list_modes_from_site_index = np.repeat(np.arange(n_hier),len(list_state_indices_by_hmode)) * n_state \
                                                + list(list_state_indices_by_hmode[:,0]) * n_hier

    # Reshape hierarchy (to matrix)
    # ------------------------------
    P1_modes = np.asarray(Φ)[list_modes_from_site_index]
    P2_pop_modes = np.sqrt(np.abs(P1_modes) ** 2).reshape(
        [n_hmodes, n_hier], order="F"
    )

    # Get flux factors
    # ----------------
    W2_bymode = np.broadcast_to(np.abs(np.asarray(list_w)).reshape((n_hmodes,1)), (n_hmodes,n_hier))
    K2aux_bymode = np.zeros([n_hmodes, n_hier])
    for aux in auxiliary_list:
        array_index = np.array([list(list_absindex_mode).index(mode) for (mode,
                                                                          value) in
                                aux.tuple_aux_vec if mode in list_absindex_mode],
                               dtype=int)
        array_values = [value for (mode, value) in aux.tuple_aux_vec if mode in list_absindex_mode]
        K2aux_bymode[array_index, aux._index] = array_values


    # Filter out fluxes beyond the hierarchy depth and to aux in basis
    # ----------------------------------------------------------------
    basis_filter = np.zeros([n_hmodes, n_hier])
    for aux in auxiliary_list:
        if aux._sum < k_maxhier:
            array_index = np.array([list(list_absindex_mode).index(mode) for mode in
                           aux.dict_aux_p1.keys()if mode in list_absindex_mode], dtype=int)

            basis_filter[array_index, aux._index] = 1
        else:
            basis_filter[:, aux._index] = 1

    F2_filter = 1-basis_filter

    # Filter out Markovian Modes
    # --------------------------
    array2D_mark_param = np.array(
        [
            np.array(param)[list_absindex_mode]
            for (name, param) in static_filter_param
            if name == "Markovian"
        ]
    )

    if len(array2D_mark_param) > 0:
        F2_filter_markov = np.ones([n_hmodes, n_hier])
        array_mark_param = np.any(array2D_mark_param, axis=0)
        F2_filter_markov[array_mark_param, 1:] = 0
        F2_filter = F2_filter * F2_filter_markov

    return F2_filter * W2_bymode * (1 + K2aux_bymode) * P2_pop_modes / hbar


def error_flux_down(Φ, n_state, n_hier, n_hmodes, list_state_indices_by_hmode,
                    list_absindex_mode, auxiliary_list, list_g, list_w, type):
    """
    A function that returns the error associated with neglecting flux from members of
    A_t to auxiliaries in A_t^C that arise due to flux from higher auxiliaries to
    lower auxiliaries. This corresponds to equation 33 and 41 in arXiv:2008.06496

    .. math::
        \sum_{n \in \mathcal{S}_{t}} \\left \\vert F[\\vec{k}-\\vec{e}_n] \\frac{g_n}{\gamma_n} N^{(\\vec{k})}_t \psi_{t,n}^{(\\vec{k})}\\right \\vert^2

    .. math::
        \sum_{\\vec{k} \in \mathcal{H}_{t}} \\left \\vert F[\\vec{k}-\\vec{e}_n] \\frac{g_n}{\gamma_n} N^{(\\vec{k})}_t \psi_{t,n}^{(\\vec{k})}\\right \\vert^2


    PARAMETERS
    ----------
    1. Φ : np.array
           The current state of the hierarchy
    2. n_state : int
                 the number of states in the system
    3. n_hier : int
                the number of auxiliary wavefunctions needed
    4. n_hmodes : int
                  the number of hierarchy modes
    5. list_state_indices_by_hmode : list
                                     list of the absolute state indices ordered by the
                                     hierarchy modes
    6. list_absindex_mode : list
                            list of the absolute indices  of the modes in current basis
    7. auxiliary_list : list
                        list of auxiliaries in the current basis
    8. list_g : list
                list of prefactors for bath correlatioin function
    9. list_w : list
                list of exponents for bath correlation functions
    10. type : str
               'H' for hierarchy and 'S' for state

    RETURNS
    -------
    1. E2_flux_down_error : np.array
                            the error induced by neglecting flux from A_t (or A_S)
                            to auxiliaries with higher summed index in A_t^C.
                            
    """                       
    basis_filter = np.zeros([n_hmodes, n_hier])
    isthereflux = False
    for aux in auxiliary_list:
        
        if list(aux.dict_aux_m1) != list(aux.keys()):
            #Filter out basis to basis interactions
            isthereflux = True 
            array_index = np.array([list(list_absindex_mode).index(mode) for mode in
                       aux.dict_aux_m1.keys() if mode in list_absindex_mode],
                               dtype=int)
            basis_filter[array_index, aux._index] = 1
        
            #Also filter out connections from modes that are not in auxiliary
            #for example, all modes in the main auxiliary will be filtered here,
            #all but one mode in first-order auxiliaries, two modes in second-order auxiliaries, etc.
            
            array_index2 = np.array([list(list_absindex_mode).index(mode) for mode in aux.keys() if mode in list_absindex_mode],dtype=int)
            array_index2 = np.setdiff1d(np.arange(n_hmodes), array_index2)
          
            basis_filter[array_index2, aux._index] = 1
        else:
            basis_filter[:,aux._index] = 1
    F2_filter_aux = 1 - basis_filter
    
    if isthereflux:
    
        # Constants
        # ---------
        list_modes_from_site_index = np.repeat(np.arange(n_hier),len(list_state_indices_by_hmode)) * n_state \
                                                + list(list_state_indices_by_hmode[:,0]) * n_hier

        # Reshape hierarchy (to matrix)
        # ------------------------------
        P2_pop_site = (
            np.abs(np.asarray(Φ).reshape([n_state, n_hier], order="F")) ** 2
        )
        P1_aux_norm = np.sqrt(np.sum(P2_pop_site, axis=0))
        P2_modes_from0 = np.asarray(Φ)[
            np.tile(list_state_indices_by_hmode[:, 0], n_hier)
        ]
        P2_pop_modes_down_1 = (np.abs(P2_modes_from0) ** 2).reshape(
            [n_hmodes, n_hier], order="F"
        )
        P1_modes = np.asarray(Φ)[list_modes_from_site_index]
        P2_pop_modes = np.abs(P1_modes).reshape([n_hmodes, n_hier], order="F")

        # Get flux factors
        # ----------------
        G2_div_W2_bymode = np.broadcast_to(np.abs(np.asarray(list_g)/np.array(list_w)).reshape((n_hmodes,1)), (n_hmodes,n_hier))
        
        # Construct basis filter
    
        if type == "H":
            # Hierarchy Type Downward Flux
            # ============================
            E2_flux_down_error = (
                np.real(
                    F2_filter_aux
                    * G2_div_W2_bymode
                    * (P2_pop_modes_down_1 * P1_aux_norm[None, :] + P2_pop_modes)
                )
                / hbar
            )
        elif type == "S":
            # State Type Downward Flux
            # ========================
            # Construct <L_m> term
            # --------------------
            E2_lm = np.tile(
                np.sum(
                    F2_filter_aux * G2_div_W2_bymode * P2_pop_modes_down_1,
                    axis=0,
                ),
                [n_state, 1],
            )
            # Map Error to States
            # -------------------
            M2_state_from_mode = np.zeros([n_state, n_hmodes])
            M2_state_from_mode[
                list_state_indices_by_hmode[:, 0],
                np.arange(np.shape(list_state_indices_by_hmode)[0]),
            ] = 1
            E2_flux_down_error = (
                M2_state_from_mode
                @ np.real(F2_filter_aux * G2_div_W2_bymode * P2_pop_modes)
                / hbar
            )
            E2_flux_down_error += E2_lm * P2_pop_site / hbar
        else:
            E2_flux_down_error = 0
            raise UnsupportedRequest(type, "error_flux_down")
    
        return E2_flux_down_error
    elif type == "H":
        return np.zeros([n_hmodes,n_hier])
    elif type == "S":
        return np.zeros([n_state,n_hier])
    else:
        E2_flux_down_error = 0
        raise UnsupportedRequest(type, "error_flux_down")
        return E2_flux_down_error


def error_deletion(Φ, delta_t, n_state, n_hier):
    """
    The error associated with setting the corresponding component of Phi to 0,
    corresponding to equation 34 and 42 in arXiv:2008.06496

    PARAMETERS
    ----------
    1. Φ : np.array
           The current position of the full hierarchy vector
    2. delta_t : float
                 the timestep for the calculation
    3. n_state : int
                 the number of states in the system
    4. n_hier : int
                the size of the hierarchy at the current time

    RETURNS
    -------
    1. E2_site_aux : np.array
                     the error induced by removing components of Φ in A_t+S_t
    """

    # Error arising from removing the auxiliary directly
    # --------------------------------------------------
    E2_site_aux = np.abs(
        np.asarray(Φ).reshape([n_state, n_hier], order="F") / delta_t
    )

    return E2_site_aux


def error_sflux_state(Φ, n_state, n_hier, H2_sparse_hamiltonian,
                      list_index_aux_stable, list_states):
    """
    The error associated with losing all flux out of n in S_t. This flux always involves
    changing the state index and, as a result, can be rewritten in terms of the -iH
    component of the self-interaction. This corresponds to equation 38 and 39 in
    arXiv:2008.06496

    PARAMETERS
    ----------

    1. Φ : np.array
           the current full hierarchy vector
    2. n_state : int
                 the number of states in the system
    3. n_hier : int
                the number of auxiliary wavefunctions needed
    4. H2_sparse_hamiltonian : sparse array
                               self.system.param["SPARSE_HAMILTONIAN"]
    5. list_index_aux_stable : list
                               a list of relative indices for the stable auxiliaries
    6. list_states : list
                     the list of current states (absolute index)

    RETURNS
    -------
    1. E1_state_flux : array
                       the error associated with flux out of each state in S_t
    """
    list_s0 = np.array(list_states)
    H2_sparse_hamiltonian = H2_sparse_hamiltonian[list_s0,:]
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")[
        :, list_index_aux_stable
    ]

    # Remove the components of the Hamiltonian that map S0-->S0
    # ---------------------------------------------------------
    H2_sparse_subset = sparse.coo_matrix(
        H2_sparse_hamiltonian[np.ix_(range(len(list_s0)),list_s0)]
    )
    H2_removal = sparse.csc_matrix(
        (
            H2_sparse_subset.data,
            (
                np.arange(len(list_s0))[H2_sparse_subset.row],
                list_s0[H2_sparse_subset.col],
            ),
        ),
        shape=H2_sparse_hamiltonian.shape,
    )
    H2_sparse_hamiltonian = H2_sparse_hamiltonian - H2_removal
    V1_norm_squared = np.array(np.sum(np.abs(H2_sparse_hamiltonian).power(2), axis=1))[:,0]
    C1_norm_squared_by_state = np.sum(np.abs(C2_phi) ** 2, axis=1)
    return np.sqrt(V1_norm_squared * C1_norm_squared_by_state) / hbar

