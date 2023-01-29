import numpy as np
import scipy.sparse as sparse
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.physical_constants import hbar

__title__ = "Adaptive Basis Functions"
__author__ = "D. I. G. Bennett, B. Citty"
__version__ = "1.2"


def error_sflux_hier(Φ, list_s0, n_state, n_hier, H2_sparse_hamiltonian):
    """
    The error associated with losing all flux terms inside the kth auxiliary to
    states not contained in S_t. This code block corresponds to section 3.1.2 in the
    group adHOPS derivation document

    Parameters
    ----------
    1. Φ : np.array
           Current full hierarchy vector.

    2. list_s0 : list
                 List of the current states (absolute index).

    3. n_state : int
                 Number of states in the current state basis.

    4. n_hier : int
                Number of auxiliary wave functions in the current auxiliary basis.

    5. H2_sparse_hamiltonian : sparse array
                               self.system.param["SPARSE_HAMILTONIAN"].

    Returns
    -------
    1. E2_flux_state : array
                       Error introduced by losing flux within k from S_t to S_t^C
                       for each k in A_t.
    """
    # Construct the 2D phi and sparse Hamiltonian
    # -------------------------------------------
    list_s0 = np.array(list_s0)
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")

    # Find elements not in the current state basis
    # --------------------------------------------
    n_state_total = H2_sparse_hamiltonian.shape[0]
    list_sc = np.setdiff1d(np.arange(n_state_total), list_s0)

    # Construct Hamiltonian S_t^c<--S_t
    # ---------------------------------
    H2_sparse_hamiltonian = H2_sparse_hamiltonian[np.ix_(list_sc, list_s0)]

    D2_derivative_abs_sq = np.abs(H2_sparse_hamiltonian @ sparse.csc_matrix(C2_phi) / hbar).power(2)
    return np.array(np.sum(D2_derivative_abs_sq, axis=0))[0]


def error_deriv(dsystem_dt, Φ, z_step, n_state, n_hier, dt, list_index_aux_stable=None):
    """
    The error associated with losing all flux terms into the k auxiliary and n state,
    where k is in A_t and n is in S_t. This code block corresponds to section 3.1.1 in
    the group adHOPS derivation document when constructing the hierarchy basis and to
    section 4.1.1 when constructing the state basis.

    Parameters
    ----------
    1. dsystem_dt : function
                    Derivative function.

    2. Φ : np.array
           Current full hierarchy vector.

    3. z_step : list
                List of noise terms (compressed) for the next timestep.

    4. n_state : int
                 Number of states in the system.

    5. n_hier : int
                Number of auxiliary wave functions needed.

    6. dt : float
            Timestep of the propagation.

    7. list_index_aux_stable : list
                               List relative indices for the stable auxiliaries.

    Returns
    -------
    1. E2_del_phi : np.array
                    Error associated with losing flux to a component (either
                    hierarchy or state basis element) in H_t direct sum S_t.
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
    # Construct the derivative
    # ------------------------
    dΦ_dt = dsystem_dt(Φ_stab, z_step[2], z_step[0], z_step[1])[0]/hbar
    
    # Add the deletion flux
    # ---------------------
    dΦ_dt += Φ_stab/dt
    
    # Reshape error into [n_state, n_hier]
    # ------------------------------------
    dΦ_dt = dΦ_dt.reshape([n_state, n_hier], order="F")
    
    # Project into list_index_aux_stable
    # ----------------------------------
    if list_index_aux_stable is not None:
        dΦ_dt = dΦ_dt[:, list_index_aux_stable]

    return np.abs(dΦ_dt)**2


def error_flux_up(Φ, n_state, n_hier, n_hmodes, list_w, K2_aux_bymode,
                  M2_mode_from_state, type, F2_filter=None):
    """
    Returns the error associated with neglecting flux from members of A_t to
    auxiliaries in A_t^C that arise due to flux from lower auxiliaries to higher
    auxiliaries. This code block corresponds to section 3.1.3 in the group adHOPS
    derivation document when constructing the hierarchy basis and to section 4.1.3
    when constructing the state basis.

    Parameters
    ----------
    1. Φ : np.array
           Current state of the hierarchy.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions needed.

    4. n_hmodes : int
                  Number of hierarchy modes.

    5. list_w : list
                List of exponents for bath correlation functions.

    6. list_absindex_mode : list
                            List of the absolute indices  of the modes in current basis.

    7. K2_aux_by_mode : np.array
                         Mode values of each auxiliary in the space of [mode,k].

    8. M2_mode_from_state : np.array
                            The L_m[s,s] value in the space of [mode,s].

    9. static_filter_param : dict
                              a dictionary of parameters for the static filter.

    10. type : str
               Hierarchy or State basis ("H" or "S").

    11. F2_filter : np.array
                    Filters out unwanted auxiliary connections in the space of [mode,k].

    Returns
    -------
    1. E2_flux_up_error : np. array
                          Error induced by neglecting flux from A_t (or A_S)
                          to auxiliaries with lower summed index in A_t^C.
    """
    # Get flux factors
    # ----------------
    W2_bymode = np.tile(list_w,[1,n_hier]).reshape([n_hmodes, n_hier], order="F")

    # Reshape hierarchy (to matrix)
    # ------------------------------
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")
    if (type == "H"):
        # \sum_s |L_m[s,s] \psi_k[s]|^2 = \sum_s |M2[m,s] * \psi_k[s]|^2
        # \sum_s |M2[m,s]|^2 * |\psi_k[s]|^2
        P2_pop_modes = (np.abs(M2_mode_from_state).power(2) @ (np.abs(C2_phi) ** 2))
        E2_error = np.abs(W2_bymode) ** 2 * (1 + K2_aux_bymode) ** 2 * P2_pop_modes / hbar ** 2
        if F2_filter is not None:
            E2_error = F2_filter*E2_error

    elif (type == "S"):
        P2_pop_state = np.abs(C2_phi) ** 2
        E2_error = np.abs(np.abs(W2_bymode) * (1 + K2_aux_bymode)) ** 2
        if F2_filter is not None:
            E2_error *= F2_filter
        E2_error = np.transpose(M2_mode_from_state.power(2)) @ E2_error
        E2_error = P2_pop_state * E2_error / hbar ** 2

    else:
        E2_error = 0
        raise UnsupportedRequest(type, f"in error_flux_up received type {type}")

    return E2_error


def error_flux_down(Φ, n_state, n_hier, n_hmodes, list_g, list_w, M2_mode_from_state,
                    type, flag_gcorr=False, F2_filter=None):
    """
    Returns the error associated with neglecting flux from members of A_t to
    auxiliaries in A_t^C that arise due to flux from higher auxiliaries to lower
    auxiliaries. This code block corresponds to section 3.1.4 in the group adHOPS
    derivation document when constructing the hierarchy basis and to section 4.1.3
    when constructing the state basis.

    Parameters
    ----------
    1. Φ : np.array
           Current state of the hierarchy.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions needed.

    4. n_hmodes : int
                  Number of hierarchy modes.

    5. list_g : list
                List of prefactors for bath correlation function.

    6. list_w : list
                List of exponents for bath correlation functions.

    7. M2_mode_from_state : np.array
                            The L_m[s,s] value in the space of [mode,s].

    8. type : str
              Hierarchy or State basis ("H" or "S").

    9. flag_gcorr : bool
                    True if using linear absorption EOM False otherwise.

    10. F2_filter : np.array
                   Filters out unwanted auxiliary connections in the space of [mode,k]

    Returns
    -------
    1. E2_flux_down_error : np.array
                            Error induced by neglecting flux from A_t (or A_S)
                            to auxiliaries with higher summed index in A_t^C.
    """
    # Constants
    # ---------
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")
    E1_Lm = M2_mode_from_state @ (np.abs(C2_phi[:, 0]) ** 2)  # This is the <L_m> term assuming psi_0 normalized
    if flag_gcorr:
        E1_Lm /= 2

    # Get flux factors
    # ----------------
    G2_bymode = np.tile(list_g,[1,n_hier]).reshape([n_hmodes, n_hier], order="F")
    W2_bymode = np.tile(list_w,[1,n_hier]).reshape([n_hmodes, n_hier], order="F")
    if type == "H":
        # Hierarchy Type Downward Flux
        # ============================
        D2_mode_from_state = np.zeros([n_hmodes,n_state])
        D2_mode_from_state[:,:] = M2_mode_from_state.toarray() - E1_Lm.reshape([len(E1_Lm),1])
        E2_flux_down_error = (
                np.real(
                    (np.abs(G2_bymode / W2_bymode) ** 2)
                    * ((np.abs(D2_mode_from_state) ** 2) @ (np.abs(C2_phi) ** 2))
                )
                / hbar**2
        )
        if F2_filter is not None:
            E2_flux_down_error = F2_filter * E2_flux_down_error

    elif type == "S":
        # State Type Downward Flux
        # ========================
        D2_state_from_mode = np.zeros([n_state,n_hmodes])
        D2_state_from_mode[:,:] = np.transpose(M2_mode_from_state).toarray() - E1_Lm
        E2_flux_down_error = np.abs(G2_bymode / W2_bymode)**2
        if F2_filter is not None:
            E2_flux_down_error *= F2_filter
        E2_flux_down_error = (
                (np.abs(D2_state_from_mode) ** 2) @ (E2_flux_down_error)
                * (np.abs(C2_phi) ** 2)/hbar**2
        )
    else:
        E2_flux_down_error = 0
        raise UnsupportedRequest(type, f"in error_flux_down received type {type}")

    return E2_flux_down_error


def error_sflux_stable_state(Φ, n_state, n_hier, H2_sparse_hamiltonian,
                      list_index_aux_stable, list_states):
    """
    The error associated with losing all flux out of n in S_t. This flux always involves
    changing the state index and, as a result, can be rewritten in terms of the -iH
    component of the self-interaction. This corresponds to section 4.1.2 in group
    adhops document.

    Parameters
    ----------
    1. Φ : np.array
           Current full hierarchy vector.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions needed.

    4. H2_sparse_hamiltonian : sparse array
                               self.system.param["SPARSE_HAMILTONIAN"].

    5. list_index_aux_stable : list
                               List of relative indices for the stable auxiliaries.

    6. list_states : list
                     List of current states (absolute index).

    Returns
    -------
    1. E1_state_flux : array
                       Error associated with flux out of each state in S_t.
    """
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")[
        :, list_index_aux_stable
    ]
    H2_sparse_couplings = H2_sparse_hamiltonian - sparse.diags(
        H2_sparse_hamiltonian.diagonal(0),
        format="csc",
        shape=H2_sparse_hamiltonian.shape,
    )
    H2_sparse_hamiltonian = H2_sparse_couplings[:, list_states]

    V1_norm_squared = np.array(np.sum(np.abs(H2_sparse_hamiltonian).power(2), axis=0))[0]
    C1_norm_squared_by_state = np.sum(np.abs(C2_phi) ** 2, axis=1)
    return V1_norm_squared * C1_norm_squared_by_state / hbar**2


def error_sflux_boundary_state(Φ, list_s0, list_sc, n_state, n_hier, H2_sparse_hamiltonian, list_index_state_stable,
                             list_index_aux_stable):
    """
    Determines the error associated with neglecting flux into n not a member of S_t.
    This corresponds to section 4.2.1 in group adhops document.

    Parameters
    ----------
    1. Φ : np.array
           Current full hierarchy vector.

    2. list_s0 : list
                 Current stable states in absolute index.

    3. list_sc : list
                 States not in the current basis in absolute index.

    3. n_state : int
                 Number of states in the current state basis.

    4. n_hier : int
                Number of auxiliary wave functions in the current auxiliary basis.

    5. H2_sparse_hamiltonian : sparse array
                               self.system.param["SPARSE_HAMILTONIAN"].

    6. list_index_state_stable : list
                                 List of stable states in relative index.

    7. list_index_aux_stable : list
                               List of stable auxiliaries in relative index.

    Returns
    -------
    1. E1_sum_indices : list
                        Indices of states in S_t^c with nonzero flux into them.

    2. E1_sum_error : list
                      Error associated with flux into state in S_t^c.
    """
    if not len(list_index_state_stable) < H2_sparse_hamiltonian.shape[0]:
        return [], []
    else:
        # Remove aux components from H0\H1
        # -------------------------------------
        C2_phi = np.array(
            Φ
        ).reshape([n_state, n_hier], order="F")[
            np.ix_(list_index_state_stable, list_index_aux_stable)
        ]

        # Construct Hamiltonian
        # =====================
        # Construct Hamiltonian S_t^c<--S_s
        # ---------------------------------
        H2_sparse_couplings = H2_sparse_hamiltonian[np.ix_(list_sc, list_s0)]

        # Determine Boundary States
        # -------------------------
        C2_phi_deriv = np.abs(
            H2_sparse_couplings @ sparse.csc_matrix(C2_phi / hbar)
        ).power(2)
        E1_sum_error = np.sum(C2_phi_deriv,axis=1)
        return (np.array(list_sc)[E1_sum_error.nonzero()[0]],
                np.array(E1_sum_error[E1_sum_error.nonzero()])[0]
        )
                
