import numpy as np
import scipy.sparse as sparse
from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.physical_constants import hbar

__title__ = "Adaptive Basis Functions"
__author__ = "J. K. Lynd, D. I. G. B. Raccah, B. Citty"
__version__ = "1.4"

def error_deriv(dsystem_dt, Φ, z_step, n_state, n_hier, dt, list_index_aux_stable=None):
    """
    The error associated with losing all flux terms into the k auxiliary and s state,
    where k is in A_t and s is in S_t. This function corresponds to Eqs S36 and S47
    from the SI of "Characterizing the Role of Peierls Vibrations in Singlet Fission
    with the Adaptive Hierarchy of Pure States," available at
    https://arxiv.org/abs/2505.02292.

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

    7. list_index_aux_stable : list(int)
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
    dΦ_dt = dsystem_dt(Φ_stab, z_step[2], z_step[0], z_step[1])[0] / hbar

    # Add the deletion flux
    # ---------------------
    dΦ_dt += Φ_stab / dt

    # Reshape error into [n_state, n_hier]
    # ------------------------------------
    dΦ_dt = dΦ_dt.reshape([n_state, n_hier], order="F")

    # Project into list_index_aux_stable
    # ----------------------------------
    if list_index_aux_stable is not None:
        dΦ_dt = dΦ_dt[:, list_index_aux_stable]

    return np.abs(dΦ_dt) ** 2

def error_sflux_hier(Φ, list_s0, list_sc, n_state, n_hier, H2_sparse_hamiltonian,
                                 T2_phys=None, T2_hier=None):
    """
    The error associated with losing all flux out of the kth auxiliary to states not in
    S_t. Assumes that the first auxiliary wave function is the physical wave
    function. This function corresponds to Eq S52 from the SI of "Characterizing
    the Role of Peierls Vibrations in Singlet Fission with the Adaptive Hierarchy of
    Pure States," available at https://arxiv.org/abs/2505.02292.

    Parameters
    ----------
    1. Φ : np.array(complex)
           Current full hierarchy vector.

    2. list_s0 : list(int)
                 List of the current states (absolute index).

    3. n_state : int(int)
                 Number of states in the current state basis.

    4. n_hier : int
                Number of auxiliary wave functions in the current auxiliary basis.

    5. H2_sparse_hamiltonian : sparse array(complex)
                               self.system.param["SPARSE_HAMILTONIAN"], augmented by
                               the noise and noise memory drift.

    6. T2_phys : sparse array(complex)
                 The low-temperature correction operator applied to the physical
                 wave function.

    7. T2_hier : sparse array(complex)
                 The low-temperature correction operator applied to all auxiliary
                 wave functions, save for the physical.

    Returns
    -------
    1. E2_flux_state : np.array(float)
                       Error introduced by losing flux within k from S_t to S_t^C
                       for each k in A_t.
    """
    # Construct the 2D phi and sparse Hamiltonian
    # -------------------------------------------
    list_s0 = np.array(list_s0)
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")

    # Find elements not in the current state basis
    # --------------------------------------------
    if T2_phys is not None:
        C1_phi_phys = C2_phi[:, 0].reshape([1, n_state]).T
        C2_phi_aux = np.zeros_like(C2_phi)
        C2_phi_aux[:, 1:] = C2_phi[:, 1:]

        # Construct Hamiltonian S_t^c<--S_t
        # ---------------------------------
        H2_sparse_phys = (H2_sparse_hamiltonian+T2_phys)[np.ix_(list_sc, list_s0)]
        H2_sparse_hier = (H2_sparse_hamiltonian+T2_hier)[np.ix_(list_sc, list_s0)]

        # 1. E[k] is the squared flux error term associated with flux inside of
        # auxiliary k out of the state basis.
        # 2. H[d,s] is the Hamiltonian element going from state s in the basis to
        # state d outside of it. Thus, H has columns corresponding to states in the
        # basis, rows corresponding to states not in the basis.
        # 3. Z noise matrix of the same shape as H. T_k is low temperature
        # correction matrix of the same shape, which is different when k=0 for the
        # physical wave function.
        # 4. \psi_k is auxiliary wave function k, represented in the state basis
        # spanning all s.

        # E[k] = \sum_d {|\sum_s {(H+Z+T_k)[d,s] * \psi_k[s]}|^2}
        # E[k] = \sum_d {|((H+Z+T_k) @ \psi_k)[d]|^2}
        D1_deriv_abs_sq = np.array(
            np.sum(np.abs(H2_sparse_hier @ sparse.csc_array(C2_phi_aux) /
                           hbar).power(2), axis = 0)
        )

        D1_deriv_abs_sq[0] = np.sum(np.abs(H2_sparse_phys @ sparse.csc_array(
            C1_phi_phys) / hbar).power(2))

        return D1_deriv_abs_sq

    else:
        H2_sparse_hamiltonian = H2_sparse_hamiltonian[np.ix_(list_sc, list_s0)]

        D2_derivative_abs_sq = np.abs(H2_sparse_hamiltonian @ sparse.csc_array(
             C2_phi) / hbar).power(2)

        return np.array(np.sum(D2_derivative_abs_sq, axis=0))

def error_flux_up_hier_stable(Φ, n_state, n_hier, n_hmodes, list_w, K2_aux_bymode,
                              M2_diagonal, list_M2_by_dest_off, F2_filter_aux =
                              None):
    """
    Returns the error associated with neglecting flux from members of A_t to
    auxiliaries in A_t^C that arise due to flux from lower auxiliaries to higher
    auxiliaries, accounting for error arising from both diagonal and off-diagonal
    elements of L-operators. Specifically, this function corresponds to the flux up
    neglected when auxiliaries in the stable auxiliary basis A_t are truncated: Eq
    S37 from the SI of "Characterizing the Role of Peierls Vibrations in Singlet
    Fission with the Adaptive Hierarchy of Pure States," available at
    https://arxiv.org/abs/2505.02292.

     Parameters
    ----------
    1. Φ : np.array(complex)
           Current HOPS wave function in the adaptive basis.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions in the hierarchy.

    4. n_hmodes : int
                  Number of hierarchy modes.

    5. list_w : list(complex)
                List of exponential decay constants for each hierarchy mode [units:
                cm^-1].

    6. K2_aux_by_mode : np.array(int)
                        Mode values of each auxiliary in the space of [mode,k].

    7. M2_diagonal : sparse matrix(complex)
                     Diagonal values of the L-operators of each mode in the space of
                     [mode,s].

    8. list_M2_by_dest_off : list(sparse matrix(complex))
                             A list indexed by destination states d of the L_m[d,s]
                             value (when s != d) in the space of [mode,s].

    9. F2_filter_aux : np.array(int or bool)
                       Filters out unwanted auxiliary connections in the space of [mode,
                       k]. See between Eqs S37 and S38 of the SI.

    Returns
    -------
    1. E2_flux_up_error : np.array(float)
                          Error induced by neglecting flux from lower-lying to
                          higher-lying auxiliaries. Expressed in the space of
                          [mode,k].
    """
    # Get flux factors
    # ----------------
    list_w = np.abs(list_w)
    KW2_flux_multiplier = list_w.reshape([n_hmodes, 1], order="F") * (1 + K2_aux_bymode)

    # Reshape hierarchy (to matrix)
    # ------------------------------
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")

    # 1. E[m,k] is the squared flux error term associated with flux up out of
    # auxiliary k along mode m.
    # 2. w_m is the frequency of mode m, k_m is the depth of auxiliary k in that
    # mode. KW[m,k] is w_m * (k_m+1).
    # 3. L_m is the system-bath projection operator associated with mode m,
    # such that M_d[m,s] = L_m[d,s] for destination state d and source state s.
    # M2_diagonal and list_M2_by_dest_off[d] contain the components of M_d.
    # 4. \Phi is the full HOPS wave function such that \Phi[s,k] is the
    # amplitude at state s of auxiliary wave function k, \psi_k[s].

    # E[m,k] = \sum_d {|(w_m * (k_m+1) * L_m @ \psi_k)[d]|^2}
    # = |KW[m,k]|^2 * \sum_d {|\sum_s {L_m[d,s] * \psi_k[s]}|^2}
    # = |KW[m,k]|^2 * \sum_d {|\sum_s {M_d[m,s] * \Phi[s,k]}|^2}
    # = |KW[m,k]|^2 * \sum_d {|(M_d @ \Phi)[m,k]|^2}, such that
    # E = |KW|^2 * \sum_d {|M_d @ \Phi|^2}
    # Note that we actually calculate an upper bound via
    # E <= |KW|^2 * \sum_d {|M_d|^2} @ |\Phi|^2, reducing the number of
    # matrix multiplications but neglecting error cancellation.
    M2_super_mode_from_state = np.abs(M2_diagonal).power(2)

    M2_super_mode_from_state += np.sum([np.abs(M2_mode_from_state).power(2) for
                                        M2_mode_from_state in list_M2_by_dest_off],
                                       axis=0)
    E2_error = np.abs(KW2_flux_multiplier) ** 2 * (M2_super_mode_from_state @
                                                   np.abs(C2_phi) ** 2) / hbar ** 2

    if F2_filter_aux is not None:
        # Filtration is carried out by disallowing flux from auxiliary k up along
        # mode m when k + e_m is not a valid member of the basis, such that,
        # for filter matrix F[m,k] = \delta_(flux allowed)
        # E[m,k] --> F[m,k] * E[m,k]
        # E --> F * E
        # Generally, we do this filtration outside of this function to reduce peak
        # memory.
        print("Warning - filtration within the auxiliary basis case of the"
              "error_flux_up_hier_stable function is less efficient than "
              "filtering the output.")
        E2_error = F2_filter_aux * E2_error

    return E2_error

def error_flux_up_state_stable(Φ, n_state, n_hier, n_hmodes, list_w, K2_aux_bymode,
                               M2_diagonal, list_M2_by_dest_off, F2_filter_diag =
                               None, F2_filter_off = None):
    """
    Returns the error associated with neglecting flux from members of A_t to
    auxiliaries in A_t^C that arise due to flux from lower auxiliaries to higher
    auxiliaries, accounting for error arising from both diagonal and off-diagonal
    elements of L-operators. Specifically, this function corresponds to the flux up
    neglected when states in the stable state basis S_t are truncated: Eqs S48 and S50
    from the SI of "Characterizing the Role of Peierls Vibrations in Singlet Fission
    with the Adaptive Hierarchy of Pure States," available at
    https://arxiv.org/abs/2505.02292.

    Parameters
    ----------
    1. Φ : np.array(complex)
           Current HOPS wave function in the adaptive basis.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions in the hierarchy.

    4. n_hmodes : int
                  Number of hierarchy modes.

    5. list_w : list(complex)
                List of exponential decay constants for each hierarchy mode [units:
                cm^-1].

    6. K2_aux_by_mode : np.array(int)
                        Mode values of each auxiliary in the space of [mode,k].

    7. M2_diagonal : sparse matrix(complex)
                     Diagonal values of the L-operators of each mode in the space of
                     [mode,s].

    8. list_M2_by_dest_off : list(sparse matrix(complex))
                             A list indexed by destination states d of the L_m[d,s]
                             value (when s != d) in the space of [mode,s].

    9. F2_filter_diag : np.array(int or bool)
                        Filters out unwanted auxiliary connections in the space of
                        [mode, k]: used for fluxes from diagonal elements of the
                        L-operators. See between Eqs S49 and S50 of the SI.

    10. F2_filter_off : np.array(int or bool)
                        Filters out unwanted auxiliary connections in the space of
                        [mode, k]: used for fluxes from off-diagonal elements of the
                        L-operators. See between Eqs S51 and S52 of the SI.

    Returns
    -------
    1. E2_flux_up_error : np.array(float)
                          Error induced by neglecting flux from lower-lying to
                          higher-lying auxiliaries. Expressed in the space of [s,k].
    """
    # Get flux factors
    # ----------------
    list_w = np.abs(list_w)
    KW2_flux_multiplier = list_w.reshape([n_hmodes, 1], order="F") * (1 + K2_aux_bymode)

    # Reshape hierarchy (to matrix)
    # ------------------------------
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")

    # 1. E[s,k] is the squared flux error term associated with flux up out of
    # auxiliary k from state s.
    # 2. w_m is the frequency of mode m, k_m is the depth of auxiliary k in that
    # mode. KW[m,k] is w_m * (k_m+1).
    # 3. L_m is the system-bath projection operator associated with mode m, such that
    # M_d[m,s] = L_m[d,s] for destination state d and source state s.
    # M2_diagonal and list_M2_by_dest_off[d] contain the components of M_d.
    # 4. M_d^T is the transpose of M_d. We use separate matrices for the diagonal and
    # off-diagonal portions of the L-operators: these fluxes are filtered
    # differently, as laid out in section S2.B.1 of the SI.
    # 5. \Phi is the full HOPS wave function such that \Phi[s,k] is \psi_k[s].
    # 6. F[m,k] is the filtration term that is 0 when k + e_m is not a valid member
    # of the basis and 1 otherwise. This filter may be different for the diagonal and
    # off-diagonal portion of the error.

    # E[s,k] = \sum_d {\sum_m {|[F[m,k] * w_m * (k_m+1) * L_m[d,s] *
    # \psi_k[s]|^2}}
    # = \sum_d {\sum_m {|M_d^T[s,m]|^2 * |F[m,k] * KW[m,k]|^2]}} * |\Phi[s,k]|^2
    # = \sum_d {\sum_m {|M_d^T[s,m]|^2 * |(F * KW)|[m,k]^2}} * |\Phi[s,k]|^2
    # = (\sum_d {|M_d^T|^2 @ |(F * KW)|^2})[s,k] * |\Phi[s,k]|^2
    # = (\sum_d {|M_d^T|^2} @ |(F * KW)|^2 * |\Phi|^2)[s,k]
    # E = \sum_d {|M_d^T|^2} @ |(F * KW)|^2 * |\Phi|^2

    # Diagonal
    # Here, we are considering only the diagonal fluxes where d is s
    if F2_filter_diag is None:
        # If F does not filter out any fluxes, it leaves KW unchanged.
        F2_filter_diag = 1
    KW2_flux_multiplier_diag = F2_filter_diag * KW2_flux_multiplier
    E2_flux_up = ((np.transpose(np.abs(M2_diagonal).power(2)) @
                              np.abs(KW2_flux_multiplier_diag) ** 2) *
                              (np.abs(C2_phi) ** 2) / (hbar** 2))

    # Off-diagonal
    # Here, we are considering only the off-diagonal fluxes where d is not s,
    # so D_d[m,s] should always be L_m[d,s] - <L_m>.
    if F2_filter_off is None:
        # If F does not filter out any fluxes, it leaves KW unchanged.
        F2_filter_off = 1
    KW2_flux_multiplier_off = F2_filter_off * KW2_flux_multiplier

    M2_mode_from_state_super_off = np.sum([np.abs(M2_mode_from_state).power(2) for
               M2_mode_from_state in list_M2_by_dest_off],
              axis=0)
    if np.sum(M2_mode_from_state_super_off) > 0:
        E2_flux_up += ((np.transpose(M2_mode_from_state_super_off) @
                                 np.abs(KW2_flux_multiplier_off) ** 2) * (
                                 np.abs(C2_phi) ** 2) / (hbar ** 2))

    return E2_flux_up

def error_flux_up_by_dest_state(Φ, n_state, n_hier, n_hmodes, list_w, K2_aux_bymode,
                                list_M2_by_dest_off, list_index_aux_stable = None,
                                F2_filter_off = None, list_state_stable = None):
    """
    Returns the error associated with neglecting flux from members of A_t to
    auxiliaries in A_t^C that arise due to flux from lower auxiliaries to higher
    auxiliaries, accounting for error arising from only off-diagonal elements of
    L-operators. Specifically, this function corresponds to the flux up neglected
    when states in the boundary state basis S_t^C are not added to the basis: Eq S55
    from the SI of "Characterizing the Role of Peierls Vibrations in Singlet Fission
    with the Adaptive Hierarchy of Pure States," available at
    https://arxiv.org/abs/2505.02292.

    Parameters
    ----------
    1. Φ : np.array(complex)
           Current HOPS wave function in the adaptive basis.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions in the hierarchy.

    4. n_hmodes : int
                  Number of hierarchy modes.

    5. list_w : list(complex)
                List of exponential decay constants for each hierarchy mode [units:
                cm^-1].

    6. K2_aux_by_mode : np.array(int)
                        Mode values of each auxiliary in the space of [mode,k].

    7. list_M2_by_dest_off : list(sparse matrix(complex))
                             A list indexed by destination states d of the L_m[d,s]
                             value (when s != d) in the space of [mode,s]. To reduce
                             computational expense, this should only contain M2
                             matrices corresponding to destination states not in S_t.

    8. list_index_aux_stable : list(int)
                               Relative indices of the stable auxiliaries that can
                               provide flux.

    9. F2_filter_off : np.array(int or bool)
                       Filters out unwanted auxiliary connections in the space of
                       [mode, k]: used for fluxes from off-diagonal elements of the
                       L-operators. See between Eqs S51 and S52 of the SI.

    10. list_state_stable : list(int)
                            List of the relative state indices that have not been
                            removed from the basis during the construction of the
                            stable state basis.

    Returns
    -------
    1. list_E_by_dest : np.array(float)
                        Total flux up to each destination state. This will be
                        correct for destination states not in the current state
                        basis, but will ignore all fluxes from the diagonal portions
                        of L-operators.
    """
    # Get flux factors
    # ----------------
    list_w = np.abs(list_w)
    KW2_flux_multiplier = list_w.reshape([n_hmodes,1], order="F") * (1 + K2_aux_bymode)

    # Reshape hierarchy (to matrix), Accounting for states that have been removed
    # from the basis during stable state basis construction.
    # ------------------------------
    if list_state_stable is None:
        C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")
    else:
        C2_phi = np.zeros([n_state, n_hier], dtype=np.complex128)
        C2_phi[list_state_stable] = np.asarray(Φ).reshape([n_state, n_hier],
                                                          order="F")[list_state_stable]

    # We need to explicitly filter out the auxiliaries that have been removed from
    # the basis during stable auxiliary basis construction before summing up squared
    # fluxes.
    if list_index_aux_stable is None:
        list_index_aux_stable = range(n_hier)

    # 1. E[d] is the squared flux error term associated with flux up into destination
    # state d.
    # 2. w_m is the frequency of mode m, k_m is the depth of auxiliary k in that
    # mode. KW[m,k] is w_m * (k_m+1).
    # 3. L_m is the system-bath projection operator associated with mode m, such that
    # M_d[m,s] = L_m[d,s] for destination state d and source state s.
    # list_M2_by_dest_off[d] = M_d.
    # 4. M_d^T is the transpose of M_d. We only consider the off-diagonal portions of
    # the L-operators, as laid out in section S2.B.2 of the SI.
    # 5. \Phi is the full HOPS wave function such that \Phi[s,k] is \psi_k[s].
    # 6. F[m,k] is the filtration term that is 0 when k + e_m is not a valid member
    # of the basis and 1 otherwise. This filter may be different for the diagonal and
    # off-diagonal portion of the error.
    # 7. \SUM represents a grand sum (sum over all elements in a matrix).

    # E[d] = \sum_k {\sum_m {|\sum_s {F[m,k] * w_m * (k_m+1) * L_m[d,s] *
    # \psi_k[s]}|^2}}
    # = \sum_k {\sum_m {|\sum_s {M_d^T[s,m] * F[m,k] * KW[m,k] *
    # \Phi[s,k]}|^2}}
    # = \sum_k {\sum_m {|\sum_s {M_d^T[s,m] * (F * KW)[m,k] *
    # \Phi[s,k]]}^2}}
    # <= \sum_k {\sum_m {\sum_s {|M_d^T[s,m]|^2 * |(F * KW)[m,k]|^2 *
    # |\Phi[s,k]|^2}}}
    # = \sum_k {\sum_s {\sum_m {|M_d^T[s,m]|^2 * |(F * KW)[m,k]|^2 *
    # |\Phi[s,k]|^2}}}
    # = \sum_k {\sum_s {(|M_d^T|^2 @ |(F * KW)|^2)[s,k] * |\Phi[s,k]|^2}}
    # = \sum_k {\sum_s {((|M_d^T|^2 @ |(F * KW)|^2) * |\Phi|^2)[s,k]}}
    # E[d] <= \SUM {(|M_d^T|^2 @ |(F * KW)|^2) * |\Phi|^2}

    list_E_by_dest = []

    if F2_filter_off is None:
        # If F does not filter out any fluxes, it leaves KW unchanged.
        F2_filter_off = 1
    else:
        # Cut out unstable auxiliaries to reduce cost of algebra below.
        F2_filter_off = F2_filter_off[:, list_index_aux_stable]
    KW2_flux_multiplier_off = F2_filter_off * KW2_flux_multiplier[:,
                                            list_index_aux_stable]

    for M2_mode_from_state in list_M2_by_dest_off:
        E2_d = ((np.transpose(np.abs(M2_mode_from_state).power(2)) @
                 np.abs(KW2_flux_multiplier_off) ** 2) *
                (np.abs(C2_phi[:, list_index_aux_stable]) ** 2) / (hbar ** 2))
        list_E_by_dest.append(E2_d.sum())

    return np.array(list_E_by_dest)

def error_flux_down_hier_stable(Φ, n_state, n_hier, n_hmodes, list_g, list_w,
                                M2_diagonal, list_M2_by_dest_off,
                                X2_exp_lop_mode_state, F2_filter_aux = None):
    """
    Returns the error associated with neglecting flux from members of A_t to
    auxiliaries in A_t^C that arise due to flux from higher auxiliaries to lower
    auxiliaries, accounting for error arising from both diagonal and off-diagonal
    elements of L-operators. Specifically, this function corresponds to the flux down
    neglected when auxiliaries in the stable auxiliary basis A_t are truncated: Eq
    S39 from the SI of "Characterizing the Role of Peierls Vibrations in Singlet
    Fission with the Adaptive Hierarchy of Pure States," available at
    https://arxiv.org/abs/2505.02292.

     Parameters
    ----------
    1. Φ : np.array(complex)
           Current HOPS wave function in the adaptive basis.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions in the hierarchy.

    4. n_hmodes : int
                  Number of hierarchy modes.

    5. list_g : list(complex)
                List of prefactors for bath correlation function.

    6. list_w : list(complex)
                List of exponential decay constants for each hierarchy mode [units:
                cm^-1].

    7. M2_diagonal : sparse matrix(complex)
                     Diagonal values of the L-operators of each mode in the space of
                     [mode,s].

    8. list_M2_by_dest_off : list(sparse matrix(complex))
                             A list indexed by destination states d of the L_m[d,s]
                             value (when s != d) in the space of [mode,s].

    9. X2_exp_lop_mode_state : list(sparse matrix(complex))
                                 A list indexed by destination states d of the
                                 expectation values of each L-operator, multiplied by
                                 identity, with value <L_m> * I[d,s] reshaped into the
                                 space of [mode, s].

    10. F2_filter_aux : np.array(int or bool)
                        Filters out unwanted auxiliary connections in the space of
                        [mode, k]. See between Eqs S39 and S40 of the SI.

    Returns
    -------
    1. E2_flux_down_error : np.array(float)
                            Error induced by neglecting flux from higher-lying to
                            lower-lying auxiliaries. Expressed in the space of
                            [mode,k].
    """
    # Get flux factors
    # ----------------
    list_g_div_w_sq = np.square(np.abs(list_g / list_w))
    G1_gw_sq_bymode = list_g_div_w_sq.reshape([n_hmodes, 1], order="F")
    # Constants
    # ---------
    C2_phi_squared = np.abs(
        np.asarray(Φ).reshape([n_state, n_hier], order="F")) ** 2

    # 1. E[m,k] is the squared error flux down from auxiliary k along mode m.
    # 2. g_m and w_m are the constant prefactor and exponential decay constant of
    # mode m. G[m] = g_m/w_m.
    # 3. L_m is the L-operator of mode m, and I is the identity, such that D_d[m,s]
    # = (L_m - <L_m> * I)[d,s] corresponds to flux out of state s into state d for
    # mode m.
    # M2_diagonal, list_M2_by_dest_off[d], and X2_exp_lop_mode_state contain the
    # components of D_d.
    # 4. \Phi is the full HOPS wave function such that Phi[s,k] is \psi_k[s].

    # E[m,k] = \sum_d {|(g_m/w_m) * \sum_s {(L_m - <L_m> * I)[d,s] @ \psi_k[s]}|^2}
    # = \sum_d {|\sum_s {G[m] *  D_d[m,s] * \Phi[s,k]}|^2}
    # = |G[m] * \sum_d {(D_d @ \Phi)[m,k]|^2}
    # = (|G|^2 * \sum_d {|(D_d @ \Phi)|^2})[m,k]
    # E = |G|^2 * \sum_d {|(D_d @ \Phi)|^2}
    # Note that we actually calculate an upper bound via
    # E <= |G|^2 * \sum_d {|D_d|^2} @ |\Phi|^2, reducing the number of
    # matrix multiplications but neglecting error cancellation.
    D2_super_mode_from_state = np.abs(
        M2_diagonal - X2_exp_lop_mode_state).power(2)
    D2_super_mode_from_state += np.sum([np.abs(M2_mode_from_state).power(2) for
                                        M2_mode_from_state in list_M2_by_dest_off],
                                       axis=0)
    E2_flux_down_error = G1_gw_sq_bymode * (D2_super_mode_from_state @
                                            C2_phi_squared) / hbar ** 2

    if F2_filter_aux is not None:
        # Filtration is carried out by disallowing flux from auxiliary k down along
        # mode m when k - e_m is not a valid member of the basis, such that,
        # for filter matrix F[m,k] = delta_(flux allowed)
        # E[m,k] --> F[m,k] * E[m,k]
        # E --> F * E
        # Generally, we do this filtration outside of this function to reduce peak
        # memory.
        print("Warning - filtration within the auxiliary basis case of the"
              "error_flux_down_hier_stable function is less efficient than "
              "filtering the output.")
        E2_flux_down_error = F2_filter_aux * E2_flux_down_error

    return E2_flux_down_error

def error_flux_down_state_stable(Φ, n_state, n_hier, n_hmodes, list_g, list_w,
                                 M2_diagonal, list_M2_by_dest_off,
                                 X2_exp_lop_mode_state, F2_filter_diag = None,
                                 F2_filter_off = None):
    """
    Returns the error associated with neglecting flux from members of A_t to
    auxiliaries in A_t^C that arise due to flux from higher auxiliaries to lower
    auxiliaries, accounting for error arising from both diagonal and off-diagonal
    elements of L-operators. Specifically, this function corresponds to the flux down
    neglected when states in the stable state basis S_t are truncated: Eqs S49 and S51
    from the SI of "Characterizing the Role of Peierls Vibrations in Singlet Fission
    with the Adaptive Hierarchy of Pure States," available at
    https://arxiv.org/abs/2505.02292.

    Parameters
    ----------
    1. Φ : np.array(complex)
           Current HOPS wave function in the adaptive basis.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions in the hierarchy.

    4. n_hmodes : int
                  Number of hierarchy modes.

    5. list_g : list(complex)
                List of prefactors for bath correlation function.

    6. list_w : list(complex)
                List of exponential decay constants for each hierarchy mode [units:
                cm^-1].

    7. M2_diagonal : sparse matrix(complex)
                     Diagonal values of the L-operators of each mode in the space of
                     [mode,s].

    8. list_M2_by_dest_off : list(sparse matrix(complex))
                             A list indexed by destination states d of the L_m[d,s]
                             value (when s != d) in the space of [mode,s].

    9. X2_exp_lop_mode_state : list(sparse matrix(complex))
                                 A list indexed by destination states d of the
                                 expectation values of each L-operator, multiplied by
                                 identity, with value <L_m> * I[d,s] reshaped into the
                                 space of [mode, s].

    10. F2_filter_diag : np.array(int or bool)
                         Filters out unwanted auxiliary connections in the space of
                         [mode, k]: used for fluxes from diagonal elements of the
                         L-operators. See between Eqs S49 and S50 of the SI.

    11. F2_filter_off : np.array(int or bool)
                        Filters out unwanted auxiliary connections in the space of
                        [mode, k]: used for fluxes from off-diagonal elements of the
                        L-operators. See between Eqs S51 and S52 of the SI.

    Returns
    -------
    1. E2_flux_up_error : np.array(float)
                          Error induced by neglecting flux from higher-lying to
                          lower-lying auxiliaries. Expressed in the space of [s,k].
    """
    # Get flux factors
    # ----------------
    list_g_div_w_sq = np.square(np.abs(list_g / list_w))
    G1_gw_sq_bymode = list_g_div_w_sq.reshape([n_hmodes, 1], order="F")

    # Constants
    # ---------
    C2_phi_squared = np.abs(
        np.asarray(Φ).reshape([n_state, n_hier], order="F")) ** 2

    # 1. E[s,k] is the squared error flux down from auxiliary k at state s.
    # 2. g_m and w_m are the constant prefactor and exponential decay constant of
    # mode m. G[m,1] = g_m/w_m - note that G is a column vector, and that NumPy
    # broadcasting ensures that matrix A[m,n] * G[m,1] = (A * G)[m,n].
    # 3. L_m is the L-operator of mode m, and I is the identity, such that D_d[m,s]
    # = (L_m - <L_m> * I)[d,s] corresponds to flux out of state s into state d for
    # mode m.
    # M2_diagonal, list_M2_by_dest_off[d], and X2_exp_lop_mode_state contain the
    # components of D_d.
    # 4. D_d^T is the transpose of D_d. We use separate matrices for the diagonal and
    # off-diagonal portions of the L-operators: these fluxes are filtered
    # differently, as laid out in section S2.B.1 of the SI.
    # 6. \Phi is the full HOPS wave function such that Phi[s,k] is \psi_k[s].
    # 7. F[m,k] is the filtration term that is 0 when k - e_m is not a valid member
    # of the basis and 1 otherwise.

    # E[s,k] = \sum_m {F[m,k]  * \sum_d {|(g_m/w_m) * ((L_m - <L_m> @ I)[d,s] *
    # \psi_k[s])|^2}}
    # = \sum_m {|F[m,k] * (g_m/w_m)|^2 * \sum_d {|D_d[m,s] * \Phi[s,k]|^2}}
    # = \sum_m {|(F * G)[m,k]|^2 * \sum_d {|D_d[m,s] * \Phi[s,k]|^2}}
    # = \sum_d {\sum_m {|D_d_T[s,m]|^2 * |(F * G)[m,k]|^2 * |\Phi[s,k]|^2}}
    # = \sum_d {(|D_d_T|^2 @ |(F * G)|^2)[s,k]} * |\Phi[s,k]|^2
    # = \sum_d {(|D_d_T|^2} @ |(F * G)|^2)[s,k] * |\Phi[s,k]|^2
    # E = \sum_d {|D_d_T|^2} @ |(F * G)|^2 * |\Phi|^2

    E2_flux_down = 0
    # Off-diagonal
    # Here, we are considering only the off-diagonal fluxes where d is not s,
    # so D_d[m,s] should simply be L_m[d,s].
    if F2_filter_off is None:
        # If F does not filter out any fluxes, (F * G)[m,k] = G[m,1] in all
        # cases, and using G as-is reduces the complexity of the algebra via
        # broadcasting.
        F2_filter_off = 1
    G1_gw_sq_bymode_off = F2_filter_off * G1_gw_sq_bymode

    M2_mode_from_state_super_off = np.sum([np.abs(M2_mode_from_state).power(2) for
                                           M2_mode_from_state in list_M2_by_dest_off],
                                          axis=0)
    if np.sum(M2_mode_from_state_super_off) != 0:
        E2_flux_down = (M2_mode_from_state_super_off.T @
                        G1_gw_sq_bymode_off) * C2_phi_squared / hbar ** 2

    # Diagonal
    # Here, we are considering only the diagonal fluxes where d is s, so D_d[m,s]
    # should always be L_m[d,s] - <L_m>.
    D2_mode_from_state_diag = np.abs(M2_diagonal - X2_exp_lop_mode_state).power(2)
    if F2_filter_diag is None:
        # If F does not filter out any fluxes, (F * G)[m,k] = G[m,1] in all
        # cases, and using G as-is reduces the complexity of the algebra via
        # broadcasting.
        F2_filter_diag = 1
    G1_gw_sq_bymode_diag = F2_filter_diag * G1_gw_sq_bymode

    E2_flux_down += (D2_mode_from_state_diag.T @
                     G1_gw_sq_bymode_diag) * C2_phi_squared / hbar ** 2

    return E2_flux_down

def error_flux_down_by_dest_state(Φ, n_state, n_hier, n_hmodes, list_g, list_w,
                                  list_M2_by_dest_off, list_index_aux_stable = None,
                                  F2_filter_off = None, list_state_stable = None):
    """
    Returns the error associated with neglecting flux from members of A_t to
    auxiliaries in A_t^C that arise due to flux from higher auxiliaries to lower
    auxiliaries, accounting for error arising from only off-diagonal elements of
    L-operators. Specifically, this function corresponds to the flux down neglected
    when states in the boundary state basis S_t^C are not added to the basis: Eq S56
    from the SI of "Characterizing the Role of Peierls Vibrations in Singlet Fission
    with the Adaptive Hierarchy of Pure States," available at
    https://arxiv.org/abs/2505.02292.

    Parameters
    ----------
    1. Φ : np.array(complex)
           Current HOPS wave function in the adaptive basis.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions in the hierarchy.

    4. n_hmodes : int
                  Number of hierarchy modes.

    5. list_g : list(complex)
                List of prefactors for bath correlation function.

    6. list_w : list(complex)
                List of exponential decay constants for each hierarchy mode [units:
                cm^-1].

    7. list_M2_by_dest_off : list(sparse matrix(complex))
                             A list indexed by destination states d of the L_m[d,s]
                             value (when s != d) in the space of [mode,s]. To reduce
                             computational expense, this should only contain M2
                             matrices corresponding to destination states not in S_t.

    8. list_index_aux_stable : list(int)
                               Relative indices of the stable auxiliaries that can
                               provide flux.

    9. F2_filter_off : np.array(int or bool)
                       Filters out unwanted auxiliary connections in the space of
                       [mode, k]: used for fluxes from off-diagonal elements of the
                       L-operators. See between Eqs S51 and S52 of the SI.

    10. list_state_stable : list(int)
                            List of the relative state indices that have not been
                            removed from the basis during the construction of the
                            stable state basis.

    Returns
    -------
    1. list_E_by_dest : np.array(float)
                        Total flux down to each destination state. This will be
                        correct for destination states not in the current state
                        basis, but will ignore all fluxes from the diagonal portions
                        of L-operators.
    """
    # Get flux factors
    # ----------------
    list_g_div_w_sq = np.square(np.abs(list_g / list_w))
    G1_gw_sq_bymode = list_g_div_w_sq.reshape([n_hmodes, 1], order="F")

    # Account for states that have been removed from the basis during stable
    # state basis construction.
    if list_state_stable is None:
        C2_phi_squared = np.abs(
            np.asarray(Φ).reshape([n_state, n_hier], order="F")) ** 2
    else:
        C2_phi_squared = np.zeros([n_state, n_hier])
        C2_phi_squared[list_state_stable] = np.abs(np.asarray(Φ).reshape([
            n_state, n_hier], order="F")[list_state_stable]) ** 2

    # 1. E[d] is the squared flux error term associated with flux down into
    # destination state d.
    # 2. g_m and w_m are the constant prefactor and exponential decay constant of
    # mode m. G[m,1] = g_m/w_m - note that G is a column vector, and that NumPy
    # broadcasting ensures that matrix A[m,n] * G[m,1] = (A * G)[m,n].
    # 3. L_m is the L-operator of mode m, and I is the identity, such that D_d[m,s]
    # = (L_m - <L_m> * I)[d,s] corresponds to flux out of state s into state d for
    # mode m.
    # list_M2_by_dest_off[d] = D_d.
    # 4. D_d^T is the transpose of D_d. We only consider the off-diagonal
    # portions of the L-operators, as laid out in section S2.B.2 of the SI.
    # 5. \Phi is the full HOPS wave function such that Phi[s,k] is \psi_k[s].
    # 6. F[m,k] is the filtration term that is 0 when k - e_m is not a valid
    # member of the basis and 1 otherwise.
    # 7. \SUM represents a grand sum (sum over all elements in a matrix).

    # E[d] = \sum_k {\sum_m {|F[m,k]  * (g_m/w_m) * \sum_s { ((L_m - <L_m> @
    # I)[d,s] * \psi_k[s])}|^2}}
    # = \sum_k {\sum_m {|F[m,k] * (g_m/w_m) * \sum_s {|D_d[m,s] *
    # \Phi[s,k]}|^2}}
    # <= \sum_k {\sum_m {|(F * G)[m,k]|^2 * \sum_s {|D_d[m,s] * \Phi[s,k]|^2}}}
    # = \sum_k {\sum_s {\sum_m {|D_d_T[s,m]|^2 * |(F * G)[m,k]|^2} *
    # |\Phi[s,k]|^2}}
    # = \sum_k {\sum_s {\sum_m {|D_d_T[s,m]|^2 * |(F * G)[m,k]|^2} *
    # |\Phi[s,k]|^2}}
    # = \sum_k {\sum_s {(|D_d_T|^2 @ |(F * G)|^2)[s,k] * |\Phi[s,k]|^2}}
    # = \sum_k {\sum_s {((|D_d_T|^2 @ |(F * G)|^2) * |\Phi|^2)[s,k]}}
    # = \sum_k {\sum_s {(|D_d_T|^2 @ |(F * G)|^2)[s,k] * \Phi[s,k]|^2}}
    # = \sum_k {\sum_s {((|D_d_T|^2 @ |(F * G)|^2) * \Phi|^2)[s,k]}}
    # E[d] <= \SUM {(|D_d_T|^2 @ |(F * G)|^2) * \Phi|^2}

    # We need to explicitly filter out the auxiliaries that have been removed
    # from the basis during stable auxiliary basis construction before
    # summing up squared fluxes.
    if list_index_aux_stable is None:
        list_index_aux_stable = range(n_hier)

    if F2_filter_off is None:
        # If F does not filter out any fluxes, (F * G)[m,k] = G[m,1] in all
        # cases, and using G as-is reduces the complexity of the algebra via
        # broadcasting.
        F2_filter_off = 1
    else:
        # Cut out unstable auxiliaries to reduce cost of algebra below.
        F2_filter_off = F2_filter_off[:, list_index_aux_stable]
    G1_gw_sq_bymode_off = F2_filter_off * G1_gw_sq_bymode

    list_E_by_dest = []

    for M2_mode_from_state in list_M2_by_dest_off:
        E2_d = ((np.abs(M2_mode_from_state.T).power(2) @ G1_gw_sq_bymode_off)
                * C2_phi_squared[:, list_index_aux_stable] / hbar ** 2)
        list_E_by_dest.append(E2_d.sum())

    return np.array(list_E_by_dest)

def error_sflux_stable_state(Φ, n_state, n_hier, H2_sparse_hamiltonian,
                                         list_index_aux_stable, list_states,
                                         T2_phys=None, T2_hier=None):
    """
    The error associated with losing all flux out of state s in S_t that does not
    change the auxiliary index. Assumes that the first auxiliary wave function is
    the physical wave function. This function corresponds to Eq S52 from the SI of
    "Characterizing the Role of Peierls Vibrations in Singlet Fission with the
    Adaptive Hierarchy of Pure States," available at https://arxiv.org/abs/2505.02292.

    Parameters
    ----------
    1. Φ : np.array(complex)
           Current full hierarchy vector.

    2. n_state : int
                 Number of states in the system.

    3. n_hier : int
                Number of auxiliary wave functions needed.

    4. H2_sparse_hamiltonian : sparse array(complex)
                               self.system.param["SPARSE_HAMILTONIAN"], augmented by
                               the noise and noise memory drift.

    5. list_index_aux_stable : list(int)
                               List of relative indices for the stable auxiliaries.

    6. list_states : list(int)
                     List of current states (absolute index).

    7. T2_phys : sparse array(complex)
                 The low-temperature correction operator applied to the physical
                 wave function.

    8. T2_hier : sparse array(complex)
                 The low-temperature correction operator applied to all auxiliary
                 wave functions, save for the physical.

    Returns
    -------
    1. E1_state_flux : np.array(float)
                       Error associated with flux out of each state in S_t.
    """
    C2_phi = np.asarray(Φ).reshape([n_state, n_hier], order="F")[
        :, list_index_aux_stable
    ]

    if T2_phys is not None:
        C1_phi_phys = C2_phi[:, 0].reshape([n_state,1])
        C2_phi_aux = np.zeros_like(C2_phi)
        C2_phi_aux[:, 1:] = C2_phi[:, 1:]

        H2_sparse_couplings = sparse.csc_array(H2_sparse_hamiltonian) - sparse.diags(
            H2_sparse_hamiltonian.diagonal(0),
            format="csc",
            shape=H2_sparse_hamiltonian.shape,
        )
        T2_phys_couplings = T2_phys - sparse.diags(T2_phys.diagonal(0), format="csc",
                                         shape=T2_phys.shape)
        T2_hier_couplings = T2_hier - sparse.diags(T2_hier.diagonal(0), format="csc",
                                                   shape=T2_hier.shape)
        H2_sparse_phys = H2_sparse_couplings + T2_phys_couplings
        H2_sparse_phys = H2_sparse_phys[:, list_states]
        H2_sparse_hier = H2_sparse_couplings + T2_hier_couplings
        H2_sparse_hier = H2_sparse_hier[:, list_states]

        V1_norm_squared_phys = np.array(np.sum(np.abs(H2_sparse_phys).power(2),
                                               axis=0))
        V1_norm_squared_hier = np.array(np.sum(np.abs(H2_sparse_hier).power(2),
                                               axis=0))
        C1_norm_squared_hier = np.sum(np.abs(C2_phi_aux) ** 2, axis=1)
        C1_norm_squared_phys = np.sum(np.abs(C1_phi_phys) ** 2, axis=1)

        # 1. E[s] is the squared flux error term associated with flux from state s to
        # all other states.
        # 2. H[d,s] is the Hamiltonian element going from state s in the basis to
        # state d in the full state basis. Thus, H has columns corresponding to
        # states in the basis, rows corresponding to the full state basis.
        # 3. Z noise matrix of the same shape as H. T_k is low temperature
        # correction matrix of the same shape, which is different when k=0 for the
        # physical wave function.
        # 4. \psi_k is auxiliary wave function k, represented in the state basis
        # spanning all s.

        # E[s] = \sum_k {\sum_d {|(H+Z+T_k)[d,s] * \psi_k[s]|^2}}
        # E[s] = \sum_k {|\sum_d {(H+Z+T_k)[d,s]}|^2 * \psi_k[s]|^2}
        # E[s] = |\sum_d {(H+Z+T_k)[d,s]}|^2 * \sum_k!=0 {\psi_k[s]|^2} +
        # |\sum_d {(H+Z+T_0)[d,s]}|^2 * \psi_0[s]|^2
        return ((V1_norm_squared_phys * C1_norm_squared_phys)
                + (V1_norm_squared_hier * C1_norm_squared_hier)) / hbar ** 2

    else:
        H2_sparse_couplings = sparse.csc_array(H2_sparse_hamiltonian) - sparse.diags(
            H2_sparse_hamiltonian.diagonal(0),
            format="csc",
            shape=H2_sparse_hamiltonian.shape,
        )
        H2_sparse_hamiltonian = H2_sparse_couplings[:, list_states]

        V1_norm_squared = np.array(np.sum(np.abs(H2_sparse_hamiltonian).power(2), axis=0))
        C1_norm_squared_by_state = np.sum(np.abs(C2_phi) ** 2, axis=1)
        return V1_norm_squared * C1_norm_squared_by_state / hbar**2

def error_sflux_boundary_state(Φ, list_s0, list_sc, n_state, n_hier,
                                           H2_sparse_hamiltonian,
                                           list_index_state_stable,
                                           list_index_aux_stable, list_sc_dest,
                                           list_flux_updown, T2_phys=None,
                                           T2_hier=None):
    """
    Determines the error associated with neglecting flux into d, not a member of S_t.
    Includes the previously-calculated upper bound on fluxes up and down to destination
    states outside the current basis. Assumes that the first auxiliary wave function
    is the physical wave function.This function corresponds to Eqs S54 and S57 from the
    SI of "Characterizing the Role of Peierls Vibrations in Singlet Fission with the
    Adaptive Hierarchy of Pure States," available at https://arxiv.org/abs/2505.02292.

    Parameters
    ----------
    1. Φ : np.array(complex)
           Current full hierarchy vector.

    2. list_s0 : list(int)
                 Current stable states in absolute index.

    3. list_sc : list(int)
                 States not in the current basis in absolute index.

    3. n_state : int
                 Number of states in the current state basis.

    4. n_hier : int
                Number of auxiliary wave functions in the current auxiliary basis.

    5. H2_sparse_hamiltonian : sparse array(complex)
                               self.system.param["SPARSE_HAMILTONIAN"], augmented by
                               the noise and noise memory drift.

    6. list_index_state_stable : list(int)
                                 List of stable states in relative index.

    7. list_index_aux_stable : list(int)
                               List of stable auxiliaries in relative index.

    8. list_sc_dest : list(int)
                             List of states not in the current basis that receive
                             flux up or down in the index of list_sc.

    9. list_flux_updown : list(float)
                          The squared total flux up and down into each state in
                          list_sc_dest.

    10. T2_phys : sparse array(complex)
                  The low-temperature correction operator applied to the physical
                  wave function.
    11. T2_hier : sparse array(complex)
                  The low-temperature correction operator applied to all auxiliary
                  wave functions, save for the physical.

    Returns
    -------
    1. E1_sum_indices : list(int)
                        Indices of states in S_t^c with nonzero flux into them.

    2. E1_sum_error : list(float)
                      Error associated with flux into state in S_t^c.
    """
    if not len(list_index_state_stable) < H2_sparse_hamiltonian.shape[0]:
        return [], []
    else:
        # Remove aux components from H0\H1
        # -------------------------------------
        C2_phi = np.array(Φ).reshape([n_state, n_hier], order="F")[
            np.ix_(list_index_state_stable, list_index_aux_stable)
        ]

        if T2_phys is not None:
            C1_phi_phys = C2_phi[:,0].reshape([1,len(list_index_state_stable)]).T
            C2_phi_aux = C2_phi[:,1:]

            # Construct Hamiltonian
            # =====================
            # Construct Hamiltonian S_t^c<--S_s
            # ---------------------------------
            H2_sparse_phys = (H2_sparse_hamiltonian+T2_phys)[np.ix_(list_sc, list_s0)]

            # Determine Boundary States
            # -------------------------
            # Note: this takes the form
            # E[d] = \sum_k {|H[d,s] * C[s,k]|^2}
            # = sum_k {(|H@C|^2)[d,k]}
            # we could do this as
            # E[d] <= |H|^2 @ \sum_k {|C|^2}
            # to reduce calculation burden but make things faster. Time profile this to
            # check.

            # 1. E[d] is the squared flux error term associated with flux into state
            # d  not in the current basis (not counting states just removed from the
            # stable state basis).
            # 2. H[d,s] is the Hamiltonian element going from state s in the stable
            # state basis to state d outside of it. Thus, H has columns corresponding
            # to states in the stable basis, rows corresponding to states not in the
            # basis.
            # 3. Z noise matrix of the same shape as H. T_k is low temperature
            # correction matrix of the same shape, which is different when k=0 for the
            # physical wave function.
            # 4. \psi_k is auxiliary wave function k, represented in the state basis
            # spanning all s.

            # E[d] = \sum_k {\sum_s {|(H+Z+T_k)[d,s] * \psi_k[s]}|^2}
            # E[d] = \sum_k {|((H+Z+T_k) @ \psi_k)[d]|^2}
            # E[d] = \sum_k!=0 {|((H+Z+T_k) @ \psi_k)[d]|^2} +
            # |((H+Z+T_0) @ \psi_0)[d]|^2
            C1_phi_deriv_phys = np.abs(
                H2_sparse_phys @ sparse.csc_array(C1_phi_phys / hbar)
            ).power(2)
            E1_sum_error = C1_phi_deriv_phys.toarray().flatten()
            if len(C2_phi_aux) > 0:
                H2_sparse_aux = (H2_sparse_hamiltonian + T2_hier)[np.ix_(list_sc,
                                                                               list_s0)]
                C2_phi_deriv = np.abs(
                    H2_sparse_aux @ sparse.csc_array(C2_phi_aux / hbar)
                ).power(2)
                E1_sum_error += np.array(np.sum(C2_phi_deriv, axis=1))


        else:
            H2_sparse_couplings = H2_sparse_hamiltonian[np.ix_(list_sc, list_s0)]
            C2_phi_deriv = np.abs(
                H2_sparse_couplings @ sparse.csc_array(C2_phi / hbar)
            ).power(2)
            E1_sum_error = np.array(np.sum(C2_phi_deriv,axis=1))

        # Add the error associated with fluxes up and down to the appropriate states
        # outside the basis.
        E1_sum_error[list_sc_dest] += list_flux_updown
        return (np.array(list_sc)[E1_sum_error.nonzero()[0]],
                np.array(E1_sum_error[E1_sum_error.nonzero()[0]])
        )
