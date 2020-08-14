import numpy as np
import scipy as sp
from scipy import sparse


__title__ = "Nonlinear Normalized Derivative Terms"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"


def _permute_aux_by_matrix(K, Pkeep):
    """
    This function updates a sparse matrix by removing all references
    to indices contained in remove_index and rotating the remaining
    indices from their old position to their new position.

    PARAMETERS
    ----------
    1. K : sparse matrix
           the matrix to permute
    2. Pkeep : sparse matrix
               the permutation matrix

    RETURNS
    -------
    1. Ktmp : sparse matrix
              the permuted sparse matrix
    """
    # Multiplication to remove indices
    Ktmp = Pkeep * K * Pkeep.transpose()
    Ktmp.eliminate_zeros()
    return Ktmp


def _add_self_interactions(
    list_add_aux,
    list_stable_aux,
    list_add_state,
    system,
    hierarchy,
    l_sparse,
    K0_data,
    K0_row,
    K0_col,
    Z0_data,
    Z0_row,
    Z0_col,
):
    """
    This is a function that builds the connections from x --> (y) into
    the super operators.

    PARAMETERS
    ----------
    1. list_add_aux : list
                      the list of new auxiliaries
    2. list_stable_aux : list
                         the list of stable auxiliaries
    3. list_add_state : list
                        the list new states
    4. system : HopsSystem class
                an instance of HopsSystem
    5. hierarchy : HopsHierarchy class
                   an instance of HopsHierarchy
    6. l_sparse : list
                  the list of sparse L operators
    7. K0_data : list
                 list of non-zero entries in time invariant self interaction super
                 operator
    8. K0_row : list
                list of row indices for non-zero entries in time invariant self
                interaction super operator
    9. K0_col : list
                list of column indices for non-zero entries in time invariant self
                interaction super operator
    10. Z0_data : list
                  list of non-zero entries in time varying self interaction super
                  operator
    11. Z0_row : list
                 ist of row indices for non-zero entries in time varying self
                 interaction super operator
    12. Z0_col : list
                 list of column indices for non-zero entries in time varying self
                 interaction super operator

    RETURNS
    -------
    1. K0_data : list
                 list of updated non-zero entries in time invariant self interaction super
                 operator
    2. K0_row : list
                list of updated row indices for non-zero entries in time invariant self
                interaction super operator
    3. K0_col : list
                list of updated column indices for non-zero entries in time invariant self
                interaction super operator
    4. Z0_data : list
                 list of updated non-zero entries in time varying self interaction super
                 operator
    5. Z0_row : list
                list of updated row indices for non-zero entries in time varying self
                interaction super operator
    6. Z0_col : list
                list of updated column indices for non-zero entries in time varying self
                interaction super operator
    """
    # Prepare Constants
    # =================
    n_site = system.size
    n_lop = system.n_l2

    # Add terms for each new member of the hierarchy
    # ==============================================
    h_sparse = sp.sparse.coo_matrix(-1j * system.hamiltonian)
    h_sparse.eliminate_zeros()

    for aux_ref in list_add_aux:

        i_aux = hierarchy._aux_index(aux_ref)

        # Self-interaction in the hierarchy
        # -iH*phi(k)
        K0_data.extend(h_sparse.data)
        K0_row.extend(i_aux * n_site + h_sparse.row)
        K0_col.extend(i_aux * n_site + h_sparse.col)

        # sum_n sum_jn k_jn omega_jm phi_k
        # Note: Because the following line is taking aux_ref (with absolute index)
        #       and the 'W' parameter from system, this needs to come from the
        #       system.param['W'] rather than system.w.
        # K0_data.extend(-np.dot(aux_ref.todense(), system.param["W"]) * np.ones(n_site))
        K0_data.extend(-aux_ref.dot(system.param["W"]) * np.ones(n_site))
        K0_row.extend(i_aux * n_site + np.arange(n_site))
        K0_col.extend(i_aux * n_site + np.arange(n_site))

        # sum_n z_n*np.dot(L_n,phi(k))
        for i_lop in range(n_lop):
            Z0_data[i_lop].extend(l_sparse[i_lop].data)
            Z0_row[i_lop].extend(i_aux * n_site + l_sparse[i_lop].row)
            Z0_col[i_lop].extend(i_aux * n_site + l_sparse[i_lop].col)

    # Add terms for each new state
    # ============================
    # If a new state has been added to the state basis, then
    # we need to update the previous self-interaction terms
    # for the 'stable' members of the hierarchy.
    irel_new_state = [list(system.state_list).index(i) for i in list_add_state]
    irel_stable_state = [
        i
        for i in np.arange(len(system.state_list))
        if not system.state_list[i] in list_add_state
    ]

    if len(irel_new_state) > 0:
        v_mask_tmp = [
            1 if i in irel_stable_state else 0
            for i in np.arange(len(system.state_list))
        ]
        h_stable_mask = np.outer(v_mask_tmp, v_mask_tmp) * np.array(system.hamiltonian)
        h_new_sparse = sp.sparse.coo_matrix(-1j * (system.hamiltonian - h_stable_mask))
        ilop_add = [
            ilop
            for ilop in range(len(system.list_absindex_L2))
            if system.state_list[system.list_state_indices_by_index_L2[ilop][0]]
            in list_add_state
        ]
        # NOTE: system.list_state_indices_by_index_L2[ilop][0] is only correct if each L-operator
        #       connects to a single state

        for aux_ref in list_stable_aux:
            i_aux = hierarchy._aux_index(aux_ref)

            # -iH*phi(k)
            K0_data.extend(h_new_sparse.data)
            K0_row.extend(i_aux * n_site + h_new_sparse.row)
            K0_col.extend(i_aux * n_site + h_new_sparse.col)

            # sum_n sum_jn k_jn omega_jm phi_k
            # Note: Because the following line is taking aux_ref (with absolute index)
            #       and the 'W' parameter from system, this needs to come from the
            #       system.param['W'] rather than system.w.
            K0_data.extend(
                -aux_ref.dot(system.param["W"]) * np.ones(len(irel_new_state))
            )
            K0_row.extend(i_aux * n_site + np.array(irel_new_state))
            K0_col.extend(i_aux * n_site + np.array(irel_new_state))

            # sum_n z_n*np.dot(L_n,phi(k)) (only new lop operators)
            for i_lop in ilop_add:
                Z0_data[i_lop].extend(l_sparse[i_lop].data)
                Z0_row[i_lop].extend(i_aux * n_site + l_sparse[i_lop].row)
                Z0_col[i_lop].extend(i_aux * n_site + l_sparse[i_lop].col)

    return K0_data, K0_row, K0_col, Z0_data, Z0_row, Z0_col


def _add_crossterms(
    aux_from_list,
    aux_to_list,
    system,
    hierarchy,
    l_sparse,
    Kp1_data,
    Kp1_row,
    Kp1_col,
    Zp1_data,
    Zp1_row,
    Zp1_col,
    Km1_data,
    Km1_row,
    Km1_col,
):
    """
    PARAMETERS
    ----------
    1. aux_from_list : list

    2. aux_to_list : list

    3. system : HopsSystem class
                an instance of HopsSystem
    4. hierarchy : HopsHierarchy class
                   an instance of HopsHierachy
    5. l_sparse : list
                  the list of sparse L operators
    6. Kp1_data : list
                  list of non-zero entries in time invariant K+1 interaction super
                  operator
    7. Kp1_row : list
                 list of row indices for non-zero entries in time invariant K+1
                 interaction super operator
    8. Kp1_col : list
                 list of column indices for non-zero entries in time invariant K+1
                 interaction super operator
    9. Zp1_data : list
                  list of non-zero entries in time varying K+1 interaction super
                  operator
    10. Zp1_row : list
                  list of row indices for non-zero entries in time varying K+1
                  interaction super operator
    11. Zp1_col : list
                  list of column indices for non-zero entries in time varying K+1
                  interaction super operator
    12. Km1_data : list
                   list of non-zero entries in time invariant K-1 interaction super
                   operator
    13. Km1_row : list
                  list of row indices for non-zero entries in time invariant K-1
                  interaction super operator
    14. Km1_col : list
                  list of column indices for non-zero entries in time invariant K-1
                  interaction super operator

    RETURNS
    -------
    1. Kp1_data : list
                  list of updated non-zero entries in time invariant K+1 interaction super
                  operator
    2. Kp1_row : list
                 list of updated row indices for non-zero entries in time invariant K+1
                 interaction super operator
    3. Kp1_col : list
                 list of updated column indices for non-zero entries in time invariant K+1
                 interaction super operator
    4. Zp1_data : list
                  list of updated non-zero entries in time varying K+1 interaction super
                  operator
    5. Zp1_row : list
                 list of updated row indices for non-zero entries in time varying K+1
                 interaction super operator
    6. Zp1_col : list
                 list of updated column indices for non-zero entries in time varying K+1
                 interaction super operator
    7. Km1_data : list
                  list of updated non-zero entries in time invariant K-1 interaction super
                  operator
    8. Km1_row : list
                 list of updated row indices for non-zero entries in time invariant K-1
                 interaction super operator
    9. Km1_col : list
                 list of updated column indices for non-zero entries in time invariant K-1
                 interaction super operator
    """
    # Prepare Constants
    # =================
    n_site = system.size
    n_mode = system.n_hmodes

    list_hash_aux_from = [aux.hash for aux in aux_from_list]

    for aux_ref in aux_to_list:

        i_aux = hierarchy._aux_index(aux_ref)

        for l_mod in range(n_mode):

            # The index of the corresponding L-Operator
            i_lop = system.list_index_L2_by_hmode[l_mod]

            # HIGHER TO LOWER AUX
            # sum_n sum_jn -(L*[n]-<L*[n]>_t) * phi_(k+e_jn)
            # integrand -= np.dot(np.conj(self.L[n]),phi[one_p])
            # integrand += np.conj(Ls[n])*phi[one_p] #empty_phi
            hash_aux_p1 = aux_ref.hash_from_e_step(system.list_absindex_mode[l_mod], 1)
            if hash_aux_p1 in list_hash_aux_from:
                aux_p1 = aux_ref.e_step(system.list_absindex_mode[l_mod], 1)
                j_aux = hierarchy._aux_index(aux_p1)

                Kp1_data.extend(
                    -(system.g[l_mod] / system.w[l_mod]) * l_sparse[i_lop].data
                )
                Kp1_row.extend(i_aux * n_site + l_sparse[i_lop].row)
                Kp1_col.extend(j_aux * n_site + l_sparse[i_lop].col)

                Zp1_data[i_lop].extend(
                    (system.g[l_mod] / system.w[l_mod]) * np.ones(n_site)
                )
                Zp1_row[i_lop].extend(i_aux * n_site + np.arange(n_site))
                Zp1_col[i_lop].extend(j_aux * n_site + np.arange(n_site))

            # LOWER TO HIGHER AUX
            # sum_n sum_jn k_jn alpha_nj phi_(k-e_jn)
            # integrand += order[iterator]*self.g[iterator]*np.dot(self.L[n],phi[one_min])

            hash_aux_m1 = aux_ref.hash_from_e_step(system.list_absindex_mode[l_mod], -1)
            if hash_aux_m1 in list_hash_aux_from:
                aux_m1 = aux_ref.e_step(system.list_absindex_mode[l_mod], -1)
                j_aux = hierarchy._aux_index(aux_m1)

                Km1_data.extend(
                    (aux_m1[system.list_absindex_mode[l_mod]] + 1)
                    * system.w[l_mod]
                    * l_sparse[i_lop].data
                )
                Km1_row.extend(i_aux * n_site + l_sparse[i_lop].row)
                Km1_col.extend(j_aux * n_site + l_sparse[i_lop].col)
    return (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    )


def _add_crossterms_stable(
    aux_stable,
    list_add_state,
    system,
    hierarchy,
    l_sparse,
    Kp1_data,
    Kp1_row,
    Kp1_col,
    Zp1_data,
    Zp1_row,
    Zp1_col,
    Km1_data,
    Km1_row,
    Km1_col,
):
    """
    This is a function that adds in the cross-terms corresponding to new states
    that were not present in the previous steps.

    PARAMETERS
    ----------
    1. aux_stable : list
                    list of stable auxiliaries
    2. list_add_state : list
                        list of new states
    3. system : HopsSystem class
                an instance of HopsSystem
    4. hierarchy : HopsHierarchy class
                   an instance of HopsHierachy
    5. l_sparse : list
                  the list of sparse L operators
    6. Kp1_data : list
                  list of non-zero entries in time invariant K+1 interaction super
                  operator
    7. Kp1_row : list
                 list of row indices for non-zero entries in time invariant K+1
                 interaction super operator
    8. Kp1_col : list
                 list of column indices for non-zero entries in time invariant K+1
                 interaction super operator
    9. Zp1_data : list
                  list of non-zero entries in time varying K+1 interaction super
                  operator
    10. Zp1_row : list
                  list of row indices for non-zero entries in time varying K+1
                  interaction super operator
    11. Zp1_col : list
                  list of column indices for non-zero entries in time varying K+1
                 interaction super operator
    12. Km1_data : list
                   list of non-zero entries in time invariant K-1 interaction super
                   operator
    13. Km1_row : list
                  list of row indices for non-zero entries in time invariant K-1
                  interaction super operator
    14. Km1_col : list
                  list of column indices for non-zero entries in time invariant K-1
                  interaction super operator

    RETURNS
    -------
    1. Kp1_data : list
                  list of updated non-zero entries in time invariant K+1 interaction super
                  operator
    2. Kp1_row : list
                 list of updated row indices for non-zero entries in time invariant K+1
                 interaction super operator
    3. Kp1_col : list
                 list of updated column indices for non-zero entries in time invariant K+1
                 interaction super operator
    4. Zp1_data : list
                  list of updated non-zero entries in time varying K+1 interaction super
                  operator
    5. Zp1_row : list
                 list of updated row indices for non-zero entries in time varying K+1
                 interaction super operator
    6. Zp1_col : list
                 list of updated column indices for non-zero entries in time varying K+1
                 interaction super operator
    7. Km1_data : list
                  list of updated non-zero entries in time invariant K-1 interaction super
                  operator
    8. Km1_row : list
                 list of updated row indices for non-zero entries in time invariant K-1
                 interaction super operator
    9. Km1_col : list
                 list of updated column indices for non-zero entries in time invariant K-1
                 interaction super operator
    """

    # Prepare Constants
    # =================
    n_site = system.size
    n_mode = system.n_hmodes
    list_hash_aux_stable = [aux.hash for aux in aux_stable]

    # Add terms for each new state
    # ============================
    # If a new state has been added to the state basis, then
    # we need to update the previous cross-terms for the 'stable'
    # members of the hierarchy.
    irel_new_state = [list(system.state_list).index(i) for i in list_add_state]
    # irel_stable_state = [
    #     i
    #     for i in np.arange(len(system.state_list))
    #     if not system.state_list[i] in list_add_state
    # ]
    # v_mask_tmp = [
    #     0 if i in irel_stable_state else 1 for i in np.arange(len(system.state_list))
    # ]
    # vmask_mat = np.outer(v_mask_tmp, v_mask_tmp)

    if len(irel_new_state) > 0:
        # NOTE: THE LINE BELOW ASSUMES THAT IF AN L-OPERATOR IS CONNECTED TO A NEW
        #        NEW STATE THEN IT IS A NEW OPERATOR, BUT THIS IS A SPECIAL CASE.
        ilop_add = [
            ilop
            for ilop in range(len(system.list_absindex_L2))
            if system.state_list[system.list_state_indices_by_index_L2[ilop][0]]
            in list_add_state
        ]
        # NOTE: system.list_state_indices_by_index_L2[ilop][0] is only correct if each L-operator
        #       connects to a single state

        for aux_ref in aux_stable:

            i_aux = hierarchy._aux_index(aux_ref)

            for l_mod in range(n_mode):

                # The index of the corresponding L-Operator
                i_lop = system.list_index_L2_by_hmode[l_mod]

                # HIGHER TO LOWER AUX
                # sum_n sum_jn -(L*[n]-<L*[n]>_t) * phi_(k+e_jn)
                # integrand -= np.dot(np.conj(self.L[n]),phi[one_p])
                # integrand += np.conj(Ls[n])*phi[one_p] #empty_phi

                hash_aux_p1 = aux_ref.hash_from_e_step(
                    system.list_absindex_mode[l_mod], 1
                )
                if hash_aux_p1 in list_hash_aux_stable:
                    aux_p1 = aux_ref.e_step(system.list_absindex_mode[l_mod], 1)
                    j_aux = hierarchy._aux_index(aux_p1)

                    if i_lop in ilop_add:
                        # if this L-operator has not been present before
                        # then we should add all the associated terms
                        Kp1_data.extend(
                            -(system.g[l_mod] / system.w[l_mod]) * l_sparse[i_lop].data
                        )
                        Kp1_row.extend(i_aux * n_site + l_sparse[i_lop].row)
                        Kp1_col.extend(j_aux * n_site + l_sparse[i_lop].col)

                        Zp1_data[i_lop].extend(
                            (system.g[l_mod] / system.w[l_mod]) * np.ones(n_site)
                        )
                        Zp1_row[i_lop].extend(i_aux * n_site + np.arange(n_site))
                        Zp1_col[i_lop].extend(j_aux * n_site + np.arange(n_site))
                    else:
                        # if the L-operator has been present before this
                        # then we need to add only the terms associated
                        # with the new states.
                        Zp1_data[i_lop].extend(
                            (system.g[l_mod] / system.w[l_mod])
                            * np.ones(len(irel_new_state))
                        )
                        Zp1_row[i_lop].extend(i_aux * n_site + np.array(irel_new_state))
                        Zp1_col[i_lop].extend(j_aux * n_site + np.array(irel_new_state))

                        # this only applies if L operators can interact with more than
                        # one state
                        # l_tmp = l_sparse[i_lop].multiply(vmask_mat)
                        # if len(l_tmp.nonzero()[0]) > 0:
                        #     l_tmp.eliminate_zeros()
                        #     Kp1_data.extend(
                        #         -(system.g[l_mod] / system.w[l_mod]) * l_tmp.data
                        #     )
                        #     Kp1_row.extend(i_aux * n_site + l_tmp.row)
                        #     Kp1_col.extend(j_aux * n_site + l_tmp.col)

                # LOWER TO HIGHER AUX
                # sum_n sum_jn k_jn alpha_nj phi_(k-e_jn)
                # integrand += order[iterator]*self.g[iterator]*np.dot(self.L[n],phi[one_min])

                hash_aux_m1 = aux_ref.hash_from_e_step(
                    system.list_absindex_mode[l_mod], -1
                )
                if hash_aux_m1 in list_hash_aux_stable:
                    aux_m1 = aux_ref.e_step(system.list_absindex_mode[l_mod], -1)
                    j_aux = hierarchy._aux_index(aux_m1)

                    if i_lop in ilop_add:
                        # if this L-operator has not been present before
                        # then we should add all the associated terms
                        Km1_data.extend(
                            (aux_m1[system.list_absindex_mode[l_mod]] + 1)
                            * system.w[l_mod]
                            * l_sparse[i_lop].data
                        )
                        Km1_row.extend(i_aux * n_site + l_sparse[i_lop].row)
                        Km1_col.extend(j_aux * n_site + l_sparse[i_lop].col)
                    else:
                        pass
                        # if thie L-operator has been present before this
                        # then we need to add only the terms associated
                        # with the new states.
                        # this only applies if L operators can interact with more than
                        # one state
                        # l_tmp = l_sparse[i_lop].multiply(vmask_mat)
                        # if len(l_tmp.nonzero()[0]) > 0:
                        #     l_tmp.eliminate_zeros()
                        #     Km1_data.extend(
                        #         (aux_m1[system.list_absindex_mode[l_mod]] + 1)
                        #         * system.w[l_mod]
                        #         * l_tmp.data
                        #     )
                        #     Km1_row.extend(i_aux * n_site + l_tmp.row)
                        #     Km1_col.extend(j_aux * n_site + l_tmp.col)
    return (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    )


def update_ksuper(
    K0,
    Z0,
    Kp1,
    Zp1,
    Km1,
    stable_aux,
    add_aux,
    stable_state,
    list_old_ilop,
    system,
    hierarchy,
    n_old,
    perm_index,
):
    """
    This function updates the EOM matrices to incorporate changes to the hierarchy.
    At the moment, this function is the key step to handling
    Dynamic Filtering of the hierarchy.

    PARAMETERS
    ----------
    1. K0 : sparse matrix
            time invariant self interaction super operator
    2. Z0 : sparse matrix
            time varying self interaction super operator
    3. Kp1 : sparse matrix
             time invariant K+1 interaction super operator
    4. Zp1 : sparse matrix
             time varying K+1 interaction super operator
    5. Km1 : sparse matrix
             time invariant K-1 interaction super operator
    6. stable_aux : list
                    list of stable auxiliaries
    7. add_aux : list
                 list of new auxiliaries
    8. stable_state : list
                      list of stable states
    9. list_old_ilop : list
                       list of the absolute indices of L operators in the last basis
    10. system : HopsSystem class
                 an instance of HopsSystem
    11. hierarchy : HopsHierarchy class
                    an instance of HopsHierarchy
    12. n_old : int
                size of the full hierarchy in the previous basis
    13. perm_index : list
                     a list of column list and row list that define the permutation
                     matrix

    RETURNS
    -------
    1. K0 : sparse matrix
            updated time invariant self interaction super operator
    2. Z0 : sparse matrix
            updated time varying self interaction super operator
    3. Kp1 : sparse matrix
             updated time invariant K+1 interaction super operator
    4. Zp1 : sparse matrix
             updated time varying K+1 interaction super operator
    5. Km1 : sparse matrix
             updated time invariant K-1 interaction super operator
    """
    # Prepare Constants
    # =================
    n_site = system.size
    n_lop = system.n_l2
    n_tot = n_site * hierarchy.size
    new_state = [i for i in system.state_list if i not in stable_state]

    l_sparse = [system.list_L2_coo[i_lop] for i_lop in range(n_lop)]

    # Permutation Matrix
    # ==================
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(perm_index[0])), (perm_index[0], perm_index[1])),
        shape=(n_tot, n_old),
        dtype=np.complex128,
    ).tocsc()

    # K-matrices
    K0 = _permute_aux_by_matrix(K0, Pmat)
    Kp1 = _permute_aux_by_matrix(Kp1, Pmat)
    Km1 = _permute_aux_by_matrix(Km1, Pmat)

    # Z Matrices
    Z0_new = [[] for i_lop in range(n_lop)]
    Zp1_new = [[] for i_lop in range(n_lop)]
    for i_lop in range(n_lop):
        if system.list_absindex_L2[i_lop] in list_old_ilop:
            Z0_new[i_lop] = _permute_aux_by_matrix(
                Z0[list(list_old_ilop).index(system.list_absindex_L2[i_lop])], Pmat
            )
            Zp1_new[i_lop] = _permute_aux_by_matrix(
                Zp1[list(list_old_ilop).index(system.list_absindex_L2[i_lop])], Pmat
            )
        else:
            Z0_new[i_lop] = sparse.coo_matrix((n_tot, n_tot), dtype=np.complex128)
            Zp1_new[i_lop] = sparse.coo_matrix((n_tot, n_tot), dtype=np.complex128)

    Z0 = Z0_new
    Zp1 = Zp1_new

    # Add new_aux --> new_aux
    # =======================
    K0_data, K0_row, K0_col, Z0_data, Z0_row, Z0_col = _add_self_interactions(
        add_aux,
        stable_aux,
        new_state,
        system,
        hierarchy,
        l_sparse,
        K0_data=[],
        K0_row=[],
        K0_col=[],
        Z0_data=[[] for i in range(n_lop)],
        Z0_row=[[] for i in range(n_lop)],
        Z0_col=[[] for i in range(n_lop)],
    )

    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    ) = _add_crossterms(
        add_aux,
        add_aux,
        system,
        hierarchy,
        l_sparse,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Zp1_data=[[] for i in range(n_lop)],
        Zp1_row=[[] for i in range(n_lop)],
        Zp1_col=[[] for i in range(n_lop)],
        Km1_data=[],
        Km1_row=[],
        Km1_col=[],
    )

    # Add new_aux --> list_stable_aux
    # ==========================
    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    ) = _add_crossterms(
        add_aux,
        stable_aux,
        system,
        hierarchy,
        l_sparse,
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    )

    # Add list_stable_aux --> new_aux
    # ==========================
    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    ) = _add_crossterms(
        stable_aux,
        add_aux,
        system,
        hierarchy,
        l_sparse,
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    )

    # Add list_stable_aux --> list_stable_aux
    # (new states only!)
    # =============================
    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    ) = _add_crossterms_stable(
        stable_aux,
        new_state,
        system,
        hierarchy,
        l_sparse,
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    )

    # Construct Sparse Matrices
    # =========================
    K0 = (
        K0
        + sparse.coo_matrix(
            (K0_data, (K0_row, K0_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsc()
    )
    Kp1 = (
        Kp1
        + sparse.coo_matrix(
            (Kp1_data, (Kp1_row, Kp1_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsc()
    )
    Km1 = (
        Km1
        + sparse.coo_matrix(
            (Km1_data, (Km1_row, Km1_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsc()
    )
    Z0 = [
        Z0[i]
        + sparse.coo_matrix(
            (Z0_data[i], (Z0_row[i], Z0_col[i])),
            shape=(n_tot, n_tot),
            dtype=np.complex128,
        ).tocsc()
        for i in range(n_lop)
    ]
    Zp1 = [
        Zp1[i]
        + sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(n_tot, n_tot),
            dtype=np.complex128,
        ).tocsc()
        for i in range(n_lop)
    ]

    return [K0, Z0, Kp1, Zp1, Km1]


def calculate_ksuper(system, hierarchy):
    """
    This calculates all the pieces we need to time-evolve the system
    EXCEPT for the normalization term.

    PARAMETERS
    ----------
    1. system : HopsSystem Class
                an instance of HopsSystem
    2. hierarchy : HopsHierarchy Class
                   an instance of HopsHierarchy

    RETURNS
    -------
    1. K0 : sparse matrix
            updated time invariant self interaction super operator
    2. Z0 : sparse matrix
            updated time varying self interaction super operator
    3. Kp1 : sparse matrix
             updated time invariant K+1 interaction super operator
    4. Zp1 : sparse matrix
             updated time varying K+1 interaction super operator
    5. Km1 : sparse matrix
             updated time invariant K-1 interaction super operator
    """
    n_site = system.size
    n_lop = system.n_l2
    n_tot = n_site * hierarchy.size
    aux_list = hierarchy.auxiliary_list

    l_sparse = [system.list_L2_coo[i_lop] for i_lop in range(n_lop)]

    # Add new_aux --> new_aux
    # =======================
    K0_data, K0_row, K0_col, Z0_data, Z0_row, Z0_col = _add_self_interactions(
        aux_list,
        [],
        [],
        system,
        hierarchy,
        l_sparse,
        K0_data=[],
        K0_row=[],
        K0_col=[],
        Z0_data=[[] for i in range(n_lop)],
        Z0_row=[[] for i in range(n_lop)],
        Z0_col=[[] for i in range(n_lop)],
    )

    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
        Km1_row,
        Km1_col,
    ) = _add_crossterms(
        aux_list,
        aux_list,
        system,
        hierarchy,
        l_sparse,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Zp1_data=[[] for i in range(n_lop)],
        Zp1_row=[[] for i in range(n_lop)],
        Zp1_col=[[] for i in range(n_lop)],
        Km1_data=[],
        Km1_row=[],
        Km1_col=[],
    )

    # Construct Sparse Matrices
    K0 = sparse.coo_matrix(
        (K0_data, (K0_row, K0_col)), shape=(n_tot, n_tot), dtype=np.complex128
    ).tocsc()
    Kp1 = sparse.coo_matrix(
        (Kp1_data, (Kp1_row, Kp1_col)), shape=(n_tot, n_tot), dtype=np.complex128
    ).tocsc()
    Km1 = sparse.coo_matrix(
        (Km1_data, (Km1_row, Km1_col)), shape=(n_tot, n_tot), dtype=np.complex128
    ).tocsc()
    Z0 = [
        sparse.coo_matrix(
            (Z0_data[i], (Z0_row[i], Z0_col[i])),
            shape=(n_tot, n_tot),
            dtype=np.complex128,
        ).tocsc()
        for i in range(n_lop)
    ]
    Zp1 = [
        sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(n_tot, n_tot),
            dtype=np.complex128,
        ).tocsc()
        for i in range(n_lop)
    ]

    return [K0, Z0, Kp1, Zp1, Km1]
