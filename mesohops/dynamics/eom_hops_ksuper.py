import numpy as np
import scipy as sp
from scipy import sparse


__title__ = "HOPS Super Operator"
__author__ = "D. I. G. Bennett, B. Citty"
__version__ = "1.2"


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
    Ktmp = Pkeep @ K @ Pkeep.transpose()
    Ktmp.eliminate_zeros()
    return Ktmp

def _add_self_interactions(
    list_add_aux,
    system,
    K0_data,
    K0_row,
    K0_col,
):
    """
    This is a function that builds the connections from x --> (y) into
    the superoperators.

    PARAMETERS
    ----------
    1. list_add_aux : list
                      the list of new auxiliaries
    2. system : HopsSystem class
                an instance of HopsSystem
    3. K0_data : list
                 list of non-zero entries in time-invariant self-interaction super
                 operator
    4. K0_row : list
                list of row indices for non-zero entries in time-invariant self
                interaction superoperator
    5. K0_col : list
                list of column indices for non-zero entries in time-invariant self
                interaction superoperator

    RETURNS
    -------
    1. K0_data : list
                 list of updated non-zero entries in time-invariant self-interaction super
                 operator
    2. K0_row : list
                list of updated row indices for non-zero entries in time-invariant self
                interaction superoperator
    3. K0_col : list
                list of updated column indices for non-zero entries in time-invariant self
                interaction superoperator
    """
    # Prepare Constants
    # =================
    n_site = system.size
    n_lop = system.n_l2

    # Add terms for each new member of the hierarchy
    # ==============================================

    for aux_ref in list_add_aux:

        i_aux = aux_ref._index
        # sum_n sum_jn k_jn omega_jm phi_k
        # Note: Because the following line is taking aux_ref (with absolute index)
        #       and the 'W' parameter from system, this needs to come from the
        #       system.param['W'] rather than system.w.
        K0_data.append(-aux_ref.dot(system.param["W"]))
        K0_row.append(i_aux)
        K0_col.append(i_aux)


    return K0_data, K0_row, K0_col


def _add_crossterms(
    aux_from_list,
    aux_to_list,
    system,
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
    4. l_sparse : list
                  the list of sparse L operators
    5. Kp1_data : list
                  list of non-zero entries in time-invariant K+1 interaction super
                  operator
    6. Kp1_row : list
                 list of row indices for non-zero entries in time-invariant K+1
                 interaction superoperator
    7. Kp1_col : list
                 list of column indices for non-zero entries in time-invariant K+1
                 interaction superoperator
    8. Zp1_data : list
                  list of non-zero entries in time-varying K+1 interaction super
                  operator
    9. Zp1_row : list
                  list of row indices for non-zero entries in time-varying K+1
                  interaction superoperator
    10. Zp1_col : list
                  list of column indices for non-zero entries in time-varying K+1
                  interaction superoperator
    11. Km1_data : list
                   list of non-zero entries in time-invariant K-1 interaction super
                   operator
    12. Km1_row : list
                  list of row indices for non-zero entries in time-invariant K-1
                  interaction superoperator
    13. Km1_col : list
                  list of column indices for non-zero entries in time-invariant K-1
                  interaction superoperator

    RETURNS
    -------
    1. Kp1_data : list
                  list of updated non-zero entries in time-invariant K+1 interaction super
                  operator
    2. Kp1_row : list
                 list of updated row indices for non-zero entries in time-invariant K+1
                 interaction superoperator
    3. Kp1_col : list
                 list of updated column indices for non-zero entries in time-invariant K+1
                 interaction superoperator
    4. Zp1_data : list
                  list of updated non-zero entries in time-varying K+1 interaction super
                  operator
    5. Zp1_row : list
                 list of updated row indices for non-zero entries in time-varying K+1
                 interaction superoperator
    6. Zp1_col : list
                 list of updated column indices for non-zero entries in time-varying K+1
                 interaction superoperator
    7. Km1_data : list
                  list of updated non-zero entries in time-invariant K-1 interaction super
                  operator
    8. Km1_row : list
                 list of updated row indices for non-zero entries in time-invariant K-1
                 interaction superoperator
    9. Km1_col : list
                 list of updated column indices for non-zero entries in time-invariant K-1
                 interaction superoperator
    """
    # Prepare Constants
    # =================
    n_site = system.size
    list_index_from_list = [aux._index for aux in aux_from_list]

    for aux_ref in aux_to_list:

        i_aux = aux_ref._index

        # HIGHER TO LOWER AUX
        # sum_n sum_jn -(L*[n]-<L*[n]>_t) * phi_(k+e_jn)
        # integrand -= np.dot(np.conj(self.L[n]),phi[one_p])
        # integrand += np.conj(Ls[n])*phi[one_p] #empty_phi
        for (l_mod_abs, aux_p1) in aux_ref.dict_aux_p1.items():
            if l_mod_abs in system.list_absindex_mode:

                # The index of the corresponding L-Operator
                l_mod = list(system.list_absindex_mode).index(l_mod_abs)
                i_lop = system.list_index_L2_by_hmode[l_mod]

                if aux_p1._index in list_index_from_list:
                    j_aux = aux_p1._index

                    Kp1_data.extend(
                        -(system.g[l_mod] / system.w[l_mod]) * l_sparse[i_lop].data
                    )
                    Kp1_row.extend(i_aux * n_site + l_sparse[i_lop].row)
                    Kp1_col.extend(j_aux * n_site + l_sparse[i_lop].col)

                    Zp1_data[i_lop].append(system.g[l_mod] / system.w[l_mod])
                    Zp1_row[i_lop].append(i_aux)
                    Zp1_col[i_lop].append(j_aux)

        # LOWER TO HIGHER AUX
        # sum_n sum_jn k_jn alpha_nj phi_(k-e_jn)
        # integrand += order[iterator]*self.g[iterator]*np.dot(self.L[n],phi[one_min])
        for (l_mod_abs, aux_m1) in aux_ref.dict_aux_m1.items():
            if l_mod_abs in system.list_absindex_mode:
                l_mod = list(system.list_absindex_mode).index(l_mod_abs)
                i_lop = system.list_index_L2_by_hmode[l_mod]

                if aux_m1._index in list_index_from_list:
                    j_aux = aux_m1._index

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
    4. l_sparse : list
                  the list of sparse L operators
    5. Kp1_data : list
                  list of non-zero entries in time-invariant K+1 interaction super
                  operator
    6. Kp1_row : list
                 list of row indices for non-zero entries in time-invariant K+1
                 interaction superoperator
    7. Kp1_col : list
                 list of column indices for non-zero entries in time-invariant K+1
                 interaction superoperator
    8. Zp1_data : list
                  list of non-zero entries in time-varying K+1 interaction super
                  operator
    9. Zp1_row : list
                  list of row indices for non-zero entries in time-varying K+1
                  interaction superoperator
    10. Zp1_col : list
                  list of column indices for non-zero entries in time-varying K+1
                 interaction superoperator
    11. Km1_data : list
                   list of non-zero entries in time-invariant K-1 interaction super
                   operator
    12. Km1_row : list
                  list of row indices for non-zero entries in time-invariant K-1
                  interaction superoperator
    13. Km1_col : list
                  list of column indices for non-zero entries in time-invariant K-1
                  interaction superoperator

    RETURNS
    -------
    1. Kp1_data : list
                  list of updated non-zero entries in time-invariant K+1 interaction super
                  operator
    2. Kp1_row : list
                 list of updated row indices for non-zero entries in time-invariant K+1
                 interaction superoperator
    3. Kp1_col : list
                 list of updated column indices for non-zero entries in time-invariant K+1
                 interaction superoperator
    4. Zp1_data : list
                  list of updated non-zero entries in time-varying K+1 interaction super
                  operator
    5. Zp1_row : list
                 list of updated row indices for non-zero entries in time-varying K+1
                 interaction superoperator
    6. Zp1_col : list
                 list of updated column indices for non-zero entries in time-varying K+1
                 interaction superoperator
    7. Km1_data : list
                  list of updated non-zero entries in time-invariant K-1 interaction super
                  operator
    8. Km1_row : list
                 list of updated row indices for non-zero entries in time-invariant K-1
                 interaction superoperator
    9. Km1_col : list
                 list of updated column indices for non-zero entries in time-invariant K-1
                 interaction superoperator
    """

    # Prepare Constants
    # =================
    n_site = system.size
    list_index_aux_stable = [aux._index for aux in aux_stable]

    # Add terms for each new state
    # ============================
    # If a new state has been added to the state basis, then
    # we need to update the previous cross-terms for the 'stable'
    # members of the hierarchy.
    list_irel_new_state = [list(system.state_list).index(i) for i in list_add_state]

    if len(list_irel_new_state) > 0:
        # NOTE: THE LINE BELOW ASSUMES THAT IF AN L-OPERATOR IS CONNECTED TO A NEW
        #        NEW STATE THEN IT IS A NEW OPERATOR, BUT THIS IS A SPECIAL CASE.
        list_ilop_add = [
            ilop
            for ilop in range(len(system.list_absindex_L2))
            if system.state_list[system.list_state_indices_by_index_L2[ilop][0]]
            in list_add_state
        ]
        # NOTE: system.list_state_indices_by_index_L2[ilop][0] is only correct if each L-operator
        #       connects to a single state

        for aux_ref in aux_stable:

            i_aux = aux_ref._index

            for (l_mod_abs, aux_p1) in aux_ref.dict_aux_p1.items():
                if l_mod_abs in system.list_absindex_mode:

                    # The index of the corresponding L-Operator
                    l_mod = list(system.list_absindex_mode).index(l_mod_abs)
                    i_lop = system.list_index_L2_by_hmode[l_mod]

                    # HIGHER TO LOWER AUX
                    # sum_n sum_jn -(L*[n]-<L*[n]>_t) * phi_(k+e_jn)
                    # integrand -= np.dot(np.conj(self.L[n]),phi[one_p])
                    # integrand += np.conj(Ls[n])*phi[one_p] #empty_phi
                    if aux_p1._index in list_index_aux_stable:
                        j_aux = aux_p1._index

                        if i_lop in list_ilop_add:
                            # if this L-operator has not been present before
                            # then we should add all the associated terms
                            Kp1_data.extend(
                                -(system.g[l_mod] / system.w[l_mod]) * l_sparse[i_lop].data
                            )
                            Kp1_row.extend(i_aux * n_site + l_sparse[i_lop].row)
                            Kp1_col.extend(j_aux * n_site + l_sparse[i_lop].col)


                            Zp1_data[i_lop].append(
                                (system.g[l_mod] / system.w[l_mod])
                            )
                            Zp1_row[i_lop].append(i_aux)
                            Zp1_col[i_lop].append(j_aux)
                        else:
                            pass
                            # if the L-operator has been present before this
                            # then we need to add only the terms associated
                            # with the new states.

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

            for (l_mod_abs, aux_m1) in aux_ref.dict_aux_m1.items():
                if l_mod_abs in system.list_absindex_mode:
                    l_mod = list(system.list_absindex_mode).index(l_mod_abs)
                    i_lop = system.list_index_L2_by_hmode[l_mod]

                    if aux_m1._index in list_index_aux_stable:
                        j_aux = aux_m1._index

                        if i_lop in list_ilop_add:
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
            time-invariant self-interaction superoperator
    2. Kp1 : sparse matrix
             time-invariant K+1 interaction superoperator
    3. Zp1 : sparse matrix
             time-varying K+1 interaction superoperator
    4. Km1 : sparse matrix
             time-invariant K-1 interaction superoperator
    5. stable_aux : list
                    list of stable auxiliaries
    6. add_aux : list
                 list of new auxiliaries
    7. stable_state : list
                      list of stable states
    8. list_old_ilop : list
                       list of the absolute indices of L operators in the last basis
    9. system : HopsSystem class
                 an instance of HopsSystem
    10. hierarchy : HopsHierarchy class
                    an instance of HopsHierarchy
    11. n_old : int
                size of the full hierarchy in the previous basis
    12. perm_index : list
                     a list of column list and row list that define the permutation
                     matrix

    RETURNS
    -------
    1. K0 : sparse matrix
            updated time-invariant self-interaction superoperator
    2. Kp1 : sparse matrix
             updated time-invariant K+1 interaction superoperator
    3. Zp1 : sparse matrix
             updated time-varying K+1 interaction superoperator
    4. Km1 : sparse matrix
             updated time-invariant K-1 interaction superoperator
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
    Pmat2 = sp.sparse.coo_matrix(
        (np.ones(len(perm_index[3])), (perm_index[3],perm_index[2])),
        shape=(hierarchy.size,perm_index[4]),
        dtype=np.complex128,
    ).tocsc()
    # K-matrices
    K0 = _permute_aux_by_matrix(K0, Pmat2)
    Kp1 = _permute_aux_by_matrix(Kp1, Pmat)
    Km1 = _permute_aux_by_matrix(Km1, Pmat)
    # Z Matrices
    Zp1_new = [[] for i_lop in range(n_lop)]
    for i_lop in range(n_lop):
        if system.list_absindex_L2[i_lop] in list_old_ilop:
            Zp1_new[i_lop] = _permute_aux_by_matrix(
                Zp1[list(list_old_ilop).index(system.list_absindex_L2[i_lop])], Pmat2
            )
        else:
            Zp1_new[i_lop] = sparse.coo_matrix((hierarchy.size, hierarchy.size), dtype=np.complex128)

    Zp1 = Zp1_new

    # Add new_aux --> new_aux
    # =======================
    (
        K0_data,
        K0_row,
        K0_col,
    ) = _add_self_interactions(
        add_aux,
        system,
        K0_data=[],
        K0_row=[],
        K0_col=[],
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
            (K0_data, (K0_row, K0_col)), shape=(hierarchy.size, hierarchy.size), dtype=np.complex128
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
    Zp1 = [
        Zp1[i]
        + sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(hierarchy.size, hierarchy.size),
            dtype=np.complex128,
        ).tocsc()
        for i in range(n_lop)
    ]
    return [K0, Kp1, Zp1, Km1]


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
            updated time-invariant self-interaction superoperator
    2. Kp1 : sparse matrix
             updated time-invariant K+1 interaction superoperator
    3. Zp1 : sparse matrix
             updated time-varying K+1 interaction superoperator
    4. Km1 : sparse matrix
             updated time-invariant K-1 interaction superoperator
    """
    n_site = system.size
    n_lop = system.n_l2
    n_tot = n_site * hierarchy.size
    aux_list = hierarchy.auxiliary_list

    l_sparse = [system.list_L2_coo[i_lop] for i_lop in range(n_lop)]

    # Add new_aux --> new_aux
    # =======================
    (   
        K0_data,
        K0_row,
        K0_col,
    ) = _add_self_interactions(
        aux_list,
        system,
        K0_data=[],
        K0_row=[],
        K0_col=[],
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
        (K0_data, (K0_row, K0_col)), shape=(hierarchy.size, hierarchy.size), dtype=np.complex128
    ).tocsc()
    Kp1 = sparse.coo_matrix(
        (Kp1_data, (Kp1_row, Kp1_col)), shape=(n_tot, n_tot), dtype=np.complex128
    ).tocsc()
    Km1 = sparse.coo_matrix(
        (Km1_data, (Km1_row, Km1_col)), shape=(n_tot, n_tot), dtype=np.complex128
    ).tocsc()
    Zp1 = [
        sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(hierarchy.size, hierarchy.size),
            dtype=np.complex128,
        ).tocsc()
        for i in range(n_lop)
    ]

    return [K0, Kp1, Zp1, Km1]

