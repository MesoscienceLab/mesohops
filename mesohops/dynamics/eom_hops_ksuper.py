import numpy as np
import scipy as sp
from scipy import sparse


__title__ = "HOPS Super Operator"
__author__ = "D. I. G. Bennett, B. Citty"
__version__ = "1.2"


def _permute_aux_by_matrix(K, Pkeep):
    """
    Updates a sparse matrix by removing all references to indices contained in
    remove_index and rotating the remaining indices from their old position to their
    new position.

    Parameters
    ----------
    1. K : sparse matrix
           Matrix to permute.

    2. Pkeep : sparse matrix
               Permutation matrix.

    Returns
    -------
    1. Ktmp : sparse matrix
              Permuted sparse matrix.
    """
    # Multiplication to remove indices
    Ktmp = Pkeep @ K @ Pkeep.transpose()
    Ktmp.eliminate_zeros()
    return Ktmp


def _add_self_interactions(system, hierarchy, K0_data, K0_row, K0_col):
    """
    Constructs the list of self-decay terms given by the dot product of the indexing
    vector and vectorized list of correlation function modes for incorporation into the
    time-evolution super-operator.

    Parameters
    ----------
    1. system : instance(HopsSystem)

    2. hierarchy : instance(HopsHierarchy)

    3. K0_data : list(complex)
                 List of non-zero entries in time-invariant self-interaction super
                 operator.

    4. K0_row : list(int)
                List of row indices for non-zero entries in time-invariant self
                interaction superoperator.

    5. K0_col : list(int)
                List of column indices for non-zero entries in time-invariant self
                interaction superoperator.

    Returns
    -------
    1. K0_data : list(complex)
                 list of updated non-zero entries in time-invariant self-interaction
                 super operator.

    2. K0_row : list(int)
                List of updated row indices for non-zero entries in time-invariant self
                interaction superoperator.

    3. K0_col : list(int)
                List of updated column indices for non-zero entries in time-invariant
                self interaction superoperator.
    """

    # Add terms for each new member of the hierarchy
    # ==============================================

    for aux_ref in hierarchy.list_aux_add:

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
    system,
    hierarchy,
    mode,
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
    Adds crossterms between auxiliaries from the aux_from_list to the aux_to_list.

    Parameters
    ----------

    1. system : instance(HopsSystem)

    2. hierarchy : instance(HopsHierarchy)

    3. mode : instance(HopsMode)

    4. l_sparse : list(sparse matrix)
                  List of sparse L operators.

    5. Kp1_data : list(complex)
                  List of non-zero entries in time-invariant K+1 interaction super
                  operator.

    6. Kp1_row : list(int)
                 List of row indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    7. Kp1_col : list(int)
                 List of column indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    8. Zp1_data : list(complex)
                  List of non-zero entries in time-varying K+1 interaction super
                  operator.

    9. Zp1_row : list(int)
                  List of row indices for non-zero entries in time-varying K+1
                  interaction superoperator.

    10. Zp1_col : list(int)
                  List of column indices for non-zero entries in time-varying K+1
                  interaction superoperator.

    11. Km1_data : list(complex)
                   List of non-zero entries in time-invariant K-1 interaction super
                   operator.

    12. Km1_row : list(int)
                  List of row indices for non-zero entries in time-invariant K-1
                  interaction superoperator.

    13. Km1_col : list(int)
                  List of column indices for non-zero entries in time-invariant K-1
                  interaction superoperator.

    Returns
    -------
    1. Kp1_data : list(complex)
                  List of updated non-zero entries in time-invariant K+1 interaction
                  super operator.

    2. Kp1_row : list(int)
                 List of updated row indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    3. Kp1_col : list(int)
                 List of updated column indices for non-zero entries in time-invariant
                 K+1 interaction superoperator.

    4. Zp1_data : list(complex)
                  List of updated non-zero entries in time-varying K+1 interaction super
                  operator.

    5. Zp1_row : list(int)
                 List of updated row indices for non-zero entries in time-varying K+1
                 interaction superoperator.

    6. Zp1_col : list(int)
                 List of updated column indices for non-zero entries in time-varying K+1
                 interaction superoperator.

    7. Km1_data : list(complex)
                  List of updated non-zero entries in time-invariant K-1 interaction
                  superoperator.

    8. Km1_row : list(int)
                 List of updated row indices for non-zero entries in time-invariant K-1
                 interaction superoperator.

    9. Km1_col : list(int)
                 List of updated column indices for non-zero entries in
                 time-invariant K-1 interaction superoperator
    """
    # Prepare Constants
    # =================
    n_site = system.size
    
    for (l_mod,l_mod_abs) in enumerate(mode.list_absindex_mode):
        
        if(len(hierarchy.new_connect_p1[l_mod_abs].keys()) > 0):
            aux_indices, aux_indices_p1 = list(zip(*hierarchy.new_connect_p1[
                l_mod_abs].items()))
            i_lop = mode.list_index_L2_by_hmode[l_mod]
            #The Kp1 term has value g/w * vec(L).  
            #The total number of Kp1 entries to add (for mode l_mod) is  #aux_indices * #nonzero(l_sparse[i_lop]) 
            Kp1_data.extend(np.repeat(np.ones(len(aux_indices)) * -(mode.g[l_mod] / mode.w[l_mod]),len(l_sparse[i_lop].data)) * (list(l_sparse[i_lop].data) * len(aux_indices)))
            #To add all the row and column indices, we perform the array addition as follows:
            #1. Take the aux indices and repeat them for how many values each aux has (just 1 or zero for 1-particle)
            # Multiply this by n_site to get the starting indices
            #2. Add this by the row of l_sparse to get the result
            #e.g. if indices are [0,1,2], and n_site = 3, l_sparse[i_lop].row = l_sparse[i_lop].col = [0,1]:
            #1. = [0,0,1,1,2,2] * n_site = [0,0,3,3,6,6]
            #2. = [0,1,0,1,0,1] -> result = [0,1,3,4,6,7]
            
            Kp1_row.extend(np.repeat(np.array(aux_indices),len(l_sparse[i_lop].row)) * n_site + list(l_sparse[i_lop].row) * len(aux_indices))
            Kp1_col.extend(np.repeat(np.array(aux_indices_p1),len(l_sparse[i_lop].col)) * n_site + list(l_sparse[i_lop].col) * len(aux_indices_p1))
            
            Zp1_data[i_lop].extend(np.ones(len(aux_indices)) * (mode.g[l_mod] / mode.w[l_mod]))
            Zp1_row[i_lop].extend(aux_indices)
            Zp1_col[i_lop].extend(aux_indices_p1)

            
        if(len(hierarchy.new_connect_m1[l_mod_abs].keys()) > 0):
            aux_indices, aux_indices_m1 = list(
                zip(*hierarchy.new_connect_m1[l_mod_abs].items()))
            aux_mode_values = [hierarchy.new_mode_values_m1[l_mod_abs][aux_index]
                               for aux_index in aux_indices]
            i_lop = mode.list_index_L2_by_hmode[l_mod]
            
            Km1_data.extend(np.repeat(np.array(list(aux_mode_values)) * mode.w[l_mod],len(l_sparse[i_lop].data)) * (list(l_sparse[i_lop].data) * len(aux_mode_values)))
            Km1_row.extend(np.repeat(np.array(aux_indices),len(l_sparse[i_lop].row)) * n_site + list(l_sparse[i_lop].row) * len(aux_indices))
            Km1_col.extend(np.repeat(np.array(aux_indices_m1),len(l_sparse[i_lop].col)) * n_site + list(l_sparse[i_lop].col) * len(aux_indices_m1))           
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
    mode,
    list_ilop_rel_stable,
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
    Adds in the cross-terms corresponding to new states that were not present in the
    previous steps.

    Parameters
    ----------
    1. aux_stable : list(AuxiliaryVector)
                    List of stable auxiliaries.

    2. list_add_state : list(int)
                        List of new states.

    3. system : instance(HopsSystem)

    4. mode : instance(HopsMode)

    5. list_ilop_rel_stable : list(int)
                              List of the relative indices of stable L-operators in
                              the current list of L-operators.

    6. l_sparse : list(sparse matrix)
                  List of sparse L operators.

    7. Kp1_data : list(complex)
                  List of non-zero entries in time-invariant K+1 interaction super
                  operator.

    8. Kp1_row : list(int)
                 List of row indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    9. Kp1_col : list(int)
                 List of column indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    10. Zp1_data : list(complex)
                  List of non-zero entries in time-varying K+1 interaction super
                  operator.

    11. Zp1_row : list(int)
                 List of row indices for non-zero entries in time-varying K+1
                 interaction superoperator.

    12. Zp1_col : list(int)
                  List of column indices for non-zero entries in time-varying K+1
                  interaction superoperator.

    13. Km1_data : list(complex)
                   List of non-zero entries in time-invariant K-1 interaction super
                   operator.

    14. Km1_row : list(int)
                  List of row indices for non-zero entries in time-invariant K-1
                  interaction superoperator.

    15. Km1_col : list(int)
                  List of column indices for non-zero entries in time-invariant K-1
                  interaction superoperator.

    Returns
    -------
    1. Kp1_data : list(complex)
                  List of updated non-zero entries in time-invariant K+1 interaction
                  superoperator.

    2. Kp1_row : list(int)
                 List of updated row indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    3. Kp1_col : list(int)
                 List of updated column indices for non-zero entries in
                 time-invariant K+1 interaction superoperator.

    4. Zp1_data : list(complex)
                  List of updated non-zero entries in time-varying K+1 interaction super
                  operator.

    5. Zp1_row : list(int)
                 List of updated row indices for non-zero entries in time-varying K+1
                 interaction superoperator.

    6. Zp1_col : list(int)
                 List of updated column indices for non-zero entries in time-varying K+1
                 interaction superoperator.

    7. Km1_data : list(complex)
                  List of updated non-zero entries in time-invariant K-1 interaction
                  superoperator.

    8. Km1_row : list(int)
                 List of updated row indices for non-zero entries in time-invariant K-1
                 interaction superoperator.

    9. Km1_col : list(int)
                 List of updated column indices for non-zero entries in time-invariant
                 K-1 interaction superoperator.
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
        list_ilop_add = []
        for ilop in range(len(mode.list_absindex_L2)):
            if len(mode.list_state_indices_by_index_L2[ilop])>0:
                if system.state_list[mode.list_state_indices_by_index_L2[ilop][0]] in list_add_state:
                    list_ilop_add.append(ilop)

        # NOTE: mode.list_state_indices_by_index_L2[ilop][0] is only correct if each L-operator
        #       connects to a single state

        # NOTE: The logic above may miss some edge cases by excluding L-operators that have
        # no associated states by do have connections among hierarchy members (which should
        # be updating Zp1).

        for aux_ref in aux_stable:
            i_aux = aux_ref._index

            for (l_mod_abs, aux_p1) in aux_ref.dict_aux_p1.items():
                # The index of the corresponding L-Operator
                l_mod = list(mode.list_absindex_mode).index(l_mod_abs)
                i_lop = mode.list_index_L2_by_hmode[l_mod]

                # HIGHER TO LOWER AUX
                # sum_n sum_jn -(L*[n]-<L*[n]>_t) * phi_(k+e_jn)
                # integrand -= np.dot(np.conj(self.L[n]),phi[one_p])
                # integrand += np.conj(Ls[n])*phi[one_p] #empty_phi
                if aux_p1._index in list_index_aux_stable:
                    j_aux = aux_p1._index

                    if i_lop in list_ilop_add:
                        # if the state associated with this L-operator has not been
                        # present before then we should add all the associated terms
                        Kp1_data.extend(
                            -(mode.g[l_mod] / mode.w[l_mod]) * l_sparse[i_lop].data
                        )
                        Kp1_row.extend(i_aux * n_site + l_sparse[i_lop].row)
                        Kp1_col.extend(j_aux * n_site + l_sparse[i_lop].col)

                        # If the L-operator has been present before then we do not need
                        # to update this term (which does not depend on the state values)
                        if i_lop not in list_ilop_rel_stable:
                            Zp1_data[i_lop].append(
                                (mode.g[l_mod] / mode.w[l_mod])
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
                        #         -(mode.g[l_mod] / mode.w[l_mod]) * l_tmp.data
                        #     )
                        #     Kp1_row.extend(i_aux * n_site + l_tmp.row)
                        #     Kp1_col.extend(j_aux * n_site + l_tmp.col)

            # LOWER TO HIGHER AUX
            # sum_n sum_jn k_jn alpha_nj phi_(k-e_jn)
            # integrand += order[iterator]*self.g[iterator]*np.dot(self.L[n],phi[one_min])
            for (l_mod_abs, aux_m1) in aux_ref.dict_aux_m1.items():
                l_mod = list(mode.list_absindex_mode).index(l_mod_abs)
                i_lop = mode.list_index_L2_by_hmode[l_mod]

                if aux_m1._index in list_index_aux_stable:
                    j_aux = aux_m1._index

                    if i_lop in list_ilop_add:
                        # if this L-operator has not been present before
                        # then we should add all the associated terms
                        Km1_data.extend(
                            (aux_m1[mode.list_absindex_mode[l_mod]] + 1)
                            * mode.w[l_mod]
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
                        #         (aux_m1[mode.list_absindex_mode[l_mod]] + 1)
                        #         * mode.w[l_mod]
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
    system,
    hierarchy,
    mode,
    perm_index,
):
    """
    Updates the EOM matrices to incorporate changes to the hierarchy. At the moment,
    this function is the key step to handling Dynamic Filtering of the hierarchy.

    Parameters
    ----------
    1. K0 : sparse matrix
            Time-invariant self-interaction superoperator.

    2. Kp1 : sparse matrix
             Time-invariant K+1 interaction superoperator.

    3. Zp1 : sparse matrix
             Time-varying K+1 interaction superoperator.

    4. Km1 : sparse matrix
             Time-invariant K-1 interaction superoperator.

    5. system : instance(HopsSystem)

    6. hierarchy : instance(HopsHierarchy)

    7. mode : instance(HopsMode)

    8. perm_index : list(list(int))
                    List of column list and row list that define the permutation
                    matrix.

    Returns
    -------
    1. K0 : sparse matrix
            Updated time-invariant self-interaction superoperator.

    2. Kp1 : sparse matrix
             Updated time-invariant K+1 interaction superoperator.

    3. Zp1 : sparse matrix
             Updated time-varying K+1 interaction superoperator.

    4. Km1 : sparse matrix
             Updated time-invariant K-1 interaction superoperator.
    """
    # Prepare Constants
    # =================
    n_old = len(hierarchy.previous_auxiliary_list)*len(system.previous_state_list)
    n_site = system.size
    n_lop = len(mode.list_L2_coo)
    n_tot = system.size * hierarchy.size
    new_state = [i for i in system.state_list if i not in system.list_stable_state]

    l_sparse = [mode.list_L2_coo[i_lop] for i_lop in range(n_lop)]
    list_ilop_rel_stable = [list(mode.list_absindex_L2).index(ilop) for ilop in mode.previous_list_absindex_L2
                            if ilop in mode.list_absindex_L2]

    # Permutation Matrix
    # ==================
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(perm_index[0])), (perm_index[0], perm_index[1])),
        shape=(n_tot, n_old),
        dtype=np.complex128,
    ).tocsc()
    Pmat2 = sp.sparse.coo_matrix(
        (np.ones(len(perm_index[3])), (perm_index[3],perm_index[2])),
        shape=(hierarchy.size, perm_index[4]),
        dtype=np.complex128,
    ).tocsc()
    # K-matrices
    K0 = _permute_aux_by_matrix(K0, Pmat2)
    Kp1 = _permute_aux_by_matrix(Kp1, Pmat)
    Km1 = _permute_aux_by_matrix(Km1, Pmat)
    # Z Matrices
    Zp1_new = [[] for i_lop in range(n_lop)]
    for i_lop in range(n_lop):
        if mode.list_absindex_L2[i_lop] in mode.previous_list_absindex_L2:
            Zp1_new[i_lop] = _permute_aux_by_matrix(
                Zp1[list(mode.previous_list_absindex_L2).index(mode.list_absindex_L2[i_lop])], Pmat2
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
        system,
        hierarchy,
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
        system,
        hierarchy,
        mode,
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

    # Add list_stable_aux --> list_stable_aux
    # (new states only!)
    # =======================================
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
        hierarchy.list_aux_stable,
        new_state,
        system,
        mode,
        list_ilop_rel_stable,
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


def calculate_ksuper(system, 
                     hierarchy,
                     mode):
    """
    Calculates all the pieces needed to time-evolve the system EXCEPT for the
    normalization term. Used to create the time-evolution super-operators when they
    have not yet been constructed.

    Parameters
    ----------
    1. system : instance(HopsSystem)

    2. hierarchy : instance(HopsHierarchy)

    3. mode : instance(HopsMode)


    Returns
    -------
    1. K0 : sparse matrix
            Updated time-invariant self-interaction superoperator.

    2. Kp1 : sparse matrix
             Updated time-invariant K+1 interaction superoperator.

    3. Zp1 : sparse matrix
             Updated time-varying K+1 interaction superoperator.

    4. Km1 : sparse matrix
             Updated time-invariant K-1 interaction superoperator.
    """
    n_site = system.size
    n_lop = len(mode.list_L2_coo)
    n_tot = n_site * hierarchy.size

    l_sparse = [mode.list_L2_coo[i_lop] for i_lop in range(n_lop)]

    # Add new_aux --> new_aux
    # =======================
    (   
        K0_data,
        K0_row,
        K0_col,
    ) = _add_self_interactions(
        system,
        hierarchy,
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
        system,
        hierarchy,
        mode,
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

