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
    Kp1_data,
    Kp1_row,
    Kp1_col,
    Zp1_data,
    Zp1_row,
    Zp1_col,
    Km1_data,
):
    """
    Adds crossterms between auxiliary wave functions corresponding to the flux up and
    flux down portions of the HOPS equation-of-motion.

    Parameters
    ----------

    1. system : instance(HopsSystem)

    2. hierarchy : instance(HopsHierarchy)

    3. mode : instance(HopsMode)

    4. Kp1_data : list(complex)
                  List of non-zero entries in time-invariant K+1 interaction super
                  operator.

    5. Kp1_row : list(int)
                 List of row indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    6. Kp1_col : list(int)
                 List of column indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    7. Zp1_data : list(complex)
                  List of non-zero entries in time-varying K+1 interaction super
                  operator.

    8. Zp1_row : list(int)
                  List of row indices for non-zero entries in time-varying K+1
                  interaction superoperator.

    9. Zp1_col : list(int)
                  List of column indices for non-zero entries in time-varying K+1
                  interaction superoperator.

    10. Km1_data : list(complex)
                   List of non-zero entries in time-invariant K-1 interaction super
                   operator.

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
    """
    # Prepare Constants
    # =================
    
    n_site = system.size
    list_l_sparse = [mode.list_L2_coo[i_lop] for i_lop in range(len(mode.list_L2_coo))]
    
    for (l_mod,l_mod_abs) in enumerate(mode.list_absindex_mode):
        try:
            num_conn = len(hierarchy.new_aux_index_conn_by_mode[l_mod_abs])
        except:
            continue
        if(num_conn > 0):
            list_aux_indices, list_aux_conn_index_and_value = list(zip(*hierarchy.new_aux_index_conn_by_mode[
                l_mod_abs].items()))
            list_aux_indices_p1, list_aux_mode_values = list(zip(*list_aux_conn_index_and_value))
            i_lop = mode.list_index_L2_by_hmode[l_mod]
            L2_sparse = list_l_sparse[i_lop]
            #The Kp1 term has value g/w * vec(L).  
            #The total number of Kp1 entries to add (for mode l_mod) is  #aux_indices * #nonzero(L2_sparse)
            Kp1_data.extend(np.repeat(np.ones(len(list_aux_indices)) * -(mode.list_g[l_mod] / mode.list_w[l_mod]),len(L2_sparse.data)) \
                        * (list(L2_sparse.data) * len(list_aux_indices)))
            #To add all the row and column indices, we perform the array addition as follows:
            #1. Take the aux indices and repeat them for how many values each aux has (just 1 or zero for 1-particle)
            # Multiply this by n_site to get the starting indices
            #2. Add this by the row of l_sparse to get the result
            #e.g. if indices are [0,1,2], n_site = 3, and L2_sparse.row = L2_sparse.col
            # = [0,1]:
            #1. = [0,0,1,1,2,2] * n_site = [0,0,3,3,6,6]
            #2. = [0,1,0,1,0,1] -> result = [0,1,3,4,6,7]
            
            Kp1_row.extend(np.repeat(np.array(list_aux_indices),len(L2_sparse.row)) * n_site + list(L2_sparse.row) * len(list_aux_indices))
            Kp1_col.extend(np.repeat(np.array(list_aux_indices_p1),len(L2_sparse.col)) * n_site + list(L2_sparse.col) * len(list_aux_indices_p1))
            
            Zp1_data[i_lop].extend(np.ones(len(list_aux_indices)) * (mode.list_g[l_mod] / mode.list_w[l_mod]))
            Zp1_row[i_lop].extend(list_aux_indices)
            Zp1_col[i_lop].extend(list_aux_indices_p1)

            # The Km1 matrix must have nonzero entries in locations given by the
            # transpose of the Kp1 matrix. Taking advantage of the Hermitian property of
            # L-operators, we take the complex conjugate of any L-operator entries
            # involved to use the row and col indices of the Kp1 matrix directly.
            Km1_data.extend(
                np.repeat(np.array(list(list_aux_mode_values)) * mode.list_w[l_mod],
                          len(L2_sparse.data)) \
                * (list(np.conj(L2_sparse.data)) * len(
                    list_aux_mode_values)))
    return (
     Kp1_data,
     Kp1_row,
     Kp1_col,
     Zp1_data,
     Zp1_row,
     Zp1_col,
     Km1_data,
    )


def _add_crossterms_stable_K(
    system,
    hierarchy,
    mode,
    Kp1_data,
    Kp1_row,
    Kp1_col,
    Km1_data,
):
    """
    Adds in the crossterms between auxiliary wave functions corresponding to the flux up
    and flux down for all new states that were not present in the previous steps.

    Parameters
    ----------
    1. system : instance(HopsSystem)

    2. hierarchy : instance(HopsHierarchy)
    
    3. mode : instance(HopsMode)
    
    4. Kp1_data : list(complex)
                  List of non-zero entries in time-invariant K+1 interaction super
                  operator.

    5. Kp1_row : list(int)
                 List of row indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    6. Kp1_col : list(int)
                 List of column indices for non-zero entries in time-invariant K+1
                 interaction superoperator.

    7. Km1_data : list(complex)
                   List of non-zero entries in time-invariant K-1 interaction super
                   operator.

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

    4. Km1_data : list(complex)
                  List of updated non-zero entries in time-invariant K-1 interaction
                  superoperator.
    """
    n_site = system.size
    # Finds the relative indices of newly-included states.
    list_irel_new_state = [list(system.state_list).index(i) for i in system.list_add_state]
    if len(list_irel_new_state) > 0:
        # Finds the correlation function modes associated with newly-included states.
        list_new_mode = list(system.list_absindex_new_state_modes)
        # If an L-operator has a row or column in the newly-added states, the
        # information that interacts with that state in the entry (data, row, col) form
        # can be used to build the necessary crossterm. For each L-operator,
        # we identify the index of that entry in the sparse representation.
        list_l_sparse_entry_id = [set(list(np.where(np.isin(L2.row,
                                                        list_irel_new_state))[0]) +
                                list(np.where(np.isin(L2.col, list_irel_new_state))[0]))
                                for L2 in mode.list_L2_coo]
        for l_mode_abs in list_new_mode:
            # For each new mode, we look through our dictionary of mode connections
            # between stable auxiliaries and add the corresponding entry to K super
            # operators.
            if(len(hierarchy.stable_aux_id_conn_by_mode[l_mode_abs]) > 0):
                # Get position of auxiliaries in the list of stable auxiliaries,
                # as well as information about their connections to other auxiliaries
                # along the mode of interest.
                list_ids, list_aux_conn_id_and_value = list(zip(*hierarchy.stable_aux_id_conn_by_mode[
                    l_mode_abs].items()))
                # Get the index of those auxiliaries in the overall list of auxiliaries.
                list_aux_indices = [hierarchy._aux_index(hierarchy.dict_aux_by_id[id_]) for id_ in list_ids]
                # Use information about connections between auxiliaries to get the
                # list of auxiliaries connected to the auxiliaries in list_aux_indices
                list_ids_p1, list_aux_mode_values = list(zip(*list_aux_conn_id_and_value))
                list_aux_indices_p1 = [hierarchy._aux_index(hierarchy.dict_aux_by_id[id_]) for id_ in list_ids_p1]

                # Relative index of the mode of interest
                l_mod = list(mode.list_absindex_mode).index(l_mode_abs)
                # Relative index of the L-operator associated with the mode of interest
                i_lop = mode.list_index_L2_by_hmode[l_mod]

                # Find the indices of the entries of the L-operator of concern
                # associated with newly-added states, then get the data, row,
                # and column of those entries out of the sparse representation of the
                # L-operator.
                list_l_entry_mask = np.array(list(list_l_sparse_entry_id[i_lop]),
                                             dtype=int)
                l_data = list(mode.list_L2_coo[i_lop].data[list_l_entry_mask])
                l_col = list(mode.list_L2_coo[i_lop].col[list_l_entry_mask])
                l_row = list(mode.list_L2_coo[i_lop].row[list_l_entry_mask])

                # Add the flux-down terms in the HOPS equation-of-motion associated
                # with (g_m/w_m)L_m. Because the mode is specified in the outer
                # loop, g/w is the same for every term. L_data is a list of L_m[r,c]
                # where r, c are the row and column indices. By repeating L_data for
                # every auxiliary and multiplying by g/w, every term is accounted for
                # in the order [Lm[r1,c1],Lm[r2,c2]...] for each auxiliary.
                Kp1_data.extend(
                 np.array(l_data * len(list_aux_indices)) * -(mode.list_g[l_mod] /
                 mode.list_w[l_mod])
                )

                # Add the flux-up terms in the HOPS equation-of-motion associated with
                # (k_m*w_m)L_m. The np.repeat generates the list of k_m*w_m for all
                # auxiliaries, which is multiplied by the list of L_m[r,c] where r,
                # c are the row and column indices. This gets every term in the order
                # [Lm[c1,r1], Lm[c2,r2]...] for each auxiliary. The transposition of
                # c and r and the conjugate of l_data account for the fact that
                # L-operators are Hermitian: overall, tranposing Kp1_row and Kp1_col
                # when generating the portion of the EoM super-operator allows us to
                # skip generating unique Km1_row and Km1_col lists.
                Km1_data.extend(
                    np.repeat(np.array(list(list_aux_mode_values)) * mode.list_w[l_mod],
                              len(list_l_entry_mask)) * np.conj(np.array(list(l_data) * len(
                        list_aux_mode_values)))
                )

                # For each L-operator entry we need to consider, get the list of
                # auxiliary indices multiplied by the number of states and modulate
                # those indices by the row state indices of the L-operator entries.
                # This matches the ordering [Lm[r1,c1],Lm[r2,c2]...] for each auxiliary
                # in Kp1_data.
                Kp1_row.extend(
                    np.repeat(list_aux_indices,len(list_l_entry_mask)) * n_site + np.array(
                        l_row * len(list_aux_indices))
                )

                # For each L-operator entry we need to consider, get the list of
                # auxiliary indices multiplied by the number of states and modulate
                # those indices  by the column state indices of the L-operator entries.
                # This matches the ordering [Lm[r1,c1],Lm[r2,c2]...] for each
                # auxiliary in Kp1_data.
                Kp1_col.extend(
                    np.repeat(list_aux_indices_p1,len(list_l_entry_mask)) * n_site + np.array(
                        l_col * len(list_aux_indices_p1))
                )
    return (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Km1_data,
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
            Time-invariant self-interaction superoperator. Exists in the space of the
            auxiliary basis.

    2. Kp1 : sparse matrix
             Time-invariant K+1 interaction superoperator. Exists in the space of the
             HOPS basis.

    3. Zp1 : sparse matrix
             Time-varying K+1 interaction superoperator. Exists in the space of the
             auxiliary basis.

    4. Km1 : sparse matrix
             Time-invariant K-1 interaction superoperator. Exists in the space of the
             HOPS basis.

    5. system : instance(HopsSystem)

    6. hierarchy : instance(HopsHierarchy)

    7. mode : instance(HopsMode)

    8. perm_index : list(list(int))
                    List of column list and row list that define the permutation
                    matrix.

    Returns
    -------
    1. K0 : sparse matrix
            Updated time-invariant self-interaction superoperator. Exists in the
            space of the auxiliary basis.

    2. Kp1 : sparse matrix
             Updated time-invariant K+1 interaction superoperator. Exists in the
             space of the HOPS basis.

    3. Zp1 : sparse matrix
             Updated time-varying K+1 interaction superoperator. Exists in the
             space of the auxiliary basis.

    4. Km1 : sparse matrix
             Updated time-invariant K-1 interaction superoperator. Exists in the
             space of the HOPS basis.
    """
    # Prepare Constants
    # =================
    n_old = len(hierarchy.previous_auxiliary_list)*len(system.previous_state_list)
    n_site = system.size
    n_lop = len(mode.list_L2_coo)
    n_tot = system.size * hierarchy.size
    

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
    ) = _add_crossterms(
        system,
        hierarchy,
        mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Zp1_data=[[] for i in range(n_lop)],
        Zp1_row=[[] for i in range(n_lop)],
        Zp1_col=[[] for i in range(n_lop)],
        Km1_data=[],
    )

    # Add list_stable_aux_K --> list_stable_aux_K
    # (new states only!)
    # =======================================
    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Km1_data,
    ) = _add_crossterms_stable_K(
        system,
        hierarchy,
        mode,
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Km1_data,
    )
    # Construct Sparse Matrices
    # =========================
    K0 = (
        K0
        + sparse.coo_matrix(
            (K0_data, (K0_row, K0_col)), shape=(hierarchy.size, hierarchy.size), dtype=np.complex128
        ).tocsr()
    )
    Kp1 = (
        Kp1
        + sparse.coo_matrix(
            (Kp1_data, (Kp1_row, Kp1_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsr()
    )

    Km1 = (
        Km1
        + sparse.coo_matrix(
            (Km1_data, (Kp1_col, Kp1_row)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsr()
    )
    Zp1 = [
        Zp1[i]
        + sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(hierarchy.size, hierarchy.size),
            dtype=np.complex128,
        ).tocsr()
        for i in range(n_lop)
    ]
    
    list_hierarchy_mask_Zp1 = [
        [list(set(Zp1[i].nonzero()[0])),list(set(Zp1[i].nonzero()[1])), np.ix_(list(set(Zp1[i].nonzero()[0])),list(set(Zp1[i].nonzero()[1])))]
        for i in range(n_lop)
    ]
    
    
    return [K0, Kp1, Zp1, Km1, list_hierarchy_mask_Zp1]


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
    ) = _add_crossterms(
        system,
        hierarchy,
        mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Zp1_data=[[] for i in range(n_lop)],
        Zp1_row=[[] for i in range(n_lop)],
        Zp1_col=[[] for i in range(n_lop)],
        Km1_data=[],
    )
    # Construct Sparse Matrices
    K0 = sparse.coo_matrix(
        (K0_data, (K0_row, K0_col)), shape=(hierarchy.size, hierarchy.size), dtype=np.complex128
    ).tocsr()
    Kp1 = sparse.coo_matrix(
        (Kp1_data, (Kp1_row, Kp1_col)), shape=(n_tot, n_tot), dtype=np.complex128
    ).tocsr()
    Km1 = sparse.coo_matrix(
        (Km1_data, (Kp1_col, Kp1_row)), shape=(n_tot, n_tot), dtype=np.complex128
    ).tocsr()
    Zp1 = [
        sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(hierarchy.size, hierarchy.size),
            dtype=np.complex128,
        ).tocsr()
        for i in range(n_lop)
    ]

    list_hierarchy_mask_Zp1 = [
        [list(set(Zp1[i].nonzero()[0])),list(set(Zp1[i].nonzero()[1])), np.ix_(list(set(Zp1[i].nonzero()[0])),list(set(Zp1[i].nonzero()[1])))]
        for i in range(n_lop)
    ]

    return [K0, Kp1, Zp1, Km1, list_hierarchy_mask_Zp1]


