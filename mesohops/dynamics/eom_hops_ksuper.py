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
    Adds crossterms between auxiliaries from the aux_from_list to the aux_to_list.

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
    l_sparse = [mode.list_L2_coo[i_lop] for i_lop in range(len(mode.list_L2_coo))]
    
    for (l_mod,l_mod_abs) in enumerate(mode.list_absindex_mode):
        if(len(hierarchy.new_aux_index_conn_by_mode[l_mod_abs].keys()) > 0):
            list_aux_indices, list_aux_conn_index_and_value = list(zip(*hierarchy.new_aux_index_conn_by_mode[
                l_mod_abs].items()))
            list_aux_indices_p1, list_aux_mode_values = list(zip(*list_aux_conn_index_and_value))
            i_lop = mode.list_index_L2_by_hmode[l_mod]
            #The Kp1 term has value g/w * vec(L).  
            #The total number of Kp1 entries to add (for mode l_mod) is  #aux_indices * #nonzero(l_sparse[i_lop]) 
            Kp1_data.extend(np.repeat(np.ones(len(list_aux_indices)) * -(mode.g[l_mod] / mode.w[l_mod]),len(l_sparse[i_lop].data)) \
                            * (list(l_sparse[i_lop].data) * len(list_aux_indices)))
            #To add all the row and column indices, we perform the array addition as follows:
            #1. Take the aux indices and repeat them for how many values each aux has (just 1 or zero for 1-particle)
            # Multiply this by n_site to get the starting indices
            #2. Add this by the row of l_sparse to get the result
            #e.g. if indices are [0,1,2], and n_site = 3, l_sparse[i_lop].row = l_sparse[i_lop].col = [0,1]:
            #1. = [0,0,1,1,2,2] * n_site = [0,0,3,3,6,6]
            #2. = [0,1,0,1,0,1] -> result = [0,1,3,4,6,7]
            
            Kp1_row.extend(np.repeat(np.array(list_aux_indices),len(l_sparse[i_lop].row)) * n_site + list(l_sparse[i_lop].row) * len(list_aux_indices))
            Kp1_col.extend(np.repeat(np.array(list_aux_indices_p1),len(l_sparse[i_lop].col)) * n_site + list(l_sparse[i_lop].col) * len(list_aux_indices_p1))
            
            Zp1_data[i_lop].extend(np.ones(len(list_aux_indices)) * (mode.g[l_mod] / mode.w[l_mod]))
            Zp1_row[i_lop].extend(list_aux_indices)
            Zp1_col[i_lop].extend(list_aux_indices_p1)

            #aux_mode_values = [hierarchy.new_mode_values_m1[l_mod_abs][aux_index]
            #                   for aux_index in aux_indices_p1]
            
            Km1_data.extend(np.repeat(np.array(list(list_aux_mode_values)) * mode.w[l_mod],len(l_sparse[i_lop].data)) \
                            * (list(l_sparse[i_lop].data) * len(list_aux_mode_values)))     
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
    Adds in the cross-terms corresponding to new states that were not present in the
    previous steps.

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
    #We need to find the new modes corresponding to new states, and the relative indices of the new states
    n_site = system.size
    list_irel_new_state = [list(system.state_list).index(i) for i in system.list_add_state]
    list_new_mode = list(system.list_absindex_new_state_modes)
    l_sparse_diag = [mode.list_L2_diag[i_lop] for i_lop in range(len(mode.list_L2_coo))]
    if len(list_irel_new_state) > 0:                  
                   
        for l_mode_abs in list_new_mode:
            #For each new mode, we look through our dictionary of mode connections between stable auxiliaries and add the corresponding entry to K
            #super operators.
            if(len(hierarchy.stable_aux_id_conn_by_mode[l_mode_abs]) > 0):
                list_ids, list_aux_conn_id_and_value = list(zip(*hierarchy.stable_aux_id_conn_by_mode[
                    l_mode_abs].items()))
                list_aux_indices = [hierarchy._aux_index(hierarchy.dict_aux_by_id[id_]) for id_ in list_ids]
                list_ids_p1, list_aux_mode_values = list(zip(*list_aux_conn_id_and_value))
                list_aux_indices_p1 = [hierarchy._aux_index(hierarchy.dict_aux_by_id[id_]) for id_ in list_ids_p1]
                l_mod = list(mode.list_absindex_mode).index(l_mode_abs)
                i_lop = mode.list_index_L2_by_hmode[l_mod]
                Kp1_data.extend(
                    np.array(list(l_sparse_diag[i_lop][list_irel_new_state]) * len(list_aux_indices)) * -(mode.g[l_mod] / mode.w[l_mod])
                )
                Kp1_row.extend(np.repeat(list_aux_indices,len(list_irel_new_state)) * n_site + np.array(list_irel_new_state * len(list_aux_indices)))
                Kp1_col.extend(np.repeat(list_aux_indices_p1,len(list_irel_new_state)) * n_site + np.array(list_irel_new_state * len(list_aux_indices_p1)))
            
                Km1_data.extend(
                    np.repeat(np.array(list(list_aux_mode_values)) * mode.w[l_mod],len(list_irel_new_state)) * \
                    np.array(list(l_sparse_diag[i_lop][list_irel_new_state]) * len(list_aux_mode_values))
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

    return [K0, Kp1, Zp1, Km1]


