import numpy as np
import scipy as sp
from scipy import sparse
from pyhops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from pyhops.dynamics.bath_corr_functions import bcf_exp
from pyhops.dynamics.eom_hops_ksuper import (
    _permute_aux_by_matrix,
    _add_self_interactions,
    _add_crossterms,
    _add_crossterms_stable,
    update_ksuper,
)

__title__ = "Test of eom_hops_ksuperr"
__author__ = "D. I. G. Bennett, B. Citty"
__version__ = "1.2"
__date__ = ""

# NOTE: NEED TO TEST WHAT HAPPENS WHEN THE NUMBER OF LOPERATORS
#       IS DIFFERENT FROM THE NUMBER OF SITES!

# TEST PARAMETERS
# ===============
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 10.0,  # Units: fs
    "TAU": 0.5,  # Units: fs
}

loperator = np.zeros([2, 2, 2], dtype=np.float64)
loperator[0, 0, 0] = 1.0
loperator[1, 1, 1] = 1.0

sys_param = {
    "HAMILTONIAN": np.array([[0, 10.0], [10.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
}

hier_param = {"MAXHIER": 4}

eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None
    }

psi_0 = [1.0 + 0.0 * 1j, 0.0 + 0.0 * 1j]

hops = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
)
hops.initialize(psi_0)
km1_col = tuple(
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9,
     9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12,
     12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14,
     14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17,
     17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20,
     20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22,
     22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25,
     25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27,
     28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30,
     30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33,
     33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35,
     35, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38,
     38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 41,
     41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43,
     43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46,
     46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48,
     49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51,
     51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54,
     54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56,
     56, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59,
     59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 62,
     62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64,
     64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67,
     67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69,
    ]
)


def test_permute_aux_by_matrix():
    """
    test for _permute_aux_by_matrix
    """
    # perm_index[1] is the original state index
    # perm_index[0] is the new state index
    # ------------------------------------
    perm_index = [[1, 2, 3, 0], [0, 1, 2, 3]]
    M2_permute = sp.sparse.coo_matrix(
        (np.ones(len(perm_index[0])), (perm_index[0], perm_index[1])),
        shape=(4, 4),
        dtype=np.complex128,
    ).tocsc()

    M2_base = np.array(
        [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 6, 9]], dtype=np.complex128
    )
    M2_trans = np.array(
        [
            [9.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j, 6.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [3.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j],
            [6.0 + 0.0j, 0.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j],
        ]
    )
    M2_trans_2 = _permute_aux_by_matrix(sp.sparse.csc_matrix(M2_base), M2_permute)
    assert (M2_trans == M2_trans_2.todense()).all()


def test_permute_ksuper_by_matrix():
    """
    test to check we correctly permute super operators
    """
    auxiliary_list_2 = [hops.basis.hierarchy.auxiliary_list[i] for i in [0, 1, 3, 5]]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x, absolute=True)
    )
    stable_aux = set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list)
    permute_aux_row = []
    permute_aux_col = []
    for aux in stable_aux:
        permute_aux_row.append(
            auxiliary_list_2.index(aux)
        )
        permute_aux_col.append(
            hops.basis.hierarchy.auxiliary_list.index(aux)
        )
    # Using permutation matrix
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row, permute_aux_col)),
        shape=(4, 70),
        dtype=np.complex128,
    ).tocsc()
    K0_new = _permute_aux_by_matrix(hops.basis.eom.K2_k, Pmat)
    # Hand Reconstruction of the permutation action
    row = []
    col = []
    data = []
    for (i, inew) in enumerate(permute_aux_row):
        for (j, jnew) in enumerate(permute_aux_row):
            row.append(inew)
            col.append(jnew)
            data.append(hops.basis.eom.K2_k[permute_aux_col[i], permute_aux_col[j]])
    K0_new2 = sp.sparse.coo_matrix(
        (data, (row, col)), shape=(4, 4), dtype=np.complex128
    ).tocsc()

    assert (K0_new.todense() == K0_new2.todense()).all()



def test_add_self_interaction_remove_aux():
    """
    test _add_self_interaction() when only auxiliaries are removed
    """
    # Prepare Constants
    # =================
    n_site = hops.basis.system.param["NSTATES"]
    n_lop = hops.basis.system.param["N_L2"]
    n_mode = hops.basis.system.param["N_HMODES"]
    n_tot = hops.basis.hierarchy.size
    l_sparse = [
        sp.sparse.coo_matrix(hops.basis.system.param["LIST_L2_COO"][i_lop])
        for i_lop in range(n_lop)
    ]
    for i_lop in range(n_lop):
        l_sparse[i_lop].eliminate_zeros()

        # Remove indices
    # --------------
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x, absolute=True)
    )
    stable_aux = set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list)
    permute_aux_row = []
    permute_aux_col = []
    for aux in stable_aux:
        permute_aux_row.append(
            auxiliary_list_2.index(aux)
        )
        permute_aux_col.append(
            hops.basis.hierarchy.auxiliary_list.index(aux)
        )
    # Using permutation matrix
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row, permute_aux_col)),
        shape=(68, 70),
        dtype=np.complex128,
    ).tocsc()
    K0_new = _permute_aux_by_matrix(hops.basis.eom.K2_k, Pmat)

    # Add indices
    # --------------
    # Using permutation matrix
    Pmat = Pmat.transpose()
    K0_new = _permute_aux_by_matrix(K0_new, Pmat)


        # Add back interactions
    # ---------------------
    K0_data, K0_row, K0_col = _add_self_interactions(
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        hops.basis.system,
        K0_data=[],
        K0_row=[],
        K0_col=[],
    )

    K0 = (
        K0_new
        + sparse.coo_matrix(
            (K0_data, (K0_row, K0_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsc()
    )

    assert (K0.todense() == hops.basis.eom.K2_k.todense()).all()


# noinspection PyTupleAssignmentBalance
def test_add_cross_terms():
    """
    test add_cross_terms() with only removed aux
    """
    # Prepare Constants
    # =================
    n_site = hops.basis.system.param["NSTATES"]
    n_lop = hops.basis.system.param["N_L2"]
    n_mode = hops.basis.system.param["N_HMODES"]
    n_tot = n_site * hops.basis.hierarchy.size
    n_tot2 = hops.basis.hierarchy.size

    l_sparse = [
        sp.sparse.coo_matrix(hops.basis.system.param["LIST_L2_COO"][i_lop])
        for i_lop in range(n_lop)
    ]
    for i_lop in range(n_lop):
        l_sparse[i_lop].eliminate_zeros()

        # Remove indices
    # --------------
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x, absolute=True)
    )
    stable_aux = set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list)
    permute_aux_row = []
    permute_aux_col = []
    for aux in stable_aux:
        permute_aux_row.extend(
            auxiliary_list_2.index(aux) * hops.basis.system.param["NSTATES"]
            + np.arange(hops.basis.system.param["NSTATES"])
        )
        permute_aux_col.extend(
            hops.basis.hierarchy.auxiliary_list.index(aux)
            * hops.basis.system.param["NSTATES"]
            + np.arange(hops.basis.system.param["NSTATES"])
        )
    permute_aux_row2 = []
    permute_aux_col2 = []
    for aux in stable_aux:
        permute_aux_row2.append(
            auxiliary_list_2.index(aux)
        )
        permute_aux_col2.append(
            hops.basis.hierarchy.auxiliary_list.index(aux)
        )
    # Using permutation matrix
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row, permute_aux_col)),
        shape=(136, 140),
        dtype=np.complex128,
    ).tocsc()
    Pmat2 = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row2)), (permute_aux_row2, permute_aux_col2)),
        shape=(68,70),
        dtype=np.complex128,
    ).tocsc()
    Kp1_new = _permute_aux_by_matrix(hops.basis.eom.K2_kp1, Pmat)
    Km1_new = _permute_aux_by_matrix(hops.basis.eom.K2_km1, Pmat)

    Zp1_new = [[] for i in range(n_lop)]
    for i_lop in range(n_lop):
        Zp1_new[i_lop] = _permute_aux_by_matrix(hops.basis.eom.Z2_kp1[i_lop], Pmat2)

    # Add indices
    # --------------
    # Using permutation matrix
    Pmat = Pmat.transpose()
    Pmat2 = Pmat2.transpose()
    Kp1_new = _permute_aux_by_matrix(Kp1_new, Pmat)
    Km1_new = _permute_aux_by_matrix(Km1_new, Pmat)

    Zp1_new2 = [[] for i in range(n_lop)]
    for i_lop in range(n_lop):
        Zp1_new2[i_lop] = _permute_aux_by_matrix(Zp1_new[i_lop], Pmat2)

        # Add back interactions
    # ---------------------
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
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        [
            hops.basis.hierarchy.auxiliary_list[0],
            hops.basis.hierarchy.auxiliary_list[1],
            *hops.basis.hierarchy.auxiliary_list[4:],
        ],
        hops.basis.system,
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
        [
            hops.basis.hierarchy.auxiliary_list[0],
            hops.basis.hierarchy.auxiliary_list[1],
            *hops.basis.hierarchy.auxiliary_list[4:],
        ],
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        hops.basis.system,
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

    Kp1 = (
        Kp1_new
        + sparse.coo_matrix(
            (Kp1_data, (Kp1_row, Kp1_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsc()
    )
    Km1 = (
        Km1_new
        + sparse.coo_matrix(
            (Km1_data, (Km1_row, Km1_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsc()
    )
    Zp1 = [
        Zp1_new2[i]
        + sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(n_tot2, n_tot2),
            dtype=np.complex128,
        ).tocsc()
        for i in range(n_lop)
    ]
    assert (Kp1.todense() == hops.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops.basis.eom.K2_km1.todense()).all()
    assert (Zp1[0].todense() == hops.basis.eom.Z2_kp1[0].todense()).all()


def test_matrix_updates_with_missing_aux_and_states():
    """
    test the matrix update functions when aux and states are
    removed
    """
    # Prepare Constants
    # =================
    n_site = hops.basis.system.param["NSTATES"]
    n_lop = hops.basis.system.param["N_L2"]
    n_tot = n_site * hops.basis.hierarchy.size
    n_tot2 = hops.basis.hierarchy.size

    l_sparse = [
        sp.sparse.coo_matrix(hops.basis.system.param["LIST_L2_COO"][i_lop])
        for i_lop in range(n_lop)
    ]
    for i_lop in range(n_lop):
        l_sparse[i_lop].eliminate_zeros()

    # Determine removed indices
    # -------------------------
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x, absolute=True)
    )
    stable_aux = set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list)

    state_list_2 = [
        hops.basis.system.state_list[i] for i in range(hops.n_state) if i > 0
    ]
    stable_state = state_list_2

    permute_aux_row = []
    permute_aux_col = []
    for aux in stable_aux:
        for state in stable_state:
            permute_aux_row.append(
                auxiliary_list_2.index(aux) * len(state_list_2)
                + state_list_2.index(state)
            )
            permute_aux_col.append(
                hops.basis.hierarchy.auxiliary_list.index(aux)
                * hops.basis.system.param["NSTATES"]
                + list(hops.basis.system.state_list).index(state)
            )
    permute_aux_row2 = []
    permute_aux_col2 = []
    for aux in stable_aux:
        permute_aux_row2.append(
            auxiliary_list_2.index(aux)
        )
        permute_aux_col2.append(
            hops.basis.hierarchy.auxiliary_list.index(aux)
        )
    # Remove indices using permutation matrix
    # ---------------------------------------
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row, permute_aux_col)),
        shape=(68, 140),
        dtype=np.complex128,
    ).tocsc()
    Pmat2 = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row2, permute_aux_col2)),
        shape=(68, 70),
        dtype=np.complex128,
    ).tocsc()
    K0_new = _permute_aux_by_matrix(hops.basis.eom.K2_k, Pmat2)
    Kp1_new = _permute_aux_by_matrix(hops.basis.eom.K2_kp1, Pmat)
    Km1_new = _permute_aux_by_matrix(hops.basis.eom.K2_km1, Pmat)
    Zp1_new = [_permute_aux_by_matrix(hops.basis.eom.Z2_kp1[1], Pmat2)]

    # Now attempt to add states and auxiliaries back
    # ==============================================
    # Add indices
    # --------------
    # Using permutation matrix
    Pmat = Pmat.transpose()
    Pmat2 = Pmat2.transpose()
    Kp1_new = _permute_aux_by_matrix(Kp1_new, Pmat)
    Km1_new = _permute_aux_by_matrix(Km1_new, Pmat)
    K0_new = _permute_aux_by_matrix(K0_new, Pmat2)

    Zp1_new2 = [
        sparse.coo_matrix((70, 70), dtype=np.complex128),
        _permute_aux_by_matrix(Zp1_new[0], Pmat2),
    ]

    # Add back cross interactions
    # ---------------------------
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
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        hops.basis.system,
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
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        [
            hops.basis.hierarchy.auxiliary_list[0],
            hops.basis.hierarchy.auxiliary_list[1],
            *hops.basis.hierarchy.auxiliary_list[4:],
        ],
        hops.basis.system,
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
        [
            hops.basis.hierarchy.auxiliary_list[0],
            hops.basis.hierarchy.auxiliary_list[1],
            *hops.basis.hierarchy.auxiliary_list[4:],
        ],
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        hops.basis.system,
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
        aux_stable=[
            hops.basis.hierarchy.auxiliary_list[0],
            hops.basis.hierarchy.auxiliary_list[1],
            *hops.basis.hierarchy.auxiliary_list[4:],
        ],
        list_add_state=[0],
        system=hops.basis.system,
        l_sparse=l_sparse,
        Kp1_data=Kp1_data,
        Kp1_row=Kp1_row,
        Kp1_col=Kp1_col,
        Zp1_data=Zp1_data,
        Zp1_row=Zp1_row,
        Zp1_col=Zp1_col,
        Km1_data=Km1_data,
        Km1_row=Km1_row,
        Km1_col=Km1_col,
    )

    # Add back self interactions
    # ---------------------------
    K0_data, K0_row, K0_col = _add_self_interactions(
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        hops.basis.system,
        K0_data=[],
        K0_row=[],
        K0_col=[],
    )

    Kp1 = (
        Kp1_new
        + sparse.coo_matrix(
            (Kp1_data, (Kp1_row, Kp1_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsc()
    )
    Km1 = (
        Km1_new
        + sparse.coo_matrix(
            (Km1_data, (Km1_row, Km1_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsc()
    )
    Zp1 = [
        Zp1_new2[i]
        + sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(n_tot2, n_tot2),
            dtype=np.complex128,
        ).tocsc()
        for i in range(n_lop)
    ]
    K0 = (
        K0_new
        + sparse.coo_matrix(
            (K0_data, (K0_row, K0_col)), shape=(n_tot2, n_tot2), dtype=np.complex128
        ).tocsc()
    )

    assert (Kp1.todense() == hops.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops.basis.eom.K2_km1.todense()).all()
    assert (Zp1[0].todense() == hops.basis.eom.Z2_kp1[0].todense()).all()
    assert (K0.todense() == hops.basis.eom.K2_k.todense()).all()


def test_update_super_remove_aux():
    """
    test update_ksuper() when only aux are removed
    """
    # Prepare Constants
    # -----------------
    n_lop = hops.basis.system.param["N_L2"]
 
    # Remove Auxiliaries
    # ------------------
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x, absolute=True)
    )
    stable_aux = set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list)
    permute_aux_row = []
    permute_aux_col = []
    for aux in stable_aux:
        permute_aux_row.extend(
            auxiliary_list_2.index(aux) * hops.basis.system.param["NSTATES"]
            + np.arange(hops.basis.system.param["NSTATES"])
        )
        permute_aux_col.extend(
            hops.basis.hierarchy.auxiliary_list.index(aux)
            * hops.basis.system.param["NSTATES"]
            + np.arange(hops.basis.system.param["NSTATES"])
        )
    permute_aux_row2 = []
    permute_aux_col2 = []
    for aux in stable_aux:
        permute_aux_row2.append(
            auxiliary_list_2.index(aux)
        )
        permute_aux_col2.append(
            hops.basis.hierarchy.auxiliary_list.index(aux)
        )
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row, permute_aux_col)),
        shape=(136, 140),
        dtype=np.complex128,
    ).tocsc()
    Pmat2 = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row2)), (permute_aux_row2, permute_aux_col2)),
        shape=(68, 70),
        dtype=np.complex128,
    ).tocsc()
    K0_new = _permute_aux_by_matrix(hops.basis.eom.K2_k, Pmat2)
    Kp1_new = _permute_aux_by_matrix(hops.basis.eom.K2_kp1, Pmat)
    Km1_new = _permute_aux_by_matrix(hops.basis.eom.K2_km1, Pmat)
    Zp1_new = [[] for i in range(n_lop)]
    for i_lop in range(n_lop):
        Zp1_new[i_lop] = _permute_aux_by_matrix(hops.basis.eom.Z2_kp1[i_lop], Pmat2)

        
        
    list_stable_aux_old_index = list(np.arange(68))
    list_stable_aux_new_index = list(set(np.arange(70)) - set([2,3]))
    
    
    K0, Kp1, Zp1, Km1 = update_ksuper(
        K0_new,
        Kp1_new,
        Zp1_new,
        Km1_new,
        stable_aux,
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        hops.basis.system.state_list,
        range(n_lop),
        hops.basis.system,
        hops.basis.hierarchy,
        hops.basis.hierarchy.size * hops.basis.system.param["NSTATES"] - 4,
        [permute_aux_col, permute_aux_row, list_stable_aux_old_index, list_stable_aux_new_index, 68],
    )

    assert (K0.todense() == hops.basis.eom.K2_k.todense()).all()
    assert (Kp1.todense() == hops.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops.basis.eom.K2_km1.todense()).all()
    assert (Zp1[0].todense() == hops.basis.eom.Z2_kp1[0].todense()).all()


def test_update_super_remove_aux_and_state():
    """
    test update_ksuper when aux and states are removed
    """
    # Prepare Constants
    # =================
    n_site = hops.basis.system.param["NSTATES"]

    # Remove indices
    # --------------
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x, absolute=True)
    )
    stable_aux = set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list)

    state_list_2 = [
        hops.basis.system.state_list[i] for i in range(hops.n_state) if i > 0
    ]
    stable_state = state_list_2

    permute_aux_row = []
    permute_aux_col = []
    for aux in stable_aux:
        for state in stable_state:
            permute_aux_row.append(
                auxiliary_list_2.index(aux) * len(state_list_2)
                + state_list_2.index(state)
            )
            permute_aux_col.append(
                hops.basis.hierarchy.auxiliary_list.index(aux)
                * hops.basis.system.param["NSTATES"]
                + list(hops.basis.system.state_list).index(state)
            )
    permute_aux_row2 = []
    permute_aux_col2 = []
    for aux in stable_aux:
        permute_aux_row2.append(
            auxiliary_list_2.index(aux)
        )
        permute_aux_col2.append(
            hops.basis.hierarchy.auxiliary_list.index(aux)
        )
    # Using permutation matrix
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row, permute_aux_col)),
        shape=(68, 140),
        dtype=np.complex128,
    ).tocsc()
    
    Pmat2 = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row2)), (permute_aux_row2, permute_aux_col2)),
        shape=(68, 70),
        dtype=np.complex128,
    ).tocsc()
    K0_new = _permute_aux_by_matrix(hops.basis.eom.K2_k, Pmat2)
    Kp1_new = _permute_aux_by_matrix(hops.basis.eom.K2_kp1, Pmat)
    Km1_new = _permute_aux_by_matrix(hops.basis.eom.K2_km1, Pmat)
    Zp1_new = [_permute_aux_by_matrix(hops.basis.eom.Z2_kp1[1], Pmat2)]

    list_stable_aux_old_index = list(np.arange(68))
    list_stable_aux_new_index = list(set(np.arange(70)) - set([2,3]))


    K0, Kp1, Zp1, Km1 = update_ksuper(
        K0_new,
        Kp1_new,
        Zp1_new,
        Km1_new,
        stable_aux,
        [
            hops.basis.hierarchy.auxiliary_list[2],
            hops.basis.hierarchy.auxiliary_list[3],
        ],
        [hops.basis.system.state_list[1]],
        [1],
        hops.basis.system,
        hops.basis.hierarchy,
        hops.basis.hierarchy.size - 2,
        [permute_aux_col, permute_aux_row, list_stable_aux_old_index, list_stable_aux_new_index, 68],
    )

    assert (Kp1.todense() == hops.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops.basis.eom.K2_km1.todense()).all()
    assert (K0.todense() == hops.basis.eom.K2_k.todense()).all()
    assert (Zp1[0].todense() == hops.basis.eom.Z2_kp1[0].todense()).all()
