import numpy as np
import scipy as sp
from scipy import sparse
from mesohops.basis.hops_aux import AuxiliaryVector as AuxiliaryVector
from mesohops.eom.eom_hops_ksuper import (
    _permute_aux_by_matrix,
    _add_self_interactions,
    _add_crossterms,
    _add_crossterms_stable_K,
    update_ksuper,
)
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.util.bath_corr_functions import bcf_convert_sdl_to_exp

__title__ = "Test of eom_hops_ksuper"
__author__ = "D. I. G. Bennett, B. Citty, J. K. Lynd"
__version__ = "1.4"
__date__ = ""
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

psi_0 = np.array([1.0 + 0.0 * 1j, 0.0 + 0.0 * 1j])

hops = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
)
hops.initialize(psi_0)

def generate_eom_k_super(list_states, list_aux, list_lop, list_lop_ind, list_g, list_w,
                         psi, list_modes):
    """
    Generates the non time-dependent portion of the super-operator for an adaptive
    HOPS basis.

    Parameters
    ----------
    1. list_states : list(int)
                     Absolute index of each state in the current basis.
    2. list_aux : list(np.array(int))
                  Auxiliary indexing vector of each auxiliary wave function in the
                  current basis, represented in the full space of modes.
    3. list_lop : list(np.array(complex))
                  L-operators corresponding to the full list of modes, represented
                  in the full state basis.
    4. list_lop_ind : list(int)
                      The absolute index of the unique L-operator for each mode in the
                      full basis.
    5. list_g : list(complex)
                The constant prefactor on each correlation function mode in the full
                list of modes [units: cm^-2].
    6. list_w : list(complex)
                The complex exponential decay factor for each correlation function
                mode in the full list of modes [units: cm^-1].
    7. psi : np.array(float)
             The system wave function in the current state basis.
    8. list_modes : list(int)
                    The list of modes in the current basis.
    Returns
    -------
    1. K2_self : np.array(complex)
                 The portion of the static derivative super-operator for the
                 full HOPS wave function in the current basis that accounts for
                 self-interaction terms, expressed in the auxiliary basis.
    2. K2_up : np.array(complex)
               The portion of the static derivative super-operator for the
               full HOPS wave function in the current basis that accounts for
               flux from higher-lying auxiliaries, expressed in the full basis.
    3. Z2_up_red : list(np.array(complex))
                   The portion of the time-dependent derivative super-operator for the
                   full HOPS wave function in the current basis that accounts for flux
                   from higher-lying auxiliaries, in the space of auxiliary
                   connections only (that is, without regard for state basis),
                   for each relative index of unique L-operator.
    4. K2_down : np.array(complex)
                 The portion of the static derivative super-operator for the
                 full HOPS wave function in the current basis that accounts for flux
                 from lower-lying auxiliaries, expressed in the full basis.
    """
    n_aux = len(list_aux)
    n_states = len(list_states)
    basis_size = n_aux*n_states
    lop_reduced = [L2[np.ix_(list_states,list_states)] for L2 in list_lop]
    n_unique_lop = max(list_lop_ind)+1

    K2_self = np.zeros([n_aux,n_aux],dtype=np.complex128)
    K2_up = np.zeros([basis_size, basis_size], dtype=np.complex128)
    Z2_up_red = [np.zeros([n_aux, n_aux], dtype=np.complex128) for l in range(
        n_unique_lop)]
    K2_down = np.zeros([basis_size, basis_size], dtype=np.complex128)
    for a1 in range(n_aux):
        k1 = list_aux[a1]
        # self-interaction term
        K2_self[a1, a1] = -1*(np.conj(k1)@np.array(list_w))
        for a2 in range(n_aux):
            k2 = list_aux[a2]
            diff = k1-k2
            if np.sum(np.abs(diff)) == 1:
                # Find absolute index of the mode where k1 and k2 don't match.
                m = np.where(diff != 0)[0][0]
                if m in list_modes:
                    L2 = lop_reduced[m]
                    if diff[m] == -1:
                        # flux down (from higher-lying) terms: k2 higher-lying than k1
                        gdw = list_g[m]/list_w[m]
                        for s in range(n_states):
                            dest_ind = a1*n_states + s
                            K2_up[dest_ind, a2*n_states:(a2+1)*n_states] = (
                                    -gdw*L2[s,:])
                        # Multiplied by <L_m> in the EoM
                        Z2_up_red[list_lop_ind[m]][a1, a2] = gdw
                    elif diff[m] == 1:
                        # flux up (from lower-lying) terms: k2 lower-lying than k1
                        kmw = k1[m]*list_w[m]
                        for s in range(n_states):
                            dest_ind = a1 * n_states + s
                            K2_down[dest_ind, a2*n_states:(a2+1)*n_states] = kmw*L2[s,:]
    return K2_self, K2_up, Z2_up_red, K2_down

def test_generate_eom_k_super():
    """
    Tests that the generate_eom_k_super helper function properly constructs all
    components of the super-operator.
    """
    lop_1 = np.array([[3,0,0],[0,0,0], [0,0,1]],)
    lop_2 = np.array([[0,1j,0],[-1j,0,1], [0,1,0]])
    list_lop = [lop_1, lop_1, lop_2, lop_2]
    list_lop_ind = [0, 0, 1, 1]
    list_states = [0,1]
    list_g = [10, -20, 30, -40]
    list_w = [10, 5, 10, 5]
    list_modes = [0, 1, 2, 3]
    list_aux = [np.array([0,0,0,0]),
                np.array([1,0,0,0]),
                np.array([0,1,0,0]),
                np.array([0,0,1,0]),
                np.array([2,0,0,0]),
                np.array([1,1,0,0]),]
    psi_0 = np.array([1j,-1j])
    K2_self, K2_up, Z2_up_red, K2_down = generate_eom_k_super(list_states, list_aux,
                                                              list_lop, list_lop_ind,
                                                              list_g, list_w, psi_0,
                                                              list_modes)
    known_K2_self = np.diag([0, -10, -5, -10, -20, -15])
    np.testing.assert_allclose(known_K2_self, K2_self)

    n_basis = len(list_states)*len(list_aux)

    # Terms going down from...
    # <1,0,0,0> to <0,0,0,0> (mode 0, state 0)
    # <0,1,0,0> to <0,0,0,0> (mode 1, state 0)
    # <0,0,1,0> to <0,0,0,0> (mode 2, state 0-->1,1-->0)
    # <2,0,0,0> t0 <1,0,0,0> (mode 0, state 0)
    # <1,1,0,0> to <1,0,0,0> (mode 1, state 0)
    # <1,1,0,0> to <0,1,0,0> (mode 0, state 0)
    known_K2_up = np.zeros([n_basis, n_basis], dtype=np.complex128)
    known_K2_up[0, 2] = -list_g[0]/list_w[0] * list_lop[0][0, 0]
    known_K2_up[0, 4] = -list_g[1] / list_w[1] * list_lop[1][0, 0]
    known_K2_up[1, 6] = -list_g[2] / list_w[2] * list_lop[2][1, 0]
    known_K2_up[0, 7] = -list_g[2] / list_w[2] * list_lop[2][0, 1]
    known_K2_up[2, 8] = -list_g[0] / list_w[0] * list_lop[0][0, 0]
    known_K2_up[2, 10] = -list_g[1] / list_w[1] * list_lop[1][0, 0]
    known_K2_up[4, 10] = -list_g[0] / list_w[0] * list_lop[0][0, 0]
    np.testing.assert_allclose(known_K2_up, K2_up)

    # Time-dependent terms going down from...
    # <1,0,0,0> to <0,0,0,0> (mode 0)
    # <0,1,0,0> to <0,0,0,0> (mode 1)
    # <0,0,1,0> to <0,0,0,0> (mode 2)
    # <2,0,0,0> t0 <1,0,0,0> (mode 0)
    # <1,1,0,0> to <1,0,0,0> (mode 1)
    # <1,1,0,0> to <0,1,0,0> (mode 0)
    known_Z2_up_red = [np.zeros([len(list_aux), len(list_aux)], dtype=np.complex128)
                       for l in range(2)]
    known_Z2_up_red[0][0, 1] = list_g[0] / list_w[0]
    known_Z2_up_red[0][0, 2] = list_g[1] / list_w[1]
    known_Z2_up_red[1][0, 3] = list_g[2] / list_w[2]
    known_Z2_up_red[0][1, 4] = list_g[0] / list_w[0]
    known_Z2_up_red[0][1, 5] = list_g[1] / list_w[1]
    known_Z2_up_red[0][2, 5] = list_g[0] / list_w[0]
    np.testing.assert_allclose(known_Z2_up_red[0], Z2_up_red[0])
    np.testing.assert_allclose(known_Z2_up_red[1], Z2_up_red[1])

    # Terms going up from...
    # <0,0,0,0> to <1,0,0,0> (mode 0, state 0)
    # <0,0,0,0> to <0,1,0,0> (mode 1, state 0)
    # <0,0,0,0> to <0,0,1,0> (mode 2, state 0-->1,1-->0)
    # <1,0,0,0> t0 <2,0,0,0> (mode 0, state 0)
    # <1,0,0,0> to <1,1,0,0> (mode 1, state 0)
    # <0,1,0,0> to <1,1,0,0> (mode 0, state 0)
    known_K2_down = np.zeros([n_basis, n_basis], dtype=np.complex128)
    known_K2_down[2, 0] = list_w[0] * list_aux[1][0] * list_lop[0][0,0]
    known_K2_down[4, 0] = list_w[1] * list_aux[2][1] * list_lop[1][0, 0]
    known_K2_down[6, 1] = list_w[2] * list_aux[3][2] * list_lop[2][0, 1]
    known_K2_down[7, 0] = list_w[2] * list_aux[3][2] * list_lop[2][1, 0]
    known_K2_down[8, 2] = list_w[0] * list_aux[4][0] * list_lop[0][0, 0]
    known_K2_down[10, 2] = list_w[1] * list_aux[5][1] * list_lop[1][0, 0]
    known_K2_down[10, 4] = list_w[0] * list_aux[5][0] * list_lop[0][0, 0]
    np.testing.assert_allclose(known_K2_down, K2_down)

def test_permute_aux_by_matrix():
    """
    Tests that _permute_aux_by_matrix properly rotates indices from an old to new
    position.
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
        [[0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 4, 6],
                [0, 3, 6, 9]], dtype=np.complex128
    )
    M2_trans = np.array(
        [[9, 0, 3, 6],
                [0, 0, 0, 0],
                [3, 0, 1, 2],
                [6, 0, 2, 4]], dtype=np.complex128
    )
    M2_trans_2 = _permute_aux_by_matrix(sp.sparse.csc_matrix(M2_base), M2_permute)
    assert (M2_trans == M2_trans_2.todense()).all()


def test_permute_ksuper_by_matrix():
    """
    Tests that _permute_aux_by_matrix used directly on a time-evolution
    super-operator correctly permutes to new auxiliary indices.
    """
    auxiliary_list_2 = [hops.basis.hierarchy.auxiliary_list[i] for i in [0, 1, 3, 5]]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x)
    )
    # We do not generally assume that sets will be sorted.
    stable_aux = set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list)

    # Create a permutation matrix with rows corresponding to position of a given
    # auxiliary in the truncated list, cols corresponding to position of a given
    # auxiliary in the full list.
    permute_aux_row = []
    permute_aux_col = []
    for aux in stable_aux:
        permute_aux_row.append(
            auxiliary_list_2.index(aux)
        )
        permute_aux_col.append(
            hops.basis.hierarchy.auxiliary_list.index(aux)
        )

    # Use the permutation matrix to reconfigure the self-interaction EoM matrix.
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row, permute_aux_col)),
        shape=(4, 70),
        dtype=np.complex128,
    ).tocsc()
    K0_new = _permute_aux_by_matrix(hops.basis.eom.K2_k, Pmat)

    # Manual reconstruction of the permutation action.
    row = []
    col = []
    data = []
    # Loop over relative indices of the reduced auxiliary list.
    for (i, inew) in enumerate(permute_aux_row):
        for (j, jnew) in enumerate(permute_aux_row):
            # Note that the relative indices of the reduced auxiliary list may not be
            # ordered as the counting numbers.
            row.append(inew)
            col.append(jnew)
            # Find the absolute index of the associated auxiliary and get the
            # corresponding term from the full self-interaction EoM matrix.
            data.append(hops.basis.eom.K2_k[permute_aux_col[i], permute_aux_col[j]])
    K0_new2 = sp.sparse.coo_matrix(
        (data, (row, col)), shape=(4, 4), dtype=np.complex128
    ).tocsr()

    assert (K0_new.todense() == K0_new2.todense()).all()


def test_add_self_interaction_remove_aux():
    """
    Tests that _add_self_interaction() produces the correct time-independent
    self-interaction time-evolution matrix a) in general and b) when we remove
    auxiliaries from the basis and add them back in (that is, check accuracy and
    self-consistency of the pyHOPS architecture that manages the self-interaction
    terms).
    """
    # Prepare Constants
    # =================
    n_tot = hops.basis.hierarchy.size

    # Generate the self-interaction terms for the original HOPS object.
    aux_dense_list = [aux.todense() for aux in hops.basis.hierarchy.auxiliary_list]
    state_list = np.array([0, 1], dtype=int)
    lop_list_test = sys_param["L_HIER"]
    g_list_test = [gw[0] for gw in sys_param["GW_SYSBATH"]]
    w_list_test = [gw[1] for gw in sys_param["GW_SYSBATH"]]
    lop_ind_list_test = hops.basis.system.param["LIST_INDEX_L2_BY_HMODE"]
    psi_sb = psi_0[state_list]
    mode_list = hops.basis.mode.list_absindex_mode
    K0_ref, _, _, _ = generate_eom_k_super(state_list, aux_dense_list,
                                              lop_list_test, lop_ind_list_test,
                                        g_list_test, w_list_test, psi_sb, mode_list)

    # Generate a reduced list of auxiliaries
    # --------------
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x)
    )
    stable_aux = set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list)

    # Generate a new HOPS object with only the reduced list of auxiliaries
    hops2 = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops2.initialize(psi_0)
    hops2.basis.hierarchy.auxiliary_list = list(stable_aux)

    # Reconfigure the self-interaction terms using a permutation matrix to remove the
    # terms associated with auxiliaries that were removed from the basis.
    permute_aux_row = []
    permute_aux_col = []
    for aux in stable_aux:
        permute_aux_row.append(
            auxiliary_list_2.index(aux)
        )
        permute_aux_col.append(
            hops.basis.hierarchy.auxiliary_list.index(aux)
        )
    Pmat = sp.sparse.coo_matrix(
        (np.ones(len(permute_aux_row)), (permute_aux_row, permute_aux_col)),
        shape=(68, 70),
        dtype=np.complex128,
    ).tocsc()

    K0_new = _permute_aux_by_matrix(hops.basis.eom.K2_k, Pmat)

    # Add indices corresponding to previously removed auxiliaries back (note that the
    # associated terms are still gone).
    Pmat_t = Pmat.transpose()
    K0_new = _permute_aux_by_matrix(K0_new, Pmat_t)

    # Add the removed auxiliaries back into the hops2 basis and use the
    # _add_self_interactions function to introduce the missing self-interaction terms.
    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list
    K0_data, K0_row, K0_col = _add_self_interactions(
        hops2.basis.system,
        hops2.basis.hierarchy,
        K0_data=[],
        K0_row=[],
        K0_col=[],
    )
    K0 = (
        K0_new
        + sparse.coo_matrix(
            (K0_data, (K0_row, K0_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsr()
    )

    # Test the calculated self-interaction matrix against the one belonging to
    # the HopsTrajectory object with all removed auxiliaries re-introduced, and to the
    # matrix we have calculated.
    assert (K0.todense() == hops2.basis.eom.K2_k.todense()).all()
    np.testing.assert_allclose(K0.todense(), K0_ref)


# noinspection PyTupleAssignmentBalance
def test_add_crossterms():
    """
    Test that the cross-terms, both time-dependent and time-independent, in the
    time-evolution super-operator are properly managed when the auxiliary basis is
    altered.
    """
    # Prepare Constants
    n_site = hops.basis.system.param["NSTATES"]
    n_lop = hops.basis.system.param["N_L2"]
    n_tot = n_site * hops.basis.hierarchy.size
    n_tot2 = hops.basis.hierarchy.size

    # Generate a reduced list of auxiliaries
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x)
    )
    stable_aux = list(set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list))

    # Generate the reference time-evolution operator from the original HopsTrajectory
    # object's data.
    aux_dense_list = [aux.todense() for aux in hops.basis.hierarchy.auxiliary_list]
    state_list = [0, 1]
    lop_list_test = sys_param["L_HIER"]
    lop_ind_list_test = hops.basis.system.param["LIST_INDEX_L2_BY_HMODE"]
    g_list_test = [gw[0] for gw in sys_param["GW_SYSBATH"]]
    w_list_test = [gw[1] for gw in sys_param["GW_SYSBATH"]]
    psi_sb = psi_0[state_list]
    mode_list = hops.basis.mode.list_absindex_mode
    _, Kp1_ref, Zp1_red_ref, Km1_ref = generate_eom_k_super(np.array(
        state_list), aux_dense_list, lop_list_test, lop_ind_list_test, g_list_test,
        w_list_test, psi_sb, mode_list)

    # Generate a new HopsTrajectory with fewer auxiliaries.
    hops2 = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
    )
    hops2.initialize(psi_0)
    hops2.basis.hierarchy.auxiliary_list = stable_aux

    # Re-indexing the Kp1, Km1, and Zp1 matrices of hops with permutation matrices (for
    # both the full basis and the auxiliary basis) to delete the information
    # corresponding to removed auxiliaries.
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
        shape=(68,70),
        dtype=np.complex128,
    ).tocsc()
    Kp1_new = _permute_aux_by_matrix(hops.basis.eom.K2_kp1, Pmat)
    Km1_new = _permute_aux_by_matrix(hops.basis.eom.K2_km1, Pmat)

    Zp1_new = [[] for i in range(n_lop)]
    for i_lop in range(n_lop):
        Zp1_new[i_lop] = _permute_aux_by_matrix(hops.basis.eom.Z2_kp1[i_lop], Pmat2)

    # Add indices back via permutation. We now have Kp1, Km1, and Zp1 in the space of
    # the full basis, but with information corresponding to the deleted auxiliaries
    # lost. This lost data should be recoverable by rewriting the auxiliary_list of
    # hops2 and then using _add_crossterms.
    Pmat = Pmat.transpose()
    Pmat2 = Pmat2.transpose()
    Kp1_new = _permute_aux_by_matrix(Kp1_new, Pmat)
    Km1_new = _permute_aux_by_matrix(Km1_new, Pmat)

    Zp1_new2 = [[] for i in range(n_lop)]
    for i_lop in range(n_lop):
        Zp1_new2[i_lop] = _permute_aux_by_matrix(Zp1_new[i_lop], Pmat2)


    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list
    # Add interactions back to cross-term matrices
    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
    ) = _add_crossterms(
        hops2.basis.system,
        hops2.basis.hierarchy,
        hops2.basis.mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Zp1_data=[[] for i in range(n_lop)],
        Zp1_row=[[] for i in range(n_lop)],
        Zp1_col=[[] for i in range(n_lop)],
        Km1_data=[],
    )

    Kp1 = (
        Kp1_new
        + sparse.coo_matrix(
            (Kp1_data, (Kp1_row, Kp1_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsr()
    )
    Km1 = (
        Km1_new
        + sparse.coo_matrix(
            (Km1_data, (Kp1_col, Kp1_row)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsr()
    )
    Zp1 = [
        Zp1_new2[i]
        + sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(n_tot2, n_tot2),
            dtype=np.complex128,
        ).tocsr()
        for i in range(n_lop)
    ]

    # Test that the generated cross-terms match our "manual" reference as well as the
    # super-operator components generated by the original HopsTrajectory
    np.testing.assert_allclose(Kp1.toarray(), Kp1_ref)
    np.testing.assert_allclose(Km1.toarray(), Km1_ref)
    for (i,m) in enumerate(hops.basis.mode.list_absindex_L2):
        np.testing.assert_allclose(Zp1[i].toarray(), Zp1_red_ref[m])
    assert (Kp1.todense() == hops.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops.basis.eom.K2_km1.todense()).all()
    for (i,m) in enumerate(hops.basis.mode.list_absindex_L2):
        assert (Zp1[i].todense() == hops.basis.eom.Z2_kp1[i].todense()).all()

# Very generalized L-operators for tests not in a special case
l0 = np.zeros([10,10], dtype=np.complex128)
l0[0,0] = 1
l0[0,1] = 1+1j
l0[1,0] = 1-1j
l0[1,1] = -2
l0[2,2] = 3
l0[3,3] = -4
l0 = sp.sparse.coo_matrix(l0)
l2 = np.zeros([10, 10], dtype=np.complex128)
l2[1, 1] = 1
l2[1, 4] = 1j
l2[4, 1] = -1j
l2[4, 4] = -2
l2[5, 5] = 3
l2[6, 6] = -4
l2 = sp.sparse.coo_matrix(l2)
l4 = np.zeros([10, 10], dtype=np.complex128)
l4[2, 2] = 1
l4[2, 5] = -1j
l4[5, 2] = 1j
l4[5, 5] = -2
l4[7, 7] = 3
l4[8, 8] = -4
l4 = sp.sparse.coo_matrix(l4)
l6 = np.zeros([10, 10], dtype=np.complex128)
l6[3, 3] = 1
l6[3, 6] = -1j
l6[6, 3] = 1j
l6[6, 6] = -2
l6[8, 8] = 3
l6[9, 9] = -4
l6 = sp.sparse.coo_matrix(l6)

def test_add_crossterms_arbitrary_lop():
    """
    Test that the cross-terms, both time-dependent and time-independent, in the
    time-evolution super-operator are properly managed when the auxiliary basis is
    altered, even when the L-operators of interest have a complex structure.
    """

    noise_param = {

        "SEED": 0,  # This sets the seed for the noise
        "MODEL": "FFT_FILTER", # This sets the noise model to be used
        "TLEN": 4000.0, # Units: fs (the total time length of the noise trajectory)
        "TAU": 1.0, # Units: fs  (the time-step resolution of the noise trajectory
    }
    # 2-particle system with 4 sites and 10 states.
    nsite = 4
    kmax = 4
    e_lambda = 65.
    gamma = 53.0
    temp = 300.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)
    numsites2 = nsite*(nsite+1)/2
    gw_sysbath = []
    for i in range(nsite):
        gw_sysbath.append([g_0, w_0])
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])

    lop_list = [l0,l0,l2,l2,l4,l4,l6,l6]
    # Generate a 2-particle Hamiltonian.
    hs = np.zeros([int(numsites2), int(numsites2)])
    for site in range(int(numsites2-1)):
        if site%2 == 0:
            hs[site][site+1] = 40
            hs[site+1][site] = 40
        else:
            hs[site][site+1] = 40
            hs[site+1][site] = 40

    # System parameters
    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),  # the Hamiltonian we constructed
        "GW_SYSBATH": gw_sysbath,  # defines exponential decomposition of correlation function
        "L_HIER": lop_list,  # list of L operators
        "L_NOISE1": lop_list,  # list of noise params associated with noise1
        "ALPHA_NOISE1": bcf_exp,  # function that calculates correlation function
        "PARAM_NOISE1": gw_sysbath,  # list of noise pararms defining decomposition of noise1
    }

    # EOM parameters
    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"} # we generally pick normalized nonlinear
    # as it has better convergence properties than the linear eom

    # Integration parameters
    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA"}  # We use a Runge-Kutta method for our integrator

    # Initial wave function (in the state basis, we fully populate state 3 and no
    # others)
    psi_0 = np.array([0.0] * int(numsites2), dtype=np.complex128)
    psi_0[2] = 1.0

    hops2p = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": kmax},
        eom_param=eom_param,
    )
    hops2p.make_adaptive(0.01, 0.01)
    hops2p.initialize(psi_0)

    #Test 1:  Add Auxiliaries, Full State, Mode Bases
    n_lop = 8
    n_state = 10
    n_hier = 8
    hops2p.basis.system.state_list = [0,1,2,3,4,5,6,7,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]

    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]

    # "Manually" calculate the cross-terms
    aux_dense_list = [aux.todense() for aux in hops2p.basis.hierarchy.auxiliary_list]
    state_list = hops2p.basis.system.state_list
    lop_list_test = [L2.toarray() for L2 in sys_param["L_HIER"]]
    lop_ind_list_test = hops2p.basis.system.param["LIST_INDEX_L2_BY_HMODE"]
    g_list_test = [gw[0] for gw in sys_param["GW_SYSBATH"]]
    w_list_test = [gw[1] for gw in sys_param["GW_SYSBATH"]]
    psi_sb = psi_0[state_list]
    mode_list = hops2p.basis.mode.list_absindex_mode
    _, Kp1_ref, Zp1_red_ref, Km1_ref = generate_eom_k_super(np.array(
        state_list), aux_dense_list, lop_list_test, lop_ind_list_test, g_list_test,
        w_list_test, psi_sb, mode_list)

    # Get the cross terms of the time-derivative super-operator.
    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
    ) = _add_crossterms(
        hops2p.basis.system,
        hops2p.basis.hierarchy,
        hops2p.basis.mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Zp1_data=[[] for i in range(n_lop)],
        Zp1_row=[[] for i in range(n_lop)],
        Zp1_col=[[] for i in range(n_lop)],
        Km1_data=[],
    )

    # Build Computed Kp1,Km1 and test against "manually" calculated version.
    Kp1 = sp.sparse.coo_matrix((Kp1_data, (Kp1_row, Kp1_col)),
                               shape=(n_state * n_hier, n_state * n_hier),
                               dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data, (Kp1_col, Kp1_row)),
                               shape=(n_state * n_hier, n_state * n_hier),
                               dtype=np.complex64).toarray()
    np.testing.assert_allclose(Kp1, Kp1_ref)
    np.testing.assert_allclose(Km1, Km1_ref)

    # Build calculated Zp1 for each L-operator and test against "manually"-generated
    # version.
    Zp1_0 = sp.sparse.coo_matrix((Zp1_data[0],(Zp1_row[0],Zp1_col[0])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_1 = sp.sparse.coo_matrix((Zp1_data[1],(Zp1_row[1],Zp1_col[1])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_2 = sp.sparse.coo_matrix((Zp1_data[2],(Zp1_row[2],Zp1_col[2])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_3 = sp.sparse.coo_matrix((Zp1_data[3],(Zp1_row[3],Zp1_col[3])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()

    np.testing.assert_allclose(Zp1_0, Zp1_red_ref[0])
    np.testing.assert_allclose(Zp1_1, Zp1_red_ref[1])
    np.testing.assert_allclose(Zp1_2, Zp1_red_ref[2])
    np.testing.assert_allclose(Zp1_3, Zp1_red_ref[3])

    # Test 2: Partial State Basis, Full Mode Basis
    n_lop = 8
    n_state = 7
    n_hier = 8
    # Alter the bases of the HopsTrajectory object
    hops2p.basis.system.state_list = [0,1,3,5,6,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]

    # Overwrite auxiliary list so that all auxiliaries are "new" and _add_crossterms
    # actually gets all the crossterms.
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]

    # Generate new reference matrices
    aux_dense_list = [aux.todense() for aux in hops2p.basis.hierarchy.auxiliary_list]
    state_list = hops2p.basis.system.state_list
    lop_list_test = [L2.toarray() for L2 in sys_param["L_HIER"]]
    lop_ind_list_test = hops2p.basis.system.param["LIST_INDEX_L2_BY_HMODE"]
    g_list_test = [gw[0] for gw in sys_param["GW_SYSBATH"]]
    w_list_test = [gw[1] for gw in sys_param["GW_SYSBATH"]]
    psi_sb = psi_0[state_list]
    mode_list = hops2p.basis.mode.list_absindex_mode
    K0_ref, Kp1_ref, Zp1_red_ref, Km1_ref = generate_eom_k_super(np.array(
        state_list), aux_dense_list, lop_list_test, lop_ind_list_test, g_list_test,
        w_list_test, psi_sb, mode_list)

    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
    ) = _add_crossterms(
        hops2p.basis.system,
        hops2p.basis.hierarchy,
        hops2p.basis.mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Zp1_data=[[] for i in range(n_lop)],
        Zp1_row=[[] for i in range(n_lop)],
        Zp1_col=[[] for i in range(n_lop)],
        Km1_data=[],
    )
    # Build Computed Kp1,Km1 and test against "manually" calculated version.
    Kp1 = sp.sparse.coo_matrix((Kp1_data, (Kp1_row, Kp1_col)),
                               shape=(n_state * n_hier, n_state * n_hier),
                               dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data, (Kp1_col, Kp1_row)),
                               shape=(n_state * n_hier, n_state * n_hier),
                               dtype=np.complex64).toarray()
    np.testing.assert_allclose(Kp1, Kp1_ref)
    np.testing.assert_allclose(Km1, Km1_ref)

    # Build calculated Zp1 for each L-operator and test against "manually"-generated
    # version.
    Zp1_0 = sp.sparse.coo_matrix((Zp1_data[0], (Zp1_row[0], Zp1_col[0])),
                                 shape=(n_hier, n_hier), dtype=np.complex64).toarray()
    Zp1_1 = sp.sparse.coo_matrix((Zp1_data[1], (Zp1_row[1], Zp1_col[1])),
                                 shape=(n_hier, n_hier), dtype=np.complex64).toarray()
    Zp1_2 = sp.sparse.coo_matrix((Zp1_data[2], (Zp1_row[2], Zp1_col[2])),
                                 shape=(n_hier, n_hier), dtype=np.complex64).toarray()
    Zp1_3 = sp.sparse.coo_matrix((Zp1_data[3], (Zp1_row[3], Zp1_col[3])),
                                 shape=(n_hier, n_hier), dtype=np.complex64).toarray()

    np.testing.assert_allclose(Zp1_0, Zp1_red_ref[0])
    np.testing.assert_allclose(Zp1_1, Zp1_red_ref[1])
    np.testing.assert_allclose(Zp1_2, Zp1_red_ref[2])
    np.testing.assert_allclose(Zp1_3, Zp1_red_ref[3])

    # Test 3: Partial State Basis, Partial Mode Basis
    # Alter the bases of the HopsTrajectory object. One L operator has no populated
    # states associated with it but is present because of a mode represented in
    # auxiliaries.
    n_lop = 8
    n_state = 2
    n_hier = 8
    hops2p.basis.system.state_list = [1,5]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6]

    # Overwrite auxiliary list so that all auxiliaries are "new" and _add_crossterms
    # actually gets all the crossterms.
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]

    # Generate new reference matrices
    aux_dense_list = [aux.todense() for aux in hops2p.basis.hierarchy.auxiliary_list]
    state_list = hops2p.basis.system.state_list
    lop_list_test = [L2.toarray() for L2 in sys_param["L_HIER"]]
    lop_ind_list_test = hops2p.basis.system.param["LIST_INDEX_L2_BY_HMODE"]
    g_list_test = [gw[0] for gw in sys_param["GW_SYSBATH"]]
    w_list_test = [gw[1] for gw in sys_param["GW_SYSBATH"]]
    psi_sb = psi_0[state_list]
    mode_list = hops2p.basis.mode.list_absindex_mode
    K0_ref, Kp1_ref, Zp1_red_ref, Km1_ref = generate_eom_k_super(np.array(
        state_list), aux_dense_list, lop_list_test, lop_ind_list_test, g_list_test,
        w_list_test, psi_sb, mode_list)

    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Zp1_data,
        Zp1_row,
        Zp1_col,
        Km1_data,
    ) = _add_crossterms(
        hops2p.basis.system,
        hops2p.basis.hierarchy,
        hops2p.basis.mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Zp1_data=[[] for i in range(n_lop)],
        Zp1_row=[[] for i in range(n_lop)],
        Zp1_col=[[] for i in range(n_lop)],
        Km1_data=[],
    )

    # Build Computed Kp1,Km1 and test against "manually" calculated version.
    Kp1 = sp.sparse.coo_matrix((Kp1_data, (Kp1_row, Kp1_col)),
                               shape=(n_state * n_hier, n_state * n_hier),
                               dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data, (Kp1_col, Kp1_row)),
                               shape=(n_state * n_hier, n_state * n_hier),
                               dtype=np.complex64).toarray()
    np.testing.assert_allclose(Kp1, Kp1_ref)
    np.testing.assert_allclose(Km1, Km1_ref)

    # Build calculated Zp1 for each L-operator and test against "manually"-generated
    # version.
    Zp1_0 = sp.sparse.coo_matrix((Zp1_data[0], (Zp1_row[0], Zp1_col[0])),
                                 shape=(n_hier, n_hier), dtype=np.complex64).toarray()
    Zp1_1 = sp.sparse.coo_matrix((Zp1_data[1], (Zp1_row[1], Zp1_col[1])),
                                 shape=(n_hier, n_hier), dtype=np.complex64).toarray()
    Zp1_2 = sp.sparse.coo_matrix((Zp1_data[2], (Zp1_row[2], Zp1_col[2])),
                                 shape=(n_hier, n_hier), dtype=np.complex64).toarray()
    Zp1_3 = sp.sparse.coo_matrix((Zp1_data[3], (Zp1_row[3], Zp1_col[3])),
                                 shape=(n_hier, n_hier), dtype=np.complex64).toarray()

    np.testing.assert_allclose(Zp1_0, Zp1_red_ref[0])
    np.testing.assert_allclose(Zp1_1, Zp1_red_ref[1])
    np.testing.assert_allclose(Zp1_2, Zp1_red_ref[2])
    np.testing.assert_allclose(Zp1_3, Zp1_red_ref[3])


def test_add_crossterms_stable_arbitrary_lop():
    """
    Test that the cross-terms, both time-dependent and time-independent, in the
    time-evolution super-operator are properly managed when the state basis is altered,
    even when the L-operators have a complex structure.
    """
    # Define a HOPS object for a 2-particle system as in the test above.
    noise_param = {

        "SEED": 0,  # This sets the seed for the noise
        "MODEL": "FFT_FILTER", # This sets the noise model to be used
        "TLEN": 4000.0, # Units: fs (the total time length of the noise trajectory)
        "TAU": 1.0, # Units: fs  (the time-step resolution of the noise trajectory
    }
    nsite = 4
    kmax = 4
    e_lambda = 65.
    gamma = 53.0
    temp = 300.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)
    numsites2 = nsite*(nsite+1)/2
    gw_sysbath = []
    for i in range(nsite):
        gw_sysbath.append([g_0, w_0])
        gw_sysbath.append([-1j * np.imag(g_0), 500.0])
    lop_list = [l0,l0,l2,l2,l4,l4,l6,l6]
    hs = np.zeros([int(numsites2), int(numsites2)])
    for site in range(int(numsites2-1)):
        if site%2 == 0:
            hs[site][site+1] = 40
            hs[site+1][site] = 40
        else:
            hs[site][site+1] = 40
            hs[site+1][site] = 40

    # System parameters
    sys_param = {
        "HAMILTONIAN": np.array(hs, dtype=np.complex128),  # the Hamiltonian we constructed
        "GW_SYSBATH": gw_sysbath,  # defines exponential decomposition of correlation function
        "L_HIER": lop_list,  # list of L operators
        "L_NOISE1": lop_list,  # list of noise params associated with noise1
        "ALPHA_NOISE1": bcf_exp,  # function that calculates correlation function
        "PARAM_NOISE1": gw_sysbath,  # list of noise pararms defining decomposition of noise1
    }

    # EOM parameters
    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    # Initial wave function (we fully populate state 3 and no others)
    psi_0 = np.array([0.0] * int(numsites2), dtype=np.complex128)
    psi_0[2] = 1.0

    hops2p = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param={"MAXHIER": kmax},
        eom_param=eom_param,
    )
    hops2p.make_adaptive(0.01, 0.01)
    hops2p.initialize(psi_0)

    # Test 1: Redefine the adaptive basis with no new states - add_crossterms_stable
    # should do nothing!
    hops2p.basis.system.state_list = [0,1,2,3,4,5,6,7,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]

    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],hops2p.basis.hierarchy.auxiliary_list[1],hops2p.basis.hierarchy.auxiliary_list[2],
                                             hops2p.basis.hierarchy.auxiliary_list[3],hops2p.basis.hierarchy.auxiliary_list[4],hops2p.basis.hierarchy.auxiliary_list[5], AuxiliaryVector([(6,1)], 8)]
    # No new states added or removed
    hops2p.basis.system.state_list = [0,1,2,3,4,5,6,7,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]
    n_state = len(hops2p.basis.system.state_list)
    n_hier = len(hops2p.basis.hierarchy.auxiliary_list)

    # Use the add_crossterms_stable_K function
    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Km1_data,
    ) = _add_crossterms_stable_K(
        hops2p.basis.system,
        hops2p.basis.hierarchy,
        hops2p.basis.mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Km1_data=[],
    )

    # Build Computed Kp1,Km1 and test that they are empty
    Kp1 = sp.sparse.coo_matrix((Kp1_data,(Kp1_row,Kp1_col)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data,(Kp1_col,Kp1_row)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    np.testing.assert_allclose(Kp1, np.zeros_like(Kp1))
    np.testing.assert_allclose(Km1, np.zeros_like(Km1))

    # Test 2: Redefine the adaptive basis with new states but no new modes
    hops2p.basis.system.state_list = [0,1,3,5,6,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],hops2p.basis.hierarchy.auxiliary_list[1],hops2p.basis.hierarchy.auxiliary_list[2],
                                             hops2p.basis.hierarchy.auxiliary_list[3],hops2p.basis.hierarchy.auxiliary_list[4],hops2p.basis.hierarchy.auxiliary_list[5],
                                             hops2p.basis.hierarchy.auxiliary_list[6],hops2p.basis.hierarchy.auxiliary_list[7]]
    # Rewrite the state list - added states 2, 4, 7 (same absolute and relative index).
    hops2p.basis.system.state_list = [0,1,2,3,4,5,6,7,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]
    n_state = len(hops2p.basis.system.state_list)
    n_hier = len(hops2p.basis.hierarchy.auxiliary_list)

    # Get the list of aux, states, l-operators, g, w, modes, etc. to generate the
    # components of the K super-operator.
    aux_dense_list = [aux.todense() for aux in hops2p.basis.hierarchy.auxiliary_list]
    state_list = hops2p.basis.system.state_list
    lop_list_test = [L2.toarray() for L2 in sys_param["L_HIER"]]
    lop_ind_list_test = hops2p.basis.system.param["LIST_INDEX_L2_BY_HMODE"]
    g_list_test = [gw[0] for gw in sys_param["GW_SYSBATH"]]
    w_list_test = [gw[1] for gw in sys_param["GW_SYSBATH"]]
    psi_sb = psi_0[state_list]
    mode_list = hops2p.basis.mode.list_absindex_mode

    _, Kp1_ref, _, Km1_ref = (
        generate_eom_k_super(
        np.array(
        state_list), aux_dense_list, lop_list_test, lop_ind_list_test, g_list_test,
        w_list_test, psi_sb, mode_list))

    # Find the list of element indices in the adaptive basis that correspond to
    # newly-added states
    list_added_basis = []
    for i in range(n_hier):
        base = i * n_state
        list_added_basis = list_added_basis + [base + 2, base + 4, base + 7]

    def check_in_added_states(basis_index):
        return basis_index in list_added_basis

    # Find the list of nonzero elements in the super-operator components of interest
    # that correspond to the new states. Indices i correspond to the ith data point
    # in a matrix in terms of the np.where() indexing.
    list_ind_new_state_elements_Kp1 = [i for i in range(len(np.where(Kp1_ref)[0])) if
                                       check_in_added_states(np.where(Kp1_ref)[0][i]) or
                                       check_in_added_states(np.where(Kp1_ref)[1][i])]

    list_ind_new_state_elements_Km1 = [i for i in range(len(np.where(Km1_ref)[0])) if
                                       check_in_added_states(np.where(Km1_ref)[0][i]) or
                                       check_in_added_states(np.where(Km1_ref)[1][i])]

    # Subset the reference Kp1 and Km1 to only include the elements that correspond
    # to new states.
    Kp1_ref_new = np.zeros_like(Kp1_ref)
    for i in list_ind_new_state_elements_Kp1:
        Kp1_ref_new[np.where(Kp1_ref)[0][i], np.where(Kp1_ref)[1][i]] = Kp1_ref[
            np.where(Kp1_ref)[0][i], np.where(Kp1_ref)[1][i]]

    Km1_ref_new = np.zeros_like(Km1_ref)
    for i in list_ind_new_state_elements_Km1:
        Km1_ref_new[np.where(Km1_ref)[0][i], np.where(Km1_ref)[1][i]] = Km1_ref[
            np.where(Km1_ref)[0][i], np.where(Km1_ref)[1][i]]

    # Use the add_crossterms_stable_K function to get crossterms associated with new
    # states.
    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Km1_data,
    ) = _add_crossterms_stable_K(
        hops2p.basis.system,
        hops2p.basis.hierarchy,
        hops2p.basis.mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Km1_data=[],
    )

    # Build computed Kp1,Km1 and test against reference.
    Kp1_new = sp.sparse.coo_matrix((Kp1_data,(Kp1_row,Kp1_col)),shape=(n_state*n_hier,
                                                                    n_state*n_hier),dtype=np.complex64).toarray()
    Km1_new = sp.sparse.coo_matrix((Km1_data,(Kp1_col,Kp1_row)),shape=(n_state*n_hier,
                                                                    n_state*n_hier),dtype=np.complex64).toarray()
    np.testing.assert_allclose(Kp1_ref_new, Kp1_new)
    np.testing.assert_allclose(Km1_ref_new, Km1_new)

    # Test 3: Redefine the adaptive basis with new states and new modes, where the
    # relative and absolute state indices are not the same.
    hops2p.basis.system.state_list = [1,5]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6]

    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],hops2p.basis.hierarchy.auxiliary_list[1],hops2p.basis.hierarchy.auxiliary_list[2],
                                             hops2p.basis.hierarchy.auxiliary_list[3],hops2p.basis.hierarchy.auxiliary_list[4],hops2p.basis.hierarchy.auxiliary_list[5],
                                             hops2p.basis.hierarchy.auxiliary_list[6],hops2p.basis.hierarchy.auxiliary_list[7]]
    # New states = 0,3,6,8,9
    # Relative indices of new states = 0,2,4,5,6
    # New modes = 7
    # Rewrite the state list - added states 0, 3, 6, 8, 9 (relative indices 0, 2, 4,
    # 5, 6), and added mode 7 to the basis as well. Note that we don't have to
    # consider new connections between existing auxiliaries, as a mode MUST be in the
    # basis if it's represented in any auxiliary indexing vector!
    hops2p.basis.system.state_list = [0,1,3,5,6,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]

    n_state = len(hops2p.basis.system.state_list)
    n_hier = len(hops2p.basis.hierarchy.auxiliary_list)

    # Get the list of aux, states, l-operators, g, w, modes, etc. to generate the
    # components of the K super-operator.
    aux_dense_list = [aux.todense() for aux in hops2p.basis.hierarchy.auxiliary_list]
    state_list = hops2p.basis.system.state_list
    lop_list_test = [L2.toarray() for L2 in sys_param["L_HIER"]]
    lop_ind_list_test = hops2p.basis.system.param["LIST_INDEX_L2_BY_HMODE"]
    g_list_test = [gw[0] for gw in sys_param["GW_SYSBATH"]]
    w_list_test = [gw[1] for gw in sys_param["GW_SYSBATH"]]
    psi_sb = psi_0[state_list]
    mode_list = hops2p.basis.mode.list_absindex_mode

    _, Kp1_ref, _, Km1_ref = (
        generate_eom_k_super(
            np.array(state_list), aux_dense_list, lop_list_test, lop_ind_list_test,
            g_list_test, w_list_test, psi_sb, mode_list))

    # Find the list of element indices in the adaptive basis that correspond to
    # newly-added states. This will be in terms of relative indices.
    list_added_basis = []
    for i in range(n_hier):
        base = i * n_state
        list_added_basis = list_added_basis + [base + 0, base + 2, base + 4,
                                               base + 5, base + 6]

    def check_in_added_states(basis_index):
        return basis_index in list_added_basis

    # Find the list of nonzero elements in the super-operator components of interest
    # that correspond to the new states.
    list_ind_new_state_elements_Kp1 = [i for i in range(len(np.where(Kp1_ref)[0])) if
                                       check_in_added_states(np.where(Kp1_ref)[0][i]) or
                                       check_in_added_states(np.where(Kp1_ref)[1][i])]

    list_ind_new_state_elements_Km1 = [i for i in range(len(np.where(Km1_ref)[0])) if
                                       check_in_added_states(np.where(Km1_ref)[0][i]) or
                                       check_in_added_states(np.where(Km1_ref)[1][i])]

    # Subset the reference Kp1 and Km1 to only include the elements that correspond
    # to new states.
    Kp1_ref_new = np.zeros_like(Kp1_ref)
    for i in list_ind_new_state_elements_Kp1:
        Kp1_ref_new[np.where(Kp1_ref)[0][i], np.where(Kp1_ref)[1][i]] = Kp1_ref[
            np.where(Kp1_ref)[0][i], np.where(Kp1_ref)[1][i]]

    Km1_ref_new = np.zeros_like(Km1_ref)
    for i in list_ind_new_state_elements_Km1:
        Km1_ref_new[np.where(Km1_ref)[0][i], np.where(Km1_ref)[1][i]] = Km1_ref[
            np.where(Km1_ref)[0][i], np.where(Km1_ref)[1][i]]

    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Km1_data,
    ) = _add_crossterms_stable_K(
        hops2p.basis.system,
        hops2p.basis.hierarchy,
        hops2p.basis.mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Km1_data=[],
    )

    # Build computed Kp1, Km1 and test against reference.
    Kp1 = sp.sparse.coo_matrix((Kp1_data,(Kp1_row,Kp1_col)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data,(Kp1_col,Kp1_row)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    np.testing.assert_allclose(Kp1_ref_new, Kp1)
    np.testing.assert_allclose(Km1_ref_new, Km1)

def test_matrix_updates_with_missing_aux_and_states():
    """
    Test the matrix update functions when aux and states are removed.
    """
    # Prepare Constants
    # =================
    n_site = hops.basis.system.param["NSTATES"]
    n_lop = hops.basis.system.param["N_L2"]
    n_mode = hops.basis.system.param["N_HMODES"]
    n_tot = n_site * hops.basis.hierarchy.size
    n_tot2 = hops.basis.hierarchy.size

    # Determine removed indices
    # -------------------------
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x)
    )
    stable_aux = list(set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list))

    hops2 = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
    )
    hops2.initialize(psi_0)
    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list
    hops2.basis.hierarchy.auxiliary_list = stable_aux
    state_list_2 = [
        hops.basis.system.state_list[i] for i in range(hops.n_state) if i > 0
    ]
    hops2.basis.system.state_list = state_list_2
    hops2.basis.mode.list_absindex_mode = list(range(n_mode))
    # state_list_2 = [1]
    stable_state = state_list_2
    list_ilop_rel_stable = np.arange(len(hops.basis.mode.list_absindex_L2))


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
    Zp1_new = [_permute_aux_by_matrix(hops.basis.eom.Z2_kp1[0], Pmat2),
               _permute_aux_by_matrix(hops.basis.eom.Z2_kp1[1], Pmat2)]

    # Now attempt to add states and auxiliaries back
    # ==============================================
    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list
    hops2.basis.system.state_list = hops.basis.system.state_list
    hops2.basis.mode.list_absindex_mode = list(hops.basis.mode.list_absindex_mode)

    # Add indices
    # --------------
    # Using permutation matrix
    Pmat = Pmat.transpose()
    Pmat2 = Pmat2.transpose()
    Kp1_new = _permute_aux_by_matrix(Kp1_new, Pmat)
    Km1_new = _permute_aux_by_matrix(Km1_new, Pmat)
    K0_new = _permute_aux_by_matrix(K0_new, Pmat2)

    Zp1_new2 = [
        _permute_aux_by_matrix(Zp1_new[0], Pmat2),
        _permute_aux_by_matrix(Zp1_new[1], Pmat2),
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
    ) = _add_crossterms(
        hops2.basis.system,
        hops2.basis.hierarchy,
        hops2.basis.mode,
        Kp1_data=[],
        Kp1_row=[],
        Kp1_col=[],
        Zp1_data=[[] for i in range(n_lop)],
        Zp1_row=[[] for i in range(n_lop)],
        Zp1_col=[[] for i in range(n_lop)],
        Km1_data=[],
    )

    (
        Kp1_data,
        Kp1_row,
        Kp1_col,
        Km1_data,
    ) = _add_crossterms_stable_K(
        system=hops2.basis.system,
        mode = hops2.basis.mode,
        hierarchy = hops2.basis.hierarchy,
        Kp1_data=Kp1_data,
        Kp1_row=Kp1_row,
        Kp1_col=Kp1_col,
        Km1_data=Km1_data,
    )
    # Add back self interactions
    # ---------------------------
    K0_data, K0_row, K0_col = _add_self_interactions(
        hops2.basis.system,
        hops2.basis.hierarchy,
        K0_data=[],
        K0_row=[],
        K0_col=[],
    )

    Kp1 = (
        Kp1_new
        + sparse.coo_matrix(
            (Kp1_data, (Kp1_row, Kp1_col)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsr()
    )
    Km1 = (
        Km1_new
        + sparse.coo_matrix(
            (Km1_data, (Kp1_col, Kp1_row)), shape=(n_tot, n_tot), dtype=np.complex128
        ).tocsr()
    )
    Zp1 = [
        Zp1_new2[i]
        + sparse.coo_matrix(
            (Zp1_data[i], (Zp1_row[i], Zp1_col[i])),
            shape=(n_tot2, n_tot2),
            dtype=np.complex128,
        ).tocsr()
        for i in range(n_lop)
    ]
    K0 = (
        K0_new
        + sparse.coo_matrix(
            (K0_data, (K0_row, K0_col)), shape=(n_tot2, n_tot2), dtype=np.complex128
        ).tocsr()
    )
    assert (Kp1.todense() == hops.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops.basis.eom.K2_km1.todense()).all()
    assert (Zp1[0].todense() == hops.basis.eom.Z2_kp1[0].todense()).all()
    assert (Zp1[1].todense() == hops.basis.eom.Z2_kp1[1].todense()).all()
    assert (K0.todense() == hops.basis.eom.K2_k.todense()).all()


def test_update_super_remove_aux():
    """
    Test update_ksuper() when only auxiliaries are removed.
    """
    # Prepare Constants
    # -----------------
    n_lop = hops.basis.system.param["N_L2"]
    n_mode = hops.basis.system.param["N_HMODES"]
    # Remove Auxiliaries
    # ------------------
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x)
    )
    stable_aux = list(set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list))

    hops2 = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
    )
    hops2.initialize(psi_0)
    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list

    hops2.basis.hierarchy.auxiliary_list = stable_aux

    hops2.basis.system.state_list = [0,1]
    hops2.basis.system.state_list = [0,1]
    hops2.basis.mode.list_absindex_mode = [0,1,2,3]

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
    K0_new = _permute_aux_by_matrix(hops2.basis.eom.K2_k, Pmat2)
    Kp1_new = _permute_aux_by_matrix(hops2.basis.eom.K2_kp1, Pmat)
    Km1_new = _permute_aux_by_matrix(hops2.basis.eom.K2_km1, Pmat)
    Zp1_new = [[] for i in range(n_lop)]
    for i_lop in range(n_lop):
        Zp1_new[i_lop] = _permute_aux_by_matrix(hops2.basis.eom.Z2_kp1[i_lop], Pmat2)

    list_stable_aux_old_index = list(np.arange(68))
    list_stable_aux_new_index = list(set(np.arange(70)) - set([2,3]))

    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list

    K0, Kp1, Zp1, Km1, masks = update_ksuper(
        K0_new,
        Kp1_new,
        Zp1_new,
        Km1_new,
        hops2.basis.system,
        hops2.basis.hierarchy,
        hops2.basis.mode,
        [permute_aux_col, permute_aux_row, list_stable_aux_old_index, list_stable_aux_new_index, 68],
    )

    assert (K0.todense() == hops.basis.eom.K2_k.todense()).all()
    assert (Kp1.todense() == hops.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops.basis.eom.K2_km1.todense()).all()
    for i in range(n_lop):
        assert (Zp1[i].todense() == hops.basis.eom.Z2_kp1[i].todense()).all()


def test_update_super_remove_aux_and_state():
    """
    Test update_ksuper when both auxiliaries and states are removed.
    """
    # Prepare Constants
    # =================
    n_site = hops.basis.system.param["NSTATES"]
    n_mode = hops.basis.system.param["N_HMODES"]
    n_lop = hops.basis.system.param["N_L2"]

    # Remove indices
    # --------------
    auxiliary_list_2 = [
        hops.basis.hierarchy.auxiliary_list[i]
        for i in range(len(hops.basis.hierarchy.auxiliary_list))
        if i != 2 and i != 3
    ]
    auxiliary_list_2.sort(
        key=lambda x: hops.basis.hierarchy._aux_index(x)
    )
    stable_aux = list(set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list))

    hops2 = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param=hier_param,
    eom_param=eom_param,
    integration_param=integrator_param,
    )
    hops2.initialize(psi_0)
    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list

    hops2.basis.hierarchy.auxiliary_list = stable_aux

    hops2.basis.system.state_list = [1]
    hops2.basis.mode.list_absindex_mode = [0,1,2,3]

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
    Zp1_new = [_permute_aux_by_matrix(hops.basis.eom.Z2_kp1[0], Pmat2),
               _permute_aux_by_matrix(hops.basis.eom.Z2_kp1[1], Pmat2)]

    list_stable_aux_old_index = list(np.arange(68))
    list_stable_aux_new_index = list(set(np.arange(70)) - set([2,3]))

    # NOTE: This test breaks because we are not updating the system,
    # hierarchy, and mode objects prior to calling update_ksuper.
    
    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list
    hops2.basis.system.state_list = [0,1]
    hops2.basis.mode.list_absindex_mode = [0,1,2,3]

    # Note that update_ksuper should do all permutations into the new basis.
    K0, Kp1, Zp1, Km1, masks = update_ksuper(
        K0_new,
        Kp1_new,
        Zp1_new,
        Km1_new,
        hops2.basis.system,
        hops2.basis.hierarchy,
        hops2.basis.mode,
        [permute_aux_col, permute_aux_row, list_stable_aux_old_index, list_stable_aux_new_index, 68],
    )
    assert (Kp1.todense() == hops.basis.eom.K2_kp1.todense()).all()
    assert (Km1.todense() == hops.basis.eom.K2_km1.todense()).all()
    assert (K0.todense() == hops.basis.eom.K2_k.todense()).all()
    for i in range(n_lop):
        assert (Zp1[i].todense() == hops.basis.eom.Z2_kp1[i].todense()).all()
