import numpy as np
import scipy as sp
from scipy import sparse
from mesohops.dynamics.hops_trajectory import HopsTrajectory as HOPS
from mesohops.dynamics.bath_corr_functions import bcf_exp, bcf_convert_sdl_to_exp
from mesohops.dynamics.hops_aux import AuxiliaryVector as AuxiliaryVector
from mesohops.dynamics.eom_hops_ksuper import (
    _permute_aux_by_matrix,
    _add_self_interactions,
    _add_crossterms,
    _add_crossterms_stable_K,
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
        key=lambda x: hops.basis.hierarchy._aux_index(x)
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
    stable_aux = set(auxiliary_list_2) & set(hops.basis.hierarchy.auxiliary_list)

    hops2 = HOPS(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
    )
    hops2.initialize(psi_0)
    hops2.basis.hierarchy.auxiliary_list = list(stable_aux)

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

    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list

        # Add back interactions
    # ---------------------
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
        ).tocsc()
    )

    assert (K0.todense() == hops2.basis.eom.K2_k.todense()).all()


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
    hops2.basis.hierarchy.auxiliary_list = stable_aux
    
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


    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list
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
        ).tocsc()
    )
    Km1 = (
        Km1_new
        + sparse.coo_matrix(
            (Km1_data, (Kp1_col, Kp1_row)), shape=(n_tot, n_tot), dtype=np.complex128
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
  
    
def test_add_crossterms_2part():

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
    t_max = 2000.0
    t_step = 2.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)  
    numsites2 = nsite*(nsite+1)/2
    loperator = np.zeros([nsite, int(numsites2), int(numsites2)], dtype=np.float64) 
    gw_sysbath = []  
    for i in range(nsite):
        gw_sysbath.append([g_0, w_0])   
        gw_sysbath.append([-1j * np.imag(g_0), 500.0]) 
    l0 = sp.sparse.coo_matrix(np.diag([1,-2,3,-4,0,0,0,0,0,0]))
    l2 = sp.sparse.coo_matrix(np.diag([0,1,0,0,-2,3,-4,0,0,0]))
    l4 = sp.sparse.coo_matrix(np.diag([0,0,1,0,0,-2,0,3,-4,0]))
    l6 = sp.sparse.coo_matrix(np.diag([0,0,0,1,0,0,-2,0,3,-4]))
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
    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"} # we generally pick normalized nonlinear 
    # as it has better convergence properties than the linear eom

    # Integration parameters 
    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA"}  # We use a Runge-Kutta method for our integrator 

    # Initial wave function (in the state basis, we fully populate site 3 and no others)
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
    
    #There are 10 states, so each auxiliary starts at index with ones digit equal to 0.  Auxiliary 2, state 5 = index 25.
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
    #Build Computed Kp1,Km1
    Kp1 = sp.sparse.coo_matrix((Kp1_data,(Kp1_row,Kp1_col)),shape=(n_state * n_hier,n_state * n_hier),dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data,(Kp1_col,Kp1_row)),shape=(n_state * n_hier,n_state * n_hier),dtype=np.complex64).toarray()
    #Calculate expected Kp1,Km1
    gdw0 = g_0/w_0
    gdw1 = -1j * np.imag(g_0)/500.0
    w0 = w_0
    w1 = 500.0
    #Connections with mode 0:
    #Loperator = [1,-2,3,-4,0,0,0,0,0,0]
    Kp1_row_calc = [0,1,2,3] #From Aux{[],4} to Aux{[(0,1)],4}
    Kp1_col_calc = [10,11,12,13]
    Kp1_data_calc = list(np.array([-gdw0,-gdw0,-gdw0,-gdw0]) * np.array([1,-2,3,-4]))
    Km1_data_calc = list(np.array([w0,w0,w0,w0]) * np.array([1,-2,3,-4]))
    #Connections with mode 1:
    #L operator = [1,-2,3,-4,0,0,0,0,0,0]
    Kp1_row_calc += [0,1,2,3] #From Aux{[],4} to Aux{[(1,1)],4}
    Kp1_col_calc += [20,21,22,23]
    Kp1_data_calc += list(np.array([-gdw1,-gdw1,-gdw1,-gdw1]) * np.array([1,-2,3,-4]))
    Km1_data_calc += list(np.array([w1,w1,w1,w1]) * np.array([1,-2,3,-4]))
    Kp1_row_calc += [20,21,22,23] #From Aux{[(1,1)],4} to Aux{[(1,2)],4}
    Kp1_col_calc += [60,61,62,63]
    Kp1_data_calc += list(np.array([-gdw1,-gdw1,-gdw1,-gdw1]) * np.array([1,-2,3,-4]))
    Km1_data_calc += list(np.array([2*w1,2*w1,2*w1,2*w1]) * np.array([1,-2,3,-4]))
    Kp1_row_calc += [30,31,32,33] #From Aux{[(2,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += [70,71,72,73]
    Kp1_data_calc += list(np.array([-gdw1,-gdw1,-gdw1,-gdw1]) * np.array([1,-2,3,-4]))
    Km1_data_calc += list(np.array([w1,w1,w1,w1]) * np.array([1,-2,3,-4]))
    #Connections with mode 2:
    #L operator = [0,1,0,0,-2,3,-4,0,0,0]
    Kp1_row_calc += [1,4,5,6] #From Aux{[],4} to Aux{[(2,1)],4}
    Kp1_col_calc += [31,34,35,36]
    Kp1_data_calc += list(np.array([-gdw0,-gdw0,-gdw0,-gdw0]) * np.array([1,-2,3,-4]))
    Km1_data_calc += list(np.array([w0,w0,w0,w0]) * np.array([1,-2,3,-4]))
    Kp1_row_calc += [21,24,25,26] #From Aux{[(1,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += [71,74,75,76]
    Kp1_data_calc += list(np.array([-gdw0,-gdw0,-gdw0,-gdw0]) * np.array([1,-2,3,-4]))
    Km1_data_calc += list(np.array([w0,w0,w0,w0]) * np.array([1,-2,3,-4]))
    #Connections with mode 3:
    #Connections with mode 4:
    #Connections with mode 5:
    #Loperator = [0,0,1,0,0,-2,0,3,-4,0]
    Kp1_row_calc += [2,5,7,8] #From Aux{[],4} to Aux{[(5,1)],4}
    Kp1_col_calc += [42,45,47,48]
    Kp1_data_calc += list(np.array([-gdw1,-gdw1,-gdw1,-gdw1]) * np.array([1,-2,3,-4]))
    Km1_data_calc += list(np.array([w1,w1,w1,w1]) * np.array([1,-2,3,-4]))
    #Connections with mode 6:
    #Loperator = [0,0,0,1,0,0,-2,0,3,-4]
    Kp1_row_calc += [3,6,8,9] #From Aux{[],4} to Aux{[(6,1)],4} 
    Kp1_col_calc += [53,56,58,59]
    Kp1_data_calc += list(np.array([-gdw0,-gdw0,-gdw0,-gdw0]) * np.array([1,-2,3,-4]))
    Km1_data_calc += list(np.array([w0,w0,w0,w0]) * np.array([1,-2,3,-4]))
    #Connections with mode 7:
    #None
    Kp1_calc = sp.sparse.coo_matrix((Kp1_data_calc,(Kp1_row_calc,Kp1_col_calc)),shape=(n_state * n_hier,n_state * n_hier),dtype=np.complex64).toarray()
    Km1_calc = sp.sparse.coo_matrix((Km1_data_calc,(Kp1_col_calc,Kp1_row_calc)),shape=(n_state * n_hier,n_state * n_hier),dtype=np.complex64).toarray()
    assert np.all(Kp1_calc == Kp1)
    assert np.all(Km1_calc == Km1)
    #Build calculated Zp1
    Zp1_0 = sp.sparse.coo_matrix((Zp1_data[0],(Zp1_row[0],Zp1_col[0])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_1 = sp.sparse.coo_matrix((Zp1_data[1],(Zp1_row[1],Zp1_col[1])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_2 = sp.sparse.coo_matrix((Zp1_data[2],(Zp1_row[2],Zp1_col[2])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_3 = sp.sparse.coo_matrix((Zp1_data[3],(Zp1_row[3],Zp1_col[3])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    #Calculate expected Zp1
    #L operator 0: modes 0,1
    Zp1_row_calc0 = [0,0] #From Aux{[],4} to Aux{[(0,1),4}
    Zp1_col_calc0 = [1,2]
    Zp1_data_calc0 = [gdw0,gdw1]
    Zp1_row_calc0 += [2] #From Aux{[(1,1)],4} to Aux{[(1,2),4}
    Zp1_col_calc0 += [6]
    Zp1_data_calc0 += [gdw1]
    Zp1_row_calc0 += [3] #From Aux{[(2,1)],4} to Aux{[(1,1),(2,1),4}
    Zp1_col_calc0 += [7]
    Zp1_data_calc0 += [gdw1]
    #L operator 1: modes 2,3
    Zp1_row_calc1 = [0] #From Aux{[],4} to Aux{[(2,1)],4}
    Zp1_col_calc1 = [3]
    Zp1_data_calc1 = [gdw0]
    Zp1_row_calc1 += [2] #From Aux{[(1,1)],4} to Aux{[(1,1),(2,1)],4}
    Zp1_col_calc1 += [7]
    Zp1_data_calc1 += [gdw0]
    #L operator 2: modes 4,5
    Zp1_row_calc2 = [0] #From Aux{[],4} to Aux{[(5,1),4}
    Zp1_col_calc2 = [4]
    Zp1_data_calc2 = [gdw1]
    #L operator 3: mode 6,7
    Zp1_row_calc3 = [0] #From Aux{[],4} to Aux{[(6,1),4}
    Zp1_col_calc3 = [5]
    Zp1_data_calc3 = [gdw0]
    Zp1_calc_0 = sp.sparse.coo_matrix((Zp1_data_calc0,(Zp1_row_calc0,Zp1_col_calc0)),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_calc_1 = sp.sparse.coo_matrix((Zp1_data_calc1,(Zp1_row_calc1,Zp1_col_calc1)),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_calc_2 = sp.sparse.coo_matrix((Zp1_data_calc2,(Zp1_row_calc2,Zp1_col_calc2)),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_calc_3 = sp.sparse.coo_matrix((Zp1_data_calc3,(Zp1_row_calc3,Zp1_col_calc3)),shape=(n_hier,n_hier),dtype=np.complex64).toarray() 
    assert np.all(Zp1_0 == Zp1_calc_0)
    assert np.all(Zp1_1 == Zp1_calc_1)         
    assert np.all(Zp1_2 == Zp1_calc_2)
    assert np.all(Zp1_3 == Zp1_calc_3)
    
    #Test 2: Partial State Basis, Full Mode Basis
    n_lop = 8
    n_state = 7
    n_hier = 8
    hops2p.basis.system.state_list = [0,1,3,5,6,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]
    
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]
                                             
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
    #Build Computed Kp1,Km1
    Kp1 = sp.sparse.coo_matrix((Kp1_data,(Kp1_row,Kp1_col)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data,(Kp1_col,Kp1_row)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    #Calculate expected Kp1,Km1
    gdw0 = g_0/w_0
    gdw1 = -1j * np.imag(g_0)/500.0
    w0 = w_0
    w1 = 500.0
    #Connections with mode 0:
    #Loperator = [1,-2,-4,0,0,0,0]
    Kp1_row_calc = [0,1,2] #From Aux{[],4} to Aux{[(0,1)],4}
    Kp1_col_calc = list(np.array([1*7,1*7,1*7]) + np.array([0,1,2]))
    Kp1_data_calc = list(np.array([-gdw0,-gdw0,-gdw0]) * np.array([1,-2,-4]))
    Km1_data_calc = list(np.array([w0,w0,w0]) * np.array([1,-2,-4]))
    #Connections with mode 1:
    #L operator = [1,-2,-4,0,0,0,0]
    Kp1_row_calc += [0,1,2] #From Aux{[],4} to Aux{[(1,1)],4}
    Kp1_col_calc += list(np.array([2*7,2*7,2*7]) + np.array([0,1,2]))
    Kp1_data_calc += list(np.array([-gdw1,-gdw1,-gdw1]) * np.array([1,-2,-4]))
    Km1_data_calc += list(np.array([w1,w1,w1]) * np.array([1,-2,-4]))
    Kp1_row_calc += list(np.array([2*7,2*7,2*7]) + np.array([0,1,2])) #From Aux{[(1,1)],4} to Aux{[(1,2)],4}
    Kp1_col_calc += list(np.array([6*7,6*7,6*7]) + np.array([0,1,2]))
    Kp1_data_calc += list(np.array([-gdw1,-gdw1,-gdw1]) * np.array([1,-2,-4]))
    Km1_data_calc += list(np.array([2*w1,2*w1,2*w1]) * np.array([1,-2,-4]))
    Kp1_row_calc += list(np.array([3*7,3*7,3*7]) + np.array([0,1,2])) #From Aux{[(2,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += list(np.array([7*7,7*7,7*7]) + np.array([0,1,2]))
    Kp1_data_calc += list(np.array([-gdw1,-gdw1,-gdw1]) * np.array([1,-2,-4]))
    Km1_data_calc += list(np.array([w1,w1,w1]) * np.array([1,-2,-4]))
    #Connections with mode 2:
    #L operator = [0,1,0,3,-4,0,0]
    Kp1_row_calc += [1,3,4] #From Aux{[],4} to Aux{[(2,1)],4}
    Kp1_col_calc += list(np.array([3*7,3*7,3*7]) + np.array([1,3,4]))
    Kp1_data_calc += list(np.array([-gdw0,-gdw0,-gdw0]) * np.array([1,3,-4]))
    Km1_data_calc += list(np.array([w0,w0,w0]) * np.array([1,3,-4]))
    Kp1_row_calc += list(np.array([2*7,2*7,2*7]) + np.array([1,3,4])) #From Aux{[(1,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += list(np.array([7*7,7*7,7*7]) + np.array([1,3,4]))
    Kp1_data_calc += list(np.array([-gdw0,-gdw0,-gdw0]) * np.array([1,3,-4]))
    Km1_data_calc += list(np.array([w0,w0,w0]) * np.array([1,3,-4]))
    #Connections with mode 3:
    #Connections with mode 4:
    #Connections with mode 5:
    #Loperator = [0,0,0,-2,0,-4,0]
    Kp1_row_calc += [3,5] #From Aux{[],4} to Aux{[(5,1)],4}
    Kp1_col_calc += list(np.array([4*7,4*7]) + np.array([3,5]))
    Kp1_data_calc += list(np.array([-gdw1,-gdw1]) * np.array([-2,-4]))
    Km1_data_calc += list(np.array([w1,w1]) * np.array([-2,-4]))
    #Connections with mode 6:
    #Loperator = [0,0,1,0,-2,3,-4]
    Kp1_row_calc += [2,4,5,6] #From Aux{[],4} to Aux{[(6,1)],4} 
    Kp1_col_calc += list(np.array([5*7,5*7,5*7,5*7]) + np.array([2,4,5,6]))
    Kp1_data_calc += list(np.array([-gdw0,-gdw0,-gdw0,-gdw0]) * np.array([1,-2,3,-4]))
    Km1_data_calc += list(np.array([w0,w0,w0,w0]) * np.array([1,-2,3,-4]))
    #Connections with mode 7:
    #None
    Kp1_calc = sp.sparse.coo_matrix((Kp1_data_calc,(Kp1_row_calc,Kp1_col_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1_calc = sp.sparse.coo_matrix((Km1_data_calc,(Kp1_col_calc,Kp1_row_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    assert np.all(Kp1_calc == Kp1)
    assert np.all(Km1_calc == Km1)
    #Build calculated Zp1
    Zp1_0 = sp.sparse.coo_matrix((Zp1_data[0],(Zp1_row[0],Zp1_col[0])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_1 = sp.sparse.coo_matrix((Zp1_data[1],(Zp1_row[1],Zp1_col[1])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_2 = sp.sparse.coo_matrix((Zp1_data[2],(Zp1_row[2],Zp1_col[2])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_3 = sp.sparse.coo_matrix((Zp1_data[3],(Zp1_row[3],Zp1_col[3])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    #Calculate expected Zp1
    #L operator 0: modes 0,1
    Zp1_row_calc0 = [0,0] #From Aux{[],4} to Aux{[(0,1),4}
    Zp1_col_calc0 = [1,2]
    Zp1_data_calc0 = [gdw0,gdw1]
    Zp1_row_calc0 += [2] #From Aux{[(1,1)],4} to Aux{[(1,2),4}
    Zp1_col_calc0 += [6]
    Zp1_data_calc0 += [gdw1]
    Zp1_row_calc0 += [3] #From Aux{[(2,1)],4} to Aux{[(1,1),(2,1),4}
    Zp1_col_calc0 += [7]
    Zp1_data_calc0 += [gdw1]
    #L operator 1: modes 2,3
    Zp1_row_calc1 = [0] #From Aux{[],4} to Aux{[(2,1)],4}
    Zp1_col_calc1 = [3]
    Zp1_data_calc1 = [gdw0]
    Zp1_row_calc1 += [2] #From Aux{[(1,1)],4} to Aux{[(1,1),(2,1)],4}
    Zp1_col_calc1 += [7]
    Zp1_data_calc1 += [gdw0]
    #L operator 2: modes 4,5
    Zp1_row_calc2 = [0] #From Aux{[],4} to Aux{[(5,1),4}
    Zp1_col_calc2 = [4]
    Zp1_data_calc2 = [gdw1]
    #L operator 3: mode 6,7
    Zp1_row_calc3 = [0] #From Aux{[],4} to Aux{[(6,1),4}
    Zp1_col_calc3 = [5]
    Zp1_data_calc3 = [gdw0]
    Zp1_calc_0 = sp.sparse.coo_matrix((Zp1_data_calc0,(Zp1_row_calc0,Zp1_col_calc0)),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_calc_1 = sp.sparse.coo_matrix((Zp1_data_calc1,(Zp1_row_calc1,Zp1_col_calc1)),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_calc_2 = sp.sparse.coo_matrix((Zp1_data_calc2,(Zp1_row_calc2,Zp1_col_calc2)),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_calc_3 = sp.sparse.coo_matrix((Zp1_data_calc3,(Zp1_row_calc3,Zp1_col_calc3)),shape=(n_hier,n_hier),dtype=np.complex64).toarray() 
    assert np.all(Zp1_0 == Zp1_calc_0)
    assert np.all(Zp1_1 == Zp1_calc_1)         
    assert np.all(Zp1_2 == Zp1_calc_2)
    assert np.all(Zp1_3 == Zp1_calc_3)
    
    #Test 3: Partial State Basis, Partial Mode Basis 
    #One L operator has no populated states associated with it but is present because of Hierarchy Modes
    n_lop = 8
    n_state = 2
    n_hier = 8
    hops2p.basis.system.state_list = [1,5]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6]
    
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]
                                             
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
    #Build Computed Kp1,Km1
    Kp1 = sp.sparse.coo_matrix((Kp1_data,(Kp1_row,Kp1_col)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data,(Kp1_col,Kp1_row)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    #Calculate expected Kp1,Km1
    gdw0 = g_0/w_0
    gdw1 = -1j * np.imag(g_0)/500.0
    w0 = w_0
    w1 = 500.0
    #Connections with mode 0:
    #Loperator = [-2,0]
    Kp1_row_calc = [0] #From Aux{[],4} to Aux{[(0,1)],4}
    Kp1_col_calc = list(np.array([1*2]) + np.array([0]))
    Kp1_data_calc = list(np.array([-gdw0]) * np.array([-2]))
    Km1_data_calc = list(np.array([w0]) * np.array([-2]))
    #Connections with mode 1:
    #L operator = [-2,0]
    Kp1_row_calc += [0] #From Aux{[],4} to Aux{[(1,1)],4}
    Kp1_col_calc += list(np.array([2*2]) + np.array([0]))
    Kp1_data_calc += list(np.array([-gdw1]) * np.array([-2]))
    Km1_data_calc += list(np.array([w1]) * np.array([-2]))
    Kp1_row_calc += list(np.array([2*2]) + np.array([0])) #From Aux{[(1,1)],4} to Aux{[(1,2)],4}
    Kp1_col_calc += list(np.array([6*2]) + np.array([0]))
    Kp1_data_calc += list(np.array([-gdw1]) * np.array([-2]))
    Km1_data_calc += list(np.array([2*w1]) * np.array([-2]))
    Kp1_row_calc += list(np.array([3*2]) + np.array([0])) #From Aux{[(2,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += list(np.array([7*2]) + np.array([0]))
    Kp1_data_calc += list(np.array([-gdw1]) * np.array([-2]))
    Km1_data_calc += list(np.array([w1]) * np.array([-2]))
    #Connections with mode 2:
    #L operator = [1,3]
    Kp1_row_calc += [0,1] #From Aux{[],4} to Aux{[(2,1)],4}
    Kp1_col_calc += list(np.array([3*2,3*2]) + np.array([0,1]))
    Kp1_data_calc += list(np.array([-gdw0,-gdw0]) * np.array([1,3]))
    Km1_data_calc += list(np.array([w0,w0]) * np.array([1,3]))
    Kp1_row_calc += list(np.array([2*2,2*2]) + np.array([0,1])) #From Aux{[(1,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += list(np.array([7*2,7*2]) + np.array([0,1]))
    Kp1_data_calc += list(np.array([-gdw0,-gdw0]) * np.array([1,3]))
    Km1_data_calc += list(np.array([w0,w0]) * np.array([1,3]))
    #Connections with mode 3:
    #Connections with mode 4:
    #Connections with mode 5:
    #Loperator = [0,-2]
    Kp1_row_calc += [1] #From Aux{[],4} to Aux{[(5,1)],4}
    Kp1_col_calc += list(np.array([4*2]) + np.array([1]))
    Kp1_data_calc += list(np.array([-gdw1]) * np.array([-2]))
    Km1_data_calc += list(np.array([w1]) * np.array([-2]))
    #Connections with mode 6:
    #Loperator = [0,0]
    #Connections with mode 7:
    #None
    Kp1_calc = sp.sparse.coo_matrix((Kp1_data_calc,(Kp1_row_calc,Kp1_col_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1_calc = sp.sparse.coo_matrix((Km1_data_calc,(Kp1_col_calc,Kp1_row_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    assert np.all(Kp1_calc == Kp1)
    assert np.all(Km1_calc == Km1)
    #Build calculated Zp1
    Zp1_0 = sp.sparse.coo_matrix((Zp1_data[0],(Zp1_row[0],Zp1_col[0])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_1 = sp.sparse.coo_matrix((Zp1_data[1],(Zp1_row[1],Zp1_col[1])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_2 = sp.sparse.coo_matrix((Zp1_data[2],(Zp1_row[2],Zp1_col[2])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_3 = sp.sparse.coo_matrix((Zp1_data[3],(Zp1_row[3],Zp1_col[3])),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    #Calculate expected Zp1
    #L operator 0: modes 0,1
    Zp1_row_calc0 = [0,0] #From Aux{[],4} to Aux{[(0,1),4}
    Zp1_col_calc0 = [1,2]
    Zp1_data_calc0 = [gdw0,gdw1]
    Zp1_row_calc0 += [2] #From Aux{[(1,1)],4} to Aux{[(1,2),4}
    Zp1_col_calc0 += [6]
    Zp1_data_calc0 += [gdw1]
    Zp1_row_calc0 += [3] #From Aux{[(2,1)],4} to Aux{[(1,1),(2,1),4}
    Zp1_col_calc0 += [7]
    Zp1_data_calc0 += [gdw1]
    #L operator 1: modes 2,3
    Zp1_row_calc1 = [0] #From Aux{[],4} to Aux{[(2,1)],4}
    Zp1_col_calc1 = [3]
    Zp1_data_calc1 = [gdw0]
    Zp1_row_calc1 += [2] #From Aux{[(1,1)],4} to Aux{[(1,1),(2,1)],4}
    Zp1_col_calc1 += [7]
    Zp1_data_calc1 += [gdw0]
    #L operator 2: modes 4,5
    Zp1_row_calc2 = [0] #From Aux{[],4} to Aux{[(5,1),4}
    Zp1_col_calc2 = [4]
    Zp1_data_calc2 = [gdw1]
    #L operator 3: mode 6,7
    Zp1_row_calc3 = [0] #From Aux{[],4} to Aux{[(6,1),4}
    Zp1_col_calc3 = [5]
    Zp1_data_calc3 = [gdw0]
    Zp1_calc_0 = sp.sparse.coo_matrix((Zp1_data_calc0,(Zp1_row_calc0,Zp1_col_calc0)),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_calc_1 = sp.sparse.coo_matrix((Zp1_data_calc1,(Zp1_row_calc1,Zp1_col_calc1)),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_calc_2 = sp.sparse.coo_matrix((Zp1_data_calc2,(Zp1_row_calc2,Zp1_col_calc2)),shape=(n_hier,n_hier),dtype=np.complex64).toarray()
    Zp1_calc_3 = sp.sparse.coo_matrix((Zp1_data_calc3,(Zp1_row_calc3,Zp1_col_calc3)),shape=(n_hier,n_hier),dtype=np.complex64).toarray() 
    assert np.all(Zp1_0 == Zp1_calc_0)
    assert np.all(Zp1_1 == Zp1_calc_1)         
    assert np.all(Zp1_2 == Zp1_calc_2)
    assert np.all(Zp1_3 == Zp1_calc_3)
   
    
def test_add_crossterms_stable_2part():

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
    t_max = 2000.0
    t_step = 2.0
    (g_0, w_0) = bcf_convert_sdl_to_exp(e_lambda, gamma, 0.0, temp)  
    numsites2 = nsite*(nsite+1)/2
    loperator = np.zeros([nsite, int(numsites2), int(numsites2)], dtype=np.float64) 
    gw_sysbath = []  
    for i in range(nsite):
        gw_sysbath.append([g_0, w_0])   
        gw_sysbath.append([-1j * np.imag(g_0), 500.0]) 
    l0 = sp.sparse.coo_matrix(np.diag([1,-2,3,-4,0,0,0,0,0,0]))
    l2 = sp.sparse.coo_matrix(np.diag([0,1,0,0,-2,3,-4,0,0,0]))
    l4 = sp.sparse.coo_matrix(np.diag([0,0,1,0,0,-2,0,3,-4,0]))
    l6 = sp.sparse.coo_matrix(np.diag([0,0,0,1,0,0,-2,0,3,-4]))
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
    eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"} # we generally pick normalized nonlinear 
    # as it has better convergence properties than the linear eom

    # Integration parameters 
    integrator_param = {"INTEGRATOR": "RUNGE_KUTTA"}  # We use a Runge-Kutta method for our integrator 

    # Initial wave function (in the state basis, we fully populate site 3 and no others)
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
    
    #Test 1: No new states 
    n_lop = 8
    n_state = 10
    n_hier = 8
    hops2p.basis.system.state_list = [0,1,2,3,4,5,6,7,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]
    
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],hops2p.basis.hierarchy.auxiliary_list[1],hops2p.basis.hierarchy.auxiliary_list[2],
                                             hops2p.basis.hierarchy.auxiliary_list[3],hops2p.basis.hierarchy.auxiliary_list[4],hops2p.basis.hierarchy.auxiliary_list[5],
                                             hops2p.basis.hierarchy.auxiliary_list[6],hops2p.basis.hierarchy.auxiliary_list[7]]
    hops2p.basis.system.state_list = [0,1,2,3,4,5,6,7,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]
    
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
    #Build Computed Kp1,Km1
    Kp1 = sp.sparse.coo_matrix((Kp1_data,(Kp1_row,Kp1_col)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data,(Kp1_col,Kp1_row)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    #Calculated Expected Kp1,Km1
    Kp1_row_calc = []
    Kp1_col_calc = []
    Kp1_data_calc = []
    Km1_data_calc = []
    Kp1_calc = sp.sparse.coo_matrix((Kp1_data_calc,(Kp1_row_calc,Kp1_col_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1_calc = sp.sparse.coo_matrix((Km1_data_calc,(Kp1_col_calc,Kp1_row_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    assert np.all(Kp1_calc == Kp1)
    assert np.all(Km1_calc == Km1)
    
    #Test 2: New states, no new modes
    n_lop = 8
    n_state = 10
    n_hier = 8
    hops2p.basis.system.state_list = [0,1,3,5,6,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]
    
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],hops2p.basis.hierarchy.auxiliary_list[1],hops2p.basis.hierarchy.auxiliary_list[2],
                                             hops2p.basis.hierarchy.auxiliary_list[3],hops2p.basis.hierarchy.auxiliary_list[4],hops2p.basis.hierarchy.auxiliary_list[5],
                                             hops2p.basis.hierarchy.auxiliary_list[6],hops2p.basis.hierarchy.auxiliary_list[7]]
    hops2p.basis.system.state_list = [0,1,2,3,4,5,6,7,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]
    #New states = 2,4,7
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
    #Build Computed Kp1,Km1
    Kp1 = sp.sparse.coo_matrix((Kp1_data,(Kp1_row,Kp1_col)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data,(Kp1_col,Kp1_row)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    #Calculated Expected Kp1,Km1
    gdw0 = g_0/w_0
    gdw1 = -1j * np.imag(g_0)/500.0
    w0 = w_0
    w1 = 500.0
    #Connections with mode 0:
    #Loperator = [*,*,3,*,0,*,*,0,*,*]
    Kp1_row_calc = [2] #From Aux{[],4} to Aux{[(0,1)],4}
    Kp1_col_calc = [12]
    Kp1_data_calc = list(np.array([-gdw0]) * np.array([3]))
    Km1_data_calc = list(np.array([w0]) * np.array([3]))
    #Connections with mode 1:
    #L operator = [*,*,3,*,0,*,*,0,*,*]
    Kp1_row_calc += [2] #From Aux{[],4} to Aux{[(1,1)],4}
    Kp1_col_calc += [22]
    Kp1_data_calc += list(np.array([-gdw1]) * np.array([3]))
    Km1_data_calc += list(np.array([w1]) * np.array([3]))
    Kp1_row_calc += [22] #From Aux{[(1,1)],4} to Aux{[(1,2)],4}
    Kp1_col_calc += [62]
    Kp1_data_calc += list(np.array([-gdw1]) * np.array([3]))
    Km1_data_calc += list(np.array([2*w1]) * np.array([3]))
    Kp1_row_calc += [32] #From Aux{[(2,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += [72]
    Kp1_data_calc += list(np.array([-gdw1]) * np.array([3]))
    Km1_data_calc += list(np.array([w1]) * np.array([3]))
    #Connections with mode 2:
    #L operator = [*,*,0,*,-2,*,*,0,*,*]
    Kp1_row_calc += [4] #From Aux{[],4} to Aux{[(2,1)],4}
    Kp1_col_calc += [34]
    Kp1_data_calc += list(np.array([-gdw0]) * np.array([-2]))
    Km1_data_calc += list(np.array([w0]) * np.array([-2]))
    Kp1_row_calc += [24] #From Aux{[(1,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += [74]
    Kp1_data_calc += list(np.array([-gdw0]) * np.array([-2]))
    Km1_data_calc += list(np.array([w0]) * np.array([-2]))
    #Connections with mode 3:
    #Connections with mode 4:
    #Connections with mode 5:
    #Loperator = [*,*,1,*,0,*,*,3,*,*]
    Kp1_row_calc += [2,7] #From Aux{[],4} to Aux{[(5,1)],4}
    Kp1_col_calc += [42,47]
    Kp1_data_calc += list(np.array([-gdw1,-gdw1]) * np.array([1,3]))
    Km1_data_calc += list(np.array([w1,w1]) * np.array([1,3]))
    #Connections with mode 6:
    #Loperator = [*,*,0,*,0,*,*,0,*,*]
    #Connections with mode 7:
    #None
    Kp1_calc = sp.sparse.coo_matrix((Kp1_data_calc,(Kp1_row_calc,Kp1_col_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1_calc = sp.sparse.coo_matrix((Km1_data_calc,(Kp1_col_calc,Kp1_row_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    assert np.all(Kp1_calc == Kp1)
    assert np.all(Km1_calc == Km1)
    
    #Test 3: New states, New modes
    n_lop = 8
    n_state = 10
    n_hier = 8
    hops2p.basis.system.state_list = [1,5]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6]
    
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0]]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],AuxiliaryVector([(0, 1)], 8),AuxiliaryVector([(1,1)], 8),AuxiliaryVector([(2,1)], 8),
                                             AuxiliaryVector([(5,1)], 8), AuxiliaryVector([(6,1)], 8), AuxiliaryVector([(1,2)], 8), AuxiliaryVector([(1,1), (2,1)], 8)]
    hops2p.basis.hierarchy.auxiliary_list = [hops2p.basis.hierarchy.auxiliary_list[0],hops2p.basis.hierarchy.auxiliary_list[1],hops2p.basis.hierarchy.auxiliary_list[2],
                                             hops2p.basis.hierarchy.auxiliary_list[3],hops2p.basis.hierarchy.auxiliary_list[4],hops2p.basis.hierarchy.auxiliary_list[5],
                                             hops2p.basis.hierarchy.auxiliary_list[6],hops2p.basis.hierarchy.auxiliary_list[7]]
    hops2p.basis.system.state_list = [0,1,3,5,6,8,9]
    hops2p.basis.mode.list_absindex_mode = [0,1,2,3,4,5,6,7]
    #New states = 0,3,6,8,9
    #Relative indices of new states = 0,2,4,5,6
    #New modes = 7
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
    #Build Computed Kp1,Km1
    Kp1 = sp.sparse.coo_matrix((Kp1_data,(Kp1_row,Kp1_col)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1 = sp.sparse.coo_matrix((Km1_data,(Kp1_col,Kp1_row)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    #Calculated Expected Kp1,Km1
    gdw0 = g_0/w_0
    gdw1 = -1j * np.imag(g_0)/500.0
    w0 = w_0
    w1 = 500.0
    #Connections with mode 0:
    #Loperator = [1,*,-4,*,0,0,0]
    Kp1_row_calc = [0,2] #From Aux{[],4} to Aux{[(0,1)],4}
    Kp1_col_calc = list(np.array([1*7,1*7]) + np.array([0,2]))
    Kp1_data_calc = list(np.array([-gdw0,-gdw0]) * np.array([1,-4]))
    Km1_data_calc = list(np.array([w0,w0]) * np.array([1,-4]))
    #Connections with mode 1:
    #L operator = [1,*,-4,*,0,0,0]
    Kp1_row_calc += [0,2] #From Aux{[],4} to Aux{[(1,1)],4}
    Kp1_col_calc += list(np.array([2*7,2*7]) + np.array([0,2]))
    Kp1_data_calc += list(np.array([-gdw1,-gdw1]) * np.array([1,-4]))
    Km1_data_calc += list(np.array([w1,w1]) * np.array([1,-4]))
    Kp1_row_calc += list(np.array([2*7,2*7]) + np.array([0,2])) #From Aux{[(1,1)],4} to Aux{[(1,2)],4}
    Kp1_col_calc += list(np.array([6*7,6*7]) + np.array([0,2]))
    Kp1_data_calc += list(np.array([-gdw1,-gdw1]) * np.array([1,-4]))
    Km1_data_calc += list(np.array([2*w1,2*w1]) * np.array([1,-4]))
    Kp1_row_calc += list(np.array([3*7,3*7]) + np.array([0,2])) #From Aux{[(2,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += list(np.array([7*7,7*7]) + np.array([0,2]))
    Kp1_data_calc += list(np.array([-gdw1,-gdw1]) * np.array([1,-4]))
    Km1_data_calc += list(np.array([w1,w1]) * np.array([1,-4]))
    #Connections with mode 2:
    #L operator = [0,*,0,*,-4,0,0]
    Kp1_row_calc += [4] #From Aux{[],4} to Aux{[(2,1)],4}
    Kp1_col_calc += list(np.array([3*7]) + np.array([4]))
    Kp1_data_calc += list(np.array([-gdw0]) * np.array([-4]))
    Km1_data_calc += list(np.array([w0]) * np.array([-4]))
    Kp1_row_calc += list(np.array([2*7]) + np.array([4])) #From Aux{[(1,1)],4} to Aux{[(1,1),(2,1)],4}
    Kp1_col_calc += list(np.array([7*7]) + np.array([4]))
    Kp1_data_calc += list(np.array([-gdw0]) * np.array([-4]))
    Km1_data_calc += list(np.array([w0]) * np.array([-4]))
    #Connections with mode 3:
    #Connections with mode 4:
    #Connections with mode 5:
    #Loperator = [0,*,0,*,0,-4,0]
    Kp1_row_calc += [5] #From Aux{[],4} to Aux{[(5,1)],4}
    Kp1_col_calc += list(np.array([4*7]) + np.array([5]))
    Kp1_data_calc += list(np.array([-gdw1]) * np.array([-4]))
    Km1_data_calc += list(np.array([w1]) * np.array([-4]))
    #Connections with mode 6:
    #Loperator = [0,*,1,*,-2,3,-4]
    Kp1_row_calc += [2,4,5,6] #From Aux{[],4} to Aux{[(6,1)],4} 
    Kp1_col_calc += list(np.array([5*7,5*7,5*7,5*7]) + np.array([2,4,5,6]))
    Kp1_data_calc += list(np.array([-gdw0,-gdw0,-gdw0,-gdw0]) * np.array([1,-2,3,-4]))
    Km1_data_calc += list(np.array([w0,w0,w0,w0]) * np.array([1,-2,3,-4]))
    #Connections with mode 7:
    #None
    Kp1_calc = sp.sparse.coo_matrix((Kp1_data_calc,(Kp1_row_calc,Kp1_col_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    Km1_calc = sp.sparse.coo_matrix((Km1_data_calc,(Kp1_col_calc,Kp1_row_calc)),shape=(n_state*n_hier,n_state*n_hier),dtype=np.complex64).toarray()
    assert np.all(Kp1_calc == Kp1)
    assert np.all(Km1_calc == Km1)
    
    
def test_matrix_updates_with_missing_aux_and_states():
    """
    test the matrix update functions when aux and states are
    removed
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

    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list
    hops2.basis.system.state_list = hops.basis.system.state_list
    hops2.basis.mode.list_absindex_mode = list(hops.basis.mode.list_absindex_mode)
    
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
        ).tocsc()
    )
    Km1 = (
        Km1_new
        + sparse.coo_matrix(
            (Km1_data, (Kp1_col, Kp1_row)), shape=(n_tot, n_tot), dtype=np.complex128
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
    assert (Zp1[1].todense() == hops.basis.eom.Z2_kp1[1].todense()).all()
    assert (K0.todense() == hops.basis.eom.K2_k.todense()).all()


def test_update_super_remove_aux():
    """
    test update_ksuper() when only aux are removed
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

    # NOTE: This test breaks because we are not updating the system,
    # hierarchy, and mode objects prior to calling update_ksuper.
    
    hops2.basis.hierarchy.auxiliary_list = hops.basis.hierarchy.auxiliary_list
    
    K0, Kp1, Zp1, Km1 = update_ksuper(
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
    assert (Zp1[0].todense() == hops.basis.eom.Z2_kp1[0].todense()).all()


def test_update_super_remove_aux_and_state():
    """
    test update_ksuper when aux and states are removed
    """
    # Prepare Constants
    # =================
    n_site = hops.basis.system.param["NSTATES"]
    n_mode = hops.basis.system.param["N_HMODES"]

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
    
    K0, Kp1, Zp1, Km1 = update_ksuper(
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
    assert (Zp1[0].todense() == hops.basis.eom.Z2_kp1[0].todense()).all()
