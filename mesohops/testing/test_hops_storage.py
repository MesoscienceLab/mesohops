import numpy as np
import pytest
from mesohops.dynamics.hops_aux import AuxiliaryVector
import mesohops.dynamics.storage_functions as sf
from mesohops.dynamics.bath_corr_functions import bcf_exp
from mesohops.dynamics.hops_trajectory import HopsTrajectory
from mesohops.dynamics.hops_storage import HopsStorage
from mesohops.util.exceptions import LockedException, UnsupportedRequest



# New Hops_Storage tests
# --------------------------------------------------------------------------------------
def test_t_axis_save():
    """
    test to see if the current t_axis is saved with the proper function
    """
    # This pattern is the same for each of these functions: define a Hops Storage
    # object, both adaptive and non, as well as one non-adaptive with a fake saving
    # function at the key in question of the storage dictionary, and one adaptive
    # with the dictionary empty at the key. This should cause HS and AHS to have the
    # default saving function (or an empty dictionary, in the case that the default
    # for non-adaptive is none), as defined in storage_functions, at the key in the
    # saving dictionary, while the "broken" and "empty" Hops Storage objects will
    # have the arbitrary function and an empty dictionary at the keys listed.

    HS = HopsStorage(False, {})
    AHS = HopsStorage(True, {})

    def fake_save_func():
        return
    broken_HS = HopsStorage(False, {"t_axis": fake_save_func})
    false_AHS = HopsStorage(True, {"t_axis": False})

    assert HS.dic_save["t_axis"] == sf.save_t_axis
    assert AHS.dic_save["t_axis"] == sf.save_t_axis

    assert broken_HS.dic_save["t_axis"] == fake_save_func
    assert not "t_axis" in false_AHS.dic_save
    try:
        empty_AHS = HopsStorage(True, {"t_axis": None})
    except UnsupportedRequest as excinfo:
        assert 'The current code does not support this value' in str(excinfo)


def test_psi_traj_save():
    """
    test to see if the current psi trajectory is saved with the proper function
    input is a list
    """
    HS = HopsStorage(False, {})
    AHS = HopsStorage(True, {})
    false_AHS = HopsStorage(True, {"t_axis": False})

    def fake_save_func():
        return

    broken_HS = HopsStorage(False, {"psi_traj": fake_save_func})
    false_AHS = HopsStorage(True, {"psi_traj": False})

    assert HS.dic_save["psi_traj"] == sf.save_psi_traj
    assert AHS.dic_save["psi_traj"] == sf.save_psi_traj

    assert broken_HS.dic_save["psi_traj"] == fake_save_func
    assert not "psi_traj" in false_AHS.dic_save
    try:
        empty_AHS = HopsStorage(True, {"psi_traj": None})
    except UnsupportedRequest as excinfo:
        assert 'The current code does not support this value' in str(excinfo)


def test_phi_traj_save():
    """
       test to see if the current phi trajectory is saved with the proper function
    """
    HS = HopsStorage(False, {})
    AHS = HopsStorage(True, {})

    def fake_save_func():
        return

    broken_HS = HopsStorage(False, {"phi_traj": fake_save_func})
    false_AHS = HopsStorage(True, {"phi_traj": False})

    assert not "phi_traj" in HS.dic_save
    assert not "phi_traj" in AHS.dic_save

    assert broken_HS.dic_save["phi_traj"] == fake_save_func
    assert not "phi_traj" in false_AHS.dic_save
    try:
        empty_AHS = HopsStorage(True, {"phi_traj": None})
    except UnsupportedRequest as excinfo:
        assert 'The current code does not support this value' in str(excinfo)


def test_aux_new_save():
    """
       test to see if the current new auxiliary list is saved with the proper function
    """
    HS = HopsStorage(False, {})
    AHS = HopsStorage(True, {})

    def fake_save_func():
        return

    broken_HS = HopsStorage(False, {"aux_list": fake_save_func})
    false_AHS = HopsStorage(True, {"aux_list": False})


    assert not "aux_list" in HS.dic_save
    assert AHS.dic_save["aux_list"] == sf.save_aux_list

    assert broken_HS.dic_save["aux_list"] == fake_save_func
    assert not "aux_list" in false_AHS.dic_save
    try:
        empty_AHS = HopsStorage(True, {"aux_list": None})
    except UnsupportedRequest as excinfo:
        assert 'The current code does not support this value' in str(excinfo)


def test_state_list_save():
    """
       test to see if the current state list is saved with the proper function
    """
    HS = HopsStorage(False, {})
    AHS = HopsStorage(True, {})

    def fake_save_func():
        return

    broken_HS = HopsStorage(False, {"state_list": fake_save_func})
    false_AHS = HopsStorage(True, {"state_list": False})

    assert not "state_list" in HS.dic_save.keys()
    assert AHS.dic_save["state_list"] == sf.save_state_list

    assert broken_HS.dic_save["state_list"] == fake_save_func
    assert not "state_list" in false_AHS.dic_save.keys()
    try:
        empty_AHS = HopsStorage(True, {"state_list": None})
    except UnsupportedRequest as excinfo:
        assert 'The current code does not support this value' in str(excinfo)


def test_list_nhier_save():
    """
       test to see if the current new auxiliary list is saved with the proper function
    """
    HS = HopsStorage(False, {})
    AHS = HopsStorage(True, {})

    def fake_save_func():
        return

    broken_HS = HopsStorage(False, {"list_nhier": fake_save_func})
    false_AHS = HopsStorage(True, {"list_nhier": False})

    assert not "list_nhier" in HS.dic_save.keys()
    assert AHS.dic_save["list_nhier"] == sf.save_list_nhier

    assert broken_HS.dic_save["list_nhier"] == fake_save_func
    assert not "list_nhier" in false_AHS.dic_save.keys()
    try:
        empty_AHS = HopsStorage(True, {"list_nhier": None})
    except UnsupportedRequest as excinfo:
        assert 'The current code does not support this value' in str(excinfo)


def test_list_nstate_save():
    """
       test to see if the current new auxiliary list is saved with the proper function
    """
    HS = HopsStorage(False, {})
    AHS = HopsStorage(True, {})

    def fake_save_func():
        return

    broken_HS = HopsStorage(False, {"list_nstate": fake_save_func})
    false_AHS = HopsStorage(True, {"list_nstate": False})


    assert not "list_nstate" in HS.dic_save.keys()
    assert AHS.dic_save["list_nstate"] == sf.save_list_nstate

    assert broken_HS.dic_save["list_nstate"] == fake_save_func
    assert not "list_nstate" in false_AHS.dic_save.keys()
    try:
        empty_AHS = HopsStorage(True, {"list_nstate": None})
    except UnsupportedRequest as excinfo:
        assert 'The current code does not support this value' in str(excinfo)


def test_arbitrary_saving_function():
    """
    test to see if the architecture to save an arbitrary value works properly
    """

    def dummy_saving_function():
        return
    HS = HopsStorage(False, {"property":dummy_saving_function})
    AHS = HopsStorage(True, {"property":dummy_saving_function})
    false_AHS = HopsStorage(False,{})

    assert HS.dic_save["property"] == dummy_saving_function
    assert AHS.dic_save["property"] == dummy_saving_function
    assert not "property" in false_AHS.dic_save
    try:
        empty_AHS = HopsStorage(True, {"property": None})
    except UnsupportedRequest as excinfo:
        assert 'The current code does not support this value' in str(excinfo)


def test_store_step():
    """
    test to see if the test_store_step function updates all self.data correctly
    """
    phi_new = np.array([1,2,3,4,5,6,7])
    aux_new = [AuxiliaryVector([[0,0]],4)]
    state_new = ["state 1 at time 1", "state 2 at time 1"]
    t_new = 1.0
    zmem_new = [1,1,1]

    def fake_saving_function(**kargs):
        return np.pi

    # Non adaptive test
    HS = HopsStorage(False, {})
    false_HS = HopsStorage(False, {'t_axis': False, 'psi_traj': False,
                                   'phi_traj': False, })
    HS.store_step(phi_new=phi_new, aux_list=aux_new, state_list=state_new, t_new=t_new,
                  z_mem_new=zmem_new)
    false_HS.store_step(phi_new=phi_new, aux_list=aux_new, state_list=state_new,
                        t_new=t_new, z_mem_new=zmem_new)

    assert "phi_traj" not in HS.data.keys()
    assert 'list_nhier' not in HS.data.keys()
    assert 't_axis' not in false_HS.data.keys()

    # Adaptive test
    AHS = HopsStorage(True, {})
    broken_AHS = HopsStorage(True, {'psi_traj': fake_saving_function})

    AHS.store_step(phi_new=phi_new, aux_list=aux_new, state_list=state_new,
                   t_new=t_new, z_mem_new=zmem_new)
    broken_AHS.store_step(phi_new=phi_new, aux_list=aux_new, state_list=state_new,
                          t_new=t_new, z_mem_new=zmem_new)

    # update AHS one more time
    phi_new = np.array([7, 6, 5, 4, 3, 2, 1])
    aux_new = [AuxiliaryVector([[0,0]],4),AuxiliaryVector([[0,1]],4)]
    state_new = ["state 1 at time 2", "state 2 at time 2"]
    t_new = 2.0
    zmem_new = [2, 2, 2]

    AHS.store_step(phi_new=phi_new, aux_list=aux_new, state_list=state_new,
                   t_new=t_new, z_mem_new=zmem_new)

    assert AHS.data['t_axis'] == [1.0, 2.0]
    assert AHS.data['state_list'][1] == ["state 1 at time 2", "state 2 at time 2"]
    assert broken_AHS.data['psi_traj'][0] == np.pi


def test_get_item():
    AHS = HopsStorage(True, {})
    phi_new = np.array([1, 2, 3, 4, 5, 6, 7])
    aux_new = [AuxiliaryVector([[0,0]], 4)]
    state_new = ["state 1 at time 1", "state 2 at time 1"]
    t_new = 1.0
    zmem_new = [1, 1, 1]
    AHS.store_step(phi_new=phi_new, aux_list=aux_new, state_list=state_new,
                   t_new=t_new, z_mem_new=zmem_new)
    assert AHS['t_axis'] == [t_new]
    assert np.array_equal(AHS['state_list'], [state_new])
    assert np.allclose(AHS['aux_list'], [aux.array_aux_vec for aux in aux_new])

# Test HOPS object setup
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 40.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
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

hier_param = {"MAXHIER": 2}

eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR':'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 0,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS':None
    }
t_max = 10.0
t_step = 2.0
psi_0 = [1.0 + 0.0 * 1j, 0.0 + 0.0 * 1j]

def test_hierarchy_storage_functions():
    """
    Tests that the hierarchy storage functions save the correct data in the correct
    format.
    """
    hops_traj = HopsTrajectory(
        sys_param,
        noise_param=noise_param,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param,
        storage_param={"list_nhier": True, "aux_list": True}
    )
    hops_traj.make_adaptive(0.00001, 0.00001, 2)
    hops_traj.initialize(psi_0)
    list_aux = [AuxiliaryVector([], 4), AuxiliaryVector([[0, 1]], 4),
                                                AuxiliaryVector([[1, 2]], 4),
                                                AuxiliaryVector([[0, 1], [3, 1]], 4)]
    storage = hops_traj.storage
    storage.store_step(
        phi_new=hops_traj.phi, aux_list=list_aux,
        state_list=hops_traj.state_list,
        t_new=0, z_mem_new=hops_traj.z_mem
    )

    assert storage["list_nhier"][-1] == 4
    for i in range(4):
        assert np.allclose(list_aux[i].array_aux_vec, storage["aux_list"][-1][i])
