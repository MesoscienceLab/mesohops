import numpy as np
from mesohops.dynamics.hops_storage import AdaptiveTrajectoryStorage as AdapTrajStorage
from mesohops.dynamics.hops_storage import TrajectoryStorage as TrajStorage


# Trajectory Storage test
# --------------------------------------------------------------------------------------
def test_t_setter():
    """
    test to see if the current time is properly being stored
    """
    TS = TrajStorage()
    TS.t = 1
    t = TS.t
    known_t = 1
    assert t == known_t

    TS.t = 2
    t2 = TS.t
    known_t2 = 2
    assert t2 == known_t2


def test_z_mem_setter():
    """
    test to see if the current z_mem is properly being stored
    """
    TS = TrajStorage()
    TS.z_mem = np.array([1])
    z_mem = TS.z_mem
    known_z_mem = np.array([1])
    assert z_mem == known_z_mem

    TS.z_mem = np.array([2])
    z_mem = TS.z_mem
    known_z_mem = np.array([2])
    assert z_mem == known_z_mem


def test_phi_setter():
    """
    test to see if the current phi is properly being stored
    """
    TS = TrajStorage()
    TS.phi = np.array([0, 0, 1, 0])
    phi = TS.phi
    known_phi = np.array([0, 0, 1, 0])
    assert np.array_equal(phi, known_phi)

    TS.phi = np.array([0, 1, 0, 0])
    phi = TS.phi
    known_phi = np.array([0, 1, 0, 0])
    assert np.array_equal(phi, known_phi)


def test_psi_traj_setter_if():
    """
    test to see if the current psi_traj is properly being stored when the input is of
    type list
    """
    TS = TrajStorage()
    TS.psi_traj = [[0, 0], [1, 2]]
    TS.psi_traj = [[1, 0], [1, 2]]
    traj = TS.psi_traj
    known_traj = [[[0, 0], [1, 2]], [[1, 0], [1, 2]]]
    assert np.array_equal(traj, known_traj)


def test_psi_traj_setter_else():
    """
    test to see if the current psi_traj is properly being stored when the input is not
    a list
    """
    TS = TrajStorage()
    TS.psi_traj = [1, 2, 3]
    TS.psi_traj = [4, 5, 6]
    traj = TS.psi_traj
    known = [[1, 2, 3], [4, 5, 6]]
    assert np.array_equal(traj, known)


def test_phi_traj_setter_if():
    """
    test to see if the current phi_traj is properly being stored when the input is a
    list. If store_aux is false nothing should be getting stored but if store_aux is
    True we should store phi_traj
    """
    TS = TrajStorage()
    TS.store_aux = False
    TS.phi_traj = [[0, 0], [1, 2]]
    traj = TS.phi_traj
    known = []
    assert np.array_equal(traj, known)

    TS.store_aux = True
    TS.phi_traj = [[1, 0], [1, 2]]
    traj = TS.phi_traj
    known = [[[1, 0], [1, 2]]]
    assert np.array_equal(traj, known)


def test_phi_traj_setter_else():
    """
    test to see if the current phi_traj is properly being stored when the input is not a
    list. If store_aux is false nothing should be getting stored but if store_aux is
    True we should store phi_traj
    """
    TS = TrajStorage()
    TS.store_aux = False
    TS.phi_traj = [1, 2, 3]
    TS.phi_traj = [4, 5, 6]
    traj = TS.phi_traj
    known = []
    assert np.array_equal(traj, known)

    TS.store_aux = True
    TS.phi_traj = [1, 2, 3]
    TS.phi_traj = [4, 5, 6]
    traj = TS.phi_traj
    known = [[1, 2, 3], [4, 5, 6]]
    assert np.array_equal(traj, known)


def test_t_axis_setter_else():
    """
    test to see if the current t_axis is properly being stored when the input is not a
    list
    """
    TS = TrajStorage()
    TS.t_axis = 2
    TS.t_axis = 4
    t_axis = TS.t_axis
    known_t_axis = [2, 4]
    assert np.array_equal(t_axis, known_t_axis)


def test_list_nhier_setter():
    """
    test to see if the current N_hier is properly being stored
    """
    TS = AdapTrajStorage()
    TS.list_nhier = 1
    N_hier = TS.list_nhier
    known_N_hier = [1]
    print(TS.list_nhier, type(TS.list_nhier))
    assert N_hier == known_N_hier

    TS.list_nhier = 2
    N_hier = TS.list_nhier
    known_N_hier = [1, 2]
    assert np.array_equal(N_hier, known_N_hier)


def test_n_dim_setter():
    """
    test to see if the current N_dim is properly being stored
    """
    TS = TrajStorage()
    TS.n_dim = 1
    N_dim = TS.n_dim
    known_n_dim = 1
    assert N_dim == known_n_dim

    TS.n_dim = 2
    n_dim = TS.n_dim
    known_n_dim = 2
    assert n_dim == known_n_dim


def test_aux_list_setter():
    """
    Test to see if aux list if properly being stored. If store_aux = False nothing
    gets stored. If store_aux = True list_aux is stored
    """
    TS = TrajStorage()
    TS.store_aux = False
    TS.aux_list = [1]
    aux_list = TS.aux_list
    known_aux_list = []
    assert np.array_equal(aux_list, known_aux_list)

    TS.store_aux = True
    TS.aux_list = [2]
    aux_list = TS.aux_list
    known_aux_list = [[2]]
    assert np.array_equal(aux_list, known_aux_list)


# Adaptive Trajectory Storage test
# --------------------------------------------------------------------------------------
def test_adap_psi_traj_setter():
    """
    test to see if the current psi_traj is properly being stored in the adaptive storage
    when the input is not a list
    """
    ATS = AdapTrajStorage()
    ATS.n_dim = 2
    ATS.state_list = [0, 1]
    ATS.psi_traj = [2]
    traj = ATS.psi_traj
    known = [np.array((2, 2), dtype=np.complex128)]
    assert np.array_equal(traj, known)


# aux setter test
# ---------------------------------------
def test_adap_aux_new():
    """
    test to see if aux_new is properly being stored in the adaptive storage
    """
    ATS = AdapTrajStorage()
    ATS.aux = [[tuple((0, 1)), tuple((2, 3))]], [tuple((0, 1))], [tuple((2, 3))]
    aux_new = ATS.list_aux_new
    known_aux_new = [[[tuple((0, 1)),tuple((2, 3))]]]
    assert np.array_equal(aux_new, known_aux_new)


def test_adap_aux_pop():
    """
    test to see if aux_critical is properly being stored in the adaptive storage
    """
    ATS = AdapTrajStorage()
    ATS.aux = [[tuple((0,1)),tuple((2,3))]],[tuple((0, 1))], [tuple((2, 3))]
    aux_crit = ATS.list_aux_pop
    known_aux_crit = [[tuple((0, 1))]]
    assert np.array_equal(aux_crit, known_aux_crit)


def test_adap_aux_boundary():
    """
    test to see if aux_boundary is properly being stored in the adaptive storage
    """
    ATS = AdapTrajStorage()
    ATS.aux = [[tuple((0,1)),tuple((2,3))]],[tuple((0, 1))], [tuple((2, 3))]
    aux_bound = ATS.list_aux_boundary
    known_aux_bound = [[tuple((2, 3))]]
    assert np.array_equal(aux_bound, known_aux_bound)


def test_adap_n_hier():
    """
    test to see if the hierarchy number is properly being stored in the adaptive storage
    """
    ATS = AdapTrajStorage()
    ATS.aux = [[[0],[1],[2],[3]],[[0], [1]], [[2], [3]]]
    n_hier = ATS.list_nhier
    known = [4]
    assert np.array_equal(n_hier, known)
# ----------------------------------------


def test_adap_state_list_setter():
    """
    test to see if the current state_list is properly being stored in the adaptive
    storage when the input is not of type list
    """
    ATS = AdapTrajStorage()
    ATS.state_list = [0, 2]
    state = ATS.state_list
    known = [[0, 2]]
    assert np.array_equal(state, known)
