import numpy as np
import pytest
import warnings
from scipy import sparse
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.storage.hops_storage import HopsStorage
from mesohops.basis.hops_aux import AuxiliaryVector
from .utils import compare_dictionaries

# List of hierarchy properties that depend on the previous time step
list_hierarchy_properties_path_dependent = [
    '_new_aux_index_conn_by_mode',  # New auxiliary index connections by mode
    '_new_aux_id_conn_by_mode',  # New auxiliary ID connections by mode
    '_stable_aux_id_conn_by_mode',  # Stable auxiliary ID connections by mode
    '_auxiliary_list',  # List of current auxiliary vectors (main storage)
    '__previous_auxiliary_list',  # Auxiliary list from previous step
    '__list_aux_stable',  # Auxiliaries stable between steps
    '__list_aux_remove',  # Auxiliaries to remove in update
    '__list_aux_add',  # Auxiliaries to add in update
    '__previous_list_auxstable_index',  # Indices of stable auxiliaries from previous step
    '_previous_list_modes_in_use',  # Modes in use from previous step
]

list_hierarchy_properties_obj = ['system']

list_system_properties_path_dependent = [
    '__previous_state_list',  # Previous state list (for adaptive updates)
    '__list_add_state',  # States to add in update
    '__list_stable_state',  # States stable between updates
    '_list_boundary_state',  # States coupled to basis by Hamiltonian
    '__list_absindex_new_state_modes',  # New state mode indices (absolute)
]
list_system_properties_obj = []

list_mode_properties_path_dependent= []
list_mode_properties_obj = ['system', 'hierarchy']

list_aux_properties_path_dependent = []
list_aux_properties_obj = []


def get_private(obj, name: str):
    if name.startswith('__'):
        cls = obj.__class__
        mangled = f"_{cls.__name__}{name}"
        return getattr(obj, mangled)
    else:
        return getattr(obj, name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_hops_nonadaptive():
    """Factory fixture for creating and initialising a non-adaptive HOPS object."""
    # Parameters mirrored from test_hops_trajectory
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 500.0,
        "TAU": 0.5,
    }

    noise2_param = {
        "SEED": 1010101,
        "MODEL": "FFT_FILTER",
        "TLEN": 500.0,
        "TAU": 0.5,
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

    sys_param_with_noise2 = sys_param.copy()
    sys_param_with_noise2.update({
        "L_NOISE2": [loperator[0], loperator[0], loperator[1], loperator[1]],
        "ALPHA_NOISE2": bcf_exp,
        "PARAM_NOISE2": [[10.0, 10.0], [5.0, 5.0], [10.0, 10.0], [5.0, 5.0]],
    })

    system_ltc_params = {
        "PARAM_LT_CORR": [250.0 / 1000.0, 250.0 / 2000.0],
        "L_LT_CORR": [loperator[0], loperator[1]]
    }

    hier_param = {"MAXHIER": 3}

    eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

    integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS': None,
        'EFFECTIVE_NOISE_INTEGRATION': False,
    }

    psi_0 = [1.0 + 0.0j, 0.0 + 0.0j]

    def _factory(sys_p=[sys_param, sys_param_with_noise2],
                 system_ltc_params = system_ltc_params,
                 psi0=psi_0,
                 noise_p=noise_param,
                 hier_p=hier_param,
                 eom_p=eom_param,
                 integ_p=integrator_param,
                 storage_param=None,
                 flag_noise2=False,
                 noise2_p=noise2_param,
                 flag_lt_corr=True):
        if flag_noise2:
            sys_p = sys_p[1]
        else:
            sys_p = sys_p[0]
            noise2_p = None

        if flag_lt_corr:
            sys_p.update(system_ltc_params)

        hops = HOPS(
            sys_p,
            noise_param=noise_p,
            noise2_param=noise2_p,
            hierarchy_param=hier_p,
            eom_param=eom_p,
            integration_param=integ_p,
            storage_param=storage_param,
        )
        hops.initialize(psi0)
        return hops

    return _factory

@pytest.fixture
def make_hops_adaptive():
        nsite = 10
        coupling = 50
        loperator = np.zeros([nsite, nsite, nsite], dtype=np.float64)
        for i in range(nsite):
            loperator[i, i, i] = 1.0

        gw_sysbath = []
        lop_list = []
        for i in range(nsite):
            gw_sysbath.extend([[20.0, 20.0], [10.0, 10.0]])
            lop_list.extend([loperator[i], loperator[i]])

        hamiltonian = np.zeros([nsite, nsite], dtype=np.float64)
        for i in range(nsite - 1):
            hamiltonian[i, i + 1] = coupling
            hamiltonian[i + 1, i] = coupling

        sys_param_adapt = {
            "HAMILTONIAN": hamiltonian,
            "GW_SYSBATH": gw_sysbath,
            "L_HIER": lop_list,
            "L_NOISE1": lop_list,
            "ALPHA_NOISE1": bcf_exp,
            "PARAM_NOISE1": gw_sysbath,
        }

        psi0 = np.zeros(nsite, dtype=np.complex128)
        psi0[0] = 1.0 + 0.0j

        noise_param = {
            "SEED": 0,
            "MODEL": "FFT_FILTER",
            "TLEN": 500.0,
            "TAU": 0.5,
        }

        hier_param = {"MAXHIER": 3}

        eom_param = {
        "TIME_DEPENDENCE": False,
        "EQUATION_OF_MOTION": "NORMALIZED NONLINEAR",
        "DELTA_A": 1e-3,
        "DELTA_S": 1e-3,
        "UPDATE_STEP": 10,
        "ADAPTIVE_S": True,
        "ADAPTIVE_H": True,
        }

        integrator_param = {
            "INTEGRATOR": "RUNGE_KUTTA",
            'EARLY_ADAPTIVE_INTEGRATOR': 'INCH_WORM',
            'EARLY_INTEGRATOR_STEPS': 5,
            'INCHWORM_CAP': 5,
            'STATIC_BASIS': None,
            'EFFECTIVE_NOISE_INTEGRATION': False,
        }

        def _factory(sys_p=sys_param_adapt,
                     psi0=psi0,
                     noise_p=noise_param,
                     hier_p=hier_param,
                     eom_p=eom_param,
                     integ_p=integrator_param,
                     storage_param=None,
                     ):
            hops = HOPS(
                sys_p,
                noise_param=noise_p,
                hierarchy_param=hier_p,
                eom_param=eom_p,
                integration_param=integ_p,
                storage_param=storage_param,
            )
            hops.initialize(psi0)
            return hops

        return _factory

def test_compare_dictionaries(tmp_path, make_hops_nonadaptive):
    """
    This test checks that compare_dictionaries() correctly manages the comparison
    between dictionaries with a variety of different data types and nesting.

    It tests:
    1. Dictionaries with different keys
    2. Dictionaries with the same keys but different values
    3. Dictionaries with the same keys but different value types
    4. Dictionaries with nested structures that differ
    5. Dictionaries with arrays or lists that differ
    6. Edge cases like empty dictionaries, None values, etc.

    Specific test cases include:
    - Test 1: Identical dictionaries should compare equal
      Verifies that comparing a dictionary with itself works correctly.

    - Test 2: Dictionaries with different keys should raise an error
      Compares non-adaptive and adaptive storage dictionaries which have different keys.

    - Test 3: Dictionaries with the same keys but different values
      Modifies a time value in a copy of a dictionary and verifies that comparison fails.

    - Test 4: Dictionaries with the same keys but different value types
      Tests that a list and a numpy array with the same values are considered equal.

    - Test 5: Dictionaries with arrays that differ slightly
      Verifies that small differences (within default tolerance) are ignored.

    - Test 6: Dictionaries with arrays that differ significantly
      Confirms that larger differences cause the comparison to fail.

    - Test 7: Dictionaries with nested structures
      Tests comparison of dictionaries containing nested dictionaries with differences.

    - Test 8: Empty dictionaries
      Verifies that empty dictionaries compare equal.

    - Test 9: None values
      Tests comparison of dictionaries containing None values.

    - Test 10: Sparse matrices
      Verifies comparison of dictionaries containing scipy sparse matrices.

    - Test 11: Lists of different lengths
      Tests that lists with different lengths are detected as different.

    - Test 12: Object arrays
      Tests comparison of numpy object arrays containing dictionaries.

    - Additional tests with actual HOPS objects:
      Verifies that checkpoint data matches the original and that modifications
      to the loaded data are detected correctly.
    """
    # Create a HopsStorage object with non-adaptive mode
    storage_non_adaptive = HopsStorage(False, {
        'psi_traj': True,
        't_axis': True,
        'z_mem': True,
    })

    # Create a HopsStorage object with adaptive mode
    storage_adaptive = HopsStorage(True, {
        'psi_traj': True,
        't_axis': True,
        'z_mem': True,
        'aux_list': True,
        'state_list': True,
        'list_nhier': True,
        'list_nstate': True,
        'list_aux_norm': True,
    })

    # Set dimensions for both storage objects
    storage_non_adaptive.n_dim = 5
    storage_adaptive.n_dim = 5

    # Create test data
    phi_new = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.complex128)
    aux_list = [AuxiliaryVector([], 4), AuxiliaryVector([[0, 1]], 4)]
    state_list = [0, 1, 2, 3, 4]
    t_new = 1.0
    z_mem_new = np.array([1.0, 2.0, 3.0], dtype=np.complex128)

    # Store data in both storage objects
    storage_non_adaptive.store_step(
        phi_new=phi_new,
        aux_list=aux_list,
        state_list=state_list,
        t_new=t_new,
        z_mem_new=z_mem_new
    )

    storage_adaptive.store_step(
        phi_new=phi_new,
        aux_list=aux_list,
        state_list=state_list,
        t_new=t_new,
        z_mem_new=z_mem_new
    )

    # Create reference dictionaries for testing
    dict1 = storage_non_adaptive.data.copy()
    dict2 = storage_adaptive.data.copy()

    # Test 1: Dictionaries with the same content should compare equal
    compare_dictionaries(dict1, dict1)
    compare_dictionaries(dict2, dict2)
    print("Test 1 passed: Identical dictionaries compare equal")

    # Test 2: Dictionaries with different keys should raise an error
    with pytest.raises(AssertionError, match="Dictionaries must have the same keys"):
        compare_dictionaries(dict1, dict2)
    print("Test 2 passed: Dictionaries with different keys raise the expected error")

    # Test 3: Dictionaries with the same keys but different values
    dict3 = dict1.copy()
    dict3['t_axis'] = [2.0]  # Change the time value
    with pytest.raises(AssertionError):
        compare_dictionaries(dict1, dict3)
    print("Test 3 passed: Dictionaries with different values raise an error")

    # Test 4: Dictionaries with the same keys but different value types
    dict4 = dict1.copy()
    dict4['t_axis'] = np.array([1.0])  # Change list to numpy array
    compare_dictionaries(dict1, dict4)
    print("Test 4 passed: Dictionaries with compatible types compare equal")

    # Test 5: Dictionaries with arrays that differ slightly
    dict5 = dict1.copy()
    dict5['psi_traj'] = [np.array(dict1['psi_traj'][0]) + 1e-10]  # Small difference
    compare_dictionaries(dict1, dict5)
    print("Test 5 passed: Arrays with small differences compare equal within tolerance")

    # Test 6: Dictionaries with arrays that differ significantly
    dict6 = dict1.copy()
    dict6['psi_traj'] = [np.array(dict1['psi_traj'][0]) + 0.1]  # Larger difference
    with pytest.raises(AssertionError):
        compare_dictionaries(dict1, dict6)
    print("Test 6 passed: Arrays with significant differences raise an error")

    # Test 7: Dictionaries with nested structures
    dict7 = {'nested': {'a': 1, 'b': 2}, 'array': np.array([1, 2, 3])}
    dict8 = {'nested': {'a': 1, 'b': 3}, 'array': np.array([1, 2, 3])}
    with pytest.raises(AssertionError):
        compare_dictionaries(dict7, dict8)
    print("Test 7 passed: Nested dictionaries with differences raise an error")

    # Test 8: Edge case - empty dictionaries
    compare_dictionaries({}, {})
    print("Test 8 passed: Empty dictionaries compare equal")

    # Test 9: Edge case - None values
    dict9 = {'a': None}
    dict10 = {'a': None}
    dict11 = {'a': 0}
    compare_dictionaries(dict9, dict10)
    with pytest.raises(AssertionError):
        compare_dictionaries(dict9, dict11)
    print("Test 9 passed: Dictionaries with None values handled correctly")

    # Test 10: Sparse matrices
    sparse_mat1 = sparse.csr_array(([1, 2, 3], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    sparse_mat2 = sparse.csr_array(([1, 2, 3], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))
    sparse_mat3 = sparse.csr_array(([1, 2, 4], ([0, 1, 2], [0, 1, 2])), shape=(3, 3))

    dict12 = {'sparse': sparse_mat1}
    dict13 = {'sparse': sparse_mat2}
    dict14 = {'sparse': sparse_mat3}

    compare_dictionaries(dict12, dict13)
    with pytest.raises(AssertionError):
        compare_dictionaries(dict12, dict14)
    print("Test 10 passed: Dictionaries with sparse matrices handled correctly")

    # Test 11: Lists of different lengths
    dict15 = {'list': [1, 2, 3]}
    dict16 = {'list': [1, 2]}
    with pytest.raises(AssertionError):
        compare_dictionaries(dict15, dict16)
    print("Test 11 passed: Lists of different lengths raise an error")


    # Test 12: Object arrays
    obj_array1 = np.array([{'a': 1}, {'b': 2}], dtype=object)
    obj_array2 = np.array([{'a': 1}, {'b': 2}], dtype=object)
    obj_array3 = np.array([{'a': 1}, {'b': 3}], dtype=object)

    dict17 = {'obj_array': obj_array1}
    dict18 = {'obj_array': obj_array2}
    dict19 = {'obj_array': obj_array3}

    compare_dictionaries(dict17, dict18)
    with pytest.raises(AssertionError):
        compare_dictionaries(dict17, dict19)
    print("Test 12 passed: Object arrays handled correctly")


    # Test with actual HOPS objects
    hops = make_hops_nonadaptive()
    hops.propagate(100.0, 1.0)

    ckpt_path = tmp_path / "traj.npz"
    hops.save_checkpoint(str(ckpt_path))

    storage_mid = {k: list(v) if isinstance(v, list) else v for k, v in hops.storage.data.items()
                   if k != 'ADAPTIVE'}

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    for key in storage_mid:
        np.testing.assert_array_equal(hops_loaded.storage.data[key], storage_mid[key])
    print("Test with HOPS objects passed: Checkpoint data matches original")


    # Modify the loaded data to test failure cases
    if 't_axis' in hops_loaded.storage.data:
        original_t_axis = hops_loaded.storage.data['t_axis'].copy()
        hops_loaded.storage.data['t_axis'] = [t + 0.1 for t in hops_loaded.storage.data['t_axis']]

        with pytest.raises(AssertionError):
            for key in storage_mid:
                np.testing.assert_array_equal(hops_loaded.storage.data[key], storage_mid[key])
        print("Test with modified HOPS data passed: Modified data detected correctly")

        # Restore the original data
        hops_loaded.storage.data['t_axis'] = original_t_axis



def test_checkpoint_nonadaptive(tmp_path, make_hops_nonadaptive):
    """Checkpoints for a standard, non-adaptive trajectory."""

    hops = make_hops_nonadaptive()
    hops.propagate(100.0, 1.0)

    ckpt_path = tmp_path / "traj.npz"
    hops.save_checkpoint(str(ckpt_path))

    phi_mid = hops.phi.copy()
    t_mid = hops.t
    storage_mid = {k: list(v) if isinstance(v, list) else v for k, v in hops.storage.data.items()
                   if k != 'ADAPTIVE'}

    hops.propagate(100.0, 1.0)
    phi_final = hops.phi.copy()
    storage_final = hops.storage.data
    t_final = hops.t

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    assert hops_loaded.t == t_mid

    # Test storage_mid values
    for key in storage_mid:
        np.testing.assert_array_equal(hops_loaded.storage.data[key], storage_mid[key])

    hops_loaded.propagate(100.0, 1.0)

    np.testing.assert_allclose(hops_loaded.phi, phi_final, atol=1e-100)
    assert hops_loaded.t == t_final
    for key in storage_final:
        np.testing.assert_array_equal(hops_loaded.storage.data[key], storage_final[key])


def test_checkpoint_early_time_integration(tmp_path, make_hops_nonadaptive):
    """Ensures checkpoints work during the early-time integration phase."""

    # Create a trajectory and propagate for fewer steps than EARLY_INTEGRATOR_STEPS
    hops = make_hops_nonadaptive(integ_p={"EARLY_ADAPTIVE_INTEGRATOR": "INCH_WORM",
                                          "EARLY_INTEGRATOR_STEPS": 50})
    hops.propagate(50.0, 2.0)

    assert hops.use_early_integrator  # Early integrator should still be active
    early_counter_mid = hops._early_step_counter

    ckpt_path = tmp_path / "traj_early.npz"
    hops.save_checkpoint(str(ckpt_path))

    phi_mid = hops.phi.copy()
    t_mid = hops.t
    storage_mid = {k: list(v) if isinstance(v, list) else v
                   for k, v in hops.storage.data.items()
                   if k != 'ADAPTIVE'}

    # Continue propagation beyond the early integration window
    hops.propagate(100.0, 2.0)
    phi_final = hops.phi.copy()
    t_final = hops.t
    early_counter_final = hops._early_step_counter
    storage_final = hops.storage.data

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))

    # Confirm the loaded trajectory matches the checkpoint state
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    assert hops_loaded.t == t_mid
    assert hops_loaded._early_step_counter == early_counter_mid
    assert hops_loaded.use_early_integrator
    compare_dictionaries(storage_mid, hops_loaded.storage.data)

    # Continue propagation from the loaded state
    hops_loaded.propagate(100., 2.0)

    # Final states should match the original trajectory
    np.testing.assert_allclose(hops_loaded.phi, phi_final, atol=1e-100)
    assert hops_loaded.t == t_final
    assert hops_loaded._early_step_counter == early_counter_final
    compare_dictionaries(storage_final, hops_loaded.storage.data)


def test_checkpoint_without_lt_corr(tmp_path, make_hops_nonadaptive):
    """Tests checkpointing without low temperature corrections."""
    # Initialize HOPS with low temperature corrections
    hops = make_hops_nonadaptive(flag_lt_corr=False)

    # Propagate the system
    hops.propagate(100.0, 1.0)

    # Save the checkpoint
    ckpt_path = tmp_path / "traj_lt_corr.npz"
    hops.save_checkpoint(str(ckpt_path))

    # Save the mid-propagation state
    phi_mid = hops.phi.copy()
    t_mid = hops.t
    storage_mid = {k: list(v) if isinstance(v, list) else v for k, v in hops.storage.data.items()
                   if k != 'ADAPTIVE'}
    lt_corr_param_mid = hops.basis.system.list_lt_corr_param.copy()

    # Continue propagation
    hops.propagate(100.0, 1.0)
    phi_final = hops.phi.copy()
    t_final = hops.t
    storage_final = hops.storage.data

    # Load the checkpoint
    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))

    # Verify the loaded state matches the mid-propagation state
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    assert hops_loaded.t == t_mid

    # Test storage_mid values
    for key in storage_mid:
        np.testing.assert_array_equal(hops_loaded.storage.data[key], storage_mid[key])

    # Continue propagation from the loaded state
    hops_loaded.propagate(100.0, 1.0)

    # Verify the final state matches the expected final state
    np.testing.assert_allclose(hops_loaded.phi, phi_final, atol=1e-100)
    assert hops_loaded.t == t_final
    for key in storage_final:
        np.testing.assert_array_equal(hops_loaded.storage.data[key], storage_final[key])

def test_checkpoint_uncompressed(tmp_path, make_hops_nonadaptive):
    """Verifies checkpoints saved without compression."""

    hops = make_hops_nonadaptive()
    hops.propagate(50.0, 1.0)

    ckpt_path = tmp_path / "traj_uncompressed.npz"
    hops.save_checkpoint(str(ckpt_path), compress=False)

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    np.testing.assert_allclose(hops_loaded.phi, hops.phi)
    assert hops_loaded.t == hops.t

def test_checkpoint_missing_directory(tmp_path, make_hops_nonadaptive):
    """Ensures saving to a missing directory raises an error."""

    hops = make_hops_nonadaptive()
    ckpt_path = tmp_path / "missing" / "traj.npz"
    with pytest.raises(FileNotFoundError):
        hops.save_checkpoint(str(ckpt_path))

def test_checkpoint_adaptive_storage(tmp_path, make_hops_adaptive):
    """Check storage recovery with adaptive propagation."""
    hops = make_hops_adaptive(
        storage_param={
            'phi_traj': True,
            'psi_traj': True,
            't_axis': True,
            'z_mem': True,
            'aux_list': True,
            'state_list': True,
            'list_nhier': True,
            'list_nstate': True,
            'list_aux_norm': True,
        },
    )

    # Propagate with adaptive calculation
    hops.propagate(50.0, 2.0)

    ckpt_path = tmp_path / "traj_adaptive.npz"
    hops.save_checkpoint(str(ckpt_path))

    phi_mid = hops.phi.copy()
    t_mid = hops.t
    storage_mid = {k: list(v) if isinstance(v, list) else v
                   for k, v in hops.storage.data.items()
                   if k != 'ADAPTIVE'}


    hops.propagate(100.0, 2.0)
    phi_final = hops.phi.copy()
    storage_final = hops.storage.data
    t_final = hops.t

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    assert hops_loaded.t == t_mid

    compare_dictionaries(storage_mid, hops_loaded.storage.data)
    assert len(hops_loaded.auxiliary_list[-1]) == len(hops.auxiliary_list[-1])

    hops_loaded.propagate(100.0, 2.0)

    np.testing.assert_allclose(hops_loaded.phi, phi_final, atol=1e-100)
    assert hops_loaded.t == t_final
    compare_dictionaries(storage_final, hops_loaded.storage.data)


def test_checkpoint_adaptive_hierarchy(tmp_path, make_hops_adaptive):
    """Ensures hierarchy objects are restored correctly in adaptive runs."""

    hops = make_hops_adaptive()

    # Propagate with adaptive calculation
    hops.propagate(50.0, 2.0)
    list_hierarchy_properties = [str for str in hops.basis.hierarchy.__class__.__slots__]
    list_hierarchy_properties_mid = [str for str in list_hierarchy_properties
                                     if not (str in list_hierarchy_properties_path_dependent) and
                                        not (str in list_hierarchy_properties_obj)]
    list_hierarchy_properties_fin = [str for str in list_hierarchy_properties
                                     if not (str in list_hierarchy_properties_obj)]
    original_hierarchy_mid = {prop: get_private(hops.basis.hierarchy, prop)
                             for prop in list_hierarchy_properties_mid}

    ckpt_path = tmp_path / "traj_adaptive.npz"
    hops.save_checkpoint(str(ckpt_path))
    hops.propagate(100.0, 2.0)
    original_hierarchy_fin = {prop: get_private(hops.basis.hierarchy, prop)
                             for prop in list_hierarchy_properties_fin}


    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    loaded_hierarchy_mid = {prop: get_private(hops_loaded.basis.hierarchy, prop)
                             for prop in list_hierarchy_properties_mid}
    hops_loaded.propagate(100.0, 2.0)
    loaded_hierarchy_fin = {prop: get_private(hops_loaded.basis.hierarchy, prop)
                            for prop in list_hierarchy_properties_fin}

    # Compare the hierarchy properties
    compare_dictionaries(original_hierarchy_mid, loaded_hierarchy_mid)
    compare_dictionaries(original_hierarchy_fin, loaded_hierarchy_fin)


def test_checkpoint_adaptive_modes(tmp_path, make_hops_adaptive):
    """Check that mode information is preserved across checkpoints."""
    hops = make_hops_adaptive()

    # Propagate with adaptive calculation
    hops.propagate(50.0, 2.0)

    list_mode_properties = [str for str in hops.basis.mode.__class__.__slots__]
    list_mode_properties_mid = [str for str in list_mode_properties
                                     if not (str in list_mode_properties_path_dependent) and
                                        not (str in list_mode_properties_obj)]
    list_mode_properties_fin = [str for str in list_mode_properties
                                     if not (str in list_mode_properties_obj)]

    original_mode_mid = {prop: get_private(hops.basis.mode, prop)
                              for prop in list_mode_properties_mid}

    ckpt_path = tmp_path / "traj_adaptive.npz"
    hops.save_checkpoint(str(ckpt_path))
    hops.propagate(100.0, 2.0)
    original_mode_fin = {prop: get_private(hops.basis.mode, prop)
                              for prop in list_mode_properties_fin}

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    loaded_mode_mid = {prop: get_private(hops_loaded.basis.mode, prop)
                            for prop in list_mode_properties_mid}
    hops_loaded.propagate(100.0, 2.0)
    loaded_mode_fin = {prop: get_private(hops_loaded.basis.mode, prop)
                            for prop in list_mode_properties_fin}

    # Compare the hierarchy properties
    compare_dictionaries(original_mode_mid, loaded_mode_mid)
    compare_dictionaries(original_mode_fin, loaded_mode_fin)


def test_checkpoint_adaptive_system(tmp_path, make_hops_adaptive):
    """Verifies system data is reconstructed with adaptive parameters."""
    hops = make_hops_adaptive()

    ckpt_path = tmp_path / "traj_adaptive.npz"

    # Propagate with adaptive calculation
    hops.propagate(50.0, 2.0)

    list_system_properties = [str for str in hops.basis.system.__class__.__slots__]
    list_system_properties_mid = [str for str in list_system_properties
                                     if not (str in list_system_properties_path_dependent) and
                                        not (str in list_system_properties_obj)]
    list_system_properties_fin = [str for str in list_system_properties
                                     if not (str in list_system_properties_obj)]

    hops.save_checkpoint(str(ckpt_path))
    orig_system_mid = {prop: get_private(hops.basis.system, prop)
                       for prop in list_system_properties_mid}
    hops.propagate(100.0, 2.0)
    orig_system_fin = {prop: get_private(hops.basis.system, prop)
                       for prop in list_system_properties_fin}

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    load_system_mid = {prop: get_private(hops_loaded.basis.system, prop)
                       for prop in list_system_properties_mid}
    hops_loaded.propagate(100.0, 2.0)
    load_system_fin = {prop: get_private(hops_loaded.basis.system, prop)
                       for prop in list_system_properties_fin}

    compare_dictionaries(orig_system_mid, load_system_mid)
    compare_dictionaries(orig_system_fin, load_system_fin)

def test_checkpoint_adaptive_listaux(tmp_path, make_hops_adaptive):
    """Verifies auxiliary lists survive checkpointing in adaptive mode."""
    hops = make_hops_adaptive()

    ckpt_path = tmp_path / "traj_adaptive.npz"

    # Propagate with adaptive calculation
    hops.propagate(50.0, 2.0)
    list_aux_properties = [str for str in hops.auxiliary_list[0].__class__.__slots__]
    list_aux_properties_mid = [str for str in list_aux_properties
                                     if not (str in list_aux_properties_path_dependent) and
                                        not (str in list_aux_properties_obj)]
    list_aux_properties_fin = [str for str in list_aux_properties
                                     if not (str in list_aux_properties_obj)]
    hops.save_checkpoint(str(ckpt_path))

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    for aux_orig, aux_load in zip(hops.auxiliary_list, hops_loaded.auxiliary_list):
        dict_aux_orig = {key: get_private(aux_orig, key) for key in list_aux_properties_mid}
        dict_aux_load = {key: get_private(aux_load, key) for key in list_aux_properties_mid}
        compare_dictionaries(dict_aux_orig, dict_aux_load)

    hops.propagate(100.0, 2.0)
    hops_loaded.propagate(100.0, 2.0)
    for aux_orig, aux_load in zip(hops.auxiliary_list, hops_loaded.auxiliary_list):
        dict_aux_orig = {key: get_private(aux_orig, key) for key in list_aux_properties_fin}
        dict_aux_load = {key: get_private(aux_load, key) for key in list_aux_properties_fin}
        compare_dictionaries(dict_aux_orig, dict_aux_load)


def test_checkpoint_orphan_aux(tmp_path, make_hops_adaptive):
    """Auxiliary without parents should load and propagate correctly."""
    # Create a trajectory and propagate briefly to generate a checkpoint
    hops = make_hops_adaptive()
    hops.propagate(20.0, 2.0)


    list_aux_properties = [str for str in hops.auxiliary_list[0].__class__.__slots__]
    list_aux_properties_mid = [str for str in list_aux_properties
                                     if not (str in list_aux_properties_path_dependent) and
                                        not (str in list_aux_properties_obj)]
    list_aux_properties_fin = [str for str in list_aux_properties
                                     if not (str in list_aux_properties_obj)]


    list_hierarchy_properties = [str for str in hops.basis.hierarchy.__class__.__slots__]
    list_hierarchy_properties_mid = [str for str in list_hierarchy_properties
                                     if not (str in list_hierarchy_properties_path_dependent) and
                                        not (str in list_hierarchy_properties_obj)]
    list_hierarchy_properties_fin = [str for str in list_hierarchy_properties
                                     if not (str in list_hierarchy_properties_obj)]

    # Construct an auxiliary list composed entirely of orphans
    list_aux = [aux for aux in hops.auxiliary_list if (aux._sum == 2 or aux._sum == 0)]
    phi_tmp, dsystem_dt = hops.basis.update_basis(hops.phi, hops.state_list, list_aux)
    hops.phi = phi_tmp
    hops.dsystem_dt = dsystem_dt

    ckpt_path = tmp_path / "orig.npz"
    hops.save_checkpoint(str(ckpt_path))
    original_hierarchy_mid = {prop: get_private(hops.basis.hierarchy, prop)
                             for prop in list_hierarchy_properties_mid}

    # Loading the modified checkpoint should succeed even though the
    # auxiliary list lacks connections
    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    loaded_hierarchy_mid = {prop: get_private(hops_loaded.basis.hierarchy, prop)
                             for prop in list_hierarchy_properties_mid}
    for aux_orig, aux_load in zip(hops.auxiliary_list, hops_loaded.auxiliary_list):
        dict_aux_orig = {key: get_private(aux_orig, key) for key in list_aux_properties_mid}
        dict_aux_load = {key: get_private(aux_load, key) for key in list_aux_properties_mid}
        compare_dictionaries(dict_aux_orig, dict_aux_load)

    compare_dictionaries(hops.storage.data, hops_loaded.storage.data)
    compare_dictionaries(original_hierarchy_mid, loaded_hierarchy_mid)


    # Propagate further to ensure the trajectory is functional
    hops_loaded.propagate(50.0, 2.0)
    hops.propagate(50.0, 2.0)
    loaded_hierarchy_fin = {prop: get_private(hops_loaded.basis.hierarchy, prop)
                             for prop in list_hierarchy_properties_fin}
    original_hierarchy_fin = {prop: get_private(hops.basis.hierarchy, prop)
                             for prop in list_hierarchy_properties_fin}

    compare_dictionaries(hops.storage.data, hops_loaded.storage.data)
    compare_dictionaries(original_hierarchy_fin, loaded_hierarchy_fin)
    for aux_orig, aux_load in zip(hops.auxiliary_list, hops_loaded.auxiliary_list):
        dict_aux_orig = {key: get_private(aux_orig, key) for key in list_aux_properties_fin}
        dict_aux_load = {key: get_private(aux_load, key) for key in list_aux_properties_fin}
        compare_dictionaries(dict_aux_orig, dict_aux_load)



def test_checkpoint_with_noise2(tmp_path, make_hops_nonadaptive):
    # Initialize HOPS with updated system parameters
    hops = make_hops_nonadaptive(flag_noise2=True)

    assert hops.noise2.param["SEED"] == 1010101

    # Propagate the system
    hops.propagate(100.0, 2.0)

    # Save the checkpoint
    ckpt_path = tmp_path / "traj_noise2.npz"
    hops.save_checkpoint(str(ckpt_path))
    storage_mid = {k: list(v) if isinstance(v, list) else v
                   for k, v in hops.storage.data.items()
                   if k != 'ADAPTIVE'}

    # Save the mid-propagation state
    phi_mid = hops.phi.copy()

    # Continue propagation
    hops.propagate(100.0, 2.0)
    phi_final = hops.phi.copy()
    storage_final = hops.storage.data

    # Load the checkpoint
    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))
    assert hops_loaded.noise2.param["SEED"] == 1010101

    # Verify the loaded state matches the mid-propagation state
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    compare_dictionaries(storage_mid, hops_loaded.storage.data)

    # Continue propagation from the loaded state
    hops_loaded.propagate(100.0, 2.0)

    # Verify the final state matches the expected final state
    np.testing.assert_allclose(hops_loaded.phi, phi_final, atol=1e-100)
    compare_dictionaries(storage_final, hops_loaded.storage.data)


def test_checkpoint_interpolate(tmp_path, make_hops_nonadaptive):
    """Tests checkpointing with INTERPOLATE flag set to True."""
    # Create custom noise parameters with INTERPOLATE=True
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 500.0,
        "TAU": 0.5,
        "INTERPOLATE": True,
    }

    # Initialize HOPS with INTERPOLATE=True
    hops = make_hops_nonadaptive(noise_p=noise_param)

    # Propagate the system
    hops.propagate(100.0, 2.0)

    # Save the checkpoint
    ckpt_path = tmp_path / "traj_interpolate.npz"
    hops.save_checkpoint(str(ckpt_path))

    # Save the mid-propagation state
    phi_mid = hops.phi.copy()
    t_mid = hops.t
    storage_mid = {k: list(v) if isinstance(v, list) else v 
                  for k, v in hops.storage.data.items()
                  if k != 'ADAPTIVE'}

    # Continue propagation
    hops.propagate(100.0, 2.0)
    phi_final = hops.phi.copy()
    t_final = hops.t
    storage_final = hops.storage.data

    # Load the checkpoint
    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))

    # Verify the loaded state matches the mid-propagation state
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    assert hops_loaded.t == t_mid
    compare_dictionaries(storage_mid, hops_loaded.storage.data)

    # Verify the INTERPOLATE flag was correctly restored
    assert hops_loaded.noise1.param["INTERPOLATE"] == True

    # Continue propagation from the loaded state
    hops_loaded.propagate(100.0, 2.0)

    # Verify the final state matches the expected final state
    np.testing.assert_allclose(hops_loaded.phi, phi_final, atol=1e-100)
    assert hops_loaded.t == t_final
    compare_dictionaries(storage_final, hops_loaded.storage.data)

def test_checkpoint_adaptive_noise(tmp_path, make_hops_nonadaptive):
    """Tests checkpointing with ADAPTIVE noise flag set to True."""
    # Create custom noise parameters with ADAPTIVE=True
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 500.0,
        "TAU": 0.5,
        "ADAPTIVE": True,
    }

    # Initialize HOPS with ADAPTIVE=True
    hops = make_hops_nonadaptive(noise_p=noise_param)

    # Propagate the system
    hops.propagate(100.0, 2.0)

    # Save the checkpoint
    ckpt_path = tmp_path / "traj_adaptive_noise.npz"
    hops.save_checkpoint(str(ckpt_path))

    # Save the mid-propagation state
    phi_mid = hops.phi.copy()
    t_mid = hops.t
    storage_mid = {k: list(v) if isinstance(v, list) else v 
                  for k, v in hops.storage.data.items()
                  if k != 'ADAPTIVE'}

    # Continue propagation
    hops.propagate(100.0, 2.0)
    phi_final = hops.phi.copy()
    t_final = hops.t
    storage_final = hops.storage.data

    # Load the checkpoint
    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))

    # Verify the loaded state matches the mid-propagation state
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    assert hops_loaded.t == t_mid
    compare_dictionaries(storage_mid, hops_loaded.storage.data)

    # Verify the ADAPTIVE flag was correctly restored
    assert hops_loaded.noise1.param["ADAPTIVE"] == True

    # Continue propagation from the loaded state
    hops_loaded.propagate(100.0, 2.0)

    # Verify the final state matches the expected final state
    np.testing.assert_allclose(hops_loaded.phi, phi_final, atol=1e-100)
    assert hops_loaded.t == t_final
    compare_dictionaries(storage_final, hops_loaded.storage.data)

def test_checkpoint_store_raw_noise(tmp_path, make_hops_nonadaptive):
    """Tests checkpointing with STORE_RAW_NOISE flag set to True."""
    # Create custom noise parameters with STORE_RAW_NOISE=True
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 500.0,
        "TAU": 0.5,
        "STORE_RAW_NOISE": True,
    }

    # Initialize HOPS with STORE_RAW_NOISE=True
    hops = make_hops_nonadaptive(noise_p=noise_param)

    # Propagate the system
    hops.propagate(100.0, 2.0)

    # Save the checkpoint
    ckpt_path = tmp_path / "traj_store_raw_noise.npz"
    hops.save_checkpoint(str(ckpt_path))

    # Save the mid-propagation state
    phi_mid = hops.phi.copy()
    t_mid = hops.t
    storage_mid = {k: list(v) if isinstance(v, list) else v 
                  for k, v in hops.storage.data.items()
                  if k != 'ADAPTIVE'}

    # Continue propagation
    hops.propagate(100.0, 2.0)
    phi_final = hops.phi.copy()
    t_final = hops.t
    storage_final = hops.storage.data

    # Load the checkpoint
    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))

    # Verify the loaded state matches the mid-propagation state
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    assert hops_loaded.t == t_mid
    compare_dictionaries(storage_mid, hops_loaded.storage.data)

    # Verify the STORE_RAW_NOISE flag was correctly restored
    assert hops_loaded.noise1.param["STORE_RAW_NOISE"] == True

    # Continue propagation from the loaded state
    hops_loaded.propagate(100.0, 2.0)

    # Verify the final state matches the expected final state
    np.testing.assert_allclose(hops_loaded.phi, phi_final, atol=1e-100)
    assert hops_loaded.t == t_final
    compare_dictionaries(storage_final, hops_loaded.storage.data)

def test_checkpoint_noise_window(tmp_path, make_hops_nonadaptive):
    """Tests checkpointing with NOISE_WINDOW parameter set."""
    # Create custom noise parameters with NOISE_WINDOW=100.0
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 500.0,
        "TAU": 0.5,
        "NOISE_WINDOW": 100.0,
    }

    # Initialize HOPS with NOISE_WINDOW=100.0
    hops = make_hops_nonadaptive(noise_p=noise_param)

    # Propagate the system
    hops.propagate(100.0, 2.0)

    # Save the checkpoint
    ckpt_path = tmp_path / "traj_noise_window.npz"
    hops.save_checkpoint(str(ckpt_path))

    # Save the mid-propagation state
    phi_mid = hops.phi.copy()
    t_mid = hops.t
    storage_mid = {k: list(v) if isinstance(v, list) else v 
                  for k, v in hops.storage.data.items()
                  if k != 'ADAPTIVE'}

    # Continue propagation
    hops.propagate(100.0, 2.0)
    phi_final = hops.phi.copy()
    t_final = hops.t
    storage_final = hops.storage.data

    # Load the checkpoint
    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))

    # Verify the loaded state matches the mid-propagation state
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    assert hops_loaded.t == t_mid
    compare_dictionaries(storage_mid, hops_loaded.storage.data)

    # Verify the NOISE_WINDOW parameter was correctly restored
    assert hops_loaded.noise1.param["NOISE_WINDOW"] == 100.0

    # Continue propagation from the loaded state
    hops_loaded.propagate(100.0, 2.0)

    # Verify the final state matches the expected final state
    np.testing.assert_allclose(hops_loaded.phi, phi_final, atol=1e-100)
    assert hops_loaded.t == t_final
    compare_dictionaries(storage_final, hops_loaded.storage.data)

def test_checkpoint_seed_none(tmp_path, make_hops_nonadaptive):
    """Tests checkpointing with SEED=None."""
    # Create custom noise parameters with SEED=None
    noise_param = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 500.0,
        "TAU": 0.5,
    }

    # Initialize HOPS with SEED=None
    hops = make_hops_nonadaptive(noise_p=noise_param)

    # Propagate the system
    hops.propagate(100.0, 2.0)

    # Save the checkpoint
    ckpt_path = tmp_path / "traj_seed_none.npz"
    hops.save_checkpoint(str(ckpt_path))

    # Save the mid-propagation state
    phi_mid = hops.phi.copy()
    t_mid = hops.t
    storage_mid = {k: list(v) if isinstance(v, list) else v 
                  for k, v in hops.storage.data.items()
                  if k != 'ADAPTIVE'}

    # Load the checkpoint
    hops_loaded = HOPS.load_checkpoint(str(ckpt_path))

    # Verify the loaded state matches the mid-propagation state
    np.testing.assert_allclose(hops_loaded.phi, phi_mid)
    assert hops_loaded.t == t_mid
    compare_dictionaries(storage_mid, hops_loaded.storage.data)
    assert hops_loaded.noise1.param['SEED'] is None

def test_checkpoint_drop_seed_array(tmp_path, make_hops_nonadaptive):
    hops = make_hops_nonadaptive()
    ckpt_keep = tmp_path / "keep.npz"
    ckpt_drop = tmp_path / "drop.npz"
    hops.save_checkpoint(str(ckpt_keep), drop_seed=False)
    hops.save_checkpoint(str(ckpt_drop), drop_seed=True)

    params_keep = np.load(ckpt_keep, allow_pickle=True)["params"].item()
    params_drop = np.load(ckpt_drop, allow_pickle=True)["params"].item()

    assert np.array_equal(params_keep["noise1_param"]["SEED"], hops.noise1.param["SEED"])
    assert np.array_equal(params_keep["noise2_param"]["SEED"], hops.noise2.param["SEED"])
    assert "SEED" not in params_drop["noise1_param"]
    assert "SEED" not in params_drop["noise2_param"]


def test_load_checkpoint_add_seed(tmp_path, make_hops_nonadaptive, capsys):
    hops = make_hops_nonadaptive(flag_noise2=True)
    assert hops.noise2.param["SEED"] == 1010101
    ckpt_path = tmp_path / "add_seed.npz"
    hops.save_checkpoint(str(ckpt_path), drop_seed=True)

    new_seed1 = 7
    new_seed2 = 1010108

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path), add_seed1=new_seed1)
    assert hops_loaded.noise1.param["SEED"] == new_seed1
    assert hops_loaded.noise2.param["SEED"] == None

    hops_loaded = HOPS.load_checkpoint(str(ckpt_path), add_seed1=new_seed1,
                                       add_seed2=new_seed2)
    assert hops_loaded.noise1.param["SEED"] == 7
    assert hops_loaded.noise2.param["SEED"] == 1010108

    with warnings.catch_warnings(record=True) as w:
        hops_loaded = HOPS.load_checkpoint(str(ckpt_path), add_seed1=new_seed1,
                                          add_seed2=new_seed1)
        assert any("Using the same seed for both noise 1 and" in str(
            warning.message) for warning in w)


def test_noise1_seed_warning(tmp_path, make_hops_nonadaptive):
    """Test that a UserWarning is raised when noise1 seed is None and drop_seed is False."""
    # Create custom noise parameters with SEED=None
    noise_param = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 500.0,
        "TAU": 0.5,
    }

    # Initialize HOPS with SEED=None
    hops = make_hops_nonadaptive(noise_p=noise_param)

    # Propagate the system
    hops.propagate(50.0, 2.0)

    ckpt_path = tmp_path / "checkpoint.npz"
    with pytest.warns(UserWarning):
        hops.save_checkpoint(ckpt_path, drop_seed=False)
