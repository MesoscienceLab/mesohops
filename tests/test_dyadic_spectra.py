import numpy as np
import pytest
from scipy import sparse

from mesohops.trajectory.dyadic_spectra import DyadicSpectra as DHOPS
from mesohops.trajectory.dyadic_spectra import (prepare_spectroscopy_input_dict,
                                              prepare_chromophore_input_dict,
                                              prepare_convergence_parameter_dict)
from mesohops.util.bath_corr_functions import ishizaki_decomposition_bcf_dl


def _base_chromophore_dict(spectrum_type):
    M2_mu_ge = np.array([[0.5, 0.2, 0.1], [0.45, 0.1, 0.2]])
    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    if spectrum_type in ["ESA-R", "ESA-NR"]:
        H2_sys_hamiltonian = np.zeros((4, 4), dtype=np.complex128)
        H2_sys_hamiltonian[1:3, 1:3] = np.array([[0, -80], [-80, 0]])
        H2_sys_hamiltonian[3, 3] = 150
        # Let helper build default full-dimension L-operators for ESA.
        return prepare_chromophore_input_dict(
            M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes}
        )
    else:
        H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
        H2_sys_hamiltonian[1:, 1:] = np.array([[0, -80], [-80, 0]])
    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]
    return prepare_chromophore_input_dict(
        M2_mu_ge, H2_sys_hamiltonian, {"list_lop": list_lop, "list_modes": list_modes}
    )


def _build_dhops_for_spectrum(spectrum_type):
    cluster_dict = {
        "list_interaction_cluster_1": np.array([1, 2]),
        "list_interaction_cluster_2": np.array([1, 2]),
        "list_interaction_cluster_3": np.array([1, 2]),
    }
    if spectrum_type == "ABSORPTION":
        propagation_time_dict = {"t_1": 0.4}
        field_dict = {"E_1": np.array([0.0, 0.0, 1.0])}
    elif spectrum_type == "FLUORESCENCE":
        propagation_time_dict = {"t_2": 0.3, "t_3": 0.1}
        field_dict = {
            "E_1": np.array([0.0, 0.0, 1.0]),
            "E_sig": np.array([0.0, 0.0, 1.0]),
        }
    else:
        propagation_time_dict = {"t_1": 0.2, "t_2": 0.3, "t_3": 0.1}
        field_dict = {
            "E_1": np.array([0.0, 0.0, 1.0]),
            "E_2": np.array([0.0, 1.0, 0.0]),
            "E_3": np.array([1.0, 0.0, 0.0]),
            "E_sig": np.array([0.0, 0.0, 1.0]),
        }
    spectroscopy_dict = prepare_spectroscopy_input_dict(
        spectrum_type, propagation_time_dict, field_dict, cluster_dict
    )
    convergence_dict = prepare_convergence_parameter_dict(t_step=0.1, max_hier=2)
    return DHOPS(
        spectroscopy_dict,
        _base_chromophore_dict(spectrum_type),
        convergence_dict,
        seed=10,
    )


def test_DyadicSpectra():
    """
    Tests the DyadicSpectra class for properly unpacking input dictionaries, and ensures
    the Hamiltonian is the proper shape.
    """
    # Spectroscopy input dictionary
    seed = 10
    spectrum_type = "FLUORESCENCE"
    propagation_time_dict = {"t_2": 2.0, "t_3": 3.0}
    field_dict = {"E_1": np.array([0, 0, 1]), "E_sig": np.array([0, 0, 1])}
    cluster_dict = {
        "list_interaction_cluster_1": np.array([1, 2]),
        "list_interaction_cluster_2": np.array([1, 2]),
        "list_interaction_cluster_3": np.array([1, 2]),
    }

    spectroscopy_dict = prepare_spectroscopy_input_dict(spectrum_type,
                                                        propagation_time_dict,
                                                        field_dict, cluster_dict)

    # Chromophore input dictionary
    M2_mu_ge = np.array([np.array([0.5, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])

    list_lop = [sparse.coo_matrix(([1, 1], ([1, 2], [2, 1])), shape=(3, 3)),
                sparse.coo_matrix(([1, 1], ([2, 1], [1, 2])), shape=(3, 3))]

    # Case 1: list_modes
    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    bath_dict1 = {"list_lop": list_lop, "list_modes": list_modes}
    chromophore_dict_1 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict1)

    # Case 2: list_modes + nmodes_LTC
    nmodes_LTC = 1
    bath_dict2 = {"list_lop": list_lop, "list_modes": list_modes,
                  "nmodes_LTC": nmodes_LTC}
    chromophore_dict_2 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict2)

    # Case 3: list_modes + static_filter_list (Markovian)
    static_filter_list = [['Markovian', [True, False]]]
    bath_dict3 = {"list_lop": list_lop, "list_modes": list_modes,
                  "static_filter_list": static_filter_list}
    chromophore_dict_3 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict3)

    # Convergence parameter dictionaries
    convergence_dict_float_dt = prepare_convergence_parameter_dict(t_step=0.1,
                                                                   max_hier=12,
                                                                   delta_a=1e-10,
                                                                   delta_s=1e-10,
                                                                   set_update_step=1,
                                                                   set_f_discard=0.5)

    convergence_dict_int_dt = prepare_convergence_parameter_dict(t_step=2, max_hier=12,
                                                                 delta_a=1e-10,
                                                                 delta_s=1e-10,
                                                                 set_update_step=1,
                                                                 set_f_discard=0.5)

    # Test input dictionary unpacking

    # Case 1 + float dt
    dhops_1a = DHOPS(spectroscopy_dict, chromophore_dict_1, convergence_dict_float_dt,
                     seed)

    assert np.allclose(dhops_1a.H2_sys_hamiltonian, H2_sys_hamiltonian)
    assert np.allclose(dhops_1a.gw_sysbath_hier, chromophore_dict_1["gw_sysbath_hier"])
    for a, b in zip(dhops_1a.lop_list_hier, chromophore_dict_1["lop_list_hier"]):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(dhops_1a.gw_sysbath_noise,
                       chromophore_dict_1["gw_sysbath_noise"])
    for a, b in zip(dhops_1a.lop_list_noise, chromophore_dict_1["lop_list_noise"]):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(dhops_1a.ltc_param, chromophore_dict_1["ltc_param"])
    for a, b in zip(dhops_1a.lop_list_ltc, chromophore_dict_1["lop_list_ltc"]):
        assert np.allclose(a.toarray(), b.toarray())
    assert dhops_1a.static_filter_list is None
    assert np.allclose(dhops_1a.M2_mu_ge, M2_mu_ge)
    assert dhops_1a.n_chromophore == 2
    assert np.allclose(dhops_1a.list_interaction_cluster_1, np.array([1, 2]))
    assert np.allclose(dhops_1a.list_interaction_cluster_2, np.array([1, 2]))
    assert np.allclose(dhops_1a.list_interaction_cluster_3, np.array([1, 2]))
    assert dhops_1a.spectrum_type == spectrum_type
    assert np.allclose(dhops_1a.E_1, np.array([0, 0, 1]))
    assert np.allclose(dhops_1a.E_2, np.array([0, 0, 1]))
    assert np.allclose(dhops_1a.E_3, np.array([0, 0, 1]))
    assert np.allclose(dhops_1a.E_sig, np.array([0, 0, 1]))
    assert dhops_1a.t_1 == 0.0
    assert dhops_1a.t_2 == 2.0
    assert dhops_1a.t_3 == 3.0
    assert dhops_1a.list_t == [0.0, 2.0, 3.0]
    assert dhops_1a.t_step == 0.1
    assert dhops_1a.max_hier == 12
    assert dhops_1a.delta_a == 1e-10
    assert dhops_1a.delta_s == 1e-10
    assert dhops_1a.set_update_step == 1
    assert dhops_1a.set_f_discard == 0.5
    assert (np.shape(dhops_1a.H2_sys_hamiltonian)[0] == dhops_1a.n_state_hilb)
    assert dhops_1a.noise_param["TAU"] == 0.1 / 2

    # Case 1 + int dt
    dhops_1b = DHOPS(spectroscopy_dict, chromophore_dict_1, convergence_dict_int_dt,
                     seed)

    assert dhops_1b.noise_param["TAU"] == 0.5

    # Case 2
    dhops_2 = DHOPS(spectroscopy_dict, chromophore_dict_2, convergence_dict_float_dt,
                    seed)

    assert np.allclose(dhops_2.gw_sysbath_hier, chromophore_dict_2["gw_sysbath_hier"])
    for a, b in zip(dhops_2.lop_list_hier, chromophore_dict_2["lop_list_hier"]):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(dhops_2.gw_sysbath_noise,
                       chromophore_dict_2["gw_sysbath_noise"])
    for a, b in zip(dhops_2.lop_list_noise, chromophore_dict_2["lop_list_noise"]):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(dhops_2.ltc_param, chromophore_dict_2["ltc_param"])
    for a, b in zip(dhops_2.lop_list_ltc, chromophore_dict_2["lop_list_ltc"]):
        assert np.allclose(a.toarray(), b.toarray())
    assert dhops_2.static_filter_list is None

    # Case 3
    dhops_3 = DHOPS(spectroscopy_dict, chromophore_dict_3, convergence_dict_float_dt,
                    seed)

    assert np.allclose(dhops_3.gw_sysbath_hier, chromophore_dict_3["gw_sysbath_hier"])
    for a, b in zip(dhops_3.lop_list_hier, chromophore_dict_3["lop_list_hier"]):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(dhops_3.gw_sysbath_noise,
                       chromophore_dict_3["gw_sysbath_noise"])
    for a, b in zip(dhops_3.lop_list_noise, chromophore_dict_3["lop_list_noise"]):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(dhops_3.ltc_param, chromophore_dict_3["ltc_param"])
    for a, b in zip(dhops_3.lop_list_ltc, chromophore_dict_3["lop_list_ltc"]):
        assert np.allclose(a.toarray(), b.toarray())
    assert dhops_3.static_filter_list == [['Markovian', [True, False, True, False]]]


    # Test Hamiltonian shape compatibility with number of states

    H2_sys_hamiltonian_wrongshape = np.zeros((4, 4), dtype=np.complex128)
    bath_dict_wrongshape = {"list_lop": list_lop, "list_modes": list_modes}
    with pytest.raises(ValueError, match='Each list_lop operator must have shape'):
        prepare_chromophore_input_dict(
            M2_mu_ge, H2_sys_hamiltonian_wrongshape, bath_dict_wrongshape
        )

    # Test "INITIALIZATION_TIME" is greater than 0
    dhops_4 = DHOPS(spectroscopy_dict, chromophore_dict_1, convergence_dict_float_dt,
                    seed)
    dhops_4.calculate_spectrum()

    assert dhops_4.storage.metadata["INITIALIZATION_TIME"] > 0

    # Test "LIST_PROPAGATION_TIME" is correct length (Fluorescence)
    assert len(dhops_4.storage.metadata["LIST_PROPAGATION_TIME"]) == 2

    # Test "LIST_PROPAGATION_TIME" is correct length (Absorption)
    spectroscopy_dict_abs = (
        prepare_spectroscopy_input_dict("ABSORPTION",
                                        {"t_1": 1.0},
                                        {"E_1": np.array([0, 0, 1])},
                                        {"list_interaction_cluster_1":
                                             np.array([1, 2])}))
    dhops_5 = DHOPS(spectroscopy_dict_abs, chromophore_dict_1,
                    convergence_dict_float_dt, seed)
    dhops_5.calculate_spectrum()

    assert len(dhops_5.storage.metadata["LIST_PROPAGATION_TIME"]) == 1


def test_initialize(capsys):
    """
    Tests the initialization of the DyadicTrajectory class, and ensures that the class
    is properly initialized in both non-adaptive and adaptive cases.
    """
    # Spectroscopy input dictionary
    seed = 10
    spectrum_type = "ABSORPTION"
    propagation_time_dict = {"t_1": 1.0}
    field_dict = {"E_1": np.array([0, 0, 1])}
    cluster_dict = {"list_interaction_cluster_1": np.array([1, 2])}

    spectroscopy_dict = prepare_spectroscopy_input_dict(spectrum_type,
                                                        propagation_time_dict,
                                                        field_dict, cluster_dict)

    # Chromophore input dictionary
    M2_mu_ge = np.array([np.array([0.5, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])

    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]

    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    bath_dict = {"list_lop": list_lop, "list_modes": list_modes}
    chromophore_dict = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict)

    # Convergence parameter dictionaries
    convergence_dict = prepare_convergence_parameter_dict(t_step=0.1, max_hier=12,
                                                          delta_a=0, delta_s=0)

    # Testing initialized property works upon initialization
    dhops = DHOPS(spectroscopy_dict, chromophore_dict, convergence_dict, seed)
    assert dhops.initialized is False

    dhops.initialize()
    assert dhops.initialized is True

    # Testing that multiple calls to initialize() triggers a warning
    dhops.initialize()
    out, err = capsys.readouterr()
    assert "WARNING: DyadicTrajectory has already been initialized." in out.strip()

    # Testing delta_a/delta_s greater than 0 causes the trajectory to be ran adaptively
    convergence_dict_adaptive = prepare_convergence_parameter_dict(t_step=0.1,
                                                                   max_hier=12,
                                                                   delta_a=1e-10,
                                                                   delta_s=1e-10,
                                                                   set_update_step=2,
                                                                   set_f_discard=0.5)

    dhops_adaptive = DHOPS(spectroscopy_dict, chromophore_dict,
                           convergence_dict_adaptive, seed)
    dhops_adaptive.initialize()

    # Adaptive
    assert dhops_adaptive.basis.eom.param["ADAPTIVE"] is True
    assert dhops_adaptive.basis.eom.param["DELTA_A"] == 1e-10
    assert dhops_adaptive.basis.eom.param["DELTA_S"] == 1e-10
    assert dhops_adaptive.basis.eom.param["UPDATE_STEP"] == 2
    assert dhops_adaptive.basis.eom.param["F_DISCARD"] == 0.5

    # Nonadaptive
    assert dhops.basis.eom.param["ADAPTIVE"] is False


def test_hilb_operator():
    """
    Tests the _hilb_operator method of the DyadicTrajectory class, ensuring that the
    Hilbert raising and lowering operators are properly constructed.
    """
    # Spectroscopy input dictionary
    seed = 10
    spectrum_type = "ABSORPTION"
    propagation_time_dict = {"t_1": 1.0}
    field_dict = {"E_1": np.array([2, 3, 1])}
    cluster_dict = {"list_interaction_cluster_1": np.array([1, 2])}

    spectroscopy_dict = prepare_spectroscopy_input_dict(spectrum_type,
                                                        propagation_time_dict,
                                                        field_dict, cluster_dict)

    # Chromophore input dictionary
    M2_mu_ge = np.array([np.array([0.5, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])

    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]

    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    bath_dict = {"list_lop": list_lop, "list_modes": list_modes}
    chromophore_dict = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                      bath_dict)

    # Convergence parameter dictionaries
    convergence_dict = prepare_convergence_parameter_dict(t_step=0.1, max_hier=12,
                                                          delta_a=0, delta_s=0)

    dhops = DHOPS(spectroscopy_dict, chromophore_dict, convergence_dict, seed)

    # Testing that the Hilbert raising and lowering operators are properly constructed

    # Note that the value of 1.7 comes from the dot product of the field and the dipole
    dense_raise = np.zeros((3, 3), dtype=np.float64)
    dense_raise[np.array([1, 2]), 0] = 1.7

    dense_lower = np.zeros((3, 3), dtype=np.float64)
    dense_lower[0, np.array([1, 2])] = 1.7

    assert np.allclose(dhops._hilb_operator("g_to_e", np.array([2, 3, 1]),
                                            dhops.list_interaction_cluster_1).toarray(),
                       dense_raise)

    assert np.allclose(dhops._hilb_operator("e_to_g", np.array([2, 3, 1]),
                                            dhops.list_interaction_cluster_1).toarray(),
                       dense_lower)

    # Testing that the method raises an error if not given a valid transition_type
    with pytest.raises(
        ValueError,
        match="transition_type must be 'g_to_e', 'e_to_g', or 'e_to_ee'",
    ):
        dhops._hilb_operator("cha_cha_slide", np.array([2, 3, 1]),
                             dhops.list_interaction_cluster_1)

def test_final_dyad_operator():
    """
    Tests the _final_dyad_operator method of the DyadicTrajectory class, ensuring that
    the final dyad operator is properly constructed, and the time index is properly set.
    """
    # Spectroscopy input dictionary
    seed = 10
    spectrum_type = "FLUORESCENCE"
    propagation_time_dict = {"t_2": 2.0, "t_3": 3.0}
    field_dict = {"E_1": np.array([2, 3, 1]), "E_sig": np.array([1, 2, 3])}
    cluster_dict = {
        "list_interaction_cluster_1": np.array([1, 2]),
        "list_interaction_cluster_2": np.array([1, 2]),
        "list_interaction_cluster_3": np.array([1, 2]),
    }

    spectroscopy_dict = prepare_spectroscopy_input_dict(spectrum_type,
                                                        propagation_time_dict,
                                                        field_dict, cluster_dict)

    # Chromophore input dictionary
    M2_mu_ge = np.array([np.array([0.5, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])

    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]

    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    bath_dict = {"list_lop": list_lop, "list_modes": list_modes}
    chromophore_dict = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict)

    # Convergence parameter dictionaries
    convergence_dict = prepare_convergence_parameter_dict(t_step=0.1, max_hier=12,
                                                          delta_a=1e-10, delta_s=1e-10,
                                                          set_update_step=1,
                                                          set_f_discard=0.5)

    # Testing that the final dyad operator is properly constructed
    dhops = DHOPS(spectroscopy_dict, chromophore_dict, convergence_dict, seed)

    dyadic_f_op = np.zeros((6, 6), dtype=np.float64)
    dyadic_f_op[3, 1] = 1.2
    dyadic_f_op[3, 2] = 1.2

    assert np.allclose(dhops._final_dyad_operator()[0].toarray(), dyadic_f_op)

    # Testing that the time index is properly set
    assert dhops._final_dyad_operator()[1] == 20

def test_prepare_spectroscopy_input_dict(capsys):
    """
    Tests the prepare_spectroscopy_input_dict helper function, ensuring that the input
    dictionary is properly formatted and that errors are raised when necessary.
    """
    # spectrum_type cases
    absorption_spectrum_type = "ABSORPTION"
    fluorescence_spectrum_type = "FLUORESCENCE"
    bad_spectrum_type = "SHARKESCENCE"

    # Site definitions
    cluster_1 = np.array([1, 2])
    cluster_1_index_issue = np.array([0, 1])
    cluster_1_list = [1, 2]
    cluster_2 = np.array([1, 2])
    cluster_2_list = [1, 2]
    cluster_3 = np.array([1, 2])

    # Field definitions
    E1 = np.array([0, 0, 1])
    E1_list = [0, 0, 1]
    E1_wrong_length = np.array([0, 0])
    Esig = np.array([0, 0, 1])

    # Propagation time definitions
    t1 = 1
    t2 = 2
    t3 = 3

    # Test proper output
    abs_test = prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                               propagation_time_dict={"t_1": t1},
                                               field_dict={"E_1": E1},
                                               cluster_dict={"list_interaction_cluster_1": cluster_1})
    assert abs_test["spectrum_type"] == 'ABSORPTION'
    assert abs_test["t_1"] == t1
    assert abs_test["t_2"] == 0
    assert abs_test["t_3"] == 0
    assert np.allclose(abs_test["E_1"], E1)
    assert np.allclose(abs_test["E_sig"], E1)
    assert np.allclose(abs_test["list_interaction_cluster_1"], cluster_1)

    fluor_test = prepare_spectroscopy_input_dict(
        spectrum_type=fluorescence_spectrum_type,
        propagation_time_dict={"t_2": t2, "t_3": t3},
        field_dict={"E_1": E1, "E_sig": Esig},
        cluster_dict={
            "list_interaction_cluster_1": cluster_1,
            "list_interaction_cluster_2": cluster_2,
            "list_interaction_cluster_3": cluster_3,
        })
    assert fluor_test["spectrum_type"] == 'FLUORESCENCE'
    assert fluor_test["t_1"] == 0
    assert fluor_test["t_2"] == t2
    assert fluor_test["t_3"] == t3
    assert np.allclose(fluor_test["E_1"], E1)
    assert np.allclose(fluor_test["E_2"], E1)
    assert np.allclose(fluor_test["E_3"], Esig)
    assert np.allclose(fluor_test["E_sig"], Esig)
    assert np.allclose(fluor_test["list_interaction_cluster_1"], cluster_1)
    assert np.allclose(fluor_test["list_interaction_cluster_2"], cluster_2)
    assert np.allclose(fluor_test["list_interaction_cluster_3"], cluster_3)

    # Testing site definition errors

    # Case 1: list_interaction_cluster_1 not defined -> warning + ALL
    with pytest.warns(UserWarning, match='list_interaction_cluster_1 not defined'):
        cluster_all = prepare_spectroscopy_input_dict(
            spectrum_type=absorption_spectrum_type,
            propagation_time_dict={"t_1": t1},
            field_dict={"E_1": E1},
            cluster_dict={})
    assert cluster_all["list_interaction_cluster_1"] == "ALL"

    # Case 2: list_interaction_cluster_1 not a numpy array
    cluster_list = prepare_spectroscopy_input_dict(
        spectrum_type=absorption_spectrum_type,
        propagation_time_dict={"t_1": t1},
        field_dict={"E_1": E1},
        cluster_dict={"list_interaction_cluster_1": cluster_1_list})

    cluster_array = prepare_spectroscopy_input_dict(
        spectrum_type=absorption_spectrum_type,
        propagation_time_dict={"t_1": t1},
        field_dict={"E_1": E1},
        cluster_dict={"list_interaction_cluster_1": cluster_1})

    assert np.allclose(cluster_list["list_interaction_cluster_1"],
                       cluster_array["list_interaction_cluster_1"])

    # Case 3: sites indexed from 0
    with pytest.raises(ValueError, match="Clusters' indices should not include 0."):
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1},
                                        field_dict={"E_1": E1},
                                        cluster_dict={
                                            "list_interaction_cluster_1":
                                                cluster_1_index_issue})

    # Testing field input formatting

    # Case 1: Field not a numpy array
    with pytest.raises(ValueError, match='All field entries should be numpy arrays.'):
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1},
                                        field_dict={"E_1": E1_list},
                                        cluster_dict={"list_interaction_cluster_1":
                                                          cluster_1})

    # Case 2: Field not a numpy array with exactly 3 entries
    with pytest.raises(ValueError,
                      match='All field entries should be numpy arrays with exactly 3 '
                            'entries.'):
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1},
                                        field_dict={"E_1": E1_wrong_length},
                                        cluster_dict={"list_interaction_cluster_1":
                                                          cluster_1})

    # Testing under-defined absorption input

    # Case 1: t_1 not defined
    with pytest.raises(ValueError,
                      match='Propagation time after first field interaction \\(t_1\\) '
                            'must be defined as > 0 for absorption.'):
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={},
                                        field_dict={"E_1": E1},
                                        cluster_dict={"list_interaction_cluster_1":
                                                          cluster_1})

    # Case 2: E_1 not defined (warns but raises KeyError when accessed)
    with pytest.warns(UserWarning, match='E_1 is not defined'):
        with pytest.raises(KeyError):
            prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                            propagation_time_dict={"t_1": t1},
                                            field_dict={},
                                            cluster_dict={
                                                "list_interaction_cluster_1":
                                                    cluster_1})

    # Testing over-defined absorption input

    # Case 1: propagation_time_dict contains too many inputs
    with pytest.warns(UserWarning,
                      match='Only t_1 is necessary for absorption.'):
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1, "t_2": t2},
                                        field_dict={"E_1": E1},
                                        cluster_dict={
                                            "list_interaction_cluster_1":
                                                cluster_1})

    # Case 2: field_dict contains too many inputs
    with pytest.warns(UserWarning,
                      match='Only E_1 is necessary for absorption.'):
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1},
                                        field_dict={"E_1": E1, "E_sig": Esig},
                                        cluster_dict={
                                            "list_interaction_cluster_1":
                                                cluster_1})

    # Testing under-defined fluorescence input

    # Case 1: list_interaction_cluster_2/3 not defined -> warnings + ALL
    with pytest.warns(UserWarning) as warning_list:
        cluster_defaults = prepare_spectroscopy_input_dict(
            spectrum_type=fluorescence_spectrum_type,
            propagation_time_dict={"t_2": t2, "t_3": t3},
            field_dict={"E_1": E1, "E_sig": Esig},
            cluster_dict={"list_interaction_cluster_1": cluster_1})
    warning_messages = [str(item.message) for item in warning_list]
    assert any("list_interaction_cluster_2 not defined" in msg
               for msg in warning_messages)
    assert any("list_interaction_cluster_3 not defined" in msg
               for msg in warning_messages)
    assert cluster_defaults["list_interaction_cluster_2"] == "ALL"
    assert cluster_defaults["list_interaction_cluster_3"] == "ALL"

    # Case 2: list_interaction_cluster_2 not a numpy array
    cluster2_list = prepare_spectroscopy_input_dict(
        spectrum_type=fluorescence_spectrum_type,
        propagation_time_dict={"t_2": t2, "t_3": t3},
        field_dict={"E_1": E1, "E_sig": Esig},
        cluster_dict={
            "list_interaction_cluster_1": cluster_1,
            "list_interaction_cluster_2": cluster_2_list,
            "list_interaction_cluster_3": cluster_3,
        })
    cluster2_array = prepare_spectroscopy_input_dict(
        spectrum_type=fluorescence_spectrum_type,
        propagation_time_dict={"t_2": t2, "t_3": t3},
        field_dict={"E_1": E1, "E_sig": Esig},
        cluster_dict={
            "list_interaction_cluster_1": cluster_1,
            "list_interaction_cluster_2": cluster_2,
            "list_interaction_cluster_3": cluster_3,
        })
    assert np.allclose(cluster2_list["list_interaction_cluster_2"],
                       cluster2_array["list_interaction_cluster_2"])

    # Note: There is no need to test case if list_interaction_cluster_1 is not defined,
    # as this is already tested prior to determining the spectrum type.

    # Case 3: t_2 or t_3 not defined
    with pytest.raises(ValueError,
                       match='Propagation time after second field '
                             'interactions \\(t_2\\) must be defined for '
                             'FLUORESCENCE.'):
        prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                        propagation_time_dict={"t_3": t3},
                                        field_dict={"E_1": E1, "E_sig": Esig},
                                        cluster_dict={
                                            "list_interaction_cluster_1": cluster_1,
                                            "list_interaction_cluster_2": cluster_2,
                                            "list_interaction_cluster_3": cluster_3,
                                        })

    with pytest.raises(ValueError,
                       match='Propagation time after third field '
                             'interactions \\(t_3\\) must be defined for '
                             'FLUORESCENCE.'):
        prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                        propagation_time_dict={"t_2": t2},
                                        field_dict={"E_1": E1, "E_sig": Esig},
                                        cluster_dict={
                                            "list_interaction_cluster_1": cluster_1,
                                            "list_interaction_cluster_2": cluster_2,
                                            "list_interaction_cluster_3": cluster_3,
                                        })

    # Case 4: E_1 not defined
    with pytest.warns(UserWarning, match='E_1 is not defined'):
        fluorescence_default = prepare_spectroscopy_input_dict(
            spectrum_type=fluorescence_spectrum_type,
            propagation_time_dict={"t_2": t2, "t_3": t3},
            field_dict={"E_sig": Esig},
            cluster_dict={
                "list_interaction_cluster_1": cluster_1,
                "list_interaction_cluster_2": cluster_2,
                "list_interaction_cluster_3": cluster_3,
            })
    assert np.allclose(fluorescence_default["E_1"], np.array([0, 0, 1]))

    # Case 5: E_sig not defined
    with pytest.warns(UserWarning,
                      match='E_sig is not defined. Setting E_sig to default'):
        prepare_spectroscopy_input_dict(
            spectrum_type=fluorescence_spectrum_type,
            propagation_time_dict={"t_2": t2, "t_3": t3},
            field_dict={"E_1": E1},
            cluster_dict={
                "list_interaction_cluster_1": cluster_1,
                "list_interaction_cluster_2": cluster_2,
                "list_interaction_cluster_3": cluster_3,
            })

    # Testing over-defined fluorescence input

    # Case 1: propagation_time_dict contains too many inputs
    with pytest.warns(UserWarning,
                      match='Only t_2 and t_3 are necessary for fluorescence.'):
        prepare_spectroscopy_input_dict(
            spectrum_type=fluorescence_spectrum_type,
            propagation_time_dict={"t_1": t1, "t_2": t2, "t_3": t3},
            field_dict={"E_1": E1, "E_sig": Esig},
            cluster_dict={
                "list_interaction_cluster_1": cluster_1,
                "list_interaction_cluster_2": cluster_2,
                "list_interaction_cluster_3": cluster_3,
            })

    # Case 2: field_dict contains too many inputs
    with pytest.warns(UserWarning,
                      match='Only E_1 and E_sig are necessary for fluorescence.'):
        prepare_spectroscopy_input_dict(
            spectrum_type=fluorescence_spectrum_type,
            propagation_time_dict={"t_2": t2, "t_3": t3},
            field_dict={"E_1": E1, "E_sig": Esig, "E_2": Esig},
            cluster_dict={
                "list_interaction_cluster_1": cluster_1,
                "list_interaction_cluster_2": cluster_2,
                "list_interaction_cluster_3": cluster_3,
            })

    # Testing under-defined third-order non-fluorescence input
    with pytest.raises(
        ValueError,
        match='Propagation time after first field interactions \\(t_1\\) must be defined for GSB-R.'
    ):
        prepare_spectroscopy_input_dict(
            spectrum_type="GSB-R",
            propagation_time_dict={"t_2": t2, "t_3": t3},
            field_dict={"E_1": E1, "E_2": Esig, "E_3": Esig, "E_sig": Esig},
            cluster_dict={
                "list_interaction_cluster_1": cluster_1,
                "list_interaction_cluster_2": cluster_2,
                "list_interaction_cluster_3": cluster_3,
            },
        )

    with pytest.warns(UserWarning, match='E_2 is not defined'):
        nls_default_E2 = prepare_spectroscopy_input_dict(
            spectrum_type="GSB-R",
            propagation_time_dict={"t_1": t1, "t_2": t2, "t_3": t3},
            field_dict={"E_1": E1, "E_3": Esig, "E_sig": Esig},
            cluster_dict={
                "list_interaction_cluster_1": cluster_1,
                "list_interaction_cluster_2": cluster_2,
                "list_interaction_cluster_3": cluster_3,
            },
        )
    assert np.allclose(nls_default_E2["E_2"], np.array([0, 0, 1]))

    with pytest.warns(UserWarning, match='E_3 is not defined'):
        nls_default_E3 = prepare_spectroscopy_input_dict(
            spectrum_type="GSB-R",
            propagation_time_dict={"t_1": t1, "t_2": t2, "t_3": t3},
            field_dict={"E_1": E1, "E_2": Esig, "E_sig": Esig},
            cluster_dict={
                "list_interaction_cluster_1": cluster_1,
                "list_interaction_cluster_2": cluster_2,
                "list_interaction_cluster_3": cluster_3,
            },
        )
    assert np.allclose(nls_default_E3["E_3"], np.array([0, 0, 1]))

    # Testing incorrect spectrum_type input
    with pytest.raises(ValueError, match='spectrum_type must be one of the following:'):
        prepare_spectroscopy_input_dict(spectrum_type=bad_spectrum_type,
                                        propagation_time_dict={"t_2": t2, "t_3": t3},
                                        field_dict={"E_1": E1, "E_sig": Esig},
                                        cluster_dict={
                                            "list_interaction_cluster_1": cluster_1,
                                            "list_interaction_cluster_2": cluster_2,
                                            "list_interaction_cluster_3": cluster_3,
                                        })


def test_prepare_chromophore_input_dict():
    """
    Tests the prepare_chromophore_input_dict helper function, ensuring that the input
    dictionary is properly formatted and that errors are raised when necessary.
    """

    kmax2 = 4
    kmax2_negative = -2
    M2_mu_ge = np.array([[0.5, 0.2, 0.1], [0.5, 0.2, 0.1]])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])
    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]
    lop_list_noise = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                      sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                      sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3)),
                      sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]
    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    list_modes_by_bath = [ishizaki_decomposition_bcf_dl(40, 60, 290, 0),
                          ishizaki_decomposition_bcf_dl(35, 50, 295, 0)]
    nmodes_LTC = 1
    static_filter_list_lm = [['Markovian', [True, False]],
                             ['Triangular', [[True, False], kmax2]],
                             ['LongEdge', [[True, False], kmax2]]]
    static_filter_list_lmbb = [['Markovian', [True, False, True, False]],
                               ['Triangular', [[True, False, True, False], kmax2]],
                               ['LongEdge', [[True, False, True, False], kmax2]]]

    # Testing proper output

    # Case 1: list_lop, list_modes, and static_filter_list_lm
    bath_dict_1 = {"list_lop": list_lop, "list_modes": list_modes,
                   "static_filter_list": static_filter_list_lm}
    chromophore_dict_1 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict_1)

    assert np.allclose(chromophore_dict_1["M2_mu_ge"], M2_mu_ge)
    assert chromophore_dict_1["n_chromophore"] == 2
    assert np.allclose(chromophore_dict_1["H2_sys_hamiltonian"], H2_sys_hamiltonian)
    # Note: when nmodes_LTC is not set, lop_list_hier=lop_list_noise and lop_list_ltc=[]
    for a, b in zip(chromophore_dict_1["lop_list_hier"], lop_list_noise):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_1["lop_list_ltc"], []):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_1["lop_list_noise"], lop_list_noise):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(chromophore_dict_1["gw_sysbath_hier"], [(list_modes[0],
                                                                  list_modes[1]),
                                                                  (list_modes[2],
                                                                  list_modes[3]),
                                                                  (list_modes[0],
                                                                  list_modes[1]),
                                                                  (list_modes[2],
                                                                  list_modes[3])])
    assert np.allclose(chromophore_dict_1["gw_sysbath_noise"], [(list_modes[0],
                                                                  list_modes[1]),
                                                                  (list_modes[2],
                                                                  list_modes[3]),
                                                                  (list_modes[0],
                                                                  list_modes[1]),
                                                                  (list_modes[2],
                                                                  list_modes[3])])
    assert np.allclose(chromophore_dict_1["ltc_param"], [0, 0])
    assert chromophore_dict_1["static_filter_list"] == [
        ['Markovian', [True, False, True, False]],
        ['Triangular', [[True, False, True, False], kmax2]],
        ['LongEdge', [[True, False, True, False], kmax2]]]

    # Case 2: list_lop, list_modes_by_bath, and static_filter_list
    bath_dict_2 = {"list_lop": list_lop, "list_modes_by_bath": list_modes_by_bath,
                   "static_filter_list": static_filter_list_lmbb}
    chromophore_dict_2 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict_2)

    # Note: when nmodes_LTC is not set, lop_list_hier=lop_list_noise and lop_list_ltc=[]
    for a, b in zip(chromophore_dict_2["lop_list_hier"], lop_list_noise):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_2["lop_list_ltc"], []):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_2["lop_list_noise"], lop_list_noise):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(chromophore_dict_2["gw_sysbath_hier"],
                       [(list_modes_by_bath[0][0], list_modes_by_bath[0][1]),
                        (list_modes_by_bath[0][2], list_modes_by_bath[0][3]),
                        (list_modes_by_bath[1][0], list_modes_by_bath[1][1]),
                        (list_modes_by_bath[1][2], list_modes_by_bath[1][3])])
    assert np.allclose(chromophore_dict_2["gw_sysbath_noise"],
                       [(list_modes_by_bath[0][0], list_modes_by_bath[0][1]),
                        (list_modes_by_bath[0][2], list_modes_by_bath[0][3]),
                        (list_modes_by_bath[1][0], list_modes_by_bath[1][1]),
                        (list_modes_by_bath[1][2], list_modes_by_bath[1][3])])
    assert np.allclose(chromophore_dict_2["ltc_param"],[0, 0])
    assert chromophore_dict_2["static_filter_list"] == [
        ['Markovian', [True, False, True, False]],
        ['Triangular', [[True, False, True, False], kmax2]],
        ['LongEdge', [[True, False, True, False], kmax2]]]

    # Case 3: list_modes + list_lop + nmodes_LTC
    bath_dict_3 = {"list_modes": list_modes, "list_lop": list_lop,
                   "nmodes_LTC": nmodes_LTC}
    chromophore_dict_3 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict_3)

    for a, b in zip(chromophore_dict_3["lop_list_hier"], list_lop):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_3["lop_list_noise"], lop_list_noise):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_3["lop_list_ltc"], list_lop):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(chromophore_dict_3["gw_sysbath_hier"], [(list_modes[0],
                                                                list_modes[1]),
                                                                (list_modes[0],
                                                                list_modes[1])])
    assert np.allclose(chromophore_dict_3["gw_sysbath_noise"], [(list_modes[0],
                                                                 list_modes[1]),
                                                                 (list_modes[2],
                                                                 list_modes[3]),
                                                                 (list_modes[0],
                                                                 list_modes[1]),
                                                                 (list_modes[2],
                                                                 list_modes[3])])
    assert np.allclose(chromophore_dict_3["ltc_param"], [list_modes[2]/list_modes[3],
                                                         list_modes[2]/list_modes[3]])
    assert chromophore_dict_3["static_filter_list"] == None

    # Case 4: ESA default list_lop uses full Hilbert dimension and includes ee occupation
    H2_sys_hamiltonian_esa = np.zeros((4, 4), dtype=np.complex128)
    H2_sys_hamiltonian_esa[1:3, 1:3] = np.array([[0, -100], [-100, 0]])
    H2_sys_hamiltonian_esa[3, 3] = 150.0
    chromophore_dict_4 = prepare_chromophore_input_dict(
        M2_mu_ge, H2_sys_hamiltonian_esa, {"list_modes": list_modes}
    )
    for lop in chromophore_dict_4["lop_list_hier"]:
        assert lop.shape == (4, 4)
    # For n=2 there is one doubly-excited state at index 3; both site L-ops include it.
    diagonal_patterns = {
        tuple(np.array(lop.diagonal(), dtype=int))
        for lop in chromophore_dict_4["lop_list_hier"]
    }
    assert (0, 1, 0, 1) in diagonal_patterns
    assert (0, 0, 1, 1) in diagonal_patterns

    # Case 6: sparse Hamiltonian input is accepted and preserved
    H2_sys_hamiltonian_sparse = sparse.coo_matrix(H2_sys_hamiltonian)
    chromophore_dict_6 = prepare_chromophore_input_dict(
        M2_mu_ge, H2_sys_hamiltonian_sparse, {"list_modes": list_modes}
    )
    assert sparse.issparse(chromophore_dict_6["H2_sys_hamiltonian"])
    assert chromophore_dict_6["H2_sys_hamiltonian"].shape == (3, 3)

    # Case 5: non-ESA default list_lop (no ee manifold) keeps only site projectors
    chromophore_dict_5 = prepare_chromophore_input_dict(
        M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes}
    )
    for lop in chromophore_dict_5["lop_list_hier"]:
        assert lop.shape == (3, 3)
    non_esa_diagonal_patterns = {
        tuple(np.array(lop.diagonal(), dtype=int))
        for lop in chromophore_dict_5["lop_list_hier"]
    }
    assert (0, 1, 0) in non_esa_diagonal_patterns
    assert (0, 0, 1) in non_esa_diagonal_patterns

    # Testing M2_mu_ge input errors
    M2_mu_ge_wrongshape = np.array([np.array([0.5, 0.2]), np.array([0.5, 0.2])])

    with pytest.raises(ValueError, match='M2_mu_ge must be a numpy array with shape '
                                         '\\(n_chromophore, 3\\).'):
        prepare_chromophore_input_dict(M2_mu_ge_wrongshape, H2_sys_hamiltonian,
                                       bath_dict_1)

    # list_lop shape must match Hamiltonian Hilbert-space shape
    with pytest.raises(ValueError, match='Each list_lop operator must have shape'):
        prepare_chromophore_input_dict(
            M2_mu_ge,
            H2_sys_hamiltonian_esa,
            {"list_lop": list_lop, "list_modes": list_modes},
        )

    # Testing nmodes_LTC input errors
    nmodes_LTC_wrongtype = '1'
    nmodes_LTC_wrongvalue = -2
    nmodes_LTC_large = 2

    with pytest.raises(ValueError, match='nmodes_LTC must be an integer or None.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {
            "list_modes": list_modes,"nmodes_LTC": nmodes_LTC_wrongtype})

    with pytest.raises(ValueError, match='nmodes_LTC must be >= 0 or set to None.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {
            "list_modes": list_modes, "nmodes_LTC": nmodes_LTC_wrongvalue})

    with pytest.raises(ValueError, match='nmodes_LTC must be less than the number of '
                                         'modes in each bath.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {
            "list_modes": list_modes,"nmodes_LTC": nmodes_LTC_large})

    # Testing static_filter_list input errors
    # Case 1: static_filter_list not a list
    static_filter_list_wrongtype = 'Markovian'
    with pytest.raises(ValueError, match='static_filter_list must be a list.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {
            "list_modes": list_modes, "static_filter_list":
                static_filter_list_wrongtype})

    # Case 2: static_filter_list not a list of length 2 [filter_name, filter_params]
    static_filter_list_wronglength = [['Markovian']]
    with pytest.raises(ValueError,
                      match='each filter in static_filter_list must be a 2-element '
                            'list of the form:'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "static_filter_list":
                                           static_filter_list_wronglength})

    # Case 3: static_filter_list[0] not an allowed option ["Markovian",
    # "Triangular", "LongEdge"]
    static_filter_list_wrongoption = [['Sharkovian', [True, False]]]
    with pytest.raises(ValueError,
                      match="Error in filter 0: Filter names must be 'Markovian', "
                            "'Triangular', or 'LongEdge'."):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "static_filter_list":
                                           static_filter_list_wrongoption})

    # Case 4: filter_params not a list of length 2 [filter_bool, kmax2] for
    # "Triangular" or "LongEdge"
    static_filter_list_wrongparams = [['Triangular', [True, False, 4]]]
    with pytest.raises(ValueError,
                      match='Error in filter 0: Triangular/LongEdge filter_params '
                            'must be a list containing a list of booleans and an '
                            'integer.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "static_filter_list":
                                           static_filter_list_wrongparams})

    # Case 5: Second entry of filter_params not an integer for "Triangular" or "LongEdge"
    static_filter_list_wrongparams = [['Triangular', [[True, False], 'False']]]
    with pytest.raises(ValueError,
                      match='Error in filter 0: The second entry in filter_params for '
                            'Triangular/LongEdge filters must be an integer.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "static_filter_list": static_filter_list_wrongparams})

    # Case 6: First entry of filter_params not a list of booleans for "Triangular" or
    # "LongEdge"
    static_filter_list_wrongparams = [['Triangular', [[True, 'False'], 2]]]
    with pytest.raises(ValueError,
                      match='Error in filter 0: The first entry in filter_params for '
                            'Triangular/LongEdge filters must be a list of booleans.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "static_filter_list":
                                           static_filter_list_wrongparams})

    # Case 7: Direct test for the nested conditional block - Triangular filter with
    # negative kmax2
    static_filter_list_triangular_negative = [['Triangular', [[True, False],
                                                              kmax2_negative]]]
    with pytest.raises(ValueError,
                      match='Error in filter 0: Triangular/LongEdge filter_params must '
                            'have a positive integer as the second element.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "static_filter_list":
                                           static_filter_list_triangular_negative})

    # Case 8: static_filter_list not the right length for "Markovian" case
    static_filter_list_wronglength = [['Markovian', [True]]]
    with pytest.raises(ValueError,
                      match='Error in filter 0: The list of booleans in filter_params '
                            'must have the same length as the number of modes in each '
                            'bath.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "static_filter_list":
                                           static_filter_list_wronglength})

    # Case 9: static_filter_list not the right length for "Markovian" case
    # (list_modes_by_bath)
    static_filter_list_wronglength = [['Markovian', [True, False]]]
    with pytest.raises(ValueError,
                      match='Error in filter 0: The list of booleans in filter_params '
                            'must have the same length as the number of modes in all '
                            'baths combined.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes_by_bath": list_modes_by_bath,
                                       "static_filter_list":
                                           static_filter_list_wronglength})

    # Case 10: static_filter_list not the right length for "Triangular"/"LongEdge" cases
    static_filter_list_wronglength = [['Triangular', [[True], 2]]]
    with pytest.raises(ValueError,
                      match='Error in filter 0: The list of booleans in filter_params '
                            'must have the same length as the number of modes in each '
                            'bath.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "static_filter_list":
                                           static_filter_list_wronglength})

    # Case 11: static_filter_list not the right length for "Triangular"/"LongEdge" cases
    # (list_modes_by_bath)
    static_filter_list_wronglength = [['Triangular', [[True, False], 2]]]
    with pytest.raises(ValueError,
                      match='Error in filter 0: The list of booleans in filter_params '
                            'must have the same length as the number of modes in all '
                            'baths combined.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes_by_bath": list_modes_by_bath,
                                       "static_filter_list":
                                           static_filter_list_wronglength})
    # Case 11: Filter_params not a list of booleans for "Markovian"
    static_filter_list_wrongparams = [['Markovian', ['True', 'False']]]
    with pytest.raises(ValueError,
                      match='Error in filter 0: filter_params for Markovian filters '
                            'must be a list of booleans.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "static_filter_list": static_filter_list_wrongparams})

    # Testing that defining both nmodes_LTC and static_filter_list raises an error
    with pytest.raises(ValueError,
                      match='The use of static hierarchy filters with low-temperature '
                            'correction is not currently supported.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "nmodes_LTC": nmodes_LTC,
                                       "static_filter_list": static_filter_list_lm})

    # Testing that overdefining modes (list_modes_by_bath and list_modes) yields error
    with pytest.raises(ValueError,
                      match='list_modes_by_bath and list_modes should not both be '
                            'defined.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes": list_modes,
                                       "list_modes_by_bath": list_modes_by_bath})

    # Testing incompatible list_modes_by_bath and list_lop lengths yields an error
    list_modes_by_bath_wronglength = [ishizaki_decomposition_bcf_dl(40, 60, 290, 0)]
    list_lop_wronglength = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                      sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3)),
                      sparse.coo_matrix(([1], ([2], [2])), shape=(3, 3))]
    with pytest.raises(ValueError, match='list_modes_by_bath and list_lop must have the'
                                         ' same length.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_lop": list_lop_wronglength,
                                       "list_modes_by_bath":
                                           list_modes_by_bath_wronglength})

    # Testing list_modes_by_bath structure
    # list_modes_by_bath not a list of lists
    list_modes_by_bath_wrongstructure = ['mode1', 'mode2']
    with pytest.raises(ValueError,
                      match='list_modes_by_bath must be a list of lists.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes_by_bath":
                                           list_modes_by_bath_wrongstructure})

    # list_modes_by_bath doesn't contain paired Gs and Ws
    list_modes_by_bath_wrongpairing = [ishizaki_decomposition_bcf_dl(35, 50, 295, 0), [1, 2, 3]]
    with pytest.raises(ValueError,
                      match='sublists within list_modes_by_bath should contain paired '
                            'Gs and Ws, which guarantees an even number of elements in '
                            'each sublist.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                      {"list_modes_by_bath":
                                           list_modes_by_bath_wrongpairing})

    # Testing list_modes structure
    # list_modes doesn't contain paired Gs and Ws
    list_modes_wrongpairing = [1, 2, 3]
    with pytest.raises(ValueError, match='list_modes should contain paired Gs and Ws, '
                                         'which guarantees an even number of elements.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes_wrongpairing})

    ## Testing that an error is raised if neither list_modes nor list_modes_by_bath are defined
    with pytest.raises(ValueError, match='Either list_modes_by_bath or list_modes must '
                                         'be defined.'):
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {})

def test_prepare_convergence_parameter_dict():
    """
    Tests the prepare_convergence_parameter_dict helper function, ensuring that the
    input dictionary is properly formatted with default and custom parameters.
    """
    # Test proper output with all parameters defined
    t_step = 0.1
    max_hier = 12
    delta_a = 1e-10
    delta_s = 1e-10
    set_update_step = 1
    set_f_discard = 0.5

    convergence_dict = prepare_convergence_parameter_dict(t_step, max_hier, delta_a,
                                                          delta_s, set_update_step,
                                                          set_f_discard)
    assert convergence_dict["t_step"] == t_step
    assert convergence_dict["max_hier"] == max_hier
    assert convergence_dict["delta_a"] == delta_a
    assert convergence_dict["delta_s"] == delta_s
    assert convergence_dict["set_update_step"] == set_update_step
    assert convergence_dict["set_f_discard"] == set_f_discard

    # Test proper output with defaults
    convergence_dict = prepare_convergence_parameter_dict(t_step, max_hier)
    assert convergence_dict["t_step"] == t_step
    assert convergence_dict["max_hier"] == max_hier
    assert convergence_dict["delta_a"] == 0
    assert convergence_dict["delta_s"] == 0
    assert convergence_dict["set_update_step"] == 1
    assert convergence_dict["set_f_discard"] == 0


@pytest.mark.parametrize(
    "spectrum_type,expected_transition,expected_sides,expected_scale",
    [
        ("ABSORPTION", ["g_to_e"], ["ket"], 2),
        ("FLUORESCENCE", ["g_to_e", "g_to_e", "e_to_g"], ["bra", "ket", "bra"], 4),
        ("GSB-R", ["g_to_e", "e_to_g", "g_to_e"], ["bra", "bra", "ket"], 1),
        ("SE-R", ["g_to_e", "g_to_e", "e_to_g"], ["bra", "ket", "bra"], 1),
        ("ESA-R", ["g_to_e", "g_to_e", "e_to_ee"], ["bra", "ket", "ket"], -1),
        ("GSB-NR", ["g_to_e", "e_to_g", "g_to_e"], ["ket", "ket", "ket"], 1),
        ("SE-NR", ["g_to_e", "g_to_e", "e_to_g"], ["ket", "bra", "bra"], 1),
        ("ESA-NR", ["g_to_e", "g_to_e", "e_to_ee"], ["ket", "bra", "ket"], -1),
    ],
)
def test_get_pathway_returns_only_selected_config(
    spectrum_type, expected_transition, expected_sides, expected_scale
):
    dhops = _build_dhops_for_spectrum(spectrum_type)
    pathway = dhops._get_pathway()
    assert pathway["list_transition"] == expected_transition
    assert pathway["list_sides"] == expected_sides
    assert pathway["scaling_factor"] == expected_scale
    assert len(pathway["list_transition"]) == len(pathway["list_sides"])
    assert len(pathway["list_transition"]) == len(pathway["list_clusters"])


def test_calculate_spectrum_uses_get_pathway(monkeypatch):
    dhops = _build_dhops_for_spectrum("SE-R")
    calls = []

    monkeypatch.setattr(DHOPS, "initialize", lambda self: None)
    monkeypatch.setattr(DHOPS, "_final_dyad_operator", lambda self: (None, 0))
    monkeypatch.setattr(DHOPS, "_response_function_comp", lambda self, op, idx: 1.0)
    monkeypatch.setattr(
        DHOPS,
        "_get_pathway",
        lambda self: {
            "list_transition": ["a", "b", "c"],
            "list_sides": ["ket", "bra", "ket"],
            "scaling_factor": 7,
            "list_clusters": [np.array([1]), np.array([2]), np.array([1, 2])],
        },
    )
    monkeypatch.setattr(
        DHOPS,
        "_hilb_operator",
        lambda self, transition, field, cluster: (transition, tuple(cluster.tolist())),
    )
    monkeypatch.setattr(
        DHOPS,
        "_dyad_operator",
        lambda self, op, side: calls.append((op[0], op[1], side)),
    )
    monkeypatch.setattr(
        DHOPS,
        "propagate",
        lambda self, t, t_step, timer_checkpoint: None,
    )

    response = dhops.calculate_spectrum()

    assert response == 7.0
    assert calls == [
        ("a", (1,), "ket"),
        ("b", (2,), "bra"),
        ("c", (1, 2), "ket"),
    ]


def test_get_pathway_invalid_type_raises():
    dhops = _build_dhops_for_spectrum("ABSORPTION")
    dhops.spectrum_type = "NOT-A-PATHWAY"
    with pytest.raises(ValueError, match="Unknown spectrum_type: NOT-A-PATHWAY"):
        dhops._get_pathway()
