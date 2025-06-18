import pytest
import numpy as np
from mesohops.trajectory.dyadic_spectra import DyadicSpectra as DHOPS
from mesohops.trajectory.dyadic_spectra import (prepare_spectroscopy_input_dict,
                                            prepare_chromophore_input_dict,
                                            prepare_convergence_parameter_dict)
from mesohops.util.bath_corr_functions import ishizaki_decomposition_bcf_dl
from scipy import sparse


def test_DyadicSpectra():
    """
    Tests the DyadicSpectra class for properly unpacking input dictionaries, and ensures
    the Hamiltonian is the proper shape.
    """
    ## Spectroscopy input dictionary
    seed = 10
    spectrum_type = "FLUORESCENCE"
    propagation_time_dict = {"t_2": 2.0, "t_3": 3.0}
    field_dict = {"E_1": np.array([0, 0, 1]), "E_sig": np.array([0, 0, 1])}
    site_dict = {"list_ket_sites": np.array([1, 2]), "list_bra_sites": np.array([1, 2])}

    spectroscopy_dict = prepare_spectroscopy_input_dict(spectrum_type,
                                                        propagation_time_dict,
                                                        field_dict, site_dict)

    ## Chromophore input dictionary
    M2_mu_ge = np.array([np.array([0.5, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])

    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]

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
    static_filter_list = ['Markovian', [True, False]]
    bath_dict3 = {"list_lop": list_lop, "list_modes": list_modes,
                  "static_filter_list": static_filter_list}
    chromophore_dict_3 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict3)

    ## Convergence parameter dictionaries
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

    ## Test input dictionary unpacking

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
    assert np.allclose(dhops_1a.list_ket_sites, np.array([1, 2]))
    assert np.allclose(dhops_1a.list_bra_sites, np.array([1, 2]))
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
    assert dhops_3.static_filter_list == ['Markovian', [True, False]]


    ## Test Hamiltonian shape compatibility with number of states

    H2_sys_hamiltonian_wrongshape = np.zeros((4, 4), dtype=np.complex128)
    bath_dict_wrongshape = {"list_lop": list_lop, "list_modes": list_modes}
    chromophore_dict_wrongshape = prepare_chromophore_input_dict(M2_mu_ge,
                                                                 H2_sys_hamiltonian_wrongshape,
                                                                 bath_dict_wrongshape)

    try:
        DHOPS(spectroscopy_dict, chromophore_dict_wrongshape,
              convergence_dict_float_dt, seed)
    except ValueError as excinfo:
        if ('H2_sys_hamiltonian must be ((n_chrom + 1) x (n_chrom + 1)) to account'
            ' for each chromophore and the ground state.') not in str(excinfo):
            pytest.fail()

    # Test "INITIALIZATION_TIME" is greater than 0
    dhops_4 = DHOPS(spectroscopy_dict, chromophore_dict_1, convergence_dict_float_dt, seed)
    dhops_4.calculate_spectrum()

    assert dhops_4.storage.metadata["INITIALIZATION_TIME"] > 0

    # Test "LIST_PROPAGATION_TIME" is correct length (Fluorescence)
    assert len(dhops_4.storage.metadata["LIST_PROPAGATION_TIME"]) == 2

    # Test "LIST_PROPAGATION_TIME" is correct length (Absorption)
    spectroscopy_dict_abs = prepare_spectroscopy_input_dict("ABSORPTION",
                                                            {"t_1": 1.0},
                                                            {"E_1": np.array([0, 0, 1])},
                                                            {"list_ket_sites": np.array([1, 2])})
    dhops_5 = DHOPS(spectroscopy_dict_abs, chromophore_dict_1, convergence_dict_float_dt, seed)
    dhops_5.calculate_spectrum()

    assert len(dhops_5.storage.metadata["LIST_PROPAGATION_TIME"]) == 1


def test_initialize(capsys):
    """
    Tests the initialization of the DyadicTrajectory class, and ensures that the class
    is properly initialized in both non-adaptive and adaptive cases.
    """
    ## Spectroscopy input dictionary
    seed = 10
    spectrum_type = "ABSORPTION"
    propagation_time_dict = {"t_1": 1.0}
    field_dict = {"E_1": np.array([0, 0, 1])}
    site_dict = {"list_ket_sites": np.array([1, 2])}

    spectroscopy_dict = prepare_spectroscopy_input_dict(spectrum_type,
                                                        propagation_time_dict,
                                                        field_dict, site_dict)

    ## Chromophore input dictionary
    M2_mu_ge = np.array([np.array([0.5, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])

    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]

    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    bath_dict = {"list_lop": list_lop, "list_modes": list_modes}
    chromophore_dict = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict)

    ## Convergence parameter dictionaries
    convergence_dict = prepare_convergence_parameter_dict(t_step=0.1, max_hier=12,
                                                          delta_a=0, delta_s=0)

    ## Testing initialized property works upon initialization
    dhops = DHOPS(spectroscopy_dict, chromophore_dict, convergence_dict, seed)
    assert  dhops.__initialized__ is False

    dhops.initialize()
    assert dhops.__initialized__ is True

    ## Testing that multiple calls to initialize() triggers a warning
    dhops.initialize()
    out, err = capsys.readouterr()
    assert "WARNING: DyadicTrajectory has already been initialized." in out.strip()

    ## Testing delta_a/delta_s greater than 0 causes the trajectory to be ran adaptively
    convergence_dict_adaptive = prepare_convergence_parameter_dict(t_step=0.1,
                                                                   max_hier=12,
                                                                   delta_a=1e-10,
                                                                   delta_s=1e-10,
                                                                   set_update_step=2,
                                                                   set_f_discard=0.5)

    dhops_adaptive = DHOPS(spectroscopy_dict, chromophore_dict, convergence_dict_adaptive, seed)
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
    ## Spectroscopy input dictionary
    seed = 10
    spectrum_type = "ABSORPTION"
    propagation_time_dict = {"t_1": 1.0}
    field_dict = {"E_1": np.array([2, 3, 1])}
    site_dict = {"list_ket_sites": np.array([1, 2])}

    spectroscopy_dict = prepare_spectroscopy_input_dict(spectrum_type,
                                                        propagation_time_dict,
                                                        field_dict, site_dict)

    ## Chromophore input dictionary
    M2_mu_ge = np.array([np.array([0.5, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])

    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]

    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    bath_dict = {"list_lop": list_lop, "list_modes": list_modes}
    chromophore_dict = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                      bath_dict)

    ## Convergence parameter dictionaries
    convergence_dict = prepare_convergence_parameter_dict(t_step=0.1, max_hier=12,
                                                          delta_a=0, delta_s=0)

    dhops = DHOPS(spectroscopy_dict, chromophore_dict, convergence_dict, seed)

    ## Testing that the Hilbert raising and lowering operators are properly constructed

    # Note that the value of 1.7 comes from the dot product of the field and the dipole
    dense_raise = np.zeros((3, 3), dtype=np.float64)
    dense_raise[np.array([1, 2]), 0] = 1.7

    dense_lower = np.zeros((3, 3), dtype=np.float64)
    dense_lower[0, np.array([1, 2])] = 1.7

    assert np.allclose(dhops._hilb_operator("raise", np.array([2, 3, 1]),
                                            dhops.list_ket_sites).toarray(), dense_raise)

    assert np.allclose(dhops._hilb_operator("lower", np.array([2, 3, 1]),
                                            dhops.list_ket_sites).toarray(), dense_lower)

    ## Testing that the method raises an error if not given a valid action_type
    try:
        dhops._hilb_operator("cha_cha_slide", np.array([2, 3, 1]), dhops.list_ket_sites)
    except ValueError as excinfo:
        if "action_type must be either 'raise' or 'lower'." not in str(excinfo):
            pytest.fail()

def test_final_dyad_operator():
    """
    Tests the _final_dyad_operator method of the DyadicTrajectory class, ensuring that
    the final dyad operator is properly constructed, and the time index is properly set.
    """
    ## Spectroscopy input dictionary
    seed = 10
    spectrum_type = "FLUORESCENCE"
    propagation_time_dict = {"t_2": 2.0, "t_3": 3.0}
    field_dict = {"E_1": np.array([2, 3, 1]), "E_sig": np.array([1, 2, 3])}
    site_dict = {"list_ket_sites": np.array([1, 2]), "list_bra_sites": np.array([1, 2])}

    spectroscopy_dict = prepare_spectroscopy_input_dict(spectrum_type,
                                                        propagation_time_dict,
                                                        field_dict, site_dict)

    ## Chromophore input dictionary
    M2_mu_ge = np.array([np.array([0.5, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])

    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]

    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    bath_dict = {"list_lop": list_lop, "list_modes": list_modes}
    chromophore_dict = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                        bath_dict)

    ## Convergence parameter dictionaries
    convergence_dict = prepare_convergence_parameter_dict(t_step=0.1, max_hier=12,
                                                          delta_a=1e-10, delta_s=1e-10,
                                                          set_update_step=1,
                                                          set_f_discard=0.5)

    ## Testing that the final dyad operator is properly constructed
    dhops = DHOPS(spectroscopy_dict, chromophore_dict, convergence_dict, seed)

    dyadic_f_op = np.zeros((6, 6), dtype=np.float64)
    dyadic_f_op[3, 1] = 1.2
    dyadic_f_op[3, 2] = 1.2

    assert np.allclose(dhops._final_dyad_operator()[0].toarray(), dyadic_f_op)

    ## Testing that the time index is properly set
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
    ket_sites = np.array([1, 2])
    ket_sites_index_issue = np.array([0, 1])
    ket_sites_list = [1, 2]
    bra_sites = np.array([1, 2])
    bra_sites_list = [1, 2]

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
                                               site_dict={"list_ket_sites": ket_sites})
    assert abs_test["spectrum_type"] == 'ABSORPTION'
    assert abs_test["t_1"] == t1
    assert abs_test["t_2"] == 0
    assert abs_test["t_3"] == 0
    assert np.allclose(abs_test["E_1"], E1)
    assert np.allclose(abs_test["E_sig"], E1)
    assert np.allclose(abs_test["list_ket_sites"], ket_sites)

    fluor_test = prepare_spectroscopy_input_dict(
        spectrum_type=fluorescence_spectrum_type,
        propagation_time_dict={"t_2": t2, "t_3": t3},
        field_dict={"E_1": E1, "E_sig": Esig},
        site_dict={"list_ket_sites": ket_sites, "list_bra_sites": bra_sites})
    assert fluor_test["spectrum_type"] == 'FLUORESCENCE'
    assert fluor_test["t_1"] == 0
    assert fluor_test["t_2"] == t2
    assert fluor_test["t_3"] == t3
    assert np.allclose(fluor_test["E_1"], E1)
    assert np.allclose(fluor_test["E_2"], E1)
    assert np.allclose(fluor_test["E_3"], Esig)
    assert np.allclose(fluor_test["E_sig"], Esig)
    assert np.allclose(fluor_test["list_ket_sites"], ket_sites)
    assert np.allclose(fluor_test["list_bra_sites"], bra_sites)

    # Testing site definition errors

    # Case 1: list_ket_sites not defined
    try:
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1},
                                        field_dict={"E_1": E1},
                                        site_dict={})
    except ValueError as excinfo:
        if 'list_ket_sites must be defined.' not in str(excinfo):
            pytest.fail()

    # Case 2: list_ket_sites not a numpy array
    ket_list = prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                               propagation_time_dict={"t_1": t1},
                                               field_dict={"E_1": E1},
                                               site_dict={
                                                   "list_ket_sites": ket_sites_list})

    ket_array = prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                                propagation_time_dict={"t_1": t1},
                                                field_dict={"E_1": E1},
                                                site_dict={"list_ket_sites": ket_sites})

    assert np.allclose(ket_list["list_ket_sites"], ket_array["list_ket_sites"])

    # Case 3: sites indexed from 0
    try:
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1},
                                        field_dict={"E_1": E1},
                                        site_dict={
                                            "list_ket_sites": ket_sites_index_issue})
    except ValueError as excinfo:
        if 'Ket and Bra sites must be indexed starting from 1.' not in str(excinfo):
            pytest.fail()

    # Testing field input formatting

    # Case 1: Field not a numpy array
    try:
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1},
                                        field_dict={"E_1": E1_list},
                                        site_dict={"list_ket_sites": ket_sites})
    except ValueError as excinfo:
        if 'All field entries should be numpy arrays.' not in str(excinfo):
            pytest.fail()

    # Case 2: Field not a numpy array with exactly 3 entries
    try:
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1},
                                        field_dict={"E_1": E1_wrong_length},
                                        site_dict={"list_ket_sites": ket_sites})
    except ValueError as excinfo:
        if ('All field entries should be numpy arrays with exactly 3 entries.' not in
                str(excinfo)):
            pytest.fail()

    # Testing under-defined absorption input

    # Case 1: t_1 not defined
    try:
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={},
                                        field_dict={"E_1": E1},
                                        site_dict={"list_ket_sites": ket_sites})
    except ValueError as excinfo:
        if ('Propagation time after first field interaction (t_1) must be defined as > '
            '0 for absorption.') not in str(excinfo):
            pytest.fail()

    # Case 2: E_1 not defined
    try:
        prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                        propagation_time_dict={"t_1": t1},
                                        field_dict={},
                                        site_dict={"list_ket_sites": ket_sites})
    except ValueError as excinfo:
        if 'E_1 must be defined for absorption.' not in str(excinfo):
            pytest.fail()

    # Testing over-defined absorption input

    # Case 1: propagation_time_dict contains too many inputs
    prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                    propagation_time_dict={"t_1": t1, "t_2": t2},
                                    field_dict={"E_1": E1},
                                    site_dict={"list_ket_sites": ket_sites})
    out, err = capsys.readouterr()
    assert out.strip() == ('WARNING: Only t_1 is necessary for absorption. '
                           'Setting all other propagation times to zero.')

    # Case 2: field_dict contains too many inputs
    prepare_spectroscopy_input_dict(spectrum_type=absorption_spectrum_type,
                                    propagation_time_dict={"t_1": t1},
                                    field_dict={"E_1": E1, "E_sig": Esig},
                                    site_dict={"list_ket_sites": ket_sites})
    out, err = capsys.readouterr()
    assert out.strip() == ('WARNING: Only E_1 is necessary for absorption. E_sig is '
                           'set to E_1. All other field definitions will be discarded')

    # Testing under-defined fluorescence input

    # Case 1: list_bra_sites not defined
    try:
        prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                        propagation_time_dict={"t_2": t2, "t_3": t3},
                                        field_dict={"E_1": E1, "E_sig": Esig},
                                        site_dict={"list_ket_sites": ket_sites})
    except ValueError as excinfo:
        if 'list_bra_sites must be defined for fluorescence.' not in str(excinfo):
            pytest.fail()

    # Case 2: list_bra_sites not a numpy array
    bra_list = prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                               propagation_time_dict={"t_2": t2,
                                                                      "t_3": t3},
                                               field_dict={"E_1": E1, "E_sig": Esig},
                                               site_dict={"list_ket_sites": ket_sites,
                                                          "list_bra_sites":
                                                              bra_sites_list})
    bra_array = prepare_spectroscopy_input_dict(
        spectrum_type=fluorescence_spectrum_type,
        propagation_time_dict={"t_2": t2, "t_3": t3},
        field_dict={"E_1": E1, "E_sig": Esig},
        site_dict={"list_ket_sites": ket_sites,
                   "list_bra_sites": bra_sites})

    assert np.allclose(bra_list["list_bra_sites"], bra_array["list_bra_sites"])

    # Note: There is no need to test case if list_ket_sites is not defined, as this is
    # already tested prior to determining the spectrum type.

    # Case 3: t_2 or t_3 not defined
    try:
        prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                        propagation_time_dict={"t_3": t3},
                                        field_dict={"E_1": E1, "E_sig": Esig},
                                        site_dict={"list_ket_sites": ket_sites,
                                                   "list_bra_sites": bra_sites})
    except ValueError as excinfo:
        if ('Propagation times after second and third field interactions (t_2, t_3) '
            'must be defined as > 0 for fluorescence.') not in str(excinfo):
            pytest.fail()

    try:
        prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                        propagation_time_dict={"t_2": t2},
                                        field_dict={"E_1": E1, "E_sig": Esig},
                                        site_dict={"list_ket_sites": ket_sites,
                                                   "list_bra_sites": bra_sites})
    except ValueError as excinfo:
        if ('Propagation times after second and third field interactions (t_2, t_3) '
            'must be defined as > 0 for fluorescence.') not in str(excinfo):
            pytest.fail()

    # Case 4: E_1 not defined
    try:
        prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                        propagation_time_dict={"t_2": t2, "t_3": t3},
                                        field_dict={"E_sig": Esig},
                                        site_dict={"list_ket_sites": ket_sites,
                                                   "list_bra_sites": bra_sites})
    except ValueError as excinfo:
        if 'E_1 must be defined for fluorescence.' not in str(excinfo):
            pytest.fail()

    # Case 5: E_sig not defined
    prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                    propagation_time_dict={"t_2": t2, "t_3": t3},
                                    field_dict={"E_1": E1},
                                    site_dict={"list_ket_sites": ket_sites,
                                               "list_bra_sites": bra_sites})
    out, err = capsys.readouterr()
    assert out.strip() == ('WARNING: E_sig is not defined. Setting E_sig to default, '
                           '[0, 0, 1].')

    # Testing over-defined fluorescence input

    # Case 1: propagation_time_dict contains too many inputs
    prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                    propagation_time_dict={"t_1": t1, "t_2": t2,
                                                           "t_3": t3},
                                    field_dict={"E_1": E1, "E_sig": Esig},
                                    site_dict={"list_ket_sites": ket_sites,
                                               "list_bra_sites": bra_sites})
    out, err = capsys.readouterr()
    assert out.strip() == ('WARNING: Only t_2 and t_3 are necessary for fluorescence. '
                           'Setting all other propagation times to zero.')

    # Case 2: field_dict contains too many inputs
    prepare_spectroscopy_input_dict(spectrum_type=fluorescence_spectrum_type,
                                    propagation_time_dict={"t_2": t2, "t_3": t3},
                                    field_dict={"E_1": E1, "E_sig": Esig, "E_2": Esig},
                                    site_dict={"list_ket_sites": ket_sites,
                                               "list_bra_sites": bra_sites})
    out, err = capsys.readouterr()
    assert out.strip() == ('WARNING: Only E_1 and E_sig are necessary for fluorescence.'
                           ' All other field definitions will be discarded.')

    # Testing incorrect spectrum_type input
    try:
        prepare_spectroscopy_input_dict(spectrum_type=bad_spectrum_type,
                                        propagation_time_dict={"t_2": t2, "t_3": t3},
                                        field_dict={"E_1": E1, "E_sig": Esig},
                                        site_dict={"list_ket_sites": ket_sites,
                                                   "list_bra_sites": bra_sites})
    except ValueError as excinfo:
        if 'spectrum_type must be one of the following:' not in str(excinfo):
            pytest.fail()


def test_prepare_chromophore_input_dict():
    """
    Tests the prepare_chromophore_input_dict helper function, ensuring that the input
    dictionary is properly formatted and that errors are raised when necessary.
    """

    kmax2 = 4
    M2_mu_ge = np.array([[0.5, 0.2, 0.1], [0.5, 0.2, 0.1]])
    H2_sys_hamiltonian = np.zeros((3, 3), dtype=np.complex128)
    H2_sys_hamiltonian[1:, 1:] = np.array([[0, -100], [-100, 0]])
    list_lop = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]
    lop_list_noise_nmodes_LTC = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                      sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                      sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3)),
                      sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3))]
    list_modes = ishizaki_decomposition_bcf_dl(35, 50, 295, 0)
    list_modes_by_bath = [ishizaki_decomposition_bcf_dl(40, 60, 290, 0), ishizaki_decomposition_bcf_dl(35, 50, 295, 0)]
    nmodes_LTC = 1
    static_filter_list_markovian = ['Markovian', [True, False]]
    static_filter_list_triangular = ['Triangular', [[True, False], kmax2]]
    static_filter_list_longedge = ['LongEdge', [[True, False], kmax2]]

    ## Testing proper output

    # Case 1: list_lop, list_modes, nmodes_LTC, and static_filter_list
    bath_dict_1 = {"list_lop": list_lop, "list_modes": list_modes, "nmodes_LTC": nmodes_LTC, "static_filter_list": static_filter_list_longedge}
    chromophore_dict_1 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, bath_dict_1)

    assert np.allclose(chromophore_dict_1["M2_mu_ge"], M2_mu_ge)
    assert chromophore_dict_1["n_chromophore"] == 2
    assert np.allclose(chromophore_dict_1["H2_sys_hamiltonian"], H2_sys_hamiltonian)
    # Note: when nmodes_LTC is set to 1, lop_list_hier=list_lop=lop_list_ltc for a dimer
    for a, b in zip(chromophore_dict_1["lop_list_hier"], list_lop):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_1["lop_list_ltc"], list_lop):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_1["lop_list_noise"], lop_list_noise_nmodes_LTC):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(chromophore_dict_1["gw_sysbath_hier"], [(list_modes[0],
                                                                 list_modes[1]),
                                                                 (list_modes[0],
                                                                 list_modes[1])])
    assert np.allclose(chromophore_dict_1["gw_sysbath_noise"], [(list_modes[0],
                                                                  list_modes[1]),
                                                                  (list_modes[2],
                                                                  list_modes[3]),
                                                                  (list_modes[0],
                                                                  list_modes[1]),
                                                                  (list_modes[2],
                                                                  list_modes[3])])
    assert np.allclose(chromophore_dict_1["ltc_param"],
                       [(list_modes[2]/list_modes[3]), (list_modes[2]/list_modes[3])])
    assert chromophore_dict_1["static_filter_list"] == ['LongEdge', [[True, False], kmax2]]

    # Case 2: list_lop, list_modes_by_bath, nmode_LTC, and static_filter_list
    bath_dict_2 = {"list_lop": list_lop, "list_modes_by_bath": list_modes_by_bath, "nmodes_LTC": nmodes_LTC, "static_filter_list": static_filter_list_triangular}
    chromophore_dict_2 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, bath_dict_2)

    # Note: when nmodes_LTC is set to 1, lop_list_hier=list_lop=lop_list_ltc for a dimer
    for a, b in zip(chromophore_dict_2["lop_list_hier"], list_lop):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_2["lop_list_ltc"], list_lop):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_2["lop_list_noise"], lop_list_noise_nmodes_LTC):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(chromophore_dict_2["gw_sysbath_hier"], [(list_modes_by_bath[0][0],
                                                                 list_modes_by_bath[0][1]),
                                                                 (list_modes_by_bath[1][0],
                                                                 list_modes_by_bath[1][1])])
    assert np.allclose(chromophore_dict_2["gw_sysbath_noise"], [(list_modes_by_bath[0][0],
                                                                  list_modes_by_bath[0][1]),
                                                                  (list_modes_by_bath[0][2],
                                                                  list_modes_by_bath[0][3]),
                                                                  (list_modes_by_bath[1][0],
                                                                  list_modes_by_bath[1][1]),
                                                                  (list_modes_by_bath[1][2],
                                                                  list_modes_by_bath[1][3])])
    assert np.allclose(chromophore_dict_2["ltc_param"],
                          [(list_modes_by_bath[0][2]/list_modes_by_bath[0][3]),
                            (list_modes_by_bath[1][2]/list_modes_by_bath[1][3])])
    assert chromophore_dict_2["static_filter_list"] == ['Triangular', [[True, False], kmax2]]

    # Case 3: list_modes only
    bath_dict_3 = {"list_modes": list_modes}
    chromophore_dict_3 = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, bath_dict_3)

    for a, b in zip(chromophore_dict_3["lop_list_hier"],
                    [sparse.coo_matrix(([1], ([1], [1])), shape=(3, 3)),
                    sparse.coo_matrix(([1], ([1], [1])), shape=(3, 3)),
                    sparse.coo_matrix(([1], ([2], [2])), shape=(3, 3)),
                    sparse.coo_matrix(([1], ([2], [2])), shape=(3, 3))]):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_3["lop_list_noise"],
                    [sparse.coo_matrix(([1], ([1], [1])), shape=(3, 3)),
                    sparse.coo_matrix(([1], ([1], [1])), shape=(3, 3)),
                    sparse.coo_matrix(([1], ([2], [2])), shape=(3, 3)),
                    sparse.coo_matrix(([1], ([2], [2])), shape=(3, 3))]):
        assert np.allclose(a.toarray(), b.toarray())
    for a, b in zip(chromophore_dict_3["lop_list_ltc"],
                    [sparse.coo_matrix(([1], ([1], [1])), shape=(3, 3)),
                    sparse.coo_matrix(([1], ([2], [2])), shape=(3, 3))]):
        assert np.allclose(a.toarray(), b.toarray())
    assert np.allclose(chromophore_dict_3["gw_sysbath_hier"], [(list_modes[0],
                                                                list_modes[1]),
                                                                (list_modes[2],
                                                                list_modes[3]),
                                                                (list_modes[0],
                                                                list_modes[1]),
                                                                (list_modes[2],
                                                                list_modes[3])])
    assert np.allclose(chromophore_dict_3["gw_sysbath_noise"], [(list_modes[0],
                                                                 list_modes[1]),
                                                                 (list_modes[2],
                                                                 list_modes[3]),
                                                                 (list_modes[0],
                                                                 list_modes[1]),
                                                                 (list_modes[2],
                                                                 list_modes[3])])
    assert np.allclose(chromophore_dict_3["ltc_param"], [0, 0])
    assert chromophore_dict_3["static_filter_list"] == None

    ## Testing M2_mu_ge input errors
    M2_mu_ge_wrongshape = np.array([np.array([0.5, 0.2]), np.array([0.5, 0.2])])

    try:
        prepare_chromophore_input_dict(M2_mu_ge_wrongshape, H2_sys_hamiltonian, bath_dict_1)
    except ValueError as excinfo:
        if 'M2_mu_ge must be a numpy array with shape (n_chromophore, 3).' not in str(excinfo):
            pytest.fail()

    ## Testing nmodes_LTC input errors
    nmodes_LTC_wrongtype = '1'
    nmodes_LTC_wrongvalue = -2
    nmodes_LTC_large = 2

    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes,"nmodes_LTC": nmodes_LTC_wrongtype})
    except ValueError as excinfo:
        if 'nmodes_LTC must be an integer or None.' not in str(excinfo):
            pytest.fail()

    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "nmodes_LTC": nmodes_LTC_wrongvalue})
    except ValueError as excinfo:
        if 'nmodes_LTC must be >= 0 or set to None.' not in str(excinfo):
            pytest.fail()

    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes,"nmodes_LTC": nmodes_LTC_large})
    except ValueError as excinfo:
        if 'nmodes_LTC must be less than the number of modes in each bath.' not in str(excinfo):
            pytest.fail()

    ## Testing static_filter_list input errors
    # Case 1: static_filter_list not a list
    static_filter_list_wrongtype = 'Markovian'
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wrongtype})
    except ValueError as excinfo:
        if 'static_filter_list must be a list.' not in str(excinfo):
            pytest.fail()

    # Case 2: static_filter_list not a list of length 2 [filter_name, filter_params]
    static_filter_list_wronglength = ['Markovian']
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wronglength})
    except ValueError as excinfo:
        if 'static_filter_list must be a 2-element list of the form: [filter_name, filter_params].' not in str(excinfo):
            pytest.fail()

    # Case 3: static_filter_list[0] not an allowed option ["Markovian", "Triangular", "LongEdge"]
    static_filter_list_wrongoption = ['Sharkovian', [True, False]]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wrongoption})
    except ValueError as excinfo:
        if "Filter name must be 'Markovian', 'Triangular', or 'LongEdge'." not in str(excinfo):
            pytest.fail()

    # Case 4: filter_params not a list of length 2 [filter_bool, kmax2] for "Triangular" or "LongEdge"
    static_filter_list_wrongparams = ['Triangular', [True, False, 4]]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wrongparams})
    except ValueError as excinfo:
        if 'The Triangular and LongEdge filter_params must be a list of booleans and an integer.' not in str(excinfo):
            pytest.fail()

    # Case 5: Second entry of filter_params not an integer for "Triangular" or "LongEdge"
    static_filter_list_wrongparams = ['Triangular', [[True, False], 'False']]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wrongparams})
    except ValueError as excinfo:
        if 'The second entry in filter_params for the Triangular and LongEdge filters must be an integer.' not in str(excinfo):
            pytest.fail()

    # Case 6: First entry of filter_params not a list of booleans for "Triangular" or "LongEdge"
    static_filter_list_wrongparams = ['Triangular', [[True, 'False'], 2]]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wrongparams})
    except ValueError as excinfo:
        if 'The first entry in filter_params for the Triangular and LongEdge filters must be a list of booleans.' not in str(excinfo):
            pytest.fail()

    # Case 7: kmax2 for "Triangular" or "LongEdge" not a positive integer
    static_filter_list_wrongparams = ['Triangular', [[True, False], -2]]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wrongparams})
    except ValueError as excinfo:
        if 'Triangular and LongEdge filter_params must have a positive integer as the second element.' not in str(excinfo):
            pytest.fail()

    # Case 8: static_filter_list not the right length for "Markovian" case
    static_filter_list_wronglength = ['Markovian', [True]]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wronglength})
    except ValueError as excinfo:
        if 'The list of booleans in static_filter_list must have the same length as the number of modes.' not in str(excinfo):
            pytest.fail()

    # Case 9: static_filter_list not the right length for "Triangular"/"LongEdge" cases
    static_filter_list_wronglength = ['Triangular', [[True], 2]]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wronglength})
    except ValueError as excinfo:
        if 'The list of booleans in static_filter_list must have the same length as the number of modes.' not in str(excinfo):
            pytest.fail()

    # Case 10: Filter_params not a list of booleans for "Markovian"
    static_filter_list_wrongparams = ['Markovian', ['True', 'False']]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "static_filter_list": static_filter_list_wrongparams})
    except ValueError as excinfo:
        if 'filter_params for Markovian filter must be a list of booleans.' not in str(excinfo):
            pytest.fail()

    ## Testing that overdefining modes (list_modes_by_bath and list_modes) yields an error
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes, "list_modes_by_bath": list_modes_by_bath})
    except ValueError as excinfo:
        if 'list_modes_by_bath and list_modes should not both be defined.' not in str(excinfo):
            pytest.fail()

    ## Testing incompatible list_modes_by_bath and list_lop lengths yields an error
    list_modes_by_bath_wronglength = [ishizaki_decomposition_bcf_dl(40, 60, 290, 0)]
    list_lop_wronglength = [sparse.coo_matrix(([1], ([1], [2])), shape=(3, 3)),
                sparse.coo_matrix(([1], ([2], [1])), shape=(3, 3)), sparse.coo_matrix(([1], ([2], [2])), shape=(3, 3))]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_lop": list_lop_wronglength, "list_modes_by_bath": list_modes_by_bath_wronglength})
    except ValueError as excinfo:
        if 'list_modes_by_bath and list_lop must have the same length.' not in str(excinfo):
            pytest.fail()

    ## Testing list_modes_by_bath structure
    # list_modes_by_bath not a list of lists
    list_modes_by_bath_wrongstructure = ['mode1', 'mode2']
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes_by_bath": list_modes_by_bath_wrongstructure})
    except ValueError as excinfo:
        if 'list_modes_by_bath must be a list of lists.' not in str(excinfo):
            pytest.fail()

    # list_modes_by_bath doesn't contain paired Gs and Ws
    list_modes_by_bath_wrongpairing = [ishizaki_decomposition_bcf_dl(35, 50, 295, 0), [1, 2, 3]]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes_by_bath": list_modes_by_bath_wrongpairing})
    except ValueError as excinfo:
        if 'list_modes_by_bath should contain paired Gs and Ws, which guarantees an even number of elements in each sublist.' not in str(excinfo):
            pytest.fail()

    ## Testing list_modes structure
    # list_modes doesn't contain paired Gs and Ws
    list_modes_wrongpairing = [1, 2, 3]
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {"list_modes": list_modes_wrongpairing})
    except ValueError as excinfo:
        if 'list_modes should contain paired Gs and Ws, which guarantees an even number of elements.' not in str(excinfo):
            pytest.fail()

    ## Testing that an error is raised if neither list_modes nor list_modes_by_bath are defined
    try:
        prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, {})
    except ValueError as excinfo:
        if 'Either list_modes_by_bath or list_modes must be defined.' not in str(excinfo):
            pytest.fail()

def test_prepare_convergence_parameter_dict():
    ## Test proper output with all parameters defined
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

    ## Test proper output with defaults
    convergence_dict = prepare_convergence_parameter_dict(t_step, max_hier)
    assert convergence_dict["t_step"] == t_step
    assert convergence_dict["max_hier"] == max_hier
    assert convergence_dict["delta_a"] == 0
    assert convergence_dict["delta_s"] == 0
    assert convergence_dict["set_update_step"] == 1
    assert convergence_dict["set_f_discard"] == 0