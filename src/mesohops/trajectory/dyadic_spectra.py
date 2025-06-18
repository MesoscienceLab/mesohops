import numpy as np
from scipy import sparse
import time as timer
from mesohops.trajectory.hops_dyadic import DyadicTrajectory
from mesohops.trajectory.exp_noise import bcf_exp

__title__ = "dyadic_spectra"
__author__ = "D. I. G. B. Raccah, A. Hartzell, T. Gera, J. K. Lynd"
__version__ = "1.5"

class DyadicSpectra(DyadicTrajectory):
    """
    Acts as an interface to calculate spectra using the Dyadic HOPS method.
    """

    __slots__ = (
        # --- Initialization and tracking ---
        '__initialized',     # Initialization status flag

        # --- Spectroscopy parameters ---
        'spectrum_type',     # Type of spectrum to calculate
        't_1',               # First propagation time
        't_2',               # Second propagation time
        't_3',               # Third propagation time
        'list_t',            # List of propagation times
        'E_1',               # First field definition
        'E_2',               # Second field definition
        'E_3',               # Third field definition
        'E_sig',             # Signal field definition
        'list_ket_sites',    # Ket sites excited by field
        'list_bra_sites',    # Bra sites excited by field

        # --- Chromophore parameters ---
        'M2_mu_ge',            # Transition dipole matrix
        'n_chromophore',       # Number of chromophores
        'H2_sys_hamiltonian',  # System Hamiltonian
        'lop_list_hier',       # L-operators associated with hierarchy modes
        'gw_sysbath_hier',     # Hierarchy mode parameters
        'lop_list_noise',      # L-operators associated with noise
        'gw_sysbath_noise',    # Noise mode parameters
        'lop_list_ltc',        # L-operators associated with LTC
        'ltc_param',           # Low-temperature correction parameters

        # --- Convergence parameters ---
        't_step',              # Time step
        'max_hier',            # Maximum hierarchy depth
        'delta_a',             # Auxiliary derivative error bound
        'delta_s',             # State derivative error bound
        'set_update_step',     # Update step
        'set_f_discard',       # Discard fraction
        'static_filter_list',  # Static hierarchy filters

        # --- State dimensions ---
        'n_state_hilb',     # Hilbert space dimension
        'n_state_dyad',     # Dyadic space dimension

        # --- Noise configuration ---
        'noise_param'       # Noise parameters
    )

    def __init__(self, spectroscopy_dict, chromophore_dict, convergence_dict, seed):
        """
        Inputs
        ------
        1. spectroscopy_dict: dict
                              Dictionary of spectroscopy type, field definitions, sites
                              acted upon, and propagation times between interactions.
                              [see prepare_spectroscopy_input_dict()]

        2. chromophore_dict: dict
                             Dictionary containing transition dipole moments, system
                             hamiltonian, and bath parameters.
                             [see prepare_chromophore_input_dict()]

        3. convergence_dict: dict
                             Dictionary of various convergence parameters.
                             [see prepare_convergence_parameter_dict()]

        4. seed: int
                 Seed for noise generation.
        """
        # Setting the initialized flag to False
        self.__initialized = False

        # Extracting spectroscopy parameters from spectroscopy_dict
        self.spectrum_type = spectroscopy_dict["spectrum_type"]
        self.t_1 = spectroscopy_dict["t_1"]
        self.t_2 = spectroscopy_dict["t_2"]
        self.t_3 = spectroscopy_dict["t_3"]
        self.list_t = [self.t_1, self.t_2, self.t_3]
        self.E_1 = spectroscopy_dict["E_1"]
        self.E_2 = spectroscopy_dict.get("E_2")
        self.E_3 = spectroscopy_dict.get("E_3")
        self.E_sig = spectroscopy_dict["E_sig"]
        self.list_ket_sites = spectroscopy_dict["list_ket_sites"]
        self.list_bra_sites = spectroscopy_dict.get("list_bra_sites", None)

        # Extracting chromophore parameters from chromophore_dict
        self.M2_mu_ge = chromophore_dict["M2_mu_ge"]
        self.n_chromophore = chromophore_dict["n_chromophore"]
        self.H2_sys_hamiltonian = chromophore_dict["H2_sys_hamiltonian"]
        self.lop_list_hier = chromophore_dict["lop_list_hier"]
        self.gw_sysbath_hier = chromophore_dict["gw_sysbath_hier"]
        self.lop_list_noise = chromophore_dict["lop_list_noise"]
        self.gw_sysbath_noise = chromophore_dict["gw_sysbath_noise"]
        self.lop_list_ltc = chromophore_dict["lop_list_ltc"]
        self.ltc_param = chromophore_dict["ltc_param"]
        self.static_filter_list = chromophore_dict.get("static_filter_list", None)

        # Extracting convergence parameters from convergence_dict
        self.t_step = convergence_dict["t_step"]
        self.max_hier = convergence_dict["max_hier"]
        self.delta_a = convergence_dict["delta_a"]
        self.delta_s = convergence_dict["delta_s"]
        self.set_update_step = convergence_dict["set_update_step"]
        self.set_f_discard = convergence_dict["set_f_discard"]

        # Defining number of states in Hilbert and Dyadic spaces
        self.n_state_hilb = self.n_chromophore + 1
        self.n_state_dyad = 2 * self.n_state_hilb

        # Checking the shape of the system Hamiltonian
        if np.shape(self.H2_sys_hamiltonian) != (self.n_state_hilb, self.n_state_hilb):
            raise ValueError("H2_sys_hamiltonian must be ((n_chrom + 1) x (n_chrom + "
                             "1)) to account for each chromophore and the ground "
                             "state.")

        # Preparing system parameter dictionary
        system_param = {"HAMILTONIAN": self.H2_sys_hamiltonian,
                        "GW_SYSBATH": self.gw_sysbath_hier,
                        "L_HIER": self.lop_list_hier,
                        "L_NOISE1": self.lop_list_noise, "ALPHA_NOISE1": bcf_exp,
                        "PARAM_NOISE1": self.gw_sysbath_noise,
                        "L_LT_CORR": self.lop_list_ltc,
                        "PARAM_LT_CORR": self.ltc_param}

        # Preparing equation of motion dictionary
        eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

        # Preparing noise parameter dictionary
        self.noise_param = {"SEED": seed, "MODEL": "FFT_FILTER",
                            "TLEN": float(1000 + np.sum(self.list_t)),
                            "TAU": 0.5 if self.t_step % 1 == 0 else self.t_step / 2}

        # Preparing hierarchy parameter dictionary
        hierarchy_param = {"MAXHIER": self.max_hier}
        if self.static_filter_list:
            hierarchy_param["STATIC_FILTERS"] = self.static_filter_list

        # Preparing storage parameter dictionary
        storage_param = {}

        # Initializing DyadicTrajectory class
        super().__init__(system_param, eom_param, self.noise_param, hierarchy_param,
                         storage_param)

    def initialize(self):
        """
        Prepares the ground state initial dyadic wave function based on chromophore_dict
        definitions and passes it to DyadicTrajectory.initialize. Also makes the
        trajectory adaptive when convergence_dict parameters are defined to do so.

        Returns
        -------
        None
        """
        # Initializing DyadicTrajectory class if initialized flag is False
        if not self.__initialized:

            # Starting the initialization timer
            timer_checkpoint = timer.time()

            # Defining initial bra and ket wavefunctions
            psi_k = np.zeros(self.n_state_hilb)
            psi_k[0] = 1
            psi_b = np.zeros(self.n_state_hilb)
            psi_b[0] = 1

            # Making the trajectory adaptive if delta_a or delta_s is greater than 0
            if self.delta_a > 0 or self.delta_s > 0:
                self.make_adaptive(self.delta_a, self.delta_s, self.set_update_step,
                                   self.set_f_discard)

            # Initializing trajectory
            super().initialize(psi_k, psi_b, timer_checkpoint=timer_checkpoint)

            # Setting the initialized flag to True
            self.__initialized = True

        # Raising a warning if the DyadicTrajectory object has already been initialized
        else:
            print("WARNING: DyadicTrajectory has already been initialized.")

    def _hilb_operator(self, action_type, field, list_sites):
        """
        Constructs the Hilbert space raising or lowering operator.

        Parameters
        ----------
        1. action_type: str
                        Type of action to be performed. (Options: "raise" or "lower".)

        2. field: np.array(complex)
                  Field vector definition.

        3. list_sites: np.array(int)
                       List of sites acted on by the operator, with the left-most site
                       in the chain indexed by 1.

        Returns
        -------
        1. R2_raise_hilb_op/L2_lower_hilb_op: np.array(complex)
                                              Hilbert space raising/lowering operator.
        """
        # Calculating μ•E for the given sites
        interactions = np.dot(self.M2_mu_ge[list_sites - 1], field)

        # Constructing sparse raising operator
        if action_type == "raise":
            return sparse.coo_matrix((interactions,
                                      (list_sites, np.zeros_like(list_sites))),
                                     shape=(self.n_state_hilb, self.n_state_hilb),
                                     dtype=np.float64)

        # Constructing sparse lowering operator
        elif action_type == "lower":
            return sparse.coo_matrix((interactions,
                                      (np.zeros_like(list_sites), list_sites)),
                                     shape=(self.n_state_hilb, self.n_state_hilb),
                                     dtype=np.float64)

        # Throwing error for invalid action types
        else:
            raise ValueError("action_type must be either 'raise' or 'lower'.")

    def _final_dyad_operator(self):
        """
        Constructs the final dyadic operator for calculating the response function
        and records the time index when the operator begins its action.

        Returns
        -------
        1. F2_final_op: np.array(complex)
                        Dyadic operator to calculate the response function component.

        2. final_op_index: int
                           Time index after which the response function component is
                           calculated.
        """
        # Calculating μ•Esig
        interactions = np.dot(self.M2_mu_ge, self.E_sig)

        # Defining start index for the final response operation
        final_op_index = int(self.t_2 / self.t_step)

        # Constructing sparse final dyadic operator
        F2_final_op = sparse.coo_matrix((interactions,
                                         ([self.n_state_hilb] * self.n_chromophore,
                                          np.arange(1, self.n_state_hilb))),
                                        shape=(self.n_state_dyad, self.n_state_dyad),
                                        dtype=np.float64)
        return F2_final_op, final_op_index

    def calculate_spectrum(self):
        """
        Constructs the DyadicTrajectory object, propagates the excitation dynamics
        according to the given optical response pathway, and calculates the time-domain
        response function for the single trajectory defined by the DyadicSpectra seed.

        Returns
        -------
        1. response_t: np.array(complex)
                       Calculated time-domain response function scaled to account for
                       the degenerate and conjugate pathways.
        """
        # Initializing trajectory
        self.initialize()

        # Defaulting scaling factor to 1
        scaling_factor = 1

        # Defining final operator and time index for final response operation
        final_op, final_op_index = self._final_dyad_operator()

        # Absorption case:
        # =============================
        # Double-sided Feynman diagram:
        #            ||g><g||
        # μ•Esig <-- |------|
        #            ||e><g||
        #   μ•E1 --> |------|
        #            ||g><g||
        # =============================

        if self.spectrum_type == "ABSORPTION":
            # Set timer checkpoint
            timer_checkpoint = timer.time()

            # Raising ket sites
            self._dyad_operator(self._hilb_operator("raise", self.E_1,
                                                    self.list_ket_sites), 'ket')

            # Propagating through t_1 time
            self.propagate(self.t_1, self.t_step, timer_checkpoint)

            # Scaling factor accounting for the sum of two complex conjugate pathways
            scaling_factor = 2

        # Fluorescence case:
        # ==============================
        # Double-sided Feynman diagram:
        #            ||g><g||
        # μ•Esig <-- |------|
        #            ||g><e||
        #            |------| --> μ•Esig
        #            ||e><e||
        #   μ•E1 --> |------| <-- μ•E1
        #            ||g><g||
        # ==============================

        elif self.spectrum_type == "FLUORESCENCE":
            # Set timer checkpoint
            timer_checkpoint = timer.time()

            # Raising ket sites
            self._dyad_operator(
                self._hilb_operator("raise", self.E_1, self.list_ket_sites), 'ket')

            # Raising bra sites
            self._dyad_operator(
                self._hilb_operator("raise", self.E_2, self.list_bra_sites), 'bra')

            # Propagating through t_2 time
            self.propagate(self.t_2, self.t_step, timer_checkpoint)

            # New timer checkpoint
            timer_checkpoint = timer.time()

            # Lowering bra sites
            self._dyad_operator(self._hilb_operator(
                "lower", self.E_3, np.arange(1, self.n_state_hilb)), 'bra')

            # Propagating through t_3 time
            self.propagate(self.t_3, self.t_step, timer_checkpoint)

            # Scaling factor accounting for the two equivalent pathways under the
            # impulsive limit and their complex conjugates.
            scaling_factor = 4

        # Calculating response function
        return scaling_factor * self._response_function_comp(final_op, final_op_index)

    @property
    def initialized(self):
        return self.__initialized


def prepare_spectroscopy_input_dict(spectrum_type, propagation_time_dict, field_dict,
                                    site_dict):
    """
    Prepares the spectroscopy_dict input dictionary for DyadicSpectra.

    Parameters
    ----------
    1. spectrum_type: str
                      Type of spectrum to be calculated. (Options: "ABSORPTION" or
                      "FLUORESCENCE".)

    2. propagation_time_dict: dict
                              Dictionary of propagation times between field
                              interactions. (Key Options: "t_1", "t_2", "t_3".)

    3. field_dict: dict
                   Dictionary of field vector definitions. All field vectors must be
                   numpy arrays with exactly 3 entries.
                   (Key Options: "E_1", "E_2", "E_3", "E_sig".)

    4. site_dict: dict
                  The set of initially-excited sites on the ket and bra sides,
                  defined by numpy integer arrays with indexing starting at 1, not 0.
                  (Key Options: "list_ket_sites", "list_bra_sites".)

    Returns
    -------
    1. spectroscopy_input_dict: dict
                                Dictionary of spectroscopy parameters needed for
                                DyadicSpectra class.
    """
    # Defining allowed spectrum types
    list_allowed_spectrum_types = ["ABSORPTION", "FLUORESCENCE"]

    # Checking list_ket_site input structure
    if "list_ket_sites" not in site_dict.keys():
        raise ValueError("list_ket_sites must be defined.")

    if not isinstance(site_dict["list_ket_sites"], np.ndarray):
        site_dict["list_ket_sites"] = np.array(site_dict["list_ket_sites"])

    # Checking site indexing structure
    for key, value in site_dict.items():
        if 0 in value:
            raise ValueError("Ket and Bra sites must be indexed starting from 1.")

    # Checking field_dict input structure
    for key, value in field_dict.items():
        if not isinstance(value, np.ndarray):
            raise ValueError("All field entries should be numpy arrays.")

        elif value.shape != (3,):
            raise ValueError("All field entries should be numpy arrays with exactly "
                             "3 entries.")

    # Removing keys with None/0 values from propagation_time_dict
    for key, value in list(propagation_time_dict.items()):
        if value is None or value <= 0:
            del propagation_time_dict[key]

    # Absorption case (see DyadicSpectra.calculate_spectrum() for diagram):
    if spectrum_type == "ABSORPTION":
        # Checking necessary parameters are defined
        if "t_1" not in propagation_time_dict.keys():
            raise ValueError("Propagation time after first field interaction (t_1) "
                             "must be defined as > 0 for absorption.")

        if "E_1" not in field_dict.keys():
            raise ValueError("E_1 must be defined for absorption.")

        # Warning user if unused parameters are defined
        if len(propagation_time_dict) > 1:
            print("WARNING: Only t_1 is necessary for absorption. Setting all other "
                  "propagation times to zero.")

        if len(field_dict) > 1:
            print("WARNING: Only E_1 is necessary for absorption. E_sig is set to E_1. "
                  "All other field definitions will be discarded")

        # Returning dictionary for absorption
        return {"spectrum_type": spectrum_type, "E_1": field_dict["E_1"],
                "E_sig": field_dict["E_1"], "t_1": propagation_time_dict["t_1"],
                "t_2": 0, "t_3": 0, "list_ket_sites": site_dict["list_ket_sites"]}

    # Fluorescence case (see DyadicSpectra.calculate_spectrum() for diagram):
    elif spectrum_type == "FLUORESCENCE":
        # Checking necessary parameters are properly defined
        if "list_bra_sites" not in site_dict.keys():
            raise ValueError("list_bra_sites must be defined for fluorescence.")

        if not isinstance(site_dict["list_bra_sites"], np.ndarray):
            site_dict["list_bra_sites"] = np.array(site_dict["list_bra_sites"])

        if ("t_2" not in propagation_time_dict.keys() or "t_3" not in
                propagation_time_dict.keys()):
            raise ValueError("Propagation times after second and third field "
                             "interactions (t_2, t_3) must be defined as > 0 for "
                             "fluorescence.")

        if "E_1" not in field_dict.keys():
            raise ValueError("E_1 must be defined for fluorescence.")

        # Warning user if the signal field is not defined
        if "E_sig" not in field_dict.keys():
            print("WARNING: E_sig is not defined. Setting E_sig to default, [0, 0, 1].")

        # Warning user if unused parameters are defined
        if len(propagation_time_dict) > 2:
            print("WARNING: Only t_2 and t_3 are necessary for fluorescence. Setting "
                  "all other propagation times to zero.")

        if len(field_dict) > 2:
            print("WARNING: Only E_1 and E_sig are necessary for fluorescence. All "
                  "other field definitions will be discarded.")

        # Returning dictionary for fluorescence
        return {"spectrum_type": spectrum_type, "E_1": field_dict["E_1"],
                "E_2": field_dict["E_1"],
                "E_3": field_dict.get("E_sig", np.array([0, 0, 1])),
                "E_sig": field_dict.get("E_sig", np.array([0, 0, 1])), "t_1": 0,
                "t_2": propagation_time_dict["t_2"],
                "t_3": propagation_time_dict["t_3"],
                "list_ket_sites": site_dict["list_ket_sites"],
                "list_bra_sites": site_dict["list_bra_sites"]}

    # Throwing error for invalid spectrum types
    else:
        raise ValueError(f"spectrum_type must be one of the following: "
                         f"{list_allowed_spectrum_types}")


def prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian, bath_dict):
    """
    Prepares the chromophore_dict input dictionary for DyadicSpectra.

    Parameters
    ----------
    1. M2_mu_ge: np.array(complex)
                 Array of transition dipole moments for each chromophore. The array
                 should have shape (n_chromophore, 3).

    2. H2_sys_hamiltonian: np.array(complex)
                           System Hamiltonian in Hilbert space. The array should have
                           shape (n_chromophore + 1, n_chromophore + 1) to account for
                           the ground state.

    3. bath_dict: dict
                  Dictionary of bath parameters. (Key Options: "list_lop", "list_modes",
                  "list_modes_by_bath", "nmodes_LTC", "static_filter_list".)
                  [NOTE: Either "list_modes" or "list_modes_by_bath" must be defined,
                  but not both.]

                   Key Descriptions:
                   -----------------
                     a. list_lop: list(np.array(complex)), optional
                                  List of unique system-bath coupling operators for each
                                  independent bath. If omitted, they default to site
                                  projection operators.

                     b. list_modes: list(complex), optional
                                    List of exponential modes making up the time
                                    correlation function of all independent baths.
                                    For use in systems where the independent baths are
                                    identical. The list should be in alternating format
                                    [G_1,W_1,G_2,W_2,...] where G_j•exp(-W_j•t/hbar) is
                                    the jth exponential mode of the correlation
                                    function, with prefactor G_j [units: cm^-2] and
                                    exponential decay rate W_j [units: cm^-1].
                                    [NOTE: Input structure matches output from
                                    bath_corr_functions.py helper functions.]

                     c. list_modes_by_bath: list(list(complex)), optional
                                            List of lists containing exponential modes
                                            making up the time correlation function for
                                            each independent bath. For use in systems
                                            with non-identical baths. See list_modes
                                            above for format of exponential mode
                                            definition.
                                            Example structure:
                                            ------------------
                                            [[G_1,W_1,G_2,W_2, ...],
                                             [G_1',W_1',G_2',W_2',...],
                                             ...]
                                            where the outer nest, [[],[],...], lists
                                            baths and the inner nests,
                                            [G_1,W_1,G_2,W_2,...], list modes.
                                            [NOTE: Inner nest input structure matches
                                            output from bath_corr_functions.py helper
                                            functions.]

                     d. nmodes_LTC: int, optional
                                    Number of modes in each independent bath treated
                                    with low-temperature correction, rather than
                                    explicitly in the hierarchy. The final nmodes_LTC
                                    modes in each bath will be low-temperature
                                    corrected, so modes should be ordered by decreasing
                                    G/W ratio. Note that nmodes_LTC must be less than
                                    the number of modes in each bath.

                                    For more details on low-temperature correction, see:
                                    "MesoHOPS: Size-invariant scaling calculations of
                                    multi-excitation open quantum systems."
                                    Brian Citty, Jacob K. Lynd, et al. J. Chem. Phys.
                                    160, 144118 (2024)

                    e. static_filter_list: list, optional
                                           List of static filters applied to the
                                           hierarchy. Each filter is defined by a list
                                           of the form [filter_name, filter_params].
                                           OPTIONS:
                                           --------
                                           1. "Markovian": auxiliary wave functions
                                               associated with filtered modes are
                                               only included in the hierarchy if they
                                               are depth 1. filter_params should be a
                                               list of booleans: True for filtered
                                               modes, False for unfiltered modes.

                                           2. "Triangular": auxiliary wave functions
                                               associated with filtered modes are only
                                               included in the hierarchy if they are at
                                               or below depth kmax2. filter_params =
                                               [list_modes_filtered, kmax2], where
                                               list_modes_filtered is a list of
                                               booleans: True for filtered modes,
                                               False for unfiltered modes. Note kmax2
                                               is an integer.

                                           3. "LongEdge": auxiliary wave functions
                                               associated with filtered modes are only
                                               included in the hierarchy if they are at
                                               or below depth kmax2 OR only have depth
                                               in a single mode. filter_params =
                                               [list_modes_filtered, kmax2], where
                                               list_modes_filtered is a list of
                                               booleans: True for filtered modes,
                                               False for unfiltered modes. Note kmax2
                                               is an integer.

                                           If the list of booleans determining which
                                           modes are filtered does not match the full
                                           set of modes in the "GW_SYSBATH" key of the
                                           system parameter dictionary, the trajectory
                                           will raise an error and the simulation will
                                           be terminated. The use of any static
                                           hierarchy filter may reduce the accuracy of a
                                           given calculation; that is, static hierarchy
                                           filters must be tested like any other
                                           convergence parameters.

                                           For more details on static filters, see:
                                           "MesoHOPS: Size-invariant scaling
                                           calculations of multi-excitation open quantum
                                           systems."
                                           Brian Citty, Jacob K. Lynd, et al.
                                           J. Chem. Phys. 160, 144118 (2024)

    Returns
    -------
    1. chromophore_dict: dict
                         Dictionary of chromophore parameters needed for DyadicSpectra
                         class.
    """
    # Defining number of chromophores
    n_chromophore = len(M2_mu_ge)

    # Checking M2_mu_ge input structure
    M2_mu_ge = np.array(M2_mu_ge)
    if M2_mu_ge.shape[1] != 3:
        raise ValueError(
            "M2_mu_ge must be a numpy array with shape (n_chromophore, 3).")

    # Setting default nmodes_LTC to 0 and checking nmodes_LTC input structure
    if "nmodes_LTC" not in bath_dict.keys() or bath_dict["nmodes_LTC"] is None:
        bath_dict["nmodes_LTC"] = 0

    elif not isinstance(bath_dict["nmodes_LTC"], int):
        raise ValueError("nmodes_LTC must be an integer or None.")

    elif bath_dict["nmodes_LTC"] < 0:
        raise ValueError("nmodes_LTC must be >= 0 or set to None.")

    # Checking static_filter_list input structure
    if "static_filter_list" in bath_dict.keys():
        if not isinstance(bath_dict["static_filter_list"], list):
            raise ValueError("static_filter_list must be a list.")

        elif len(bath_dict["static_filter_list"]) != 2:
            raise ValueError("static_filter_list must be a 2-element list of the form: "
                             "[filter_name, filter_params].")

        elif bath_dict["static_filter_list"][0] not in ["Markovian", "Triangular",
                                                        "LongEdge"]:
            raise ValueError("Filter name must be 'Markovian', 'Triangular', or "
                             "'LongEdge'.")

        elif (bath_dict["static_filter_list"][0] == "Markovian" and
              not all(isinstance(boolean, bool) for boolean in
                      bath_dict["static_filter_list"][1])):
            raise ValueError(
                "filter_params for Markovian filter must be a list of booleans.")

        elif (bath_dict["static_filter_list"][0] in ["Triangular", "LongEdge"] and
              len(bath_dict["static_filter_list"][1]) != 2):
            raise ValueError("The Triangular and LongEdge filter_params must be a "
                             "list of booleans and an integer.")

        elif (bath_dict["static_filter_list"][0] in ["Triangular", "LongEdge"] and
              not isinstance(bath_dict["static_filter_list"][1][1], int)):
            raise ValueError("The second entry in filter_params for the Triangular "
                             "and LongEdge filters must be an integer.")

        elif bath_dict["static_filter_list"][0] in ["Triangular", "LongEdge"]:
            if not all(isinstance(boolean, bool) for boolean in
                       bath_dict["static_filter_list"][1][0]):
                raise ValueError("The first entry in filter_params for the "
                                 "Triangular and LongEdge filters must be a "
                                 "list of booleans.")

        elif (bath_dict["static_filter_list"][0] in ["Triangular", "LongEdge"] and
              bath_dict["static_filter_list"][1][1] < 0):
            raise ValueError("Triangular and LongEdge filter_params must have a "
                             "positive integer as the second element.")

    # Checking bath_dict input structure
    for key, value in list(bath_dict.items()):
        # Setting bath_dict arrays to lists
        if type(value) == np.ndarray:
            bath_dict[key] = list(value)

        # Removing keys with None/0 values from bath_dict (except nmodes_LTC)
        elif (key != "nmodes_LTC") and (value is None or value == 0):
            del bath_dict[key]

    # Checking list_modes/list_modes_by_bath over-definition
    if "list_modes_by_bath" in bath_dict.keys() and "list_modes" in bath_dict.keys():
        raise ValueError(
            "list_modes_by_bath and list_modes should not both be defined.")

    # Setting default list_lop if not defined
    if "list_lop" not in bath_dict.keys():
        # Site-projection L-operators
        bath_dict["list_lop"] = [sparse.coo_matrix(([1], ([chrom + 1], [chrom + 1])),
                                                   shape=(n_chromophore + 1,
                                                          n_chromophore + 1)) for
                                 chrom in range(n_chromophore)]

    # Checking list_modes_by_bath/list_modes structure
    if "list_modes_by_bath" in bath_dict:
        # Checking list_modes_by_bath/list_lop compatibility
        if len(bath_dict["list_modes_by_bath"]) != len(bath_dict["list_lop"]):
            raise ValueError(
                "list_modes_by_bath and list_lop must have the same length.")

        # Checking that list_modes_by_bath is a list of lists of paired Gs and Ws
        for sublist in bath_dict["list_modes_by_bath"]:
            if not isinstance(sublist, list):
                raise ValueError("list_modes_by_bath must be a list of lists.")
            elif len(sublist) % 2 != 0:
                raise ValueError("list_modes_by_bath should contain paired Gs and Ws, "
                                 "which guarantees an even number of elements in each "
                                 "sublist.")

    elif "list_modes" in bath_dict:
        # Checking that list_modes is a list of paired Gs and Ws
        if len(bath_dict["list_modes"]) % 2 != 0:
            raise ValueError("list_modes should contain paired Gs and Ws, which "
                             "guarantees an even number of elements.")
    else:
        raise ValueError("Either list_modes_by_bath or list_modes must be defined.")

    # Preparing empty chromophore dictionary
    gw_sysbath = []
    list_lop_sysbath_by_mode = []
    gw_noise = []
    list_lop_noise_by_mode = []
    list_lop_ltc = []
    list_ltc_param = []

    # Populating chromophore dictionary
    for bath in range(len(bath_dict["list_lop"])):
        # Unpacking list of Gs and Ws for each bath in the list_modes_by_bath case
        if "list_modes_by_bath" in bath_dict.keys():
            bath_dict["list_modes"] = bath_dict["list_modes_by_bath"][bath]

        # Converting list of Gs and Ws to list of coupled G-W tuples
        list_modes_as_tuples = [(bath_dict["list_modes"][i],
                                 bath_dict["list_modes"][i + 1]) for i in
                                range(0, len(bath_dict["list_modes"]), 2)]

        # Checking nmodes_LTC and static_filter_list compatibility with number of modes
        if bath_dict["nmodes_LTC"] >= len(list_modes_as_tuples):
            raise ValueError("nmodes_LTC must be less than the number of "
                             "modes in each bath.")

        elif "static_filter_list" in bath_dict.keys():
            if bath_dict["static_filter_list"][0] == 'Markovian':
                if len(bath_dict["static_filter_list"][1]) != len(list_modes_as_tuples):
                    raise ValueError("The list of booleans in static_filter_list must "
                                     "have the same length as the number of modes.")

            elif bath_dict["static_filter_list"][0] in ['Triangular', 'LongEdge']:
                if len(bath_dict["static_filter_list"][1][0]) != len(
                        list_modes_as_tuples):
                    raise ValueError("The list of booleans in static_filter_list must "
                                     "have the same length as the number of modes.")

        ltc_param = 0
        list_lop_ltc.append(bath_dict["list_lop"][bath])
        # Appending gw_sysbath and lop_list for each G-W tuple
        for nmode, mode in enumerate(list_modes_as_tuples):
            gw_noise.append(mode)
            list_lop_noise_by_mode.append(bath_dict["list_lop"][bath])

            # Checking if mode belongs to the low-temperature corrected modes
            if len(list_modes_as_tuples) - nmode > bath_dict["nmodes_LTC"]:
                # Appending each G-W tuple treated in the hierarchy
                gw_sysbath.append(mode)
                list_lop_sysbath_by_mode.append(bath_dict["list_lop"][bath])

            # Updating ltc parameter for each low-temperature corrected mode
            else:
                ltc_param += mode[0] / mode[1]

        # Appending ltc parameter to the bath
        list_ltc_param.append(ltc_param)

    # Returning chromophore dictionary
    return {"M2_mu_ge": M2_mu_ge, "n_chromophore": n_chromophore,
            "H2_sys_hamiltonian": H2_sys_hamiltonian,
            "lop_list_hier": list_lop_sysbath_by_mode, "gw_sysbath_hier": gw_sysbath,
            "lop_list_noise": list_lop_noise_by_mode, "gw_sysbath_noise": gw_noise,
            "lop_list_ltc": list_lop_ltc, "ltc_param": list_ltc_param,
            "static_filter_list": bath_dict.get("static_filter_list", None)}


def prepare_convergence_parameter_dict(t_step, max_hier, delta_a=0, delta_s=0,
                                       set_update_step=1, set_f_discard=0):
    """
    Prepares the convergence_dict input dictionary for DyadicSpectra.

    Parameters
    ----------
    1. t_step: float
               Time step of the simulation.

    2. max_hier: int
                 Maximum hierarchy depth.

    3. delta_a: float, optional
                Threshold value for the adaptive auxiliary basis (Options: >= 0).

    4. delta_s: float, optional
                Threshold value for the adaptive state basis (Options: >= 0).

    5. set_update_step: int, optional
                        Update step for the adaptive basis (Options: >= 0).

    6. set_f_discard: float, optional
                      Discard threshold for the adaptive basis (Options: >= 0).

    Returns
    -------
    1. convergence_dict: dict
                         Dictionary of convergence parameters needed for DyadicSpectra
                         class.
    """
    # Returning convergence dictionary
    return {"t_step": t_step, "max_hier": max_hier, "delta_a": delta_a,
            "delta_s": delta_s, "set_update_step": set_update_step,
            "set_f_discard": set_f_discard}

