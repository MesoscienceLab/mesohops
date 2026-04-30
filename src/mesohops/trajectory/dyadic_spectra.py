import numpy as np
from scipy import sparse
import time as timer
import copy
from mesohops.trajectory.hops_dyadic import DyadicTrajectory
from mesohops.trajectory.exp_noise import bcf_exp
import warnings

__title__ = "dyadic_spectra"
__author__ = "D. I. G. B. Raccah, A. Hartzell, T. Gera, J. K. Lynd"
__version__ = "1.5"


class DyadicSpectra(DyadicTrajectory):
    """
    Acts as an interface to calculate spectra using the Dyadic HOPS method.
    """

    __slots__ = (
        # --- Initialization and tracking ---
        '__initialized', # Initialization status flag

        # --- Spectroscopy parameters ---
        'spectrum_type', # Type of spectrum to calculate
        't_1',           # First propagation time
        't_2',           # Second propagation time
        't_3',           # Third propagation time
        'list_t',        # List of propagation times
        'E_1',           # First field definition
        'E_2',           # Second field definition
        'E_3',           # Third field definition
        'E_sig',         # Signal field definition
        'list_interaction_cluster_1', # Chromophores excited/de-excited by first operator
        'list_interaction_cluster_2', # Chromophores excited/de-excited by second operator
        'list_interaction_cluster_3', # Chromophores excited/de-excited by third operator

        # --- Chromophore parameters ---

        'M2_mu_ge',      # Transition dipole matrix
        'n_chromophore', # Number of chromophores
        'H2_sys_hamiltonian', # System Hamiltonian
        'lop_list_hier', # L-operators associated with hierarchy modes
        'gw_sysbath_hier', # Hierarchy mode parameters
        'lop_list_noise', # L-operators associated with noise
        'gw_sysbath_noise', # Noise mode parameters
        'lop_list_ltc', # L-operators associated with LTC
        'ltc_param', # Low-temperature correction parameters
        'n_ee_states', # Number of doubly-excited states
        'list_ee_states',# List of doubly-excited states

        # --- Convergence parameters ---
        't_step', # Time step
        'max_hier',# Maximum hierarchy depth
        'delta_a',# Auxiliary derivative error bound
        'delta_s',# State derivative error bound
        'set_update_step', # Update step
        'set_f_discard', # Discard fraction
        'static_filter_list', # Static hierarchy filters

        # --- State dimensions ---
        'n_state_hilb', # Full Hilbert-space dimension
        'n_state_dyad', # Dyadic space dimension

        # --- Noise configuration ---
        'noise_param' # Noise parameters
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
        self.list_interaction_cluster_1 = spectroscopy_dict["list_interaction_cluster_1"]
        self.list_interaction_cluster_2 = spectroscopy_dict.get("list_interaction_cluster_2", None)
        self.list_interaction_cluster_3 = spectroscopy_dict.get("list_interaction_cluster_3", None)

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

        if self.spectrum_type in ['ESA-R', 'ESA-NR']:
            i_idx, j_idx = np.triu_indices(self.n_chromophore, k=1)
            self.list_ee_states = list(zip(i_idx + 1, j_idx + 1))
            self.n_ee_states = len(self.list_ee_states)
        else:
            self.list_ee_states = []
            self.n_ee_states = 0

        # Full Hilbert-space and dyadic-space dimensions.
        self.n_state_hilb = self.n_chromophore + 1 + self.n_ee_states
        self.n_state_dyad = 2 * self.n_state_hilb
        # Checking the shape of the system Hamiltonian

        expected_hilbert_shape = (
            self.n_state_hilb,
            self.n_state_hilb,
        )
        if np.shape(self.H2_sys_hamiltonian) != expected_hilbert_shape:
            raise ValueError(f"H2_sys_hamiltonian must be {expected_hilbert_shape}")
        for list_name in ("lop_list_hier", "lop_list_noise", "lop_list_ltc"):
            for lop in getattr(self, list_name):
                if np.shape(lop) != expected_hilbert_shape:
                    raise ValueError(
                        f"All operators in {list_name} must have shape "
                        f"{expected_hilbert_shape}."
                    )
        for cluster in ("list_interaction_cluster_1", "list_interaction_cluster_2",
                     "list_interaction_cluster_3"):
            cluster_val = getattr(self, cluster)
            if isinstance(cluster_val, str) and cluster_val == "ALL":
                setattr(self, cluster, np.arange(1, self.n_chromophore + 1))

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

        # Noise 2 not currently supported.
        self.noise2_param = None

        # Preparing hierarchy parameter dictionary
        hierarchy_param = {"MAXHIER": self.max_hier}
        if self.static_filter_list:
            hierarchy_param["STATIC_FILTERS"] = self.static_filter_list
        # Preparing storage parameter dictionary
        storage_param = {}

        # Initializing DyadicTrajectory class
        super().__init__(system_param, eom_param, self.noise_param,
                         self.noise2_param, hierarchy_param, storage_param)

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

                # list_permanent_sites preserves the ground state in the adaptive calc
                list_permanent_sites = [0, self.n_state_hilb]

                self.make_adaptive(self.delta_a, self.delta_s, self.set_update_step,
                                   self.set_f_discard, list_permanent_sites)
            # Initializing trajectory
            super().initialize(psi_k, psi_b, timer_checkpoint=timer_checkpoint)

            # Setting the initialized flag to True
            self.__initialized = True

        # Raising a warning if the DyadicTrajectory object has already been initialized
        else:
            print("WARNING: DyadicTrajectory has already been initialized.")

    def _hilb_operator(self, transition_type, field, list_sites):
        """
        Constructs the Hilbert-space transition operator.

        Parameters
        ----------
        1. transition_type: str
                            Type of transition to perform.
                            Options: "g_to_e", "e_to_g", "e_to_ee".

        2. field: np.array(complex)
                  Field vector definition.

        3. list_sites: np.array(int)
                       List of sites acted on by the operator, with the left-most site
                       in the chain indexed by 1.

        Returns
        -------
        1. transition_operator: np.array(complex)
                                Hilbert-space operator implementing the requested
                                transition.

        Notes
        -----
        The Hilbert basis ordering is
        ``[|g>, |e_1>, ..., |e_N>, |e_1 e_2>, ..., |e_{N-1} e_N>]``.
        State index 0 is the ground state, indices 1..N are single-excitation
        states, and remaining indices are doubly-excited states (when present).
        """
        # Calculating μ•E for the given sites
        interactions = np.dot(self.M2_mu_ge[list_sites - 1], field)

        # Constructing sparse raising operator
        if transition_type == "g_to_e":
            return sparse.coo_matrix((interactions,
                                      (list_sites, np.zeros_like(list_sites))),
                                     shape=(self.n_state_hilb,
                                            self.n_state_hilb),
                                     dtype=np.float64)

        # Constructing sparse lowering operator
        elif transition_type == "e_to_g":
            return sparse.coo_matrix((interactions,
                                      (np.zeros_like(list_sites), list_sites)),
                                     shape=(self.n_state_hilb,
                                            self.n_state_hilb),
                                     dtype=np.float64)
        elif transition_type == "e_to_ee":
            interactions = np.dot(self.M2_mu_ge, field)
            dim_hilbert = self.n_state_hilb

            list_row = []
            list_col = []
            list_data = []

            # This operator includes only single -> double transitions (e -> ee).
            # Basis layout note: doubly-excited states are indexed after all
            # single-excitation states in the Hilbert vector.
            # For each pair (e_n, e_m), we add matrix elements:
            #   |e_m> -> |e_n,e_m> with weight (mu_n · E), when site e_n is in list_sites
            #   |e_n> -> |e_n,e_m> with weight (mu_m · E), when site e_m is in list_sites
            # Example: for pair (1, 2), exciting site 2 contributes
            #          <e_1,e_2|Op|e_1> = mu_2 · E.
            for ee_idx, (e_n, e_m) in enumerate(self.list_ee_states):
                ee_state_idx = self.n_chromophore + 1 + ee_idx
                if e_n in list_sites:
                    list_row.append(ee_state_idx)
                    list_col.append(e_m)
                    list_data.append(interactions[e_n - 1])

                if e_m in list_sites:
                    list_row.append(ee_state_idx)
                    list_col.append(e_n)
                    list_data.append(interactions[e_m - 1])
            return sparse.coo_matrix(
                (list_data, (list_row, list_col)),
                shape=(dim_hilbert, dim_hilbert),
            )
             
        # Throwing error for invalid transition types
        else:
            raise ValueError(
                "transition_type must be 'g_to_e', 'e_to_g', or 'e_to_ee', "
                f"got '{transition_type}'."
            )

    def _final_dyad_operator(self):
        """
        Constructs the final dyadic operator for calculating the response function
        and records the time index when the operator begins its action.

        Notes
        -----
        For ESA pathways, this operator maps doubly-excited ket amplitudes to
        singly-excited bra amplitudes.

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
        if self.spectrum_type == "ABSORPTION":
            final_op_index = 0
        else:
            final_op_index = int((self.t_1 + self.t_2) / self.t_step)

        # Constructing sparse final dyadic operator
        if self.spectrum_type in ["ESA-R","ESA-NR"]:

            dim_hilbert = self.n_state_hilb
            dim_dyad = 2 * dim_hilbert

            list_row = []
            list_col = []
            list_data = []

            # The bra block starts at index `dim_hilbert` in dyadic space.
            # For each pair (e_n, e_m), this adds matrix elements:
            #   |e_n,e_m>_ket -> |e_n>_bra with weight (mu_m · E_sig)
            #   |e_n,e_m>_ket -> |e_m>_bra with weight (mu_n · E_sig)
            # Example: for pair (1, 2),
            #          <e_1|_bra F |e_1,e_2>_ket = mu_2 · E_sig
            #          <e_2|_bra F |e_1,e_2>_ket = mu_1 · E_sig.
            for ee_idx, (e_n, e_m) in enumerate(self.list_ee_states):
                ee_state_idx = self.n_chromophore + 1 + ee_idx

                list_row.append(e_n + dim_hilbert)
                list_col.append(ee_state_idx)
                list_data.append(interactions[e_m - 1])

                list_row.append(e_m + dim_hilbert)
                list_col.append(ee_state_idx)
                list_data.append(interactions[e_n - 1])

            F2_final_op = sparse.coo_matrix(
                (list_data, (list_row, list_col)),
                shape=(dim_dyad, dim_dyad),
            )

        else:
            F2_final_op = sparse.coo_matrix((interactions,
                                         ([self.n_state_hilb] * self.n_chromophore,
                                          np.arange(1, self.n_state_hilb))),
                                        shape=(self.n_state_dyad, self.n_state_dyad),
                                        dtype=np.float64)


        return F2_final_op, final_op_index

    def _get_pathway(self):
        """
        Return a pathway dictionary for the selected spectrum type.

        Returns
        -------
        1. pathway: dict
                    Pathway definition with sequential transition types, sequential
                    ket/bra operation sides, sequential interaction clusters, and a
                    pathway scaling factor from degeneracy/conjugate-pathway counting.
        """
        if self.spectrum_type == "ABSORPTION":
            return dict(
                list_transition=["g_to_e"],
                list_sides=["ket"],
                scaling_factor=2,
                list_clusters=[self.list_interaction_cluster_1],
            )
        if self.spectrum_type == "FLUORESCENCE":
            return dict(
                list_transition=["g_to_e", "g_to_e", "e_to_g"],
                list_sides=["bra", "ket", "bra"],
                scaling_factor=4,
                list_clusters=[self.list_interaction_cluster_1,
                               self.list_interaction_cluster_2,
                               self.list_interaction_cluster_3],
            )
        if self.spectrum_type == "GSB-R":
            return dict(
                list_transition=["g_to_e", "e_to_g", "g_to_e"],
                list_sides=["bra", "bra", "ket"],
                scaling_factor=1,
                list_clusters=[self.list_interaction_cluster_1,
                               self.list_interaction_cluster_2,
                               self.list_interaction_cluster_3],
            )
        if self.spectrum_type == "SE-R":
            return dict(
                list_transition=["g_to_e", "g_to_e", "e_to_g"],
                list_sides=["bra", "ket", "bra"],
                scaling_factor=1,
                list_clusters=[self.list_interaction_cluster_1,
                               self.list_interaction_cluster_2,
                               self.list_interaction_cluster_3],
            )
        if self.spectrum_type == "ESA-R":
            return dict(
                list_transition=["g_to_e", "g_to_e", "e_to_ee"],
                list_sides=["bra", "ket", "ket"],
                scaling_factor=-1,
                list_clusters=[self.list_interaction_cluster_1,
                               self.list_interaction_cluster_2,
                               self.list_interaction_cluster_3],
            )
        if self.spectrum_type == "GSB-NR":
            return dict(
                list_transition=["g_to_e", "e_to_g", "g_to_e"],
                list_sides=["ket", "ket", "ket"],
                scaling_factor=1,
                list_clusters=[self.list_interaction_cluster_1,
                               self.list_interaction_cluster_2,
                               self.list_interaction_cluster_3],
            )
        if self.spectrum_type == "SE-NR":
            return dict(
                list_transition=["g_to_e", "g_to_e", "e_to_g"],
                list_sides=["ket", "bra", "bra"],
                scaling_factor=1,
                list_clusters=[self.list_interaction_cluster_1,
                               self.list_interaction_cluster_2,
                               self.list_interaction_cluster_3],
            )
        if self.spectrum_type == "ESA-NR":
            return dict(
                list_transition=["g_to_e", "g_to_e", "e_to_ee"],
                list_sides=["ket", "bra", "ket"],
                scaling_factor=-1,
                list_clusters=[self.list_interaction_cluster_1,
                               self.list_interaction_cluster_2,
                               self.list_interaction_cluster_3],
            )
        raise ValueError(f"Unknown spectrum_type: {self.spectrum_type}")

    def calculate_spectrum(self):
        """
        Construct and propagate one dyadic trajectory for the selected pathway,
        then evaluate the corresponding time-domain response.

        Workflow
        --------
        1. Initialize the dyadic trajectory in the ground-state density matrix.
        2. Apply the pathway interaction operators sequentially (ket/bra side and
           transition type from ``_get_pathway``), propagating through each delay
           interval in ``list_t`` between interactions.
        3. Evaluate the response using the final detection operator returned by
           ``_final_dyad_operator`` and scale by the pathway prefactor.

        Returns
        -------
        1. response_t: np.array(complex)
                       Calculated time-domain response function scaled to account for
                       the degenerate and conjugate pathways.
        """
        # Build the initial dyadic trajectory state once.
        self.initialize()
        # Build the final detection operator and the starting time index for response evaluation.
        final_op, final_op_index = self._final_dyad_operator()
        # Keep only the explicitly defined interaction fields in interaction order.
        list_E_field = [E for E in (self.E_1, self.E_2, self.E_3, self.E_sig) if
                        E is not None]
        # Select transition sequence, ket/bra sides, pathway scaling, and site clusters.
        pathway = self._get_pathway()

        # Apply each interaction operator and propagate through the following delay interval.
        for index_E_field in range(len(list_E_field)-1):
            # Construct the Hilbert-space interaction operator and apply it to ket/bra side.
            self._dyad_operator(
                self._hilb_operator(pathway["list_transition"][index_E_field], list_E_field[index_E_field],
                                    pathway["list_clusters"][index_E_field]), pathway["list_sides"][index_E_field])
            # Propagate for the corresponding delay if nonzero.
            if self.list_t[index_E_field] > 0:
                timer_checkpoint = timer.time()
                self.propagate(self.list_t[index_E_field], self.t_step, timer_checkpoint)


        # Evaluate and scale the response component for this pathway.
        return pathway["scaling_factor"] * self._response_function_comp(final_op, final_op_index)

    @property
    def initialized(self):
        return self.__initialized


def prepare_spectroscopy_input_dict(spectrum_type, propagation_time_dict, field_dict,
                                    cluster_dict):
    """
    Prepares the spectroscopy_dict input dictionary for DyadicSpectra.

    Parameters
    ----------
    1. spectrum_type: str
                      Type of spectrum to be calculated.
                      Options: "ABSORPTION", "FLUORESCENCE", "GSB-R", "SE-R",
                      "ESA-R", "GSB-NR", "SE-NR", "ESA-NR".

    2. propagation_time_dict: dict
                              Dictionary of propagation times between field
                              interactions. (Key Options: "t_1", "t_2", "t_3".)

    3. field_dict: dict
                   Dictionary of field vector definitions. All field vectors must be
                   numpy arrays with exactly 3 entries.
                   (Key Options: "E_1", "E_2", "E_3", "E_sig".)

    4. cluster_dict: dict
                  The set of initially-excited clusters on the ket and bra sides,
                  defined by numpy integer arrays with indexing starting at 1, not 0.
                  (Key Options: "list_interaction_cluster_1",
                  "list_interaction_cluster_2", "list_interaction_cluster_3".)

    Returns
    -------
    1. spectroscopy_input_dict: dict
                                Dictionary of spectroscopy parameters needed for
                                DyadicSpectra class.
    """
    propagation_time_dict = copy.deepcopy(propagation_time_dict)
    field_dict = copy.deepcopy(field_dict)
    cluster_dict = copy.deepcopy(cluster_dict)

    # Defining allowed spectrum types
    list_allowed_spectrum_types = ["ABSORPTION", "FLUORESCENCE","GSB-R","ESA-R","SE-R",
                                   "GSB-NR","ESA-NR","SE-NR"]

    # Checking list_interaction_cluster_1 input structure
    if "list_interaction_cluster_1" not in cluster_dict.keys():
        cluster_dict["list_interaction_cluster_1"] = "ALL"
        warnings.warn(
            "list_interaction_cluster_1 not defined; setting it to ALL.")
    else:
        if not isinstance(cluster_dict["list_interaction_cluster_1"], np.ndarray):
            cluster_dict["list_interaction_cluster_1"] = (
                np.array(cluster_dict["list_interaction_cluster_1"]))

    # Checking cluster indexing structure
    for key, value in cluster_dict.items():
        if isinstance(value, str):
            continue
        if 0 in value:
            raise ValueError("Clusters' indices should not include 0.")

    # Checking field_dict input structure
    for key, value in field_dict.items():
        if not isinstance(value, np.ndarray):
            raise ValueError("All field entries should be numpy arrays.")

        elif value.shape != (3,):
            raise ValueError("All field entries should be numpy arrays with exactly "
                             "3 entries.")
    if "E_1" not in field_dict.keys():
        warnings.warn("E_1 is not defined. Setting E_1 to default, [0, 0, 1].")

    # Removing keys with None values from propagation_time_dict
    for key, value in list(propagation_time_dict.items()):
        if value is None:
            del propagation_time_dict[key]

    # First-order response case.
    if spectrum_type == "ABSORPTION":
        # Checking necessary parameters are defined
        if "t_1" not in propagation_time_dict.keys():
            raise ValueError("Propagation time after first field interaction (t_1) "
                             "must be defined as > 0 for absorption.")

        # Warning user if unused parameters are defined
        if len(propagation_time_dict) > 1:
            warnings.warn("Only t_1 is necessary for absorption. "
                          "Setting all other propagation times to zero.")

        if len(field_dict) > 1:
            warnings.warn("Only E_1 is necessary for absorption. E_sig is set "
                          "to E_1. All other field definitions will be discarded.")

        # Returning dictionary for absorption
        return {"spectrum_type": spectrum_type, "E_1": field_dict["E_1"],
                "E_sig": field_dict["E_1"], "t_1": propagation_time_dict["t_1"],
                "t_2": 0, "t_3": 0, "list_interaction_cluster_1":
                    cluster_dict["list_interaction_cluster_1"]}

    # Third-order response cases.

    elif spectrum_type in ["FLUORESCENCE","GSB-R","ESA-R","SE-R",
                           "GSB-NR","ESA-NR","SE-NR"]:
        # Checking necessary parameters are properly defined
        for i in ("list_interaction_cluster_2", "list_interaction_cluster_3"):
            if i not in cluster_dict.keys():
                cluster_dict[i] = "ALL"
                warnings.warn(f"{i} not defined; setting it to ALL.")
            elif not isinstance(cluster_dict[i], np.ndarray):
                cluster_dict[i] = np.array(cluster_dict[i])
        if "t_2" not in propagation_time_dict.keys():
            raise ValueError("Propagation time after second field "
                             "interactions (t_2) must be defined for "
                             f"{spectrum_type}.")

        if  "t_3" not in propagation_time_dict.keys():
            raise ValueError("Propagation time after third field "
                             "interactions (t_3) must be defined for "
                             f"{spectrum_type}.")

        if spectrum_type == "FLUORESCENCE":

            # Warning user if unused parameters are defined
            if len(propagation_time_dict) > 2:
                warnings.warn(
                    "Only t_2 and t_3 are necessary for fluorescence. Setting "
                    "all other propagation times to zero.")

            if len(field_dict) > 2:
                warnings.warn("Only E_1 and E_sig are necessary for fluorescence. All "
                      "other field definitions will be discarded.")
        else:
            if "t_1" not in propagation_time_dict.keys():
                raise ValueError(
                    "Propagation time after first field "
                             "interactions (t_1) must be defined for "
                             f"{spectrum_type}.")
            for E_field in ["E_2","E_3"]:
                if E_field not in field_dict.keys():
                    warnings.warn(
                        f"{E_field} is not defined. Setting "
                      "them to default, [0, 0, 1].")
        if "E_sig" not in field_dict.keys():
            warnings.warn("E_sig is not defined. Setting E_sig to default, [0, 0, 1].")

        # Build the base dictionary (used directly for fluorescence)
        spec_dict={"spectrum_type": spectrum_type,
                   "E_1": field_dict.get("E_1", np.array([0, 0, 1])),
                   "E_2": field_dict.get("E_1", np.array([0, 0, 1])),
                   "E_3": field_dict.get("E_sig", np.array([0, 0, 1])),
                   "E_sig": field_dict.get("E_sig", np.array([0, 0, 1])),
                   "t_1": 0,"t_2": propagation_time_dict["t_2"],
                   "t_3": propagation_time_dict["t_3"],
                   "list_interaction_cluster_1": cluster_dict["list_interaction_cluster_1"],
                   "list_interaction_cluster_2": cluster_dict["list_interaction_cluster_2"],
                   "list_interaction_cluster_3": cluster_dict["list_interaction_cluster_3"]}

        # Extend the base dictionary to handle generic third-order pathways.
        if spectrum_type in ["GSB-R","SE-R","GSB-NR","SE-NR", "ESA-R", "ESA-NR"]:
            spec_dict["E_2"]= field_dict.get("E_2", np.array([0, 0, 1]))
            spec_dict["E_3"] = field_dict.get("E_3", np.array([0, 0, 1]))
            spec_dict["t_1"]= propagation_time_dict["t_1"]

        return spec_dict

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
                           shape (n_state_hilb, n_state_hilb), where
                           n_state_hilb = n_chromophore + 1 for ground+single manifolds,
                           and includes additional doubly-excited states when present.

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
                                  projection operators in the Hilbert space dimension
                                  set by H2_sys_hamiltonian.

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

                     e. static_filter_list: list(list), optional
                                            List of static filters applied to the
                                            hierarchy. Each filter is defined by a list
                                            of the form [filter_name, filter_params].
                                            This means the full structure of 
                                            static_filter_list is:
                                            [filter1, filter2, ...] where each
                                            filter is a list of the form 
                                            [filter_name, filter_params]
                                            where filter_name is a string and 
                                            filter_params is a list of parameters
                                            defining the filter. The length of the
                                            boolean list in filter_params should match
                                            the number of modes in list_modes or
                                            list_modes_by_bath, depending on which
                                            is defined in bath_dict.
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

                                            The use of any static
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

    # Define number of chromophores and validate M2_mu_ge structure
    n_chromophore = len(M2_mu_ge)
    M2_mu_ge = np.array(M2_mu_ge)
    if M2_mu_ge.shape[1] != 3:
        raise ValueError(
            "M2_mu_ge must be a numpy array with shape (n_chromophore, 3).")
    if (len(np.shape(H2_sys_hamiltonian)) != 2 or
            np.shape(H2_sys_hamiltonian)[0] != np.shape(H2_sys_hamiltonian)[1]):
        raise ValueError("H2_sys_hamiltonian must be a square 2D array.")

    n_state_single = n_chromophore + 1
    n_state_hilb = np.shape(H2_sys_hamiltonian)[0]
    if n_state_hilb < n_state_single:
        raise ValueError(
            "H2_sys_hamiltonian has fewer states than n_chromophore + 1."
        )
    n_ee_states = n_state_hilb - n_state_single

    # Clean up bath_dict: convert arrays to lists and remove None/0 values
    for key, value in list(bath_dict.items()):
        # Convert numpy arrays to lists for consistency
        if type(value) == np.ndarray:
            bath_dict[key] = list(value)
        # Remove keys with None/0 values (except nmodes_LTC)
        elif (key != "nmodes_LTC") and (value is None or value == 0):
            del bath_dict[key]

    # Set default value for nmodes_LTC if not provided or None
    if "nmodes_LTC" not in bath_dict.keys() or bath_dict["nmodes_LTC"] is None:
        bath_dict["nmodes_LTC"] = 0
    # Validate nmodes_LTC if provided
    elif not isinstance(bath_dict["nmodes_LTC"], int):
        raise ValueError("nmodes_LTC must be an integer or None.")
    elif bath_dict["nmodes_LTC"] < 0:
        raise ValueError("nmodes_LTC must be >= 0 or set to None.")

    # Check that users don't try to use LTC with static filters
    if bath_dict["nmodes_LTC"] > 0 and "static_filter_list" in bath_dict:
        raise ValueError("The use of static hierarchy filters with low-temperature "
                         "correction is not currently supported.")

    # Check that list_modes and list_modes_by_bath are not both defined
    if "list_modes_by_bath" in bath_dict.keys() and "list_modes" in bath_dict.keys():
        raise ValueError(
            "list_modes_by_bath and list_modes should not both be defined.")

    # Set default list_lop if not defined (site-occupation operators)
    if "list_lop" not in bath_dict.keys():
        expected_n_ee_states = n_chromophore * (n_chromophore - 1) // 2
        if n_ee_states > 0 and n_ee_states != expected_n_ee_states:
            raise ValueError(
                f"Cannot infer default L-operators: H2_sys_hamiltonian implies "
                f"n_ee_states={n_ee_states}, but for n_chromophore={n_chromophore} "
                f"the canonical doubles count is {expected_n_ee_states}. "
                f"Either fix the Hamiltonian shape or provide list_lop explicitly."
            )

        site_to_double_state_indices = [[] for _ in range(n_chromophore)]
        if n_ee_states > 0:
            # Build doubly-excited-state pairs in canonical lexicographic order:
            # (0,1), (0,2), ..., (N-2,N-1). This matches the basis ordering used
            # in the Hilbert-space construction for default operators.
            i_idx, j_idx = np.triu_indices(n_chromophore, k=1)

            # Precompute a mapping from each site to the canonical-pair positions
            # (0..N(N-1)/2 - 1) of doubly-excited states containing that site.
            # The full Hilbert basis index of each double is n_state_single + position.
            # This avoids repeated O(N^2) scans for each site.
            for idx, (i_site, j_site) in enumerate(zip(i_idx, j_idx)):
                site_to_double_state_indices[i_site].append(idx)
                site_to_double_state_indices[j_site].append(idx)

        list_lop_default = []
        for site in range(n_chromophore):
            double_state_indices_for_site = site_to_double_state_indices[site]

            # Single-excitation state at Hilbert basis index (1 + site), followed by
            # every doubly-excited state containing this site.
            diag_indices = np.asarray(
                [1 + site] + [n_state_single + idx for idx in double_state_indices_for_site],
                dtype=np.int32,
            )

            vals = np.ones(diag_indices.shape[0], dtype=np.float64)
            list_lop_default.append(
                sparse.coo_matrix(
                    (vals, (diag_indices, diag_indices)),
                    shape=(n_state_hilb, n_state_hilb),
                )
            )
        bath_dict["list_lop"] = list_lop_default
    else:
        for lop in bath_dict["list_lop"]:
            if np.shape(lop) != (n_state_hilb, n_state_hilb):
                raise ValueError(
                    "Each list_lop operator must have shape "
                    f"({n_state_hilb}, {n_state_hilb}) to match H2_sys_hamiltonian."
                )

    # Process list_modes if provided
    if "list_modes" in bath_dict:
        # Validate list_modes structure (must have paired Gs and Ws)
        if len(bath_dict["list_modes"]) % 2 != 0:
            raise ValueError("list_modes should contain paired Gs and Ws, which "
                             "guarantees an even number of elements.")

        # Create list_modes_by_bath by repeating list_modes for each bath
        bath_dict["list_modes_by_bath"] = [bath_dict["list_modes"] for _ in
                                           range(len(bath_dict["list_lop"]))]

    # Process list_modes_by_bath if provided
    elif "list_modes_by_bath" in bath_dict:
        # Validate compatibility with list_lop
        if len(bath_dict["list_modes_by_bath"]) != len(bath_dict["list_lop"]):
            raise ValueError(
                "list_modes_by_bath and list_lop must have the same length.")

        # Validate structure of each sublist (must have paired Gs and Ws)
        for sublist in bath_dict["list_modes_by_bath"]:
            if not isinstance(sublist, list):
                raise ValueError("list_modes_by_bath must be a list of lists.")
            elif len(sublist) % 2 != 0:
                raise ValueError("sublists within list_modes_by_bath should contain "
                                 "paired Gs and Ws, which guarantees an even number of "
                                 "elements in each sublist.")

    # Require either list_modes or list_modes_by_bath is defined
    else:
        raise ValueError("Either list_modes_by_bath or list_modes must be defined.")

    # Process static_filter_list if provided
    if "static_filter_list" in bath_dict.keys():
        # Validate that static_filter_list is a list
        if not isinstance(bath_dict["static_filter_list"], list):
            raise ValueError("static_filter_list must be a list.")

        # Validate each filter in the list
        for filter_idx, filter_item in enumerate(bath_dict["static_filter_list"]):
            # Check each filter is a list of the form [filter_name, filter_params]
            if not isinstance(filter_item, list):
                raise ValueError(
                    f"Error in filter {filter_idx}: static_filter_list must be a list.")
            elif len(filter_item) != 2:
                raise ValueError(
                    f"Error in filter {filter_idx}: each filter in static_filter_list "
                    f"must be a 2-element list of the form: "
                    f"[filter_name, filter_params].")
            # Check filter_name is a string and is one of the allowed types
            elif filter_item[0] not in ["Markovian", "Triangular", "LongEdge"]:
                raise ValueError(
                    f"Error in filter {filter_idx}: Filter names must be 'Markovian', "
                    f"'Triangular', or 'LongEdge'.")

            # Validate filter parameters for Markovian filters
            if filter_item[0] == "Markovian":
                # Markovian filter_params should be a list of booleans
                if not all(isinstance(boolean, bool) for boolean in filter_item[1]):
                    raise ValueError(
                        f"Error in filter {filter_idx}: filter_params for Markovian "
                        f"filters must be a list of booleans.")

                # Check that the length of filter_item[1] matches the number of modes
                # in each bath if given list_modes input
                if "list_modes" in bath_dict.keys():
                    if len(filter_item[1]) != len(bath_dict["list_modes"]) // 2:
                        raise ValueError(f"Error in filter {filter_idx}: The list of "
                                         f"booleans in filter_params must have the "
                                         f"same length as the number of modes in "
                                         f"each bath.")
                    else:
                        # Expand filter to cover all baths
                        filter_item[1] = filter_item[1] * len(bath_dict["list_lop"])

                # If list_modes_by_bath is given, check against total number of modes
                elif len(filter_item[1]) != np.sum([len(mode_list) for mode_list in
                                                    bath_dict["list_modes_by_bath"]])//2:
                    raise ValueError(f"Error in filter {filter_idx}: The list of "
                                     f"booleans in filter_params must have "
                                     f"the same length as the number of modes in all "
                                     f"baths combined.")

            # Validate filter parameters for Triangular and LongEdge filters
            elif filter_item[0] in ["Triangular", "LongEdge"]:
                # Triangular and LongEdge filter_params should be a list containing a
                # list of booleans and an integer
                if len(filter_item[1]) != 2:
                    raise ValueError(
                        f"Error in filter {filter_idx}: Triangular/LongEdge "
                        f"filter_params must be a list containing a list of booleans "
                        f"and an integer.")
                elif not isinstance(filter_item[1][1], int):
                    raise ValueError(
                        f"Error in filter {filter_idx}: The second entry in "
                        f"filter_params for Triangular/LongEdge filters must be an "
                        f"integer.")
                elif not all(
                        isinstance(boolean, bool) for boolean in filter_item[1][0]):
                    raise ValueError(f"Error in filter {filter_idx}: The first entry in"
                                     f" filter_params for Triangular/LongEdge filters "
                                     f"must be a list of booleans.")
                elif filter_item[1][1] < 0:
                    raise ValueError(f"Error in filter {filter_idx}: "
                                     f"Triangular/LongEdge filter_params must have a "
                                     f"positive integer as the second element.")

                # Check that the length of filter_item[1][0] matches the number of modes
                # in each bath if given list_modes input
                if "list_modes" in bath_dict.keys():
                    if len(filter_item[1][0]) != len(bath_dict["list_modes"]) // 2:
                        raise ValueError(f"Error in filter {filter_idx}: The list of "
                                         f"booleans in filter_params must have the same"
                                         f" length as the number of modes in each bath.")
                    else:
                        # Expand filter to cover all baths
                        filter_item[1][0] = (filter_item[1][0] *
                                             len(bath_dict["list_lop"]))

                # If list_modes_by_bath is given, check against total number of modes
                elif len(filter_item[1][0]) != np.sum([len(mode_list) for mode_list in
                                                       bath_dict["list_modes_by_bath"]])//2:
                    raise ValueError(f"Error in filter {filter_idx}: The list of "
                                     f"booleans in filter_params must have the same "
                                     f"length as the number of modes in all baths "
                                     f"combined.")

    # Initialize empty lists for chromophore dictionary components
    gw_sysbath = []  # G-W tuples for hierarchy
    list_lop_sysbath_by_mode = []  # L-operators for hierarchy
    gw_noise = []  # G-W tuples for noise
    list_lop_noise_by_mode = []  # L-operators for noise
    list_lop_ltc = []  # L-operators for low-temperature correction
    list_ltc_param = []  # Low-temperature correction parameters

    # Process each bath
    for bath in range(len(bath_dict["list_lop"])):
        # Convert list of Gs and Ws to list of coupled G-W tuples
        list_modes_as_tuples = [(bath_dict["list_modes_by_bath"][bath][i],
                                 bath_dict["list_modes_by_bath"][bath][i + 1]) for i in
                                range(0, len(bath_dict["list_modes_by_bath"][bath]), 2)]

        # Validate nmodes_LTC compatibility with number of modes
        if bath_dict["nmodes_LTC"] >= len(list_modes_as_tuples):
            raise ValueError("nmodes_LTC must be less than the number of modes in each "
                             "bath.")

        # Initialize LTC parameter for this bath
        ltc_param = 0
        list_lop_ltc.append(bath_dict["list_lop"][bath])

        # Process each mode in the bath
        for nmode, mode in enumerate(list_modes_as_tuples):
            # Append G-W tuples and L-operators to the appropriate lists
            gw_noise.append(mode)
            list_lop_noise_by_mode.append(bath_dict["list_lop"][bath])

            # Determine if mode is treated in hierarchy or with LTC
            if len(list_modes_as_tuples) - nmode > bath_dict["nmodes_LTC"]:
                # Mode is treated in hierarchy
                gw_sysbath.append(mode)
                list_lop_sysbath_by_mode.append(bath_dict["list_lop"][bath])
            else:
                # Mode is treated with low-temperature correction
                ltc_param += mode[0] / mode[1]

        # Add LTC parameter for this bath
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
