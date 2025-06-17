# Imports
import os
import sys
import numpy as np
from mesohops.trajectory.dyadic_spectra import DyadicSpectra as DHOPS
from mesohops.trajectory.dyadic_spectra import (prepare_spectroscopy_input_dict,
                                            prepare_chromophore_input_dict,
                                            prepare_convergence_parameter_dict)
from mesohops.util.bath_corr_functions import bcf_convert_sdl_to_exp
from mesohops.timing.helper_functions.hamiltonian_generation import (
    generate_spectroscopy_hamiltonian)
from mesohops.timing.helper_functions.loperator_generation import (
     generate_spectroscopy_loperators)

# Gets the seed and number of sites from command line arguments
seed = int(sys.argv[1])
nsite = int(sys.argv[2])

# Parameter control center
kmax = 15                       # Hierarchy truncation depth
dt = 4                          # Simulation time step [fs]
t_1 = 200                       # t1 length of simulation [fs]
delta_a = 0.0005                # Auxiliary basis derivative error bound
delta_s = 0.001                 # State basis derivative error bound

# Hamiltonian parameters
gamma = 50                      # Reorganization timescale [cm^-1]
e_lambda = 50                   # Reorganization energy [cm^-1]
temp = 300                      # Temperature [K]
V = 50                          # Inter-pigment coupling [cm^-1]

# Prepare Spectroscopy Input Dictionary
spectrum_type = "ABSORPTION"
time_dict = {"t_1": t_1}
field_dict = {"E_1": np.array([0, 0, 1])}
site_dict = {"list_ket_sites": [1],}
spec_input = prepare_spectroscopy_input_dict(spectrum_type, time_dict,
                                             field_dict, site_dict)

# Prepare Chromophore Input Dictionary
M2_mu_ge = np.tile(np.array([0, 0, 1]), (nsite, 1))
H2_sys_hamiltonian = generate_spectroscopy_hamiltonian(nsite, V)
list_loperators = generate_spectroscopy_loperators(nsite)
list_modes = bcf_convert_sdl_to_exp(e_lambda, gamma, 0, temp)
chromophore_input = prepare_chromophore_input_dict(M2_mu_ge, H2_sys_hamiltonian,
                                                   bath_dict={"list_lop":
                                                                  list_loperators,
                                                              "list_modes":
                                                                  list_modes})

# Prepare Convergence Parameter Dictionary
convergence_dict = prepare_convergence_parameter_dict(dt, kmax,
                                                      delta_a=delta_a, delta_s=delta_s)

# Initialize Dyadic Spectra Object
dyadic_spectra = DHOPS(spec_input, chromophore_input, convergence_dict, seed)
dyadic_spectra.calculate_spectrum()

# Save Timing Data
save_dir = os.getcwd()
data_dir = f'/absorption_timing/nsite-{nsite}/'
if seed == 1010101010101010:
    pass
else:
    os.makedirs(save_dir + data_dir, exist_ok=True)
    np.save(save_dir + data_dir + f'/initialization_time_seed{seed}.npy',
            dyadic_spectra.storage.metadata["INITIALIZATION_TIME"])
    np.save(save_dir + data_dir + f'/propagation_time_seed{seed}.npy',
            dyadic_spectra.storage.metadata["LIST_PROPAGATION_TIME"][0])

