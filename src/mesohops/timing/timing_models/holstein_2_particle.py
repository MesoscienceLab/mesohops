# Imports
import os
import sys
import numpy as np
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.bath_corr_functions import bcf_convert_dl_to_exp
from mesohops.timing.helper_functions.hamiltonian_generation import (
    generate_2_particle_hamiltonian)
from mesohops.timing.helper_functions.loperator_generation import (
     generate_holstein_2_particle_loperators)

# Gets the seed and number of sites from command line arguments
seed = int(sys.argv[1])
nstate = int(sys.argv[2])

# Define the number of sites based on the number of states
nsite = round((np.sqrt(8 * nstate + 1) - 1) / 2)

# Parameter control center
kmax = 15                       # Hierarchy truncation depth
dt = 4                          # Simulation time step [fs]
tmax = 2000.0                   # Time length of simulation [fs]
tau = 0.5                       # Noise time step [fs]
tlen = 3000.0                   # Time length of noise [fs]
delta_a = 0.0005                # Auxiliary basis derivative error bound
delta_s = 0.001                 # State basis derivative error bound

# Hamiltonian parameters
gamma = 50                      # Reorganization timescale [cm^-1]
e_lambda = 50                   # Reorganization energy [cm^-1]
temp = 300                      # Temperature [K]
V = 50                          # Inter-pigment coupling [cm^-1]

# Hamiltonian Generation
H2_sys_hamiltonian = generate_2_particle_hamiltonian(nsite, V)

# L-Operator Generation
list_loperators = generate_holstein_2_particle_loperators(nsite)

# Gets the list of bath correlation function modes for each independent environment
list_dl_modes = bcf_convert_dl_to_exp(e_lambda, gamma, temp)

def prepare_gw_sysbath(list_lop, list_modes):
    """
    A helper function that builds the lists taken in by the HopsTrajectory object as
    the parameters of the correlation functions.

    Parameters
    ----------
    1. list_lop : list(sparse matrix)
                  A list of site-projection operators for the system-bath interaction.
    2. list_modes : list(complex)
                    A list of complex exponential modes in (g, w) form (see below).
                    Assumed here to be the same for all baths.

    Returns
    -------
    1. list_gw_noise : list(tuple)
                        A list of (g, w) modes that make up the bath correlation
                        function for the noise in the form g*np.exp(-w*t/hbar) [
                        units: cm^-2, cm^-1].
    2. list_l_noise: list(sparse matrix)
                     A list of site-projection operators for the noise, matched to
                     the modes in list_gw_noise.

    """
    # Initialize lists for noise parameters
    list_gw_noise = []
    list_l_noise = []

    # Get the first mode
    (g_0, w_0) = list_modes

    # Append the first mode to the noise lists
    for L2_ind_bath in list_lop:
        list_gw_noise.append([g_0, w_0])
        list_l_noise.append(L2_ind_bath)

    return list_gw_noise, list_l_noise


def build_HOPS_dictionaries(list_lop, list_modes, seed=None):
    """
    Constructs the dictionaries that will define the HOPS trajectory object.

    Parameters
    ----------
    1. list_lop : list(sparse matrix)
                  A list of site-projection operators for the system-bath interaction.
    2. list_modes : list(complex)
                    A list of complex exponential modes in (g, w) form. Assumed here to
                    be the same for all baths.
    3. seed : int
              The integer seed that makes the calculation reproducible.

    Returns
    -------
    1. sys_param : dictionary
                   The parameters of the system, bath, and overall simulation
    2. noise_param : dictionary
                     The parameters that define how noise is handled
    3. hierarchy_param : dictionary
                         The parameters that define how the hierarchy is handled
    4. eom_param : dictionary
                   The parameters that define the equation-of-motion
    """
    # Get the description of the bath and system-bath interaction terms
    list_gw_noise, list_l_noise = prepare_gw_sysbath(list_lop, list_modes)

    # Define sys_param
    sys_param = {'HAMILTONIAN': H2_sys_hamiltonian,
                 'GW_SYSBATH': list_gw_noise,
                 'L_HIER': list_l_noise,
                 'L_NOISE1': list_l_noise,
                 'ALPHA_NOISE1': bcf_exp,
                 'PARAM_NOISE1': list_gw_noise,
                 }

    # Define noise parameters.
    noise_param = {'SEED': seed,
                   'MODEL': 'FFT_FILTER',
                   'TLEN': tlen,  # Units: fs
                   'TAU': tau,  # Units: fs
                   }

    # Define hierarchy parameters
    hierarchy_param = {'MAXHIER': kmax}

    # Define equation-of-motion parameters.
    eom_param = {'EQUATION_OF_MOTION': "NORMALIZED NONLINEAR"}

    # Return all dictionaries in the proper order.
    return sys_param, noise_param, hierarchy_param, eom_param


# Gets the parameters of a HOPS object.
sys_param, noise_param, hierarchy_param, eom_param = (
    build_HOPS_dictionaries(list_loperators, list_dl_modes, seed))

# Initialize the HOPS trajectory object
trajectory = HOPS(sys_param, noise_param=noise_param, hierarchy_param=
        hierarchy_param, eom_param=eom_param)

# Make the trajectory adaptive if necessary
if delta_a > 0 or delta_s > 0:
    trajectory.make_adaptive(delta_a, delta_s)

# Initial state consists of an excitation localized on the edge of the linear chain
psi_0 = np.zeros([nstate])
psi_0[nstate-nsite] = 1

# Initialize the trajectory with the initial state
trajectory.initialize(psi_0)

# Propagate the trajectory
trajectory.propagate(tmax, dt)

# Save Timing Data
save_dir = os.getcwd()
data_dir = f'/holstein_2_particle_timing/nsite-{nsite}/'
if seed == 1010101010101010:
    pass
else:
    os.makedirs(save_dir + data_dir, exist_ok=True)
    np.save(save_dir + data_dir + f'/initialization_time_seed{seed}.npy',
            trajectory.storage.metadata["INITIALIZATION_TIME"])
    np.save(save_dir + data_dir + f'/propagation_time_seed{seed}.npy',
            trajectory.storage.metadata["LIST_PROPAGATION_TIME"][0])

