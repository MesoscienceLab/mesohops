import os
import glob
import numpy as np


def load_and_average_timing_data(list_nsites, timing_model):
    base_dir = os.getcwd()

    init_results = []
    prop_results = []

    for nsite in list_nsites:
        data_dir = os.path.join(base_dir, f'{timing_model}_timing/nsite-{nsite}')
        if not os.path.isdir(data_dir):
            print(f"Directory {data_dir} does not exist. Skipping.")
            continue

        # Load initialization times
        init_files = glob.glob(os.path.join(data_dir, 'initialization_time_seed*.npy'))
        init_times = [np.load(file) for file in init_files]

        # Load propagation times
        prop_files = glob.glob(os.path.join(data_dir, 'propagation_time_seed*.npy'))
        prop_times = [np.load(file) for file in prop_files]

        # Compute averages
        avg_init_time = np.mean(init_times) if init_times else 'Value not found.'
        avg_prop_time = np.mean(prop_times) if prop_times else 'Value not found.'

        # Append results
        init_results.append(avg_init_time)
        prop_results.append(avg_prop_time)

    # Save the results to two separate files
    np.save(os.path.join(base_dir, f'{timing_model}_avg_initialization_times.npy'),
            np.array(init_results))
    np.save(os.path.join(base_dir, f'{timing_model}_avg_propagation_times.npy'),
            np.array(prop_results))

    return init_results, prop_results


list_timing_models = ["markovian_filter", "longedge_filter", "triangular_filter",
                     "absorption", "fluorescence", "holstein_1_particle", "peierls",
                     "holstein_2_particle"]

# Define nsites for all tests except 2 particle
list_nsites = [3, 6, 10, 55, 91, 120, 465, 990, 1128, 5050, 9870, 11325, 51360, 92665,
          113050, 500500, 994755]

# 2 Particle nsites
list_nsites_2_particle = [2, 3, 4, 10, 13, 15, 30, 44, 47, 100, 140, 150, 320, 430, 475,
                     1000, 1410]

for model in list_timing_models:
    # Determine the list of nsites based on the model
    if model == "holstein_2_particle":
        list_nsites = list_nsites_2_particle
    else:
        list_nsites = list_nsites

    # Load, average, and save timing data for each model
    load_and_average_timing_data(list_nsites, model)

