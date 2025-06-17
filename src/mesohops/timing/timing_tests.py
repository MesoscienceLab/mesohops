import os
import multiprocessing.pool as pool

list_script_names = ["markovian_filter.py", "longedge_filter.py",
                     "triangular_filter.py", "absorption.py", "fluorescence.py",
                     "holstein_1_particle.py", "peierls.py", "holstein_2_particle.py"]

path_scripts = os.path.realpath(__file__).replace("timing_tests.py","timing_models/")

run_string = "python3 {}{} {} {}"

# Define the number of CPU cores available for parallel processing
n_cores = os.cpu_count()

# Set the number of trajectories to run in parallel
n_traj = 10


def task(list_id):
    traj = list_id[0]
    n_states = list_id[1]
    os.environ['OMP_NUM_THREADS'] = '1'
    os.system(run_string.format(path_scripts, list_id[2], traj, n_states))


def run_timing_tests():
    print("Running timing scripts!")
    for script_name in list_script_names:
        for n_states in [1, 3, 6, 10, 55, 91, 120, 465, 990, 1128, 5050, 9870, 11325,
                         51360, 92665, 113050, 500500, 994755]:
            with pool.Pool(min(n_traj, n_cores)) as mypool:
                for _ in mypool.imap(task, [(i, n_states, script_name) for i in
                                                 range(n_traj)]):
                    pass

