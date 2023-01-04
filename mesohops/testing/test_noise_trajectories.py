import numpy as np
from mesohops.dynamics.bath_corr_functions import bcf_exp
from mesohops.dynamics.noise_fft import FFTFilterNoise
from mesohops.dynamics.noise_trajectories import NumericNoiseTrajectory

__title__ = "Test of noise_trajectories"
__author__ = "J. K. Lynd"
__version__ = "1.2"
__date__ = "July 7 2021"

# Test Noise Model
# ----------------
noise_param = {
    "SEED": 0,
    "MODEL": "FFT_FILTER",
    "TLEN": 10.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
}

loperator = np.zeros([2, 2, 2], dtype=np.float64)
loperator[0, 0, 0] = 1.0
loperator[1, 1, 1] = 1.0
sys_param = {
    "HAMILTONIAN": np.array([[0, 10.0], [10.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[10.0, 10.0], [5.0, 5.0]],
    "L_HIER": loperator,
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[10.0, 10.0], [5.0, 5.0]],
    "L_NOISE1": loperator,
}


sys_param["NSITE"] = len(sys_param["HAMILTONIAN"][0])
sys_param["NMODES"] = len(sys_param["GW_SYSBATH"][0])
sys_param["N_L2"] = 2
sys_param["L_IND_BY_NMODE1"] = [0, 1]
sys_param["LIND_DICT"] = {0: loperator[0, :, :], 1: loperator[1, :, :]}

noise_param = {
    "SEED": None,
    "MODEL": "FFT_FILTER",
    "TLEN": 1000.0,  # Units: fs
    "TAU": 1.0,  # Units: fs,
    "INTERPOLATE": False
}

noise_corr = {
    "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
    "N_L2": sys_param["N_L2"],
    "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
    "CORR_PARAM": sys_param["PARAM_NOISE1"],
}

t_axis = np.arange(0, 1001.0, 1.0)

noise_obj = FFTFilterNoise(noise_param, noise_corr)
noise_obj.prepare_noise()
test_noise = noise_obj.get_noise(t_axis)

# PLEASE NOTE: THIS MAY NOT SEEM VERY UNIT-LIKE FOR A UNIT TEST, BUT THE SOURCE OF
# THE TIME AXIS AND THE NOISE ITSELF ARE IRRELEVANT. THERE IS A HIDDEN PURPOSE TO
# USING THE NOISE OBJECT TO GENERATE THE NOISE: CHECKING THAT THE PROPER DATA TYPES
# ARE COMPATIBLE WITH THE CODE.

def test_noise_traj():
    """
    Tests that a noise trajectory object initializes properly and returns the same
    noise and t_axis it took in with the requisite helper functions.
    """
    noise_traj = NumericNoiseTrajectory(test_noise, t_axis)
    assert np.allclose(noise_traj.get_taxis(), t_axis)
    assert np.allclose(noise_traj.get_noise(t_axis), test_noise)



