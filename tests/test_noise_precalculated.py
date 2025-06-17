import os
import numpy as np
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.noise.hops_noise import HopsNoise



__title__ = "Test of FFT_FILTER noise model"
__author__ = "J. K. Lynd"
__version__ = "1.3"
__date__ = "Mar 15 2023"

path_data = os.path.realpath(__file__)[: -len("test_noise_precalculated.py")]
path_seed = path_data + "/pre_calculated_correlated_noise_test.npy"

# Test FFT_FILTER Noise Model of the HopsNoise object.
# ----------------------------------------------------
noise_param_string_seed = {
    "SEED": path_seed,
    "MODEL": "PRE_CALCULATED",
    "TLEN": 10.0,  # Units: fs
    "TAU": 1.0,  # Units: fs
}

noise_param_array_seed = {
    "SEED": np.load(path_seed),
    "MODEL": "PRE_CALCULATED",
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


def test_noiseModel():
    """
    Tests that HopsNoise with the pre_calculated model properly fetches a saved noise.
    """
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    noiseModel_string_seed = HopsNoise(noise_param_string_seed, noise_corr)
    noiseModel_string_seed._prepare_noise(list(np.arange(sys_param["N_L2"])))
    noiseModel_array_seed = HopsNoise(noise_param_array_seed, noise_corr)
    noiseModel_array_seed._prepare_noise(list(np.arange(sys_param["N_L2"])))
    testnoise1 = noiseModel_string_seed.get_noise(np.arange(10.0))[0, :]
    testnoise2 = HopsNoise(noise_param_string_seed, noise_corr).get_noise(np.arange(10.0))[
                 0, :]
    testnoise3 = noiseModel_array_seed.get_noise(np.arange(10.0))[0, :]
    testnoise4 = HopsNoise(noise_param_array_seed, noise_corr).get_noise(
        np.arange(10.0))[
                 0, :]

    assert noiseModel_string_seed.param["T_AXIS"][-1] == 10.0
    assert np.allclose(testnoise1, np.load(path_seed)[0, :10])
    assert np.allclose(testnoise1, testnoise2)
    assert np.allclose(testnoise1, testnoise3)
    assert np.allclose(testnoise1, testnoise4)
    assert np.size(noiseModel_string_seed.get_noise(noiseModel_string_seed.param["T_AXIS"])) == \
           np.size(noiseModel_string_seed.param["T_AXIS"])*noiseModel_string_seed.param["N_L2"]
