import pytest
import numpy as np
from mesohops.dynamics.bath_corr_functions import bcf_exp
from mesohops.dynamics.hops_noise import HopsNoise
from mesohops.util.exceptions import UnsupportedRequest, LockedException
from mesohops.util.physical_constants import hbar


__title__ = "Test of ZERO noise model"
__author__ = "J. K. Lynd"
__version__ = "1.2"
__date__ = "Oct 19 2022"




# Test ZERO Noise Model of the HopsNoise object.
# ----------------------------------------------------
noise_param = {
    "SEED": 0,
    "MODEL": "ZERO",
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
    Tests that HopsNoise with the ZERO model produces reproducible noise
    trajectories when seeded, that the correlated and uncorrelated noise are both
    simply 0, and that the noise trajectory is the correct size and shape.
    """
    noise_param = {
        "SEED": 0,
        "MODEL": "ZERO",
        "TLEN": 10.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_param_interpolated = {
        "SEED": 0,
        "MODEL": "ZERO",
        "TLEN": 10.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "INTERPOLATE" : True,
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    noiseModel = HopsNoise(noise_param, noise_corr)
    noiseModel.prepare_noise()
    assert noiseModel.param["T_AXIS"][-1] == 10.0
    testnoise1 = noiseModel.get_noise(np.arange(10.0))[0, :]
    testnoise2 = HopsNoise(noise_param, noise_corr).get_noise(np.arange(10.0))[
                 0, :]
    noise_param["SEED"] = 1
    testnoise3 = HopsNoise(noise_param, noise_corr).get_noise(np.arange(10.0))[
                 0, :]
    testnoise_interp = HopsNoise(noise_param_interpolated, noise_corr).get_noise(
        np.arange(10.0))[0, :]
    testnoise_interp_step = HopsNoise(noise_param_interpolated,
                                      noise_corr).get_noise([0.1, 0.3, 1.2, 2.4, 3.1,
                                                             4.5, .6, 6.3, 7.9,
                                                             8.7])[0, :]

    assert np.allclose(testnoise1, testnoise2)
    assert np.allclose(testnoise1, testnoise3)
    assert np.allclose(testnoise1, testnoise_interp)
    assert np.allclose(testnoise1, testnoise_interp_step)
    assert np.allclose(testnoise1, np.zeros_like(testnoise1))
    assert np.size(noiseModel.get_noise(noiseModel.param["T_AXIS"])) == np.size(
        noiseModel.param["T_AXIS"])*noiseModel.param["N_L2"]

