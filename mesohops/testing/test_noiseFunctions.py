import numpy as np
from mesohops.dynamics.bath_corr_functions import bcf_exp
from mesohops.dynamics.noise_fft import FFTFilterNoise
from mesohops.dynamics.noise_trajectories import NumericNoiseTrajectory

__title__ = "Test of noiseFunctions"
__author__ = "D. I. G. Bennett"
__version__ = "0.1"
__date__ = "Feb. 10, 2019"

# Test Noise Classes
# ------------------
t_axis = np.array([0, 1, 2, 3, 4], dtype=np.float64)
noise = np.array([[1, 2, 2, 1, 3], [2, 3, 3, 2, 4]], dtype=np.float64)


def test_NumericNoise():
    """
    this test checks numeric noise trajectories
    """
    numNoise = NumericNoiseTrajectory(noise, t_axis)
    assert (
        numNoise.get_noise(np.array([0.0, 1.0]))[0, 1]
        == np.array([[1, 2], [2, 3]], dtype=np.float64)[0, 1]
    )
    assert list(numNoise.get_taxis()) == list(t_axis)


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


def test_noiseModel():
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    noiseModel = FFTFilterNoise(noise_param, noise_corr)
    noiseModel.prepare_noise()
    assert noiseModel.param["T_AXIS"][-1] == 10.0
    testnoise = noiseModel.get_noise(np.arange(10.0))[0, :]
    testnoise_answer = np.array(
        [
            3.82810911 + 0.9397952j,
            3.75969812 + 0.91927177j,
            3.73492567 + 1.04445045j,
            3.62975737 + 0.96089954j,
            3.45986897 + 0.81667851j,
            3.57567558 + 0.75886821j,
            3.91953564 + 0.6094234j,
            3.84677822 + 0.45471175j,
            3.92130167 + 0.45716379j,
            3.94064793 + 0.52815227j,
        ]
    )
    assert np.allclose(testnoise, testnoise_answer, rtol=1e-7)


def test_fft_filter_noise_diagonal():
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }

    noiseModel = FFTFilterNoise(noise_param, noise_corr)
    t_axis = np.arange(0, 5000.0, 1.0)
    alpha = noiseModel._corr_func_by_lop_taxis(t_axis)
    alpha_test = np.array([10.0 + 0.0j, 9.98118165 + 0.0j, 9.96239871 + 0.0j])
    assert np.allclose(alpha[0][0:3], alpha_test)
