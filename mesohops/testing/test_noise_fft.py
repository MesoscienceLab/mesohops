import pytest
import numpy as np
from pyhops.dynamics.bath_corr_functions import bcf_exp
from pyhops.dynamics.noise_fft import FFTFilterNoise
from pyhops.util.exceptions import UnsupportedRequest, LockedException
from pyhops.util.physical_constants import hbar


__title__ = "Test of noise_fft"
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


def test_noiseModel():
    """
    Tests that FFTFilterNoise produces reproducible noise trajectories when seeded,
    each seed produces a unique noise trajectories, and that the noise trajectory is
    the correct size and shape
    """
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    noiseModel = FFTFilterNoise(noise_param, noise_corr)
    noiseModel.prepare_noise()
    assert noiseModel.param["T_AXIS"][-1] == 10.0
    testnoise1 = noiseModel.get_noise(np.arange(10.0))[0, :]
    testnoise2 = FFTFilterNoise(noise_param, noise_corr).get_noise(np.arange(10.0))[
                 0, :]
    noise_param["SEED"] = 1
    testnoise3 = FFTFilterNoise(noise_param, noise_corr).get_noise(np.arange(10.0))[
                 0, :]

    assert np.allclose(testnoise1, testnoise2)
    assert not np.allclose(testnoise1, testnoise3)
    assert np.size(noiseModel.get_noise(noiseModel.param["T_AXIS"])) == np.size(
        noiseModel.param["T_AXIS"])*noiseModel.param["N_L2"]


def test_construct_indexing():
    """
    Tests that the _construct_indexing function constructs the correct indexing.
    """
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    test_noise = FFTFilterNoise(noise_param, noise_corr)
    biglength = 1000
    for smalllength in range(0, biglength):
        g_indexbig, phi_indexbig = test_noise._construct_indexing(biglength)
        g_indexsmall, phi_indexsmall = test_noise._construct_indexing(smalllength)
        assert g_indexbig[0:smalllength] == g_indexsmall[0:smalllength]
        assert phi_indexbig[0:smalllength] == phi_indexsmall[0:smalllength]

def test_correlated_noise_consistency():
    """
    Tests that the correlated noise trajectory is the same regardless of noise
    trajectory length. The noise trajectories produced should be extremely close.
    """
    noise_param_short = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 100000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_param_long = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 200000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    test_noise_short = FFTFilterNoise(noise_param_short, noise_corr)
    test_noise_long = FFTFilterNoise(noise_param_long, noise_corr)
    np.testing.assert_allclose(test_noise_short.get_noise(np.arange(10)),
                       test_noise_long.get_noise(np.arange(10)), rtol=1E-4)

def test_uncorrelated_noise_consistency():
    """
    Tests that the uncorrelated noise trajectory is the same regardless of noise
    trajectory length. The noise trajectories produced should be exactly identical.
    """
    noise_param_short = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_param_long = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 20000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    test_noise_short = FFTFilterNoise(noise_param_short, noise_corr)
    test_noise_long = FFTFilterNoise(noise_param_long, noise_corr)
    assert np.shape(test_noise_short._prepare_rand()) == (2,20000)
    assert np.shape(test_noise_long._prepare_rand()) == (2,40000)
    np.testing.assert_allclose(test_noise_short._prepare_rand()[:,:10],
                       test_noise_long._prepare_rand()[:,:10], atol=1e-15, rtol=1e-15)


def test_prepare_rand():
    # Test all of the seeding type cases
    # Test that the return is what we expect
    # Test that string breaks it
    # If seed is an integer, check that it's exactly the same as predicted by
    # RandomState - and if it's None, make sure it doesn't die horribly :D
    # You can steal lines 150-164 to check by-hand for a few random numbers.
    """
    Tests that the prepare_rand function properly prepares a random Gaussian
    distribution, and that it is seeded correctly.
    """
    noise_param = {
        "MODEL": "FFT_FILTER",
        "TLEN": 10.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    noise_param["SEED"] = 0
    test_noise = FFTFilterNoise(noise_param, noise_corr)
    test_rand_int_seed = test_noise._prepare_rand()
    assert np.allclose(test_rand_int_seed, FFTFilterNoise(noise_param,
                                                          noise_corr)._prepare_rand())

    # Manually test that the code does as it says with a certain seed
    randstate = np.random.RandomState(seed=noise_param['SEED'])

    random_numbers = randstate.rand(4 * (len(test_noise.param['T_AXIS']) - 1),
                                    test_noise.param["N_L2"]).T
    g_index, phi_index = test_noise._construct_indexing(len(test_noise.param['T_AXIS']))
    g_rands = random_numbers[:, np.array(g_index)]
    phi_rands = random_numbers[:, np.array(phi_index)]
    test_prepared_random = np.complex64(
        np.sqrt(-np.log(g_rands)) * np.exp(2.0j * np.pi * phi_rands))
    assert np.allclose(test_prepared_random, test_rand_int_seed)


    noise_param["SEED"] = None
    test_noise = FFTFilterNoise(noise_param, noise_corr)
    test_rand_unseeded = test_noise._prepare_rand()
    assert not np.allclose(test_rand_int_seed, test_rand_unseeded)

    noise_param["SEED"] = np.array([np.arange(2*(len(test_noise.param["T_AXIS"])-1)),
                                    np.arange(2*(len(test_noise.param["T_AXIS"])-1))])
    test_noise = FFTFilterNoise(noise_param, noise_corr)
    test_rand_list_seed = test_noise._prepare_rand()
    assert np.allclose(test_rand_list_seed,noise_param["SEED"])

    noise_param["SEED"] = np.array([np.arange(17), np.arange(17)])
    test_noise = FFTFilterNoise(noise_param, noise_corr)
    with pytest.raises(UnsupportedRequest) as excinfo:
        test_rand_improper_list_seed = test_noise._prepare_rand()
        assert 'Noise.param[SEED] is an array of the wrong length' in str(excinfo.value)

    noise_param["SEED"] = "string"
    test_noise = FFTFilterNoise(noise_param, noise_corr)
    with pytest.raises(UnsupportedRequest) as excinfo:
        test_rand_string_seed = test_noise._prepare_rand()
        assert 'Noise.param[SEED] of type {} not supported'.format(
                    type(noise_param['SEED'])) in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        noise_param['SEED'] = FFTFilterNoise({}, {})
        test_noise = FFTFilterNoise(noise_param, noise_corr)
        test_rand_FFTFilterNoise = test_noise._prepare_rand()
        assert 'is of type' in str(excinfo.value)