import os
import numpy as np
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.noise.hops_noise import HopsNoise
from mesohops.util.exceptions import UnsupportedRequest


__title__ = "Test of FFT_FILTER noise model"
__author__ = "J. K. Lynd"
__version__ = "1.2"
__date__ = "July 7 2021"

path_data = os.path.realpath(__file__)[: -len("test_noise_fft.py")]
path_seed = path_data + "/pre_calculated_uncorrelated_noise_test.npy"

# Test FFT_FILTER Noise Model of the HopsNoise object.
# ----------------------------------------------------
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
sys_param["NMODE1_BY_L_IND"] = [[0],[1]]
sys_param["LIND_DICT"] = {0: loperator[0, :, :], 1: loperator[1, :, :]}

eom_param = {"EQUATION_OF_MOTION": "NORMALIZED NONLINEAR"}

# To re-save hard-coded dynamics if noise generation changes
'''
hops = HOPS(
    sys_param,
    noise_param=noise_param,
    hierarchy_param={"MAXHIER": 2},
    eom_param=eom_param,
)
n_lop = 2
nsite = sys_param["NSITE"]
psi_0 = np.array([0.0] * nsite, dtype=np.complex128)
psi_0[0] = 1.0
psi_0 = psi_0 / np.linalg.norm(psi_0)
hops.initialize(psi_0)
uncorrelated_noise = hops.noise1._prepare_rand()
np.save(path_seed,uncorrelated_noise)
'''

def test_noiseModel():
    """
    Tests that HopsNoise with the FFT_Filter model produces reproducible noise
    trajectories when seeded, each seed produces a unique noise trajectories,
    and that the noise trajectory is the correct size and shape.
    """
    noise_param = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
    }
    noise_param_interpolated = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "INTERPOLATE": True
    }
    noise_param_array_seed = {
        "SEED": np.load(path_seed),
        "MODEL": "FFT_FILTER",
        "TLEN": 10.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "INTERPOLATE": True
    }
    noise_param_string_seed = {
        "SEED": path_seed,
        "MODEL": "FFT_FILTER",
        "TLEN": 10.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "INTERPOLATE": True
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "NMODE_BY_LIND": sys_param["NMODE1_BY_L_IND"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    noiseModel = HopsNoise(noise_param, noise_corr)
    assert noiseModel.param["T_AXIS"][-1] == 10.0
    testnoise1 = noiseModel.get_noise(np.arange(10.0))[0, :]
    testnoise2 = HopsNoise(noise_param, noise_corr).get_noise(np.arange(10.0))[
                 0, :]
    assert np.allclose(testnoise1, testnoise2)
    
    testnoise_array_seed = HopsNoise(noise_param_array_seed, noise_corr).get_noise(
        np.arange(10.0))[0, :]
    assert np.allclose(testnoise1, testnoise_array_seed)
    
    testnoise_string_seed = HopsNoise(noise_param_string_seed, noise_corr).get_noise(
        np.arange(10.0))[0, :]
    assert np.allclose(testnoise1, testnoise_string_seed)
    
    noise_param["SEED"] = 1
    testnoise3 = HopsNoise(noise_param, noise_corr).get_noise(np.arange(10.0))[
                 0, :]
    assert not np.allclose(testnoise1, testnoise3)
    testnoise_interp = HopsNoise(noise_param_interpolated, noise_corr).get_noise(
        np.arange(10.0))[0, :]
    assert np.allclose(testnoise1, testnoise_interp)
    testnoise_interp_step = HopsNoise(noise_param_interpolated,
                                      noise_corr).get_noise([0.3, 1.2, 2.4, 3.1, 4.5,
                                                             5.6, 6.3, 7.9, 8.7])[0, :]
    
    for i in range(len(testnoise_interp_step)):
        sample_noise = testnoise_interp_step[i]
        assert (np.real(sample_noise) > np.real(testnoise1)[i] and np.real(
            sample_noise) < np.real(testnoise1)[i+1]) or \
               (np.real(sample_noise) < np.real(testnoise1)[i] and np.real(
                   sample_noise) > np.real(testnoise1[i+1]))  or \
            np.allclose(np.real(testnoise1[i]), np.real(sample_noise), rtol=5e-2) or \
               np.allclose(np.real(testnoise1[i+1]), np.real(sample_noise), rtol=5e-2)
        assert (np.imag(sample_noise) > np.imag(testnoise1)[i] and np.imag(
            sample_noise) < np.imag(testnoise1)[i + 1]) or \
               (np.imag(sample_noise) < np.imag(testnoise1)[i] and np.imag(
                   sample_noise) > np.imag(testnoise1[i + 1])) or \
               np.allclose(np.imag(testnoise1[i]), np.imag(sample_noise), rtol=5e-2) or \
               np.allclose(np.imag(testnoise1[i + 1]), np.imag(sample_noise), rtol=5e-2)
    assert np.size(noiseModel.get_noise(noiseModel.param["T_AXIS"])) == np.size(
        noiseModel.param["T_AXIS"])*noiseModel.param["N_L2"]

def test_correlated_noise_consistency():
    """
    Tests that the correlated noise trajectory is the same regardless of noise
    trajectory length. The noise trajectories produced should be extremely close.
    """
    # Our version of the Box-Muller
    noise_param_short = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 100000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "BOX_MULLER",
    }
    noise_param_long = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 200000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "BOX_MULLER",
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "NMODE_BY_LIND": sys_param["NMODE1_BY_L_IND"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    test_noise_short = HopsNoise(noise_param_short, noise_corr)
    test_noise_long = HopsNoise(noise_param_long, noise_corr)
    np.testing.assert_allclose(test_noise_short.get_noise(np.arange(10)),
                               test_noise_long.get_noise(np.arange(10)), rtol=2e-7)

    # Sum Gaussian version of the Box-Muller
    noise_param_short = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 100000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "SUM_GAUSSIAN",
    }
    noise_param_long = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 200000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "SUM_GAUSSIAN",
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "NMODE_BY_LIND": sys_param["NMODE1_BY_L_IND"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    test_noise_short = HopsNoise(noise_param_short, noise_corr)
    test_noise_long = HopsNoise(noise_param_long, noise_corr)
    np.testing.assert_allclose(test_noise_short.get_noise(np.arange(10)),
                               test_noise_long.get_noise(np.arange(10)), rtol=2e-7)

def test_uncorrelated_noise_consistency():
    """
    Tests that the uncorrelated noise trajectory is the same regardless of noise
    trajectory length. The noise trajectories produced should be exactly identical.
    """
    


    # Sum Gaussian version of the Box-Muller
    noise_param_short = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "SUM_GAUSSIAN",
    }
    noise_param_long = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 20000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "SUM_GAUSSIAN",
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "NMODE_BY_LIND": sys_param["NMODE1_BY_L_IND"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    test_noise_short = HopsNoise(noise_param_short, noise_corr)._prepare_rand()
    test_noise_long = HopsNoise(noise_param_long, noise_corr)._prepare_rand()
    assert np.shape(test_noise_short) == (2, 20000)
    assert np.shape(test_noise_long) == (2, 40000)
    # Must re-initialize HopsNoise object or use the noise trajectory directly,
    # re-running _prepare_rand will cause the values to change.
    np.testing.assert_allclose(test_noise_short[:, :10],
                               test_noise_long[:, :10], atol=1e-15, rtol=1e-15)


    # Our version of the Box-Muller
    noise_param_short = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "BOX_MULLER",
    }
    noise_param_long = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 20000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "BOX_MULLER",
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "NMODE_BY_LIND": sys_param["NMODE1_BY_L_IND"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    test_noise_short = HopsNoise(noise_param_short, noise_corr)._prepare_rand()
    test_noise_long = HopsNoise(noise_param_long, noise_corr)._prepare_rand()
    assert np.shape(test_noise_short) == (2, 20000)
    assert np.shape(test_noise_long) == (2, 40000)
    # Must re-initialize HopsNoise object or use the noise trajectory directly,
    # re-running _prepare_rand will cause the values to change.
    np.testing.assert_allclose(test_noise_short[:, :10],
                               test_noise_long[:, :10], atol=1e-15, rtol=1e-15)

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
        "NMODE_BY_LIND": sys_param["NMODE1_BY_L_IND"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    noise_param["SEED"] = 0
    test_noise = HopsNoise(noise_param, noise_corr)
    test_rand_int_seed = test_noise._prepare_rand()
    assert np.allclose(test_rand_int_seed, HopsNoise(noise_param,
                                                     noise_corr)._prepare_rand())

    noise_param["RAND_MODEL"] = "BOX_MULLER"
    test_noise_BM_old = HopsNoise(noise_param, noise_corr)
    test_rand_int_seed_BM_old = test_noise_BM_old._prepare_rand()
    assert np.allclose(test_rand_int_seed_BM_old, HopsNoise(noise_param,
                                                     noise_corr)._prepare_rand())

    assert not np.allclose(test_rand_int_seed, test_rand_int_seed_BM_old)

    # Manually test that the code does as it says with a certain seed - our
    # implementation of sum Gaussian
    index_a, index_b = test_noise._construct_indexing(len(test_noise.param['T_AXIS']))
    n_times = len(test_noise.param['T_AXIS'])
    n_l2 = test_noise.param["N_L2"]
    random_gauss = np.zeros((n_l2, 4*(n_times-1))) 
    for (i,lop) in enumerate(np.arange(n_l2)):
        bitgenerator = np.random.PCG64(seed=noise_param['SEED']).jumped(lop)
        randstate = np.random.default_rng(bitgenerator)
        random_gauss[i,:] = randstate.normal(loc=0, size=4*(n_times-1))
    
    random_real = random_gauss[:, np.array(index_a)]
    random_imag = random_gauss[:, np.array(index_b)]
    test_prepared_random = np.complex64(random_real + 1j*random_imag)
    assert np.allclose(test_prepared_random, test_rand_int_seed)

    # Manually test that the code does as it says with a certain seed - our
    # implementation of Box Muller
    n_times = len(test_noise.param['T_AXIS'])
    n_l2 = test_noise.param["N_L2"]
    random_numbers = np.zeros((n_l2, 4*(n_times-1))) 
    for (i,lop) in enumerate(np.arange(n_l2)):
        bitgenerator = np.random.PCG64(seed=noise_param['SEED']).jumped(lop)
        randstate = np.random.default_rng(bitgenerator)
        random_numbers[i,:] = randstate.random(size=4*(n_times-1))                              
    
    g_rands = random_numbers[:, np.array(index_a)]
    phi_rands = random_numbers[:, np.array(index_b)]
    test_prepared_random = np.complex64(
        np.sqrt(-2*np.log(g_rands)) * np.exp(2.0j * np.pi * phi_rands))
    assert np.allclose(test_prepared_random, test_rand_int_seed_BM_old)

    noise_param["SEED"] = None
    test_noise = HopsNoise(noise_param, noise_corr)
    test_rand_unseeded = test_noise._prepare_rand()
    assert not np.allclose(test_rand_int_seed, test_rand_unseeded)

    noise_param["SEED"] = np.array([np.arange(2*(len(test_noise.param["T_AXIS"])-1)),
                                    np.arange(2*(len(test_noise.param["T_AXIS"])-1))])
    test_noise = HopsNoise(noise_param, noise_corr)
    test_rand_list_seed = test_noise._prepare_rand()
    assert np.allclose(test_rand_list_seed,noise_param["SEED"])

    noise_param["SEED"] = np.array([np.arange(17), np.arange(17)])
    test_noise = HopsNoise(noise_param, noise_corr)
    try:
        test_rand_improper_list_seed = test_noise._prepare_rand()
    except UnsupportedRequest as excinfo:
        assert "Noise.param[SEED] is an array of the wrong length" in str(excinfo)

    noise_param["SEED"] = "string"
    test_noise = HopsNoise(noise_param, noise_corr)
    try:
        test_rand_string_seed = test_noise._prepare_rand()
    except UnsupportedRequest as excinfo:
        assert "is not the address of a valid file" in str(excinfo)

    try:
        noise_param['SEED'] = HopsNoise({}, {"N_L2": 1})
        test_noise = HopsNoise(noise_param, noise_corr)
        test_rand_HopsNoise = test_noise._prepare_rand()
    except TypeError as excinfo:
        assert 'is of type' in str(excinfo)

def test_construct_indexing():
    """
    Tests that the _construct_indexing function constructs the correct indexing for
    Box-Mueller complex Gaussian raw noise generation.
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
        "NMODE_BY_LIND": sys_param["NMODE1_BY_L_IND"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    test_noise = HopsNoise(noise_param, noise_corr)
    biglength = 1000
    for smalllength in range(0, biglength):
        g_indexbig, phi_indexbig = test_noise._construct_indexing(biglength)
        g_indexsmall, phi_indexsmall = test_noise._construct_indexing(smalllength)
        assert g_indexbig[0:smalllength] == g_indexsmall[0:smalllength]
        assert phi_indexbig[0:smalllength] == phi_indexsmall[0:smalllength]

