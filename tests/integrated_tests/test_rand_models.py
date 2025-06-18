import pytest
import numpy as np
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.noise.hops_noise import HopsNoise
from mesohops.trajectory.hops_trajectory import HopsTrajectory as HOPS
from mesohops.util.physical_constants import hbar


__title__ = "Test of FFT_FILTER noise model"
__author__ = "J. K. Lynd"
__version__ = "1.2"
__date__ = "Jan 17 2023"

g1 = 10000.0-1000.0j
w1 = 100.0
g2 = 1000.0j
w2 = 1000.0
loperator = np.zeros([2, 2, 2], dtype=np.float64)
loperator[0, 0, 0] = 1.0
loperator[1, 1, 1] = 1.0
sys_param = {
    "HAMILTONIAN": np.array([[0, 10.0], [10.0, 0]], dtype=np.float64),
    "GW_SYSBATH": [[g1, w1], [g2, w2], [g1, w1], [g2, w2]],
    "L_HIER": [loperator[0], loperator[0], loperator[1], loperator[1]],
    "ALPHA_NOISE1": bcf_exp,
    "PARAM_NOISE1": [[g1, w1], [g2, w2], [g1, w1], [g2, w2]],
    "L_NOISE1": [loperator[0], loperator[0], loperator[1], loperator[1]],
}


sys_param["NSITE"] = len(sys_param["HAMILTONIAN"][0])
sys_param["NMODES"] = len(sys_param["GW_SYSBATH"][0])
sys_param["N_L2"] = 2
sys_param["L_IND_BY_NMODE1"] = [0, 1]
sys_param["LIND_DICT"] = {0: loperator[0, :, :], 1: loperator[1, :, :]}

hier_param = {"MAXHIER": 1}

integrator_param = {
        "INTEGRATOR": "RUNGE_KUTTA",
        'EARLY_ADAPTIVE_INTEGRATOR':'INCH_WORM',
        'EARLY_INTEGRATOR_STEPS': 5,
        'INCHWORM_CAP': 5,
        'STATIC_BASIS':None
}

eom_param = {"TIME_DEPENDENCE": False, "EQUATION_OF_MOTION": "LINEAR"}


@pytest.mark.level(3)
def test_uncorrelated_noise_averages():
    """
    Tests that the uncorrelated noise trajectory has the correct statistical qualities.
    """
    print("Expected properties")
    print("Mean of noise")
    print(0 + 0j)
    print("Mean absolute value of noise")
    print(np.sqrt(np.pi/2))
    print("Mean absolute-squared value of noise")
    print(2)
    # Our version of the Box-Muller
    noise_param_long = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10000000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "BOX_MULLER",
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    print("Old implementation of Box-Muller")
    test_noise_long = HopsNoise(noise_param_long, noise_corr)._prepare_rand()
    print("Mean of noise")
    print(np.mean(test_noise_long))
    print("Mean absolute value of noise")
    print(np.mean(np.abs(test_noise_long)))
    print("Mean absolute-squared value of noise")
    print(np.mean(np.abs(test_noise_long)**2))
    assert np.allclose(np.mean(test_noise_long), 0, rtol=1e-3, atol=1e-3)
    assert np.allclose(np.mean(np.abs(test_noise_long)), np.sqrt(np.pi / 2), rtol=1e-3,
                       atol=1e-3)
    assert np.allclose(np.mean(np.abs(test_noise_long) ** 2), 2, rtol=1e-3, atol=1e-3)

    # Sum Gaussian version of the Box-Muller
    noise_param_long = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 10000000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "SUM_GAUSSIAN",
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    print("Sum Gaussian implementation of Box-Muller")
    test_noise_long = HopsNoise(noise_param_long, noise_corr)._prepare_rand()
    print("Mean of noise")
    print(np.mean(test_noise_long))
    print("Mean absolute value of noise")
    print(np.mean(np.abs(test_noise_long)))
    print("Mean absolute-squared value of noise")
    print(np.mean(np.abs(test_noise_long) ** 2))

    assert np.allclose(np.mean(test_noise_long), 0, rtol=1e-3, atol=1e-3)
    assert np.allclose(np.mean(np.abs(test_noise_long)), np.sqrt(np.pi/2), rtol=1e-3,
                       atol=1e-3)
    assert np.allclose(np.mean(np.abs(test_noise_long)**2), 2, rtol=1e-3, atol=1e-3)

@pytest.mark.level(3)
def test_correlated_noise_ensembles():
    """
    Tests that the correlated noise trajectories reproduce the proper noise
    correlation function.
    """
    t_axis = np.arange(1000.0)
    cf_analytic = g1*np.exp(-t_axis*w1/hbar) + g2*np.exp(-t_axis*w2/hbar)

    noise_param_short = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 2500.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "BOX_MULLER",
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    hops_box_muller = HOPS(
            system_param=sys_param,
            noise_param=noise_param_short,
            hierarchy_param=hier_param,
            eom_param=eom_param,
            integration_param=integrator_param,)
    cf_box_muller = hops_box_muller.construct_noise_correlation_function(0, 1000)[0]

    noise_param_short = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 2500.0,  # Units: fs
        "TAU": 1.0,  # Units: fs
        "RAND_MODEL": "SUM_GAUSSIAN",
    }
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }
    hops_sum_gaussian = HOPS(
        system_param=sys_param,
        noise_param=noise_param_short,
        hierarchy_param=hier_param,
        eom_param=eom_param,
        integration_param=integrator_param, )
    cf_sum_gaussian = hops_sum_gaussian.construct_noise_correlation_function(0,
                                                                             1000)[0]

    assert np.allclose(cf_box_muller[:10], cf_analytic[:10], rtol=2e-2)
    assert np.allclose(cf_sum_gaussian[:10], cf_analytic[:10], rtol=2e-2)
