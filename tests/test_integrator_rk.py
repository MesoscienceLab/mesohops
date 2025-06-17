import numpy as np
from mesohops.integrator.integrator_rk import runge_kutta_variables
from mesohops.noise.hops_noise import HopsNoise
from mesohops.trajectory.exp_noise import bcf_exp

noise_param = {
    "SEED": np.array([np.arange(-10,10.5,0.5), -1*np.arange(-10,10.5,0.5)]),
    "MODEL": "PRE_CALCULATED",
    "TLEN": 10.0,  # Units: fs
    "TAU": 0.25,  # Units: fs
}

noise_param_two = {
    "SEED": np.array([np.arange(-10,10.5,0.5)/2, -1*np.arange(-10,10.5,0.5)/2]),
    "MODEL": "PRE_CALCULATED",
    "TLEN": 10.0,  # Units: fs
    "TAU": 0.25,  # Units: fs
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

noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }

def test_effective_noise_integration():
    """
    Tests that the effective noise integration that averages the noise over all time
    points within the range of an integration time step works properly.
    """
    # Using the runge_kutta_variables function
    test_noise = HopsNoise(noise_param, noise_corr)
    test_noise2 = HopsNoise(noise_param_two, noise_corr)
    rk_var_control = runge_kutta_variables("a", "b", 5.0, test_noise, test_noise2, 1.5,
                                           "c", [0,1], effective_noise_integration=False)
    rk_var_integrated = runge_kutta_variables("a", "b", 5.0, test_noise, test_noise2, 1.5,
                                           "c", [0,1], effective_noise_integration=True)

    # By-hand solutions
    known_noise_1 = noise_param["SEED"][:, 5*4:8*4]
    known_control_1 = known_noise_1[:, np.array([0, 3, 6])]
    known_integrated_1 = np.array([np.mean(known_noise_1[:,:3],axis=1),
                                   np.mean(known_noise_1[:,3:6],axis=1),
                                   np.mean(known_noise_1[:,6:9],axis=1)]).T
    known_noise_2 = noise_param_two["SEED"][:, 5*4:8*4]
    known_control_2 = known_noise_2[:, np.array([0, 3, 6])]
    known_integrated_2 = np.array([np.mean(known_noise_2[:,:3],axis=1),
                                   np.mean(known_noise_2[:,3:6],axis=1),
                                   np.mean(known_noise_2[:,6:9],axis=1)]).T

    assert np.allclose(rk_var_control['z_rnd'], known_control_1)
    assert np.allclose(rk_var_control['z_rnd2'], known_control_2)
    assert np.allclose(rk_var_integrated['z_rnd'], known_integrated_1)
    assert np.allclose(rk_var_integrated['z_rnd'], known_integrated_1)
