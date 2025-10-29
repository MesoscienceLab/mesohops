import numpy as np
import pytest

from mesohops.noise.hops_noise import HopsNoise
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.exceptions import UnsupportedRequest

__title__ = "Test of hops_noise"
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
sys_param["NMODE1_BY_LIND"] = [[0], [1]]
sys_param["LIND_DICT"] = {0: loperator[0, :, :], 1: loperator[1, :, :]}

def test_initialize():
    """
    Test the initialization of the HopsNoise class (via the FFTFilterNoise class,
    which inherits from it) and thus param, its setter, and the update_param function.
    """
    noise_param = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False
    }

    noise_corr_working = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }

    #noise_corr_empty = {}
    noise_corr_empty = {"N_L2": sys_param["N_L2"]}
    
    t_axis = np.arange(0, 1001.0, 1.0)

    # Initialize a) turns noise_param into HopsNoise.param and b) updates
    # HopsNoise.param with all the key, value pairs from noise_corr. Finally,
    # it builds the noise t_axis.

    # Test that initialization of a HopsNoise properly moves the parameters from
    # noise_param into the param dictionary and constructs the noise correlation
    # function
    test_noise = HopsNoise(noise_param, noise_corr_empty)

    assert noise_param['SEED'] == test_noise.param['SEED']
    assert noise_param['MODEL'] == test_noise.param['MODEL']
    assert noise_param['TLEN'] == test_noise.param['TLEN']
    assert noise_param['TAU'] == test_noise.param['TAU']
    assert noise_param['INTERPOLATE'] == test_noise.param['INTERPOLATE']
    assert np.allclose(t_axis, test_noise.param['T_AXIS'])

    # Now test that the param dictionary is updated with all items from the
    # noise_corr dictionary

    test_noise = HopsNoise(noise_param, noise_corr_working)

    assert noise_param['SEED'] == test_noise.param['SEED']
    assert noise_param['MODEL'] == test_noise.param['MODEL']
    assert noise_param['TLEN'] == test_noise.param['TLEN']
    assert noise_param['TAU'] == test_noise.param['TAU']
    assert noise_param['INTERPOLATE'] == test_noise.param['INTERPOLATE']
    assert np.allclose(t_axis, test_noise.param['T_AXIS'])
    assert noise_corr_working['CORR_FUNCTION'] == test_noise.param['CORR_FUNCTION']
    assert noise_corr_working['N_L2'] == test_noise.param['N_L2']
    assert noise_corr_working['LIND_BY_NMODE'] == test_noise.param['LIND_BY_NMODE']
    assert noise_corr_working['CORR_PARAM'] == test_noise.param['CORR_PARAM']
    # Test that FLAG_REAL defaults to False
    assert test_noise.param["FLAG_REAL"] == False
    # Test that the keys overlap excepting T_AXIS (added by HopsNoise) and
    # STORE_RAW_NOISE (added by FFTFilterNoise)
    assert set(list(noise_param.keys()) + list(noise_corr_working.keys()) + [
        'T_AXIS', 'RAND_MODEL', 'STORE_RAW_NOISE', 'NOISE_WINDOW', 'ADAPTIVE',
        'FLAG_REAL' ]) == set(test_noise.param.keys())


def test_get_noise(capsys):
    """
    Tests that the get_noise function gets the correct noise, both with and without
    windowing.
    """
    noise_param = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False
    }

    noise_corr_working = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "NMODE_BY_LIND": sys_param["NMODE1_BY_LIND"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }

    noise_param_broken = {
        "SEED": None,
        "MODEL": "NONEXISTENT_NOISE",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False
    }

    noise_param_windowed = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False,
        "NOISE_WINDOW": 100.0
    }
    
    noise_param_windowed_adaptive = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False,
        "NOISE_WINDOW": 100.0,
        "ADAPTIVE": True
    }
    t_axis = np.arange(0, 1001.0, 1.0)
    test_noise = HopsNoise(noise_param, noise_corr_working)
    noise = np.arange(2*len(t_axis)).reshape([2,len(t_axis)])
    test_noise._noise = noise
    #test_noise._lock()
    test_noise._lop_active = list(np.arange(sys_param["N_L2"]))

    # Tests only that the unwindowed get_noise function returns the correct noise
    # subsection. Does NOT test whether the noise is generated by the correct formula.

    assert np.allclose(test_noise.get_noise(t_axis[:2])[0,:], test_noise._noise[0,:2])

    # Tests that get_noise raises an UnsupportedRequest if using a nonexistent
    # NOISE_MODEL
    with pytest.raises(UnsupportedRequest) as excinfo:
        HopsNoise(noise_param_broken, noise_corr_working).get_noise(t_axis[:2])
    assert ('does not support Noise.param[MODEL] NONEXISTENT_NOISE in the ' in
            str(excinfo.value))

    # Windowed noise test
    test_noise_windowed = HopsNoise(noise_param_windowed, noise_corr_working)
    noise_windowed = np.arange(2 * len(t_axis)).reshape([2, len(t_axis)])
    test_noise_windowed._noise = noise_windowed
    test_noise_windowed._lop_active = list(np.arange(sys_param["N_L2"]))
    nsteps_window = int(noise_param_windowed["NOISE_WINDOW"] / noise_param_windowed[
        "TAU"])

    # Check that the windowed and unwindowed noise are the same, and that the
    # windowed noise is as we expect: initial window.
    assert np.allclose(test_noise.get_noise(t_axis[:2]),
                       test_noise_windowed.get_noise(t_axis[:2]))
    assert np.allclose(test_noise_windowed.Z2_windowed,test_noise_windowed._noise[:,
                                                  :nsteps_window+1])
    # Start and end outside of initial window.
    assert np.allclose(test_noise.get_noise(t_axis[102:104]),
                       test_noise_windowed.get_noise(t_axis[102:104]))
    assert np.allclose(test_noise_windowed.Z2_windowed,
                       test_noise_windowed._noise[:, 102:104+nsteps_window])
    # Start only out of current window.
    assert np.allclose(test_noise.get_noise(t_axis[101:103]),
                       test_noise_windowed.get_noise(t_axis[101:103]))
    assert np.allclose(test_noise_windowed.Z2_windowed,
                       test_noise_windowed._noise[:, 101:103 + nsteps_window])
    # End only out of current window.
    assert np.allclose(test_noise.get_noise(t_axis[102:301]),
                       test_noise_windowed.get_noise(t_axis[102:301]))
    assert np.allclose(test_noise_windowed.Z2_windowed,
                       test_noise_windowed._noise[:, 102:301 + nsteps_window])
    # Within current window.
    assert np.allclose(test_noise.get_noise(t_axis[151:201]),
                       test_noise_windowed.get_noise(t_axis[151:201]))
    assert np.allclose(test_noise_windowed.Z2_windowed,
                       test_noise_windowed._noise[:, 102:301 + nsteps_window])
    # Running up against end of time axis.
    assert np.allclose(test_noise.get_noise(t_axis[-2:]),
                       test_noise_windowed.get_noise(t_axis[-2:]))
    assert np.allclose(test_noise_windowed.Z2_windowed,test_noise_windowed._noise[:,-2:])
    # Check that unwindowed noise does not create a noise window.
    assert np.allclose(test_noise.Z2_windowed, test_noise._noise)



    # Interpolated noise test
    noise_param_interp = {
        "SEED": noise,
        "MODEL": "PRE_CALCULATED",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": True,
    }
    test_noise_interp = HopsNoise(noise_param_interp, noise_corr_working)
    assert np.allclose(test_noise_interp.get_noise([0, 0.25, 0.5, 0.75, 1]),
                       np.array([[0, 0.25, 0.5, 0.75, 1.0],
                                 [1001, 1001.25, 1001.5, 1001.75, 1002]]))

    # Tests that we get a warning when using windowing with interpolation
    noise_param_interp_with_windowing = {
        "SEED": noise,
        "MODEL": "PRE_CALCULATED",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": True,
        "NOISE_WINDOW": 100.0
    }
    test_noise_interp = HopsNoise(noise_param_interp_with_windowing, noise_corr_working)
    test_noise_interp.get_noise([0, 0.25, 0.5, 0.75, 1])
    out, err = capsys.readouterr()
    assert ("Warning: noise windowing is not supported while using interpolated "
            "noise") in out

    # Tests of FLAG_REAL
    noise_param_real = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False,
        "FLAG_REAL": True,
    }
    test_noise_real = HopsNoise(noise_param_real, noise_corr_working)
    # if FLAG_REAL, noise should be purely real.
    assert np.allclose(test_noise_real.get_noise(t_axis[:2])[0, :],
                       np.real(test_noise_real._noise[0, :2]), atol=1e-8)

    noise_param_complex = {
        "SEED": 0,
        "MODEL": "FFT_FILTER",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False,
        "FLAG_REAL": False,
    }
    test_noise_complex = HopsNoise(noise_param_complex, noise_corr_working)
    # If not FLAG_REAL, noise is not set to real.
    assert not np.allclose(test_noise_complex.get_noise(t_axis[:2])[0, :],
                           np.real(test_noise_complex._noise[0, :2]), atol=1e-8)

    # Same test for interpolated noise
    noise_param_interp_real = {
        "SEED": 1j*noise,
        "MODEL": "PRE_CALCULATED",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": True,
        "FLAG_REAL": True,
    }
    test_noise_interp_real = HopsNoise(noise_param_interp_real, noise_corr_working)
    assert np.allclose(test_noise_interp_real.get_noise([0, 0.25, 0.5, 0.75, 1]),
                       np.zeros([2,5]), atol=1e-8)

    noise_param_interp_complex = {
        "SEED": 1j * noise,
        "MODEL": "PRE_CALCULATED",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": True,
        "FLAG_REAL": False,
    }
    test_noise_interp_complex = HopsNoise(noise_param_interp_complex, noise_corr_working)
    assert not np.allclose(test_noise_interp_complex.get_noise([0, 0.25, 0.5, 0.75, 1]),
                       np.zeros([2, 5]), atol=1e-8)


def test_noise_adaptivity():
    """
    Tests that the noise generated adaptively by L-operator matches the 
    noise generated all at once.
    """
    tlen = 1000.0
    random_seed = 3333
    noise_param_full = {
        "SEED": random_seed,
        "MODEL": "FFT_FILTER",
        "TLEN": tlen,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False,
        "ADAPTIVE": False
    }    
    noise_param_adaptive = {
        "SEED": random_seed,
        "MODEL": "FFT_FILTER",
        "TLEN": tlen,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False,
        "ADAPTIVE": True
    }
    noise_param_adaptive_window = {
        "SEED": random_seed,
        "MODEL": "FFT_FILTER",
        "TLEN": tlen,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False,
        "NOISE_WINDOW": 100.0,
        "ADAPTIVE": True
    }
    nmode_by_lind = []
    num_lop = 2*5
    for i in range(num_lop):
        nmode_by_lind.append([i])  
    param_noise1 = []
    for i in range(int(num_lop/2)):
        param_noise1.append([10.0, 10.0])
        param_noise1.append([5.0, 5.0])  
    noise_corr = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": num_lop,
        "LIND_BY_NMODE": list(np.arange(num_lop)),
        "NMODE_BY_LIND": nmode_by_lind,
        "CORR_PARAM": param_noise1,
    }
    
    noise_full = HopsNoise(noise_param_full, noise_corr)
    noise_adaptive = HopsNoise(noise_param_adaptive, noise_corr)
    
    list_lop_full = list(np.arange(num_lop))
    t_axis = list(np.arange(tlen))
    Z_noise_full = noise_full.get_noise(t_axis,list_lop_full)
    
    list_lop_adap = []
    #Add random l_operators to list_lop, 
    #call get_noise and check that it matches, until all l_operators are added.
    list_lop_index = [9, 4, 3, 2, 0, 1, 5, 6, 8, 7]
    for i in range(num_lop):
        lop_index = list_lop_index[i]
        lop = list_lop_full[lop_index]
        list_lop_adap.append(lop)
        list_lop_adap = sorted(list_lop_adap)
        Z_noise_adap = noise_adaptive.get_noise(t_axis,list_lop_adap)
        assert np.allclose(Z_noise_adap,Z_noise_full[list_lop_adap,:])

def test_corr_func_builder():
    """
    Tests that the _corr_func_by_lop_taxis returns the correct correlation function.
    """
    noise_param = {
        "SEED": None,
        "MODEL": "FFT_FILTER",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": False
    }

    noise_corr_working = {
        "CORR_FUNCTION": sys_param["ALPHA_NOISE1"],
        "N_L2": sys_param["N_L2"],
        "LIND_BY_NMODE": sys_param["L_IND_BY_NMODE1"],
        "NMODE_BY_LIND": sys_param["NMODE1_BY_LIND"],
        "CORR_PARAM": sys_param["PARAM_NOISE1"],
    }

    t_axis = np.arange(0, 1001.0, 1.0)
    test_noise = HopsNoise(noise_param, noise_corr_working)

    # Compares the correlation function over both sites calculated manually with the
    # correlation function over both sites generated by the FFTFilterNoise object.
    corr_func_site_0 = bcf_exp(t_axis, sys_param["PARAM_NOISE1"][0][0], sys_param[
        "PARAM_NOISE1"][0][1])
    corr_func_site_1 = bcf_exp(t_axis, sys_param["PARAM_NOISE1"][1][0], sys_param[
        "PARAM_NOISE1"][1][1])
    corr_func = np.array([corr_func_site_0, corr_func_site_1])
    assert np.allclose(corr_func, test_noise._corr_func_by_lop_taxis(t_axis,list(np.arange(sys_param["N_L2"]))))
