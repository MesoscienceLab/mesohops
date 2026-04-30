import numpy as np
import pytest

from mesohops.noise.hops_noise import HopsNoise
from mesohops.trajectory.exp_noise import bcf_exp
from mesohops.util.exceptions import UnsupportedRequest
from scipy.interpolate import CubicSpline

__title__ = "Test of hops_noise"
__author__ = "J. K. Lynd"
__version__ = "1.6"
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
        'T_AXIS', 'RAND_MODEL', 'STORE_RAW_NOISE', 'NOISE_WINDOW', 'ADAPTIVE', 'FLAG_REAL']) == set(
        test_noise.param.keys())


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
    test_noise._list_activel2idx_abs = list(np.arange(sys_param["N_L2"]))

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
    test_noise_windowed._list_activel2idx_abs = list(np.arange(sys_param["N_L2"]))
    nsteps_window = int(noise_param_windowed["NOISE_WINDOW"] / noise_param_windowed[
        "TAU"])

    # Check that the windowed and unwindowed noise are the same, and that the
    # windowed noise is as we expect: initial window.
    assert np.allclose(test_noise.get_noise(t_axis[:2]),
                       test_noise_windowed.get_noise(t_axis[:2]))
    assert np.allclose(test_noise_windowed.Z2_noise_windowed,test_noise_windowed._noise[:,
                                                  :nsteps_window+1])
    # Start and end outside of initial window.
    assert np.allclose(test_noise.get_noise(t_axis[102:104]),
                       test_noise_windowed.get_noise(t_axis[102:104]))
    assert np.allclose(test_noise_windowed.Z2_noise_windowed,
                       test_noise_windowed._noise[:, 102:104+nsteps_window])
    # Start only out of current window.
    assert np.allclose(test_noise.get_noise(t_axis[101:103]),
                       test_noise_windowed.get_noise(t_axis[101:103]))
    assert np.allclose(test_noise_windowed.Z2_noise_windowed,
                       test_noise_windowed._noise[:, 101:103 + nsteps_window])
    # End only out of current window.
    assert np.allclose(test_noise.get_noise(t_axis[102:301]),
                       test_noise_windowed.get_noise(t_axis[102:301]))
    assert np.allclose(test_noise_windowed.Z2_noise_windowed,
                       test_noise_windowed._noise[:, 102:301 + nsteps_window])
    # Within current window.
    assert np.allclose(test_noise.get_noise(t_axis[151:201]),
                       test_noise_windowed.get_noise(t_axis[151:201]))
    assert np.allclose(test_noise_windowed.Z2_noise_windowed,
                       test_noise_windowed._noise[:, 102:301 + nsteps_window])
    # Running up against end of time axis.
    assert np.allclose(test_noise.get_noise(t_axis[-2:]),
                       test_noise_windowed.get_noise(t_axis[-2:]))
    assert np.allclose(test_noise_windowed.Z2_noise_windowed,test_noise_windowed._noise[:,-2:])
    # Check that unwindowed noise does not create a noise window.
    assert np.allclose(test_noise.Z2_noise_windowed, test_noise._noise)



    # Interpolated noise test
    noise_param_interp = {
        "SEED": noise,
        "MODEL": "PRE_CALCULATED",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": True,
    }
    test_spline_noise = HopsNoise(noise_param_interp, noise_corr_working)
    assert np.allclose(test_spline_noise.get_noise([0, 0.25, 0.5, 0.75, 1]),
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
    test_spline_noise = HopsNoise(noise_param_interp_with_windowing, noise_corr_working)
    test_spline_noise.get_noise([0, 0.25, 0.5, 0.75, 1])
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
    test_spline_noise_real = HopsNoise(noise_param_interp_real, noise_corr_working)
    assert np.allclose(test_spline_noise_real.get_noise([0, 0.25, 0.5, 0.75, 1]),
                       np.zeros([2,5]), atol=1e-8)

    noise_param_interp_complex = {
        "SEED": 1j * noise,
        "MODEL": "PRE_CALCULATED",
        "TLEN": 1000.0,  # Units: fs
        "TAU": 1.0,  # Units: fs,
        "INTERPOLATE": True,
        "FLAG_REAL": False,
    }
    test_spline_noise_complex = HopsNoise(noise_param_interp_complex, noise_corr_working)
    assert not np.allclose(test_spline_noise_complex.get_noise([0, 0.25, 0.5, 0.75, 1]),
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
    # Add random L-operators to list_l2idx_abs,
    #call get_noise and check that it matches, until all l_operators are added.
    list_lop_index = [9, 4, 3, 2, 0, 1, 5, 6, 8, 7]
    for i in range(num_lop):
        lop_index = list_lop_index[i]
        lop = list_lop_full[lop_index]
        list_lop_adap.append(lop)
        list_lop_adap = sorted(list_lop_adap)
        Z_noise_adap = noise_adaptive.get_noise(t_axis,list_lop_adap)
        assert np.allclose(Z_noise_adap,Z_noise_full[list_lop_adap,:])

# Test Noise Eviction
# -------------------

def test_noise_eviction():
    """
    Tests that stale L-operators are evicted from the noise arrays when the
    adaptive basis moves on, and that evicted L-operators can be regenerated
    identically via the PCG64 jumped-seed scheme.
    """
    tlen = 1000.0
    random_seed = 3333
    noise_param_full = {
        'SEED': random_seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': tlen,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': False,
    }
    noise_param_adaptive = {
        'SEED': random_seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': tlen,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': True,
    }
    nmode_by_lind = []
    num_lop = 2 * 5
    for i in range(num_lop):
        nmode_by_lind.append([i])
    param_noise1 = []
    for i in range(int(num_lop / 2)):
        param_noise1.append([10.0, 10.0])
        param_noise1.append([5.0, 5.0])
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': num_lop,
        'LIND_BY_NMODE': list(np.arange(num_lop)),
        'NMODE_BY_LIND': nmode_by_lind,
        'CORR_PARAM': param_noise1,
    }
    t_axis = list(np.arange(tlen))

    # Full (non-adaptive) reference
    noise_full = HopsNoise(noise_param_full, noise_corr)
    Z_noise_full = noise_full.get_noise(t_axis, list(np.arange(num_lop)))

    # --- Case: generate [0,1,2], then shift to [1,2,3] ---
    noise = HopsNoise(noise_param_adaptive, noise_corr)

    # Step 1: generate noise for lops [0, 1, 2]
    Z_012 = noise.get_noise(t_axis, [0, 1, 2])
    assert np.allclose(Z_012, Z_noise_full[[0, 1, 2], :])

    # Save lop 1 and 2 noise for later comparison
    Z_lop1_before = Z_012[1, :].copy()
    Z_lop2_before = Z_012[2, :].copy()

    # Step 2: shift to lops [1, 2, 3] — lop 0 should be evicted, lop 3 added
    Z_123 = noise.get_noise(t_axis, [1, 2, 3])
    assert np.allclose(noise._list_activel2idx_abs, np.array([1, 2, 3]))
    assert noise._noise.shape[0] == 3

    # Noise for lops 1 and 2 should be unchanged
    assert np.allclose(Z_123[0, :], Z_lop1_before)
    assert np.allclose(Z_123[1, :], Z_lop2_before)

    # Noise for lop 3 matches full reference
    assert np.allclose(Z_123[2, :], Z_noise_full[3, :])

    # Step 3: re-add lop 0 — regenerated noise should match full reference
    Z_0123 = noise.get_noise(t_axis, [0, 1, 2, 3])
    assert np.allclose(Z_0123, Z_noise_full[[0, 1, 2, 3], :])


def test_noise_eviction_with_window():
    """
    Tests that eviction correctly updates Z2_noise_windowed when NOISE_WINDOW is
    active.
    """
    tlen = 1000.0
    random_seed = 3333
    noise_param_full = {
        'SEED': random_seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': tlen,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': False,
    }
    noise_param_adaptive_window = {
        'SEED': random_seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': tlen,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'NOISE_WINDOW': 100.0,
        'ADAPTIVE': True,
    }
    nmode_by_lind = []
    num_lop = 2 * 5
    for i in range(num_lop):
        nmode_by_lind.append([i])
    param_noise1 = []
    for i in range(int(num_lop / 2)):
        param_noise1.append([10.0, 10.0])
        param_noise1.append([5.0, 5.0])
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': num_lop,
        'LIND_BY_NMODE': list(np.arange(num_lop)),
        'NMODE_BY_LIND': nmode_by_lind,
        'CORR_PARAM': param_noise1,
    }
    t_axis = list(np.arange(tlen))

    # Full reference
    noise_full = HopsNoise(noise_param_full, noise_corr)
    Z_noise_full = noise_full.get_noise(t_axis, list(np.arange(num_lop)))

    # Adaptive with windowing
    noise = HopsNoise(noise_param_adaptive_window, noise_corr)
    Z_012 = noise.get_noise(t_axis[:5], [0, 1, 2])
    assert noise._list_activel2idx_abs == [0, 1, 2]
    assert np.allclose(Z_012, Z_noise_full[[0, 1, 2], :5])

    # Evict lop 0, add lop 3
    Z_123 = noise.get_noise(t_axis[:5], [1, 2, 3])
    assert noise._list_activel2idx_abs == [1, 2, 3]
    assert noise._noise.shape[0] == 3
    assert noise.Z2_noise_windowed.shape[0] == 3
    assert np.allclose(Z_123, Z_noise_full[[1, 2, 3], :5])

    # Regenerate lop 0
    Z_0123 = noise.get_noise(t_axis[:5], [0, 1, 2, 3])
    assert noise._list_activel2idx_abs == [0, 1, 2, 3]
    assert np.allclose(Z_0123, Z_noise_full[[0, 1, 2, 3], :5])


def test_noise_eviction_with_interpolation():
    """
    Tests that eviction correctly rebuilds the cubic spline interpolant.
    """
    tlen = 10.0
    tau = 1.0
    random_seed = 3333
    noise_param_full = {
        'SEED': random_seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': tlen,
        'TAU': tau,
        'INTERPOLATE': False,
        'ADAPTIVE': False,
    }
    noise_param_adaptive_interp = {
        'SEED': random_seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': tlen,
        'TAU': tau,
        'INTERPOLATE': True,
        'ADAPTIVE': True,
    }
    nmode_by_lind = []
    num_lop = 2 * 5
    for i in range(num_lop):
        nmode_by_lind.append([i])
    param_noise1 = []
    for i in range(int(num_lop / 2)):
        param_noise1.append([10.0, 10.0])
        param_noise1.append([5.0, 5.0])
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': num_lop,
        'LIND_BY_NMODE': list(np.arange(num_lop)),
        'NMODE_BY_LIND': nmode_by_lind,
        'CORR_PARAM': param_noise1,
    }

    nstep_min = int(np.ceil(tlen / tau)) + 1
    t_axis = np.arange(nstep_min) * tau
    interp_axis = np.array(t_axis) + 0.5

    # Full reference with interpolation
    noise_full = HopsNoise(noise_param_full, noise_corr)
    Z_noise_full = noise_full.get_noise(t_axis, list(np.arange(num_lop)))
    Z_spline_full = CubicSpline(t_axis, Z_noise_full, axis=1)
    Z_interp_full = Z_spline_full(interp_axis)

    # Adaptive with interpolation
    noise = HopsNoise(noise_param_adaptive_interp, noise_corr)

    # Generate for lops [0, 1, 2]
    noise.get_noise(t_axis, [0, 1, 2])
    assert noise._list_activel2idx_abs == [0, 1, 2]
    interp_before = noise._spline_noise

    # Evict lop 0, add lop 3
    Z_123_interp = noise.get_noise(interp_axis, [1, 2, 3])
    assert noise._list_activel2idx_abs == [1, 2, 3]
    assert noise._spline_noise is not interp_before
    assert noise._spline_noise(t_axis).shape[0] == 3
    assert noise._noise.shape[0] == 3
    assert np.allclose(Z_123_interp, Z_interp_full[[1, 2, 3], :])

    # Regenerate lop 0
    interp_before = noise._spline_noise
    Z_0123_interp = noise.get_noise(interp_axis, [0, 1, 2, 3])
    assert noise._list_activel2idx_abs == [0, 1, 2, 3]
    assert noise._spline_noise is not interp_before
    assert noise._spline_noise(t_axis).shape[0] == 4
    assert np.allclose(Z_0123_interp, Z_interp_full[[0, 1, 2, 3], :])


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

def test_noise_adaptivity_with_interpolation():
    tlen = 10.0
    tau = 1.0
    random_seed = 3333
    noise_param_full = {
        "SEED": random_seed,
        "MODEL": "FFT_FILTER",
        "TLEN": tlen,  # Units: fs
        "TAU": tau,  # Units: fs,
        "INTERPOLATE": False,
        "ADAPTIVE": False
    }
    noise_param_adaptive = {
        "SEED": random_seed,
        "MODEL": "FFT_FILTER",
        "TLEN": tlen,  # Units: fs
        "TAU": tau,  # Units: fs,
        "INTERPOLATE": False,
        "ADAPTIVE": True
    }
    noise_param_adaptive_interpolation = {
        "SEED": random_seed,
        "MODEL": "FFT_FILTER",
        "TLEN": tlen,  # Units: fs
        "TAU": tau,  # Units: fs,
        "INTERPOLATE": True,
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
    noise_adaptive_interpolation = HopsNoise(noise_param_adaptive_interpolation, noise_corr)



    list_lop_full = list(np.arange(num_lop))
    #t_axis = list(np.arange(tlen))
    nstep_min = int(np.ceil(tlen / 1)) + 1
    t_axis = np.arange(nstep_min) *tau
    Z_noise_full = noise_full.get_noise(t_axis,list_lop_full)
    Z_noise_Spline = CubicSpline(t_axis,Z_noise_full,axis=1)
    interp_axis = np.array(t_axis) + 0.5
    Z_noise_full_interp = Z_noise_Spline(interp_axis)

    list_lop_adap = []
    # Add random L-operators to list_l2idx_abs,
    #call get_noise and check that it matches, until all l_operators are added.
    list_lop_index = [9, 4, 3, 2, 0, 1, 5, 6, 8, 7]
    for i in range(num_lop):
        lop_index = list_lop_index[i]
        lop = list_lop_full[lop_index]
        list_lop_adap.append(lop)
        list_lop_adap = sorted(list_lop_adap)
        Z_noise_adap = noise_adaptive.get_noise(t_axis,list_lop_adap)
        Z_noise_adap_interp = noise_adaptive_interpolation.get_noise(t_axis,list_lop_adap)
        Z_noise_adap_interp2 = noise_adaptive_interpolation.get_noise(interp_axis, list_lop_adap)
        assert np.allclose(Z_noise_adap,Z_noise_full[list_lop_adap,:])
        assert np.allclose(Z_noise_adap_interp2, Z_noise_full_interp[list_lop_adap])
        assert np.allclose(Z_noise_adap_interp, Z_noise_adap)


# ============================================================
# TEST SUITE: _prepare_noise() gap coverage
# ============================================================

# ------------------------------------------------------------
# TEST: Adaptive ZERO model does not crash
# ------------------------------------------------------------

def test_prepare_noise_adaptive_zero():
    """
    Tests that _prepare_noise does not crash when ADAPTIVE=True and
    MODEL='ZERO'. Previously, a bare pass on the ZERO branch failed to skip
    the adaptive array assembly, causing a TypeError on Z2_corrnoise[:,:].
    """
    noise_param_adaptive_zero = {
        'SEED': 0,
        'MODEL': 'ZERO',
        'TLEN': 10.0,
        'TAU': 1.0,
        'ADAPTIVE': True,
    }
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_adaptive_zero, noise_corr)

    # This case tests that _prepare_noise completes without error
    noise._prepare_noise([0, 1])
    assert noise._noise == 0
    assert noise._list_activel2idx_abs == [0, 1]

    # This case tests that get_noise still returns zeros for ZERO model
    t_axis = np.arange(10.0)
    Z2_noise = noise.get_noise(t_axis, [0, 1])
    assert np.allclose(Z2_noise, np.zeros([2, len(t_axis)]))


# ------------------------------------------------------------
# TEST: STORE_RAW_NOISE with ZERO and FFT_FILTER models
# ------------------------------------------------------------

def test_prepare_noise_store_raw_zero(capsys):
    """
    Tests that STORE_RAW_NOISE=True with MODEL='ZERO' prints a warning that
    raw noise is identical to correlated noise.
    """
    noise_param_raw_zero = {
        'SEED': 0,
        'MODEL': 'ZERO',
        'TLEN': 10.0,
        'TAU': 1.0,
        'STORE_RAW_NOISE': True,
    }
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_raw_zero, noise_corr)
    noise._prepare_noise([0, 1])
    out, _ = capsys.readouterr()
    assert 'Raw noise is identical to correlated noise' in out


def test_prepare_noise_store_raw_fft():
    """
    Tests that STORE_RAW_NOISE=True with MODEL='FFT_FILTER' stores the
    uncorrelated noise in param['Z_UNCORRELATED'].
    """
    noise_param_raw_fft = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'STORE_RAW_NOISE': True,
    }
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_raw_fft, noise_corr)
    noise._prepare_noise([0, 1])

    # This case tests that Z_UNCORRELATED is stored and has the expected shape
    assert 'Z_UNCORRELATED' in noise.param
    n_taus = len(noise.param['T_AXIS'])
    assert noise.param['Z_UNCORRELATED'].shape == (sys_param['N_L2'],
                                                    2 * (n_taus - 1))


# ------------------------------------------------------------
# TEST: PRE_CALCULATED error paths
# ------------------------------------------------------------

def test_prepare_noise_precalc_wrong_shape_array():
    """
    Tests that PRE_CALCULATED with an array seed of wrong shape raises
    UnsupportedRequest.
    """
    wrong_shape_seed = np.zeros((3, 5))
    noise_param_wrong = {
        'SEED': wrong_shape_seed,
        'MODEL': 'PRE_CALCULATED',
        'TLEN': 10.0,
        'TAU': 1.0,
    }
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_wrong, noise_corr)
    with pytest.raises(UnsupportedRequest, match='array of the wrong length'):
        noise._prepare_noise([0, 1])


def test_prepare_noise_precalc_nonexistent_file():
    """
    Tests that PRE_CALCULATED with a string seed pointing to a nonexistent
    file raises UnsupportedRequest.
    """
    noise_param_nofile = {
        'SEED': '/nonexistent/path/noise.npy',
        'MODEL': 'PRE_CALCULATED',
        'TLEN': 10.0,
        'TAU': 1.0,
    }
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_nofile, noise_corr)
    with pytest.raises(UnsupportedRequest, match='is not the address of a valid file'):
        noise._prepare_noise([0, 1])


def test_prepare_noise_precalc_non_npy_file(tmp_path):
    """
    Tests that PRE_CALCULATED with a string seed pointing to a non-.npy file
    raises UnsupportedRequest.
    """
    fake_file = tmp_path / 'noise.txt'
    fake_file.write_text('not a numpy file')
    noise_param_txt = {
        'SEED': str(fake_file),
        'MODEL': 'PRE_CALCULATED',
        'TLEN': 10.0,
        'TAU': 1.0,
    }
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_txt, noise_corr)
    with pytest.raises(UnsupportedRequest, match='filetype .txt is not supported'):
        noise._prepare_noise([0, 1])


def test_prepare_noise_precalc_npy_wrong_shape(tmp_path):
    """
    Tests that PRE_CALCULATED with a .npy file containing an array of wrong
    shape raises UnsupportedRequest.
    """
    wrong_shape = np.zeros((3, 5))
    npy_path = tmp_path / 'noise.npy'
    np.save(str(npy_path), wrong_shape)
    noise_param_npy = {
        'SEED': str(npy_path),
        'MODEL': 'PRE_CALCULATED',
        'TLEN': 10.0,
        'TAU': 1.0,
    }
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_npy, noise_corr)
    with pytest.raises(UnsupportedRequest, match='array of the wrong length'):
        noise._prepare_noise([0, 1])


def test_prepare_noise_precalc_unsupported_seed_type():
    """
    Tests that PRE_CALCULATED with an integer seed (valid type for FFT_FILTER
    but not for PRE_CALCULATED) raises UnsupportedRequest.
    """
    noise_param_int = {
        'SEED': 42,
        'MODEL': 'PRE_CALCULATED',
        'TLEN': 10.0,
        'TAU': 1.0,
    }
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_int, noise_corr)
    with pytest.raises(UnsupportedRequest, match='Noise.param\\[SEED\\] of type'):
        noise._prepare_noise([0, 1])


# ------------------------------------------------------------
# TEST: Adaptive PRE_CALCULATED noise matches full reference
# ------------------------------------------------------------

def test_prepare_noise_adaptive_precalculated():
    """
    Tests that PRE_CALCULATED noise with ADAPTIVE=True raises a warning
    and forces ADAPTIVE to False.
    """
    n_taus = len(np.arange(0, 11.0, 1.0))
    num_lop = 4
    full_noise = np.arange(num_lop * n_taus, dtype=np.complex128).reshape(
        num_lop, n_taus)
    noise_param_adaptive = {
        'SEED': full_noise,
        'MODEL': 'PRE_CALCULATED',
        'TLEN': 10.0,
        'TAU': 1.0,
        'ADAPTIVE': True,
    }

    nmode_by_lind = [[i] for i in range(num_lop)]
    param_noise1 = [[10.0, 10.0], [5.0, 5.0]] * (num_lop // 2)
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': num_lop,
        'LIND_BY_NMODE': list(np.arange(num_lop)),
        'NMODE_BY_LIND': nmode_by_lind,
        'CORR_PARAM': param_noise1,
    }

    with pytest.warns(UserWarning, match='PRE_CALCULATED noise does not support adaptive mode'):
        noise = HopsNoise(noise_param_adaptive, noise_corr)
    assert noise.param['ADAPTIVE'] is False


# ------------------------------------------------------------
# TEST: FFT_FILTER ndarray SEED value comparison
# ------------------------------------------------------------

def test_prepare_noise_fft_ndarray_seed_value_comparison():
    """
    Tests that FFT_FILTER noise generated from an integer seed and then
    re-generated from the stored Z_UNCORRELATED ndarray produces identical
    correlated noise.
    """
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    t_axis = np.arange(0, 11.0, 1.0)

    # Generate noise from integer seed, storing raw uncorrelated noise
    noise_param_int = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'STORE_RAW_NOISE': True,
    }
    noise_int = HopsNoise(noise_param_int, noise_corr)
    Z2_from_int = noise_int.get_noise(t_axis, list(range(sys_param['N_L2'])))
    z_uncorrelated = noise_int.param['Z_UNCORRELATED']

    # Re-generate noise using the uncorrelated array as SEED
    noise_param_arr = {
        'SEED': z_uncorrelated,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
    }
    noise_arr = HopsNoise(noise_param_arr, noise_corr)
    Z2_from_arr = noise_arr.get_noise(t_axis, list(range(sys_param['N_L2'])))

    np.testing.assert_allclose(Z2_from_arr, Z2_from_int, atol=1e-6)


# ------------------------------------------------------------
# TEST: FFT_FILTER ndarray SEED + ADAPTIVE warning
# ------------------------------------------------------------

def test_prepare_noise_fft_ndarray_seed_adaptive_warning(capsys):
    """
    Tests that FFT_FILTER with an ndarray SEED and ADAPTIVE=True prints a
    warning about bypassing adaptive subsetting, and that _list_activel2idx_abs covers
    all L-operators despite requesting only a subset.
    """
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }

    # Build an ndarray SEED of the correct shape for _prepare_rand
    n_taus = len(np.arange(0, 11.0, 1.0))
    z_seed = np.ones((sys_param['N_L2'], 2 * (n_taus - 1)), dtype=np.complex128)

    noise_param_adaptive = {
        'SEED': z_seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'ADAPTIVE': True,
    }
    noise = HopsNoise(noise_param_adaptive, noise_corr)
    noise._prepare_noise([0])

    out, _ = capsys.readouterr()
    assert 'Warning: ADAPTIVE is True but SEED is an array' in out
    assert noise._list_activel2idx_abs == list(range(sys_param['N_L2']))


# ------------------------------------------------------------
# TEST: PRE_CALCULATED list SEED handling
# ------------------------------------------------------------

def test_prepare_noise_precalc_list_seed_accepted_and_converted():
    """
    Tests that PRE_CALCULATED with a Python list SEED (correct shape) is
    accepted and converted to a numpy array.
    """
    n_taus = len(np.arange(0, 11.0, 1.0))
    num_lop = sys_param['N_L2']
    noise_data = np.arange(num_lop * n_taus, dtype=np.complex128).reshape(
        num_lop, n_taus)
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': num_lop,
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise_param_list = {
        'SEED': noise_data.tolist(),
        'MODEL': 'PRE_CALCULATED',
        'TLEN': 10.0,
        'TAU': 1.0,
    }
    noise = HopsNoise(noise_param_list, noise_corr)
    noise._prepare_noise([0, 1])
    assert isinstance(noise.param['SEED'], np.ndarray)
    assert noise.param['SEED'].dtype == np.complex64


def test_prepare_noise_precalc_list_seed_wrong_shape_rejected():
    """
    Tests that PRE_CALCULATED with a Python list SEED of wrong shape raises
    UnsupportedRequest during _prepare_noise.
    """
    wrong_shape_data = [[1.0, 2.0, 3.0]]
    noise_param_list = {
        'SEED': wrong_shape_data,
        'MODEL': 'PRE_CALCULATED',
        'TLEN': 10.0,
        'TAU': 1.0,
    }
    noise_corr = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_list, noise_corr)
    with pytest.raises(UnsupportedRequest, match='array of the wrong length'):
        noise._prepare_noise([0, 1])


# ============================================================
# TEST SUITE: get_noise() gap coverage
# ============================================================

# ------------------------------------------------------------
# TEST: Out-of-range t_axis raises UnsupportedRequest
# ------------------------------------------------------------

def test_get_noise_t_axis_below_range():
    """
    Tests that get_noise raises UnsupportedRequest when t_axis contains
    values below min(T_AXIS).
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)

    # This case tests that a negative time triggers the out-of-range error
    with pytest.raises(UnsupportedRequest, match='t-samples outside of the defined '
                                                  't-axis'):
        noise.get_noise([-1.0, 0.0, 1.0])


def test_get_noise_t_axis_above_range():
    """
    Tests that get_noise raises UnsupportedRequest when t_axis contains
    values above max(T_AXIS).
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)

    # This case tests that a time beyond TLEN triggers the out-of-range error
    with pytest.raises(UnsupportedRequest, match='t-samples outside of the defined '
                                                  't-axis'):
        noise.get_noise([9.0, 10.0, 11.0])


# ------------------------------------------------------------
# TEST: Out-of-range t_axis with windowing active
# ------------------------------------------------------------

def test_get_noise_t_axis_below_range_windowed():
    """
    Tests that get_noise raises UnsupportedRequest for below-range t values
    even when NOISE_WINDOW is active and the window has been initialized.
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'NOISE_WINDOW': 5.0,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)

    # This case tests that below-range times are caught after window creation
    with pytest.raises(UnsupportedRequest, match='t-samples outside of the defined '
                                                  't-axis'):
        noise.get_noise([-1.0, 0.0, 1.0])


def test_get_noise_t_axis_above_range_windowed():
    """
    Tests that get_noise raises UnsupportedRequest for above-range t values
    even when NOISE_WINDOW is active and the window has been re-created.
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'NOISE_WINDOW': 5.0,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)

    # This case initializes the window with a valid call first
    noise.get_noise([0.0, 1.0])

    # This case tests that above-range times are caught after window re-creation
    with pytest.raises(UnsupportedRequest, match='t-samples outside of the defined '
                                                  't-axis'):
        noise.get_noise([9.0, 10.0, 11.0])


# ------------------------------------------------------------
# TEST: Off-axis t-samples raise UnsupportedRequest
# ------------------------------------------------------------

def test_get_noise_off_axis_t_samples():
    """
    Tests that get_noise raises UnsupportedRequest when INTERPOLATE=False
    and t_axis contains values that do not align with T_AXIS grid points.
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)

    # This case tests that a mid-step time value is rejected
    with pytest.raises(UnsupportedRequest, match='Off axis t-samples'):
        noise.get_noise([0.0, 0.5, 1.0])


def test_get_noise_off_axis_t_samples_windowed():
    """
    Tests that get_noise raises UnsupportedRequest for off-axis t values
    when NOISE_WINDOW is active.
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'NOISE_WINDOW': 5.0,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)

    # This case tests off-axis rejection after window initialization
    with pytest.raises(UnsupportedRequest, match='Off axis t-samples'):
        noise.get_noise([0.0, 0.5, 1.0])


# ------------------------------------------------------------
# TEST: Oversized NOISE_WINDOW falls back to full axis
# ------------------------------------------------------------

def test_get_noise_oversized_noise_window():
    """
    Tests that when NOISE_WINDOW exceeds max(T_AXIS), get_noise behaves
    identically to NOISE_WINDOW=None (full time axis is used).
    """
    noise_param_no_window = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
    }
    noise_param_big_window = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'NOISE_WINDOW': 9999.0,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise_no_win = HopsNoise(noise_param_no_window, noise_corr_local)
    noise_big_win = HopsNoise(noise_param_big_window, noise_corr_local)
    t_axis = np.arange(0, 11.0, 1.0)

    Z2_no_win = noise_no_win.get_noise(t_axis)
    Z2_big_win = noise_big_win.get_noise(t_axis)

    # This case tests that the noise values match
    np.testing.assert_allclose(Z2_big_win, Z2_no_win, atol=1e-10)

    # This case tests that the windowed axis covers the full T_AXIS
    assert np.allclose(noise_big_win.t_ax_windowed, noise_big_win.param['T_AXIS'])


# ------------------------------------------------------------
# TEST: Non-adaptive list_l2idx_abs subset (non-interpolated)
# ------------------------------------------------------------

def test_get_noise_list_lop_subset_non_adaptive():
    """
    Tests that passing a subset of list_l2idx_abs in non-adaptive mode returns
    only the requested L-operators' noise.
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': False,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)
    t_axis = np.arange(0, 11.0, 1.0)

    Z2_all = noise.get_noise(t_axis)
    Z2_lop0 = noise.get_noise(t_axis, list_l2idx_abs=[0])
    Z2_lop1 = noise.get_noise(t_axis, list_l2idx_abs=[1])

    # This case tests that requesting lop 0 returns only row 0
    assert Z2_lop0.shape == (1, len(t_axis))
    np.testing.assert_allclose(Z2_lop0[0, :], Z2_all[0, :], atol=1e-10)

    # This case tests that requesting lop 1 returns only row 1
    assert Z2_lop1.shape == (1, len(t_axis))
    np.testing.assert_allclose(Z2_lop1[0, :], Z2_all[1, :], atol=1e-10)


# ------------------------------------------------------------
# TEST: Non-adaptive list_l2idx_abs subset (interpolated)
# ------------------------------------------------------------

def test_get_noise_list_lop_subset_non_adaptive_interpolated():
    """
    Tests that passing a subset of list_l2idx_abs in non-adaptive interpolated
    mode returns only the requested L-operators' noise.
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': True,
        'ADAPTIVE': False,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)
    t_axis = [0.0, 0.5, 1.0, 1.5, 2.0]

    Z2_all = noise.get_noise(t_axis)
    Z2_lop0 = noise.get_noise(t_axis, list_l2idx_abs=[0])
    Z2_lop1 = noise.get_noise(t_axis, list_l2idx_abs=[1])

    # This case tests that requesting lop 0 returns only row 0
    assert Z2_lop0.shape == (1, len(t_axis))
    np.testing.assert_allclose(Z2_lop0[0, :], Z2_all[0, :], atol=1e-10)

    # This case tests that requesting lop 1 returns only row 1
    assert Z2_lop1.shape == (1, len(t_axis))
    np.testing.assert_allclose(Z2_lop1[0, :], Z2_all[1, :], atol=1e-10)


# ============================================================
# TEST SUITE: _noise_to_array()
# ============================================================

# ------------------------------------------------------------
# TEST: Adaptive + FLAG_REAL returns real noise
# ------------------------------------------------------------

def test_noise_to_array_adaptive_flag_real():
    """
    Tests that get_noise returns purely real noise when ADAPTIVE=True
    and FLAG_REAL=True, exercising the np.real(_noise_to_array(...))
    path.
    """
    random_seed = 3333
    noise_param_real = {
        'SEED': random_seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': True,
        'FLAG_REAL': True,
    }
    noise_param_complex = {
        'SEED': random_seed,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': True,
        'FLAG_REAL': False,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    t_axis = np.arange(0, 11.0, 1.0)

    noise_real = HopsNoise(noise_param_real, noise_corr_local)
    noise_complex = HopsNoise(noise_param_complex, noise_corr_local)

    Z2_real = noise_real.get_noise(t_axis, [0, 1])
    Z2_complex = noise_complex.get_noise(t_axis, [0, 1])

    # This case tests that FLAG_REAL=True returns real part of complex noise
    np.testing.assert_allclose(Z2_real, np.real(Z2_complex), atol=1e-6)

    # This case tests that the result is purely real
    assert np.all(np.imag(Z2_real) == 0)


# ------------------------------------------------------------
# TEST: Output dtype is complex64
# ------------------------------------------------------------

def test_noise_to_array_dtype_complex64():
    """
    Tests that _noise_to_array returns an array with dtype np.complex64
    for both adaptive and non-adaptive modes.
    """
    noise_param_non_adaptive = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': False,
    }
    noise_param_adaptive = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': True,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    Z2_noise = np.array([[1+1j, 2+2j, 3+3j],
                          [4+4j, 5+5j, 6+6j]], dtype=np.complex128)

    # This case tests non-adaptive mode returns complex64
    noise_na = HopsNoise(noise_param_non_adaptive, noise_corr_local)
    result_na = noise_na._noise_to_array(Z2_noise, [0, 2], [0, 1])
    assert result_na.dtype == np.complex64

    # This case tests adaptive mode returns complex64. list_l2idx_abs is omitted
    # because adaptive mode does not use it for row selection.
    noise_a = HopsNoise(noise_param_adaptive, noise_corr_local)
    result_a = noise_a._noise_to_array(Z2_noise, [0, 2])
    assert result_a.dtype == np.complex64


# ------------------------------------------------------------
# TEST: Non-adaptive slicing selects correct rows and columns
# ------------------------------------------------------------

def test_noise_to_array_non_adaptive_slicing():
    """
    Tests that _noise_to_array with ADAPTIVE=False selects the correct
    L-operator rows and time columns from the noise array.
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': False,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)

    Z2_noise = np.array([[10+1j, 20+2j, 30+3j, 40+4j],
                          [50+5j, 60+6j, 70+7j, 80+8j],
                          [90+9j, 100+10j, 110+11j, 120+12j]],
                         dtype=np.complex128)

    # This case tests that list_l2idx_abs selects the correct rows and t_axis
    # selects the correct columns
    result = noise._noise_to_array(Z2_noise, [0, 2], [0, 2])
    expected = np.complex64(np.array([[10+1j, 30+3j],
                                       [90+9j, 110+11j]]))
    np.testing.assert_allclose(result, expected, atol=1e-6)

    # This case tests single L-operator selection
    result_single = noise._noise_to_array(Z2_noise, [1, 3], [1])
    expected_single = np.complex64(np.array([[60+6j, 80+8j]]))
    np.testing.assert_allclose(result_single, expected_single, atol=1e-6)


# ------------------------------------------------------------
# TEST: Adaptive slicing returns all rows for selected columns
# ------------------------------------------------------------

def test_noise_to_array_adaptive_slicing():
    """
    Tests that _noise_to_array with ADAPTIVE=True returns all rows
    for the selected time columns without requiring list_l2idx_abs. In
    adaptive mode, self._noise is already pruned to only the active
    L-operators (via _prepare_noise and _evict_noise), so list_l2idx_abs
    is unnecessary.
    """
    noise_param_local = {
        'SEED': 0,
        'MODEL': 'FFT_FILTER',
        'TLEN': 10.0,
        'TAU': 1.0,
        'INTERPOLATE': False,
        'ADAPTIVE': True,
    }
    noise_corr_local = {
        'CORR_FUNCTION': sys_param['ALPHA_NOISE1'],
        'N_L2': sys_param['N_L2'],
        'LIND_BY_NMODE': sys_param['L_IND_BY_NMODE1'],
        'NMODE_BY_LIND': sys_param['NMODE1_BY_LIND'],
        'CORR_PARAM': sys_param['PARAM_NOISE1'],
    }
    noise = HopsNoise(noise_param_local, noise_corr_local)

    Z2_noise = np.array([[10+1j, 20+2j, 30+3j, 40+4j],
                          [50+5j, 60+6j, 70+7j, 80+8j],
                          [90+9j, 100+10j, 110+11j, 120+12j]],
                         dtype=np.complex128)

    # This case tests that all rows are returned when list_l2idx_abs is not
    # provided, since adaptive mode does not need it for row selection.
    result = noise._noise_to_array(Z2_noise, [0, 2])
    expected = np.complex64(np.array([[10+1j, 30+3j],
                                       [50+5j, 70+7j],
                                       [90+9j, 110+11j]]))
    np.testing.assert_allclose(result, expected, atol=1e-6)

    # This case tests that passing list_l2idx_abs in adaptive mode has no
    # effect -- the result is identical whether or not it is provided.
    result_with_lop = noise._noise_to_array(Z2_noise, [0, 2], [2])
    np.testing.assert_allclose(result, result_with_lop, atol=1e-6)
