from mesohops.util.bath_corr_functions import *
from mesohops.util.physical_constants import kB

def test_bcf_convert_to_sdl():
    """
    Tests that bcf_convert_to_sdl returns the correct exponential in the correct format.
    """
    e_lambda = 100
    gamma = 20
    omega = 33
    temp = 300
    bcf_overdamped = bcf_convert_sdl_to_exp(e_lambda, gamma, 0, temp)
    overdamped_analytic = (complex(2*e_lambda*temp*kB - 1j*e_lambda*gamma),
                           complex(gamma))
    assert overdamped_analytic == bcf_overdamped
    bcf_underdamped = bcf_convert_sdl_to_exp(e_lambda, gamma, omega, temp)
    underdamped_analytic = (overdamped_analytic[0], gamma-1j*omega)
    assert underdamped_analytic == bcf_underdamped

def test_bcf_convert_dl_to_exp_with_Matsubara():
    """
    Tests that bcf_convert_dl_to_exp_with_Matsubara returns the correct list of
    exponentials in the correct format, and that it approaches the analytic limit of
    the hyperbolic cotangent real portion of the low-temperature mode.
    """
    e_lambda = 100
    gamma = 20
    temp = 300
    mats_const = np.pi*2*kB*temp
    # Tests that the 0-Matsubara mode limit reproduces the pure high temperature
    # approximation
    assert bcf_convert_dl_to_exp_with_Matsubara(e_lambda, gamma, temp, 0) == \
           list(bcf_convert_sdl_to_exp(e_lambda, gamma, 0, temp))
    # Tests that the function returns a list with 2 entries per mode
    kmats_10000 = bcf_convert_dl_to_exp_with_Matsubara(e_lambda, gamma, temp, 10000)
    assert len(kmats_10000) == 20002
    # Test that Matsubara modes are correct
    for n in range(3,len(kmats_10000),2):
        assert kmats_10000[n] == mats_const*(n-1)/2
    # Test that 10000 Matsubara modes properly returns the correct hyperbolic
    # cotangent real portion of the low-temperature mode's constant prefactor
    np.testing.assert_allclose(np.real(kmats_10000[0]), e_lambda*gamma/np.tan(
        gamma/kB/temp/2))
