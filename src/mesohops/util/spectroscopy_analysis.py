import numpy as np
from mesohops.util.physical_constants import h, c
from scipy import sparse

def zeropadded_fft(R_t, t_step, w_res=c*(10**-7)):
    """
    Helper function which performs a zero-padded Fast Fourier Transform (FFT) on a
    time-domain response function.

    Motivation: Zero-padding is used to increase the resolution of the frequency-domain
    response function. Therefore, when zero-padding is performed during dt convergence
    scans, the extent of zero-padding needs to be adjusted to ensure convergence is
    solely due to the time step, not as a consequence of signal processing technique.
    The frequency resolution in the fourier domain is given by 1/(N*t_step), where N is
    the number of time points in the zero-padded signal. As a result, the time points in
    the zero-padded signal should be chosen such that: N = 1/(w_res*t_step), where w_res
    is kept constant. Note that N must be a whole number, so zero-padding is only
    possible when the t_step of each calculation fulfills the condition t_step*w_res is
    the inverse of a positive integer.

    Defaults: The frequency resolution is defaulted to 2.9979e-5 fs^-1, which
    corresponds to 1 cm^-1.

    Use Cases: This function can be used on the time-domain response functions of either
    single trajectories OR ensemble averages.

    For more details on zero-padding, see:

    J. Hoch and A. Stern, NMR Data Processing (Wiley, 1996)

    Parameters
    ----------
    1. R_t : np.array(complex)
             Time-domain response function.

    2. w_res : float
               Resolution in the frequency domain [units: fs^-1].

    3. t_step : float
                Time step [units: fs].

    Returns
    -------
    1. R_w : np.array(float)
             The real portion of the response function in the frequency domain.

    2. w_axis : np.array(float)
                Frequency axis.
    """
    data_t_point_count = len(R_t)
    zeropad_t_point_count = int(1 / (w_res * t_step))
    # Ensure that the zeropadded signal has more time points than the data signal
    if zeropad_t_point_count < data_t_point_count:
        zeropad_t_point_count = data_t_point_count
        print("WARNING: Defined frequency resolution is less than inherent resolution "
              "from the time step. FFT will continue with resolution provided by time "
              "step.")
    R_t_zeropad = np.zeros(zeropad_t_point_count, dtype=complex)
    R_t_zeropad[:data_t_point_count] = R_t[:]

    # Scale the FFT for consistent amplitudes
    R_w = np.real(np.fft.fftshift(np.fft.fft(R_t_zeropad))) / zeropad_t_point_count
    w_axis = -np.fft.fftshift(h * np.fft.fftfreq(zeropad_t_point_count, t_step))
    return R_w, w_axis


def _response_function_calc(F_op, index_t, traj, list_response_norm_sq):
    '''
    Calculates the normalized response function component using a dyadic operator
    at each time point beyond a user-defined index in the trajectory with
    _response_function_calc from spectroscopy_analysis.py.

    Parameters
    ----------
    1. F_op : np.array(float)
              Dyadic operator to calculate the response function component.

    2. index_t : int
                 Time index after which the response function component is
                 calculated.

    3. traj : np.array(complex)
              Compelete trajectory in the shape [t_axis, n_site].

    4.list_response_norm_sq :  np.array(float)
                               List of normalization correction factors for each
                               preceding raising/lowering operator.

    Returns
    -------
    1. list_response_func_comp : np.array(complex)
                                 Calculated response function component for each
                                 time point beyond index_t.
    '''

    if sparse.issparse(F_op):
        F_op = F_op.tocsr()
    else:
        F_op = sparse.csr_array(F_op)

    if (sparse.issparse(traj)):
        traj_csr = sparse.csr_array(traj)
        # Transposed wave function for calculating inner product <psi|F_op|psi>.
        traj_t = traj_csr.transpose()

        return np.ravel([(np.prod(list_response_norm_sq) / (np.linalg.norm(traj_t[:, [col]].data) ** 2)) *
                               (np.conj(traj_t[:, [col]].T) @ (F_op @ traj_t[:, [col]])).todense()[0] for col in
                               range(int(index_t) + 1, traj_t.shape[1])])
    else:
        traj_t = np.array(traj).T
        return np.array([(np.prod(list_response_norm_sq) / (np.linalg.norm(traj_t[:, col]) ** 2)) *
                (np.conj(traj_t[:, col]).T @ (F_op @ traj_t[:, col])) for col in
                range(int(index_t) + 1, traj_t.shape[1])])
