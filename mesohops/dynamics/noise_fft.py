import numpy as np
from scipy.interpolate import interp1d
from mesohops.dynamics.hops_noise import HopsNoise
from mesohops.util.exceptions import UnsupportedRequest, LockedException

__title__ = "pyhops noise"
__author__ = "D. I. G. Bennett, B. Citty, J. K. Lynd"
__version__ = "1.2"

FFT_FILTER_DICT_DEFAULT = {'STORE_RAW_NOISE': False}
FFT_FILTER_DICT_TYPE = {'STORE_RAW_NOISE': [type(False)]}


class FFTFilterNoise(HopsNoise):
    """
    This is a class that describes the noise function for a calculation.

    NOTE: This class is only well defined within the context of a specific
    calculation where the system-bath parameters have been set.
    YOU SHOULDN'T INSTANTIATE ONE OF THESE CLASSES YOURSELF.
    IF YOU FEEL THE NEED TO, YOU ARE PROBABLY DOING SOMETHING STRANGE.
    """

    def __init__(self, noise_param, noise_corr):
        """
        Inputs
        ------
        1. noise_param : A dictionary that defines the noise trajectory for
                     the calculation.
                     * SEED:   an integer valued seed (or None). The noise
                              trajectories for all modes are defined by one seed.
                     * MODEL:  The name of the noise model to be used. Allowed
                              names include: 'FFT_FILTER', 'ZERO'
                     * TLEN:   The length of the time axis. Units: fs
                     * TAU:    The smallest timestep used for direct noise
                              calculations.
                     * INTERP: Boolean. If True, then off-grid calls for noise
                              values will be determined by interpolation.
                              Allowed values: False [True not implemented!]
        2. noise_corr : dict
                        A dictionary that defines the noise correlation function for
                        the calculation.
                    ===================  USER INPUTS ====================
                    * CORR_FUNCTION:  A pointer to the function that defines alpha(t)
                                      for the noise term given CORR_PARAM[i] inputs
                    * N_L2:          The number of L operators (and hence number of
                                      final z(t) trajectories)
                    * L_IND_BY_NMODE: The L-indices associated with each CORR_PARAM
                    * CORR_PARAM:     The parameters that define the components of
                                      alpha(t)
        """
        self._default_param, self._param_types = self._prepare_default(
            FFT_FILTER_DICT_DEFAULT, FFT_FILTER_DICT_TYPE
        )
        super().__init__(noise_param, noise_corr)

    def prepare_noise(self):
        """
        This function is defined for each specific noise model (children classes of
        HopsNoise class) and provides the specific rules for calculating a noise
        trajectory using.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Check for locked status
        # -----------------------
        if self.__locked__:
            raise LockedException("NoiseModel.prepare_noise()")

        # Initialize uncorrelated noise
        # -----------------------------
        z_uncorrelated = self._prepare_rand()

        # Initialize correlated noise noise
        # ---------------------------------
        alpha = np.complex64(self._corr_func_by_lop_taxis(self.param['T_AXIS']))
        z_correlated = self._construct_correlated_noise(alpha, z_uncorrelated)

        # Remove 'Z_UNCORRELATED' for memory savings
        if self.param['STORE_RAW_NOISE']:
            self.param['Z_UNCORRELATED'] = z_uncorrelated

        if self.param['INTERPOLATE']:
            self._noise = interp1d(self.param['T_AXIS'], z_correlated, kind='cubic',axis=1)
        else:
            self._noise = np.complex64(z_correlated)

        # Lock Noise Instance
        # -------------------
        # A noise model has been explicitly calculated. All further modifications
        # to this class should be blocked.
        self.__locked__ = True

    def _prepare_rand(self):
        """
        Constructs the uncorrelated complex gaussian distributions that
        define the noise trajectory. Average of this uncorrelated noise trajectory is 0,
        and average of the absolute values of is sqrt(pi)/2 (assuming an arbitrarily
        long trajectory).

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        # Basic Calculation Parameters
        # ----------------------------
        n_lop = self.param["N_L2"]
        ntaus = len(self.param['T_AXIS'])

        # Initialize un-correlated noise
        # ------------------------------
        if (type(self.param['SEED']) is list) or (
                type(self.param['SEED']) is np.ndarray):
            print('Noise Model initialized from input array.')
            # This is where we need to write the code to use an array of of random noise variables input in place of
            # the SEED parameter.
            if len(self.param['SEED'][0]) == 2 * (len(self.param['T_AXIS']) - 1):
                return self.param['SEED']
            else:
                raise UnsupportedRequest('Noise.param[SEED] is an array of the wrong length',
                    'Noise._prepare_rand')

        elif type(self.param["SEED"]) is str:
            print("Noise Model intialized from file: {}".format(self.param['SEED']))
            # This is where we need to write a function that imports the uncorrelated noise trajectory.
            # Question: What would be the best interface for this?
            raise UnsupportedRequest(
                'Noise.param[SEED] of type {} not supported'.format(
                    type(self.param['SEED'])),
                'Noise._prepare_rand')

        elif (type(self.param['SEED']) is int) or (self.param['SEED'] is None):
            print("Noise Model initialized with SEED = ", self.param["SEED"])

            # Prepare Random State
            # --------------------
            randstate = np.random.RandomState(seed=self.param['SEED'])

            # Box-Muller Method: Gaussian Random Number
            # ---------------------------------------------
            # We use special indexing to ensure that changes in the length of the time axis does not change the
            # random numbers assigned to each time point for a fixed seed value.
            # Z_t = (G1 + 1j*G2)/sqrt(2) = sqrt(-ln(g))*exp(2j*pi*phi)
            # G1 = sqrt(-2*ln(g))*Cos(2*pi*phi)
            # G2 = sqrt(-2*ln(g))*Sin(2*pi*phi)
            random_numbers = randstate.rand(4 * (ntaus - 1), n_lop).T
            g_index, phi_index = self._construct_indexing(ntaus)
            g_rands = random_numbers[:, np.array(g_index)]
            phi_rands = random_numbers[:, np.array(phi_index)]
            return np.complex64(
                np.sqrt(-2.0 * np.log(g_rands)) * np.exp(2.0j * np.pi * phi_rands))
        else:
            raise UnsupportedRequest('Noise.param[SEED] of type {} not supported'.format(type(self.param['SEED'])),
                'Noise._prepare_rand')

    @staticmethod
    def _construct_correlated_noise(c_t, z_t):
        """
        Calculates a noise trajectory using the bath correlation function (c_t).
        This function is based on the description given by:

        "Exact simulation of complex-valued
        Gaussian stationary Processes via circulant embedding."
        Donald B. Percival Signal Processing 86, p. 1470-1476 (2006)
        [Section 3, p. 5-7]

        Parameters
        ----------
        1. c_t : list
                 List of correlation function sampled at specific time points.

        2. z_t : list
                  List of complex-valued, uncorrelated random noise
                  trajectories with real and imaginary components at
                  each time having mean 0 and variance 1
                  [Re(z_t) ~ N(0,1), Im(z_t) ~ N(0,1)].


        Returns
        -------
        1. corr_noise : list
                        List of correlated noise trajectory
        """
        # Rotate stochastic process such that final entry is real
        # -------------------------------------------------------
        list_angles = np.array(np.angle(c_t), dtype=np.float64)
        list_nu = list_angles[:, -1]/((len(c_t[0, :]) - 1.0) * 2.0 * np.pi)
        E2_expmatrix = np.exp(-1.0j * 2.0 * np.pi * np.outer(list_nu,np.arange(len(c_t[0, :]))))
        tildec_t = np.abs(c_t) * np.exp(1.0j * list_angles) * E2_expmatrix

        # Construct the embedding
        # -----------------------
        
        s_w = np.real(np.fft.fft(np.array(
            np.concatenate([tildec_t, np.conj(np.flip(tildec_t[:, 1:-1], axis=1))], axis=1)),
                                 axis=1))

        # Check that the embedding is positive semidefinite
        # -------------------------------------------------
        if np.min(s_w) < 0:
            print('WARNING: circulant embedding is NOT positive semidefinite.')
            print('* Negative values will be set to 0.')
            print(f'max negative: {np.min(s_w)}')
            print(f'fractional negative: {-np.min(s_w) / np.max(s_w)}')
            print(f'negative number: {len(np.where(s_w < 0)[1])}')
            # Set negative entries to zero to obtain approximate result
            s_w[np.where(s_w < 0)] = 0
        
        # Construct the frequency domain components
        # -----------------------------------------
        # To allow for controlling noise trajectories in specific time windows,
        # we take our uncorrelated noise in the time axis, convert to the frequency
        # domain, and then proceed with the standard algorithm.
        # The Fourier transform of Gaussian noise with a std. dev. (\sigma)
        # has a std. dev. of \sigma*\sqrt(N/2) where N is the length of the trajectory.
        # As a result, the equation from the paper is slightly modified below.
        # Stack Exchange: https://dsp.stackexchange.com/questions/24170/what-are-the-statistics-of-the-discrete-fourier-transform-of-white-gaussian-nois
        z_w = np.fft.fft(z_t, axis=1)
        
        tildey_t = np.fft.ifft(np.array(np.abs(z_w) * np.exp(1.0j * np.angle(z_w))
                                        * np.sqrt(s_w/2.0)), axis=1)[:, :len(c_t[0, :])]

        # Undo initial phase rotation
        # ---------------------------
        return (np.abs(tildey_t) * np.exp(1.0j * np.angle(tildey_t)) * np.conj(E2_expmatrix))

    @staticmethod
    def _construct_indexing(ntaus):
        """
        Constructs a self-consistent indexing scheme that controls assignment of
        random numbers to different time points.

        Parameters
        ----------
        1. ntaus : int
                   Length of the noise time-axis (that is, number of time points).

        Returns
        -------
        1. g_index : list(int)
                     List of indices where the Gaussian-determined norm of an
                     uncorrelated random noise point is placed.

        2. phi_index : list(int)
                       List of indices where the phase of an uncorrelated random
                       noise point is placed.
        """
        # Construct indexing scheme
        # -------------------------
        # We want a scheme that keeps a consistent assignment of the random numbers to different time points
        # even as the length of the total time axis changes.
        ntaus_m1 = ntaus - 1
        # simple test: [0, 2, 6, 10, 7, 3]
        g_index = [0]
        g_index.extend(np.arange(2, 4 * (ntaus - 2), 4))
        g_index.append(4 * ntaus_m1 - 2)
        g_index.extend(np.arange(4 * (ntaus - 2), 3, -4))
        # simple test: [1,4,8,11, 9,5]
        phi_index = [1]
        phi_index.extend(np.arange(3, 4 * (ntaus - 1) - 1, 4))
        phi_index.append(4 * ntaus_m1 - 1)
        phi_index.extend(np.arange(4 * (ntaus_m1 - 1) + 1, 4, -4))
        return g_index, phi_index

    @property
    def param(self):
        return self.__param

    @param.setter
    def param(self, param_usr):
        if self.__locked__:
            raise LockedException("NoiseModel.param.setter")
        self.__param = self._initialize_dictionary(
            param_usr, self._default_param, self._param_types, type(self).__name__
        )

    def update_param(self, param_usr):
        if self.__locked__:
            raise LockedException("NoiseModel.update_param()")
        self.__param.update(param_usr)
