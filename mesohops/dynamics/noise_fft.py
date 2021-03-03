import numpy as np
from mesohops.dynamics.hops_noise import HopsNoise
from mesohops.dynamics.noise_trajectories import NumericNoiseTrajectory
from mesohops.util.exceptions import UnsupportedRequest, LockedException

__title__ = "mesohops noise"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"

FFT_FILTER_DICT_DEFAULT = {"DIAGONAL": True}
FFT_FILTER_DICT_TYPE = {"DIAGONAL": [type(True)]}


class FFTFilterNoise(HopsNoise):
    """
    This is a class that describes the noise function for a calculation.

    INPUTS
    ------
    1. noise_param : A dictionary that defines the noise trajectory for
                     the calculation.
                     * SEED:   an integer valued seed (or None). The noise
                              trajectoriesfor all modes is defined by one seed.
                     * MODEL:  The name of the noise model to be used. Allowed
                              names include: 'FFT_FILTER', 'ZERO'
                     * TLEN:   The length of the time axis. Units: fs
                     * TAU:    The smallest timestep used for direct noise
                              calculations.
                     * INTERP: Boolean. If True, then off-grid calls for noise
                              values will be determined by interpolation.
                              Allowed values: False [True not implemented!]
    2. system_param : This is the parameter dictionary that defines the system-
                      bath interactions. The following key words are required
                      for proper functioning:
                      * N_L2, LIND_BY_MODE, GW_SYSBATH, CORRELATION_FUNCTION
                      * For the definition of these key words, please see
                       hops_system.py

    Functions:
    ----------
    1. prepare_noise() : This function runs all the calculations needed
                        to initialize the noise trajectory and make it
                        accessible.
    2. get_noise(t_axis) : A function that returns the noise terms
                              for each t in t_axis.



    NOTE: This class is only well defined within the context of a specific
    calculation where the system-bath parameters have been set.
    YOU SHOULDN'T INSTANTIATE ONE OF THESE CLASSES YOURSELF.
    IF YOU FEEL THE NEED TO, YOU ARE PROBABLY DOING SOMETHING STRANGE.
    """

    def __init__(self, noise_param, noise_corr):
        self._default_param, self._param_types = self._prepare_default(
            FFT_FILTER_DICT_DEFAULT, FFT_FILTER_DICT_TYPE
        )
        super().__init__(noise_param, noise_corr)

    def prepare_noise(self):
        """
        This function is defined for each specific noise model (children classes of
        HopsNoise class) and provides the specific rules for calculating a noise
        trajectory using

        PARAMETERS
        ----------
        None

        RETURNS
        -------
        None
        """
        # Check for locked status
        # -----------------------
        if self.__locked__:
            raise LockedException("NoiseModel.prepare_noise()")

        # Basic Calculation Parameters
        # ----------------------------
        n_lop = self.param["N_L2"]
        nstep_min = int(np.ceil(self.param["TLEN"] / self.param["TAU"])) + 1

        # FFT-FILTER NOISE MODEL
        # ----------------------
        if self.param["DIAGONAL"]:
            t_axis = np.arange(nstep_min) * self.param["TAU"]
            self.param["T_AXIS"] = t_axis

            # Define correlation function for each L-operator
            alpha = self._corr_func_by_lop_taxis(t_axis)

            # Initialize noise
            etas = np.zeros((n_lop, nstep_min), dtype=np.complex128)
            if self.param["SEED"] is not None:
                print("Noise Model initialized with SEED = ", self.param["SEED"])
                rand_state = np.random.RandomState(seed=self.param["SEED"])

            for i in range(n_lop):
                if self.param["SEED"] is not None:
                    seed_i = rand_state.randint(0, 2 ** 30)
                    print(i, "th seed is: ", seed_i)
                else:
                    seed_i = None
                etas[i, :] = self.fft_filter_noise_diagonal(alpha[i, :], seed=seed_i)
            self._noise = NumericNoiseTrajectory(etas, t_axis)

            # A noise model has be explicitly calculated. All further modifications
            # to this class should be blocked.
            self.__locked__ = True
        else:
            raise UnsupportedRequest("Non-diagonal FFT Filter in", type(self).__name__)

    @staticmethod
    def fft_filter_noise_diagonal(s_zz, seed=None):
        """
        This function calculates a noise trajectory using the
        bath correlation function (s_zz). This function is
        based on the description given by:

        "Exact Simulation of Noncircular or Improper Complex-Valued
        Stationary Gaussian Processes using circulant embedding."
        Adam M. Sykulski and Donald B. Percival
        IEEE Internation Workship on Machine Learning for Signal
        Processing (2016)

        PARAMETERS
        ----------
        1. s_zz : list
                  correlation function sampled at specific time points
        2. seed : int
                  the seed for the random number generator

        RETURNS
        -------
        1. noise : list
                   the random noise trajectory sampled at the same time points as s_zz
        """
        randstate = np.random.RandomState(seed=seed)
        ntaus = len(s_zz)
        lxx = np.array(np.concatenate([s_zz, np.conj(np.flip(s_zz[1:-1], axis=0))]))
        sqrt_j = np.sqrt(np.real(np.fft.fft(lxx)) + 0j)
        ntaus_m1 = ntaus - 1
        ww = np.sqrt(-np.log(randstate.rand(2 * ntaus_m1))) * np.exp(
            2.0j * np.pi * randstate.rand(2 * ntaus_m1)
        )
        return np.fft.ifft(np.multiply(np.fft.fft(ww), sqrt_j))[:ntaus]

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
