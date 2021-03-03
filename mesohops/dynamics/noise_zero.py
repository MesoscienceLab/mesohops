import numpy as np
from mesohops.dynamics.hops_noise import HopsNoise
from mesohops.util.exceptions import LockedException

__title__ = "mesohops Noise"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"

ZERO_DICT_DEFAULT = {}
ZERO_DICT_TYPE = {}


class ZeroNoise(HopsNoise):
    """
    This is a class that describes the noise function for a calculation.

    INPUTS:
    -------
    1. noise_param : A dictionary that defines the noise trajectory for
                     the calculation.
                     * SEED:   an integer valued seed (or None). The noise
                               trajectoriesfor all modes is defined by one seed.
                     * MODEL:  The name of the noise model to be used. Allowed
                               names include: 'FFT_FILTER'
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
    2. get_noise(t_axis) :  A function that returns the noise terms
                            for each t in t_axis.

    NOTE: This class is only well defined within the context of a specific
    calculation where the system-bath parameters have been set.
    YOU SHOULDN'T INSTANTIATE ONE OF THESE CLASSES YOURSELF.
    IF YOU FEEL THE NEED TO, YOU ARE PROBABLY DOING SOMETHING STRANGE.
    """

    def __init__(self, noise_param, noise_corr):
        self._default_param, self._param_types = self._prepare_default(
            ZERO_DICT_DEFAULT, ZERO_DICT_TYPE
        )
        super().__init__(noise_param, noise_corr)
        self.n_lop = self.param["N_L2"]

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

        pass

    def get_noise(self, t_axis):
        """
        This is a function that simply returns a zero array of the correct length.

        PARAMETERS
        ----------
        1. taxis : list
                   list of time points

        RETURNS
        -------
        1. list_zeros : list
                        a list of zeros
        """
        return np.zeros([self.n_lop, len(t_axis)])

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
