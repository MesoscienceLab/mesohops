import numpy as np
from mesohops.dynamics.hops_noise import HopsNoise
from mesohops.util.exceptions import LockedException

__title__ = "Pyhops Noise"
__author__ = "D. I. G. Bennett"
__version__ = "1.2"

ZERO_DICT_DEFAULT = {}
ZERO_DICT_TYPE = {}


class ZeroNoise(HopsNoise):
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
       1. noise_param : dict
                        A dictionary that defines the noise trajectory for
                        the calculation.
                    ===================  USER INPUTS ====================
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
            ZERO_DICT_DEFAULT, ZERO_DICT_TYPE
        )
        super().__init__(noise_param, noise_corr)
        self.n_lop = self.param["N_L2"]

    def prepare_noise(self):
        """
        This function is defined for each specific noise model (children classes of
        HopsNoise class) and provides the specific rules for calculating a noise
        trajectory using

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

        pass

    def get_noise(self, t_axis):
        """
        Returns a zero array of the correct length.

        Parameters
        ----------
        1. taxis : list
                   List of time points.

        Returns
        -------
        1. list_zeros : list
                        List of zeros.
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
