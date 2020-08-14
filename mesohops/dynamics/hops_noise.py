import copy
import numpy as np
from mesohops.util.dynamic_dict import Dict_wDefaults
from mesohops.util.exceptions import LockedException

__title__ = "Pyhops Noise"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"

# NOISE MODELS:
# =============


NOISE_DICT_DEFAULT = {
    "SEED": None,
    "MODEL": "FFT_FILTER",
    "TLEN": 1000.0,  # Units: fs
    "TAU": 1.0,  # Units: fs,
}

NOISE_TYPE_DEFAULT = {
    "SEED": [type(int()), type(None)],
    "MODEL": [type(str())],
    "TLEN": [type(float())],
    "TAU": [type(float())],
}


class HopsNoise(Dict_wDefaults):
    """
    This is the BaseClass for defining a hops noise trajectory. All noise classes
    will inherit from here. Anything that defines the input-output structure of a
    noise trajectory should be controlled from this class - rather than any of the
    model-specific child classes.


    INPUTS:
    -------
    1. noise_param: A dictionary that defines the noise trajectory for
                    the calculation.
                    ===================  USER INPUTS ====================
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

                    ===================  CODE PARAMETERS ====================
                    * CORR_FUNCTION:  A pointer to the function that defines alpha(t)
                                      for the noise term given CORR_PARAM[i] inputs
                    * N_L2:          The number of L operators (and hence number of
                                      final z(t) trajectories)
                    * L_IND_BY_NMODE: The L-indices associated with each CORR_PARAM
                    * CORR_PARAM:     The parameters that define the components of
                                      alpha(t)



    Functions:
    ----------
    1. prepare_noise(): This function runs all the calculations needed
                        to initialize the noise trajectory and make it
                        accessible.
    2. get_noise(t_axis): A function that returns the noise terms
                              for each t in t_axis.



    NOTE: This class is only well defined within the context of a specific
    calculation where the system-bath parameters have been set.
    YOU SHOULDN'T INSTANTIATE ONE OF THESE CLASSES YOURSELF.
    IF YOU FEEL THE NEED TO, YOU ARE PROBABLY DOING SOMETHING STRANGE.
    """

    def __init__(self, noise_param, noise_corr):

        # In order to ensure that each NoiseModel instance is used to
        # calculate precisely one trajectory, there is a __locked__
        # property that tracks when the NoiseModel actually calculates
        # a noise trajectory. Before the noise trajectory is calculated,
        # the user is free to play with parameters, etc. After a noise
        # trajectory is calculated the class instance is locked.
        #
        # Only play with this parameter if you know what you are doing.
        self.__locked__ = False

        # Initialize the noise and system dictionaries
        self.param = noise_param
        self.update_param(noise_corr)

    def _corr_func_by_lop_taxis(self, t_axis):
        """
        This function calculates the correlation function for each L 
        operator by combining all of the system-bath components that 
        have the same L-operator.

        PARAMETERS
        ----------
        1. t_axis : list
                    list of time points

        RETURNS
        -------
        1. alpha : list
                   a list of list of correlation functions evaluated at each time point
        """
        alpha = np.zeros([self.param["N_L2"], len(t_axis)], dtype=np.complex128)
        for l_ind in set(self.param["LIND_BY_NMODE"]):
            i_mode_all = [
                i for (i, x) in enumerate(self.param["LIND_BY_NMODE"]) if x == l_ind
            ]
            for i in i_mode_all:
                alpha[l_ind, :] += self.param["CORR_FUNCTION"](
                    t_axis, *self.param["CORR_PARAM"][i]
                )
        return alpha

    def get_noise(self, t_axis):
        """
        Gets the noise

        PARAMETERS
        ----------
        1. t_axis : list
                    a list of time points

        RETURNS
        -------
        1. noise : list
                   a list of list of noise values sampled at the given time points
        """
        if not self.__locked__:
            self.prepare_noise()

        return self._noise.get_noise(t_axis)

    @staticmethod
    def _prepare_default(method_defaults, method_types):
        """
        Creates the full default parameter dictionary, by merging default dictionary of
        the base HopsNoise class with the child class

        PARAMETERS
        ----------
        1. method_defaults : dict
                             a dictionary of default parameter values
        2. method_types : dict
                          a dictionary of parameter types

        RETURNS
        -------
        1. default_params : dict
                            the full default dictionary
        2. param_types : dict
                         the full default dictionary of parameter types
        """
        default_params = copy.deepcopy(NOISE_DICT_DEFAULT)
        default_params.update(method_defaults)
        param_types = copy.deepcopy(NOISE_TYPE_DEFAULT)
        param_types.update(method_types)
        return default_params, param_types

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
