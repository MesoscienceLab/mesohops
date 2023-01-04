import copy
import numpy as np
from mesohops.util.dynamic_dict import Dict_wDefaults
from mesohops.util.exceptions import LockedException, UnsupportedRequest
from mesohops.util.physical_constants import precision  # constant

__title__ = "Pyhops Noise"
__author__ = "D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"

# NOISE MODELS:
# =============


NOISE_DICT_DEFAULT = {
    "SEED": None,
    "MODEL": "FFT_FILTER",
    "TLEN": 1000.0,  # Units: fs
    "TAU": 1.0,  # Units: fs,
    "INTERPOLATE": False
}

NOISE_TYPE_DEFAULT = {
    "SEED": [int, type(None), str, np.ndarray],
    "MODEL": [str],
    "TLEN": [float],
    "TAU": [float],
    "INTERPOLATE": [bool]
}


class HopsNoise(Dict_wDefaults):
    """
    This is the BaseClass for defining a hops noise trajectory. All noise classes
    will inherit from here. Anything that defines the input-output structure of a
    noise trajectory should be controlled from this class - rather than any of the
    model-specific child classes.


    NOTE: This class is only well defined within the context of a specific
    calculation where the system-bath parameters have been set.
    YOU SHOULDN'T INSTANTIATE ONE OF THESE CLASSES YOURSELF.
    IF YOU FEEL THE NEED TO, YOU ARE PROBABLY DOING SOMETHING STRANGE.
    """

    def __init__(self, noise_param, noise_corr):
        """
        Initializes the HopsNoise object with the parameters that it will use to
        construct the noise.

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

        Returns
        -------
        None
        """
        # In order to ensure that each NoiseModel instance is used to
        # calculate precisely one trajectory, there is a __locked__
        # property that tracks when the NoiseModel actually calculates
        # a noise trajectory. Before the noise trajectory is calculated,
        # the user is free to play with parameters, etc. After a noise
        # trajectory is calculated the class instance is locked.
        #
        # Only play with this parameter if you know what you are doing.
        self.__locked__ = False
        if type(self) == HopsNoise:
            self._default_param, self._param_types = self._prepare_default(
            NOISE_DICT_DEFAULT, NOISE_TYPE_DEFAULT
            )
        # Initialize the noise and system dictionaries
        self.param = noise_param
        self.update_param(noise_corr)
        nstep_min = int(np.ceil(self.param["TLEN"] / self.param["TAU"])) + 1
        t_axis = np.arange(nstep_min) * self.param["TAU"]
        self.param["T_AXIS"] = t_axis

    def _corr_func_by_lop_taxis(self, t_axis):
        """
        Calculates the correlation function for each L operator by combining all the
        system-bath components that have the same L-operator.

        Parameters
        ----------
        1. t_axis : list
                    List of time points.

        Returns
        -------
        1. alpha : List
                   List of lists of correlation functions evaluated at each time point.
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
        Gets the noise.

        Parameters
        ----------
        1. t_axis : list
                    List of time points.

        Returns
        -------
        1. noise : list
                   List of lists of noise values sampled at the given time points.
        """
        if not self.__locked__:
            self.prepare_noise()

        if not self.param["INTERPOLATE"]:
            it_list = []
            for t in t_axis:
                test = np.abs(self.param["T_AXIS"] - t) < precision
                if np.sum(test) == 1:
                    it_list.append(np.where(test)[0][0])
                else:
                    raise UnsupportedRequest(
                        "Off axis t-samples when INTERPOLATE = False",
                        "NoiseModel.get_noise()",
                    )

            return self._noise[:, np.array(it_list)]
        else:
            return self._noise(t_axis)

    @staticmethod
    def _prepare_default(method_defaults, method_types):
        """
        Creates the full default parameter dictionary, by merging default dictionary of
        the base HopsNoise class with the child class

        Parameters
        ----------
        1. method_defaults : dict
                             Dictionary of default parameter values.

        2. method_types : dict
                          Dictionary of parameter types.

        Returns
        -------
        1. default_params : dict
                            Full default dictionary

        2. param_types : dict
                         Full default dictionary of parameter types
        """
        default_params = copy.deepcopy(NOISE_DICT_DEFAULT)
        default_params.update(method_defaults)
        param_types = copy.deepcopy(NOISE_TYPE_DEFAULT)
        param_types.update(method_types)
        return default_params, param_types

    def _unlock(self):
        self.__locked__ = False

    def _lock(self):
        self.__locked__ = True

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
