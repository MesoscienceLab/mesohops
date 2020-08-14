import numpy as np
from mesohops.util.exceptions import UnsupportedRequest
from abc import ABC, abstractmethod
from mesohops.util.physical_constants import precision  # constant

__title__ = "Noise Trajectories"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"


class NoiseTrajectory(ABC):
    """
    This is the abstract base class for Noise objects.

    A noise object has two guaranteed functions:
    - get_noise(t_axis)
    - get_taxis()

    """

    def __init__(self):
        pass

    @abstractmethod
    def get_noise(self, t_axis):
        pass

    @abstractmethod
    def get_taxis(self):
        pass


class NumericNoiseTrajectory(NoiseTrajectory):
    """
    This is the class for noise that is explicitly calculated and does not support
    interpolation.
    """

    def __init__(self, noise, t_axis):
        self.__t_axis = t_axis
        self.__noise = noise

    def get_noise(self, taxis_req):
        """
        This function simply returns the noise values for the selected times.

        NOTE: INTERPOLATION SHOULD BE IMPLEMENTED BY DEFAULT. USE FCSPLINE
              FROM RICHARD TO DO IT!

        PARAMETERS
        ----------
        1. taxis_req : list
                       a list of requested time points

        RETURNS
        -------
        1. noise : list
                   a list of list of noise at the requested time points
        """
        # Check that to within 'precision' resolution, all timesteps
        # requested are present on the calculated t-axis.
        it_list = []
        for t in taxis_req:
            test = np.abs(self.__t_axis - t) < precision
            if np.sum(test) == 1:
                it_list.append(np.where(test)[0][0])
            else:
                raise UnsupportedRequest(
                    "Off axis t-samples",
                    "when INTERP = False in the NoiseModel._get_noise()",
                )

        return self.__noise[:, np.array(it_list)]

    def get_taxis(self):
        return self.__t_axis
