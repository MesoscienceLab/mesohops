from abc import ABC, abstractmethod

import numpy as np
from scipy import interpolate

from mesohops.util.exceptions import UnsupportedRequest
from mesohops.util.physical_constants import precision  # constant

__title__ = "Noise Trajectories"
__author__ = "D. I. G. Bennett, J. K. Lynd"
__version__ = "1.2"


class NoiseTrajectory(ABC):
    """
    Abstract base class for Noise objects.

    A noise object has two guaranteed functions:
    - get_noise(t_axis)
    - get_taxis()

    """

    __slots__ = (
        # No slots needed
    )

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
    Defines explicitly calculated noise.
    """

    __slots__ = (
        # --- Time and noise data ---
        '_t_axis',     # Time axis
        '_noise',      # Noise data

        # --- Interpolation ---
        '_noise_interpolation'  # Interpolation function
    )

    def __init__(self, noise, t_axis, spline_interpolation=False):
        """
        Inputs
        ------
        1. noise : list(complex)
                   Noise trajectory [units: cm^-1].

        2. t_axis : list(float)
                    List of time points [units: fs].

        3. spline_interpolation : bool
                                  True indicates that off-grid calls for noise
                                  values will be determined by interpolation while
                                  False indicates otherwise (options: False).

        Returns
        -------
        None
        """
        self._t_axis = t_axis
        self._noise = noise
        if spline_interpolation:
            print("WARNING: spline interpolation of noise trajectories is untested")
            # If this is a list, it will register as True, which is important later.
            # If it is an array, it will throw an error.
            self._noise_interpolation = [interpolate.splrep(t_axis, state_noise)
                                          for state_noise in noise]
        else:
            self._noise_interpolation = False

    def get_noise(self, taxis_req):
        """
        Returns the noise values for the selected times.

        NOTE: INTERPOLATION IS CURRENTLY NOT IMPLEMENTED

        Parameters
        ----------
        1. taxis_req : list(float)
                       List of requested time points [units: fs].

        Returns
        -------
        1. noise : list(complex)
                   List of lists of noise at the requested time points [units: cm^-1].
        """
        # Check that to within 'precision' resolution, all timesteps
        # requested are present on the calculated t-axis.


        if not self._noise_interpolation:
            it_list = []
            for t in taxis_req:
                test = np.abs(self._t_axis - t) < precision
                if np.sum(test) == 1:
                    it_list.append(np.where(test)[0][0])
                else:
                    raise UnsupportedRequest(
                        "Off axis t-samples",
                        "when INTERPOLATE = False in the NoiseModel._get_noise()",
                    )

            return self._noise[:, np.array(it_list)]
        else:
            return np.array([[interpolate.splev(taxis_req, spline)] for spline in
                     self._noise_interpolation])

    def get_taxis(self):
        return self._t_axis
