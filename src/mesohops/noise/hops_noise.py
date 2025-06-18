import copy
import os

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

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
    "INTERPOLATE": False,
    "RAND_MODEL": "SUM_GAUSSIAN",  # SUM_GAUSSIAN or BOX_MULLER
    "STORE_RAW_NOISE": False,
    "NOISE_WINDOW": None,
    "ADAPTIVE": False
}

NOISE_TYPE_DEFAULT = {
    "SEED": [int, type(None), str, np.ndarray],
    "MODEL": [str],
    "TLEN": [float],
    "TAU": [float],
    "INTERPOLATE": [bool],
    "RAND_MODEL": [str],
    "STORE_RAW_NOISE": [type(False)],
    "NOISE_WINDOW": [type(None), type(1.0), type(1)],
    "ADAPTIVE": [bool]
}


class HopsNoise(Dict_wDefaults):
    """
    Defines and manages a HOPS noise trajectory. Allows multiple methods for the
    generation of both the uncorrelated and correlated noise to ensure reproducibility of
    HOPS calculations.


    NOTE: This class is only well defined within the context of a specific
    calculation where the system-bath parameters have been set.
    YOU SHOULDN'T INSTANTIATE ONE OF THESE CLASSES YOURSELF.
    IF YOU FEEL THE NEED TO, YOU ARE PROBABLY DOING SOMETHING STRANGE.
    """

    __slots__ = (
        # --- Sparse matrix components for adaptive noise storage ---
        '_row',            # Row indices for sparse noise storage
        '_col',            # Column indices for sparse noise storage
        '_data',           # Matrix data for sparse noise storage

        # --- Locking mechanism ---
        '__locked__',      # Lock status to prevent parameter changes after noise is generated

        # --- Parameter management ---
        'masterseed',      # Master random seed for reproducibility
        '_default_param',  # Default parameter dictionary
        '_param_types',    # Parameter type dictionary
        '__param',         # Current parameters (internal)

        # --- Noise trajectory data ---
        '_noise',          # Main noise array or interpolation function
        '_lop_active',     # List of active L-operators for which noise is prepared

        # --- Noise windowing (for memory efficiency) ---
        'Z2_windowed',     # Windowed noise array (current window)
        't_ax_windowed',   # Time axis for the current noise window
        'list_window_mask' # Indices for the current noise window
    )

    def __init__(self, noise_param, noise_corr):
        """
        Inputs
        ------
        1. noise_param : dict
                         Dictionary that defines the noise trajectory for
                         the calculation.
            a. SEED : int, str, or np.array
                      Seed that predefines the noise trajectory.
            b. MODEL : str
                       Name of the noise model to be used (options: FFT_FILTER, ZERO,
                       PRE_CALCULATED).
            c. TLEN : float
                      Length of the time axis [units: fs].
            d. TAU : float
                     Smallest timestep used for direct noise calculations [units: fs].
            e. INTERP : bool
                        True indicates that off-grid calls for noise values will be
                        determined by interpolation while False indicates otherwise
                        (options: False).
            f. RAND_MODEL: str
                           Name of the raw noise generation model to be used. (options:
                           BOX_MULLER, SUM_GAUSSIAN)
            g. STORE_RAW_NOISE: bool
                                True indicates that the uncorrelated noise trajectories
                                will be stored while False indicates that they will be
                                discarded.
            h. ADAPTIVE : bool
                          True indicates that noise will be adaptively generated in
                          sparse format only for L-operators that are requested in the 
                          "get_noise" function.  Otherwise, noise for all L-operators 
                          are generated in dense format.


        2. noise_corr : dict
                        Defines the noise correlation function for the calculation.
            a. CORR_FUNCTION : function
                               Defines alpha(t) for the noise term given CORR_PARAM[i].
            b. N_L2 : int
                      Number of L operators (and final z(t) trajectories).
            c. L_IND_BY_NMODE : list(int)
                                L-indices associated with each CORR_PARAM.
            d. CORR_PARAM : list(complex)
                            Parameters that define the components of alpha(t).

        Returns
        -------
        None
        """
        self._row = []
        self._col = []
        self._data = []
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
        if type(self.param["SEED"]) == int:
            self.masterseed = self.param["SEED"]

        self._noise = None
        self._lop_active = [] 
        self.Z2_windowed = None
        self.t_ax_windowed = None
        if self.param["NOISE_WINDOW"] is not None and self.param["NOISE_WINDOW"] > self.param["TLEN"]:
            self.param["NOISE_WINDOW"] = self.param["TLEN"]
            
    def _corr_func_by_lop_taxis(self, t_axis, lind_new):
        """
        Calculates the correlation function for each L operator by combining all the
        system-bath components that have the same L-operator.

        Parameters
        ----------
        1. t_axis : list(float)
                    List of time points.
        2. lind_new : list(int)
                      List of L-operator indices

        Returns
        -------
        1. alpha : list(list(complex))
                   List of lists of correlation functions evaluated at each time point.
        """
        n_lop = len(lind_new)
        alpha = np.zeros([n_lop, len(t_axis)], dtype=np.complex128)
        for (rel_ind,l_ind) in enumerate(set(lind_new)):
            i_mode_all = self.param["NMODE_BY_LIND"][l_ind]
            for i in i_mode_all:
                alpha[rel_ind, :] += self.param["CORR_FUNCTION"](
                    t_axis, *self.param["CORR_PARAM"][i]
                )
        return alpha

    def _prepare_noise(self, new_lop):
        """
        Generates the correlated noise trajectory based on the choice of noise model.
        Options include generating a zero noise trajectory, using an FFT filter
        model to correlate a complex white noise, and using a numpy array (or loading
        a .npy file) to serve as the correlated noise directly.

        Parameters
        ----------
        1.  new_lop : list(int)
                      Absolute indices of L-operators for which noise is prepared.

        Returns
        -------
        None
        """
        if not self.param["ADAPTIVE"]:
            new_lop = list(np.arange(self.param["N_L2"]))
        
        n_l2 = len(new_lop)
        n_taus = len(self.param["T_AXIS"])

        # Zero noise case:
        if self.param["MODEL"] == "ZERO":
            if self.param['STORE_RAW_NOISE']:
                print("Raw noise is identical to correlated noise in the ZERO noise "
                      "model.")
            z_correlated = np.zeros([n_l2, n_taus], dtype=np.complex64)
            if self.param['INTERPOLATE']:
                self._noise = interp1d(self.param['T_AXIS'], z_correlated, kind='cubic',
                                       axis=1)
            elif self.param['ADAPTIVE']:
                new_noise = np.complex64(z_correlated)                           
            else:
                self._noise = np.complex64(z_correlated)

        # FFTfilter case:
        elif self.param["MODEL"] == "FFT_FILTER":
            # Initialize uncorrelated noise
            # -----------------------------
            
            #If SEED is an array, we just calculate everything, like before (for now?)
            if(type(self.param['SEED']) is np.ndarray):
                new_lop = list(np.arange(self.param['N_L2']))
            
            z_uncorrelated = self._prepare_rand(new_lop)

            # Initialize correlated noise
            # ---------------------------
            alpha = np.complex64(self._corr_func_by_lop_taxis(self.param['T_AXIS'], new_lop))
            z_correlated = self._construct_correlated_noise(alpha, z_uncorrelated)

            # Remove 'Z_UNCORRELATED' for memory savings
            if self.param['STORE_RAW_NOISE']:
                self.param['Z_UNCORRELATED'] = z_uncorrelated
            if self.param['INTERPOLATE']:
                self._noise = interp1d(self.param['T_AXIS'], z_correlated, kind='cubic',axis=1)
            elif self.param['ADAPTIVE']:
                new_noise = np.complex64(z_correlated)
            else:
                self._noise = np.complex64(z_correlated)

        # Precalculated case
        elif self.param["MODEL"] == "PRE_CALCULATED":
            # If SEED is an iterable
            if (type(self.param['SEED']) is list) or (type(self.param['SEED']) is
                                                       np.ndarray):
                print('Correlated noise initialized from input array.')
                # This is where we need to write the code to use an array of correlated
                # noise variables input in place of the SEED parameter.
                if np.shape(self.param['SEED']) == (self.param['N_L2'],
                                                    len(self.param['T_AXIS'])):
                    self._noise = np.complex64(self.param['SEED'])
                    if self.param['INTERPOLATE']:
                        self._noise = interp1d(self.param['T_AXIS'], self.param['SEED'],
                                               kind='cubic', axis=1)
                    else:
                        self._noise = self.param['SEED']

                else:
                    raise UnsupportedRequest(
                        'Noise.param[SEED] is an array of the wrong length',
                        'Noise.prepare_noise', True)

            # if seed is a file address
            elif type(self.param["SEED"]) is str:
                print("Noise Model intialized from file: {}".format(self.param['SEED']))
                if os.path.isfile(self.param["SEED"]):
                    if self.param["SEED"][-4:] == ".npy":
                        corr_noise = np.complex64(np.load(self.param["SEED"]))
                        if np.shape(corr_noise) == (self.param['N_L2'],
                                                    len(self.param['T_AXIS'])):
                            if self.param['INTERPOLATE']:
                                self._noise = interp1d(self.param['T_AXIS'], corr_noise,
                                                       kind='cubic', axis=1)
                            else:
                                self._noise = corr_noise
                        else:
                            raise UnsupportedRequest(
                                'The file loaded at address Noise.param[SEED] is an '
                                'array of the wrong length', 'Noise.prepare_noise',
                                True)

                    else:
                        raise UnsupportedRequest(
                            'Noise.param[SEED] of filetype {} is not supported'.format(
                                type(self.param['SEED']))[-4:],
                            'Noise.prepare_noise', True)
                else:
                    raise UnsupportedRequest(
                        'Noise.param[SEED] {} is not the address of a valid file'.format(
                            self.param['SEED']),
                        'Noise.prepare_noise', True)


            else:
                raise UnsupportedRequest(
                    'Noise.param[SEED] of type {}'.format(
                        type(self.param['SEED'])),
                    'Noise.prepare_noise')

        else:
            raise UnsupportedRequest(
                'Noise.param[MODEL] {}'.format(
                    self.param['MODEL']),
                'Noise.prepare_noise')

        #Add new noise to self._noise
        if self.param['ADAPTIVE']:
            for (i,lop) in enumerate(new_lop):
                self._row += [lop]*n_taus
                self._col += list(np.arange(n_taus))
                self._data += list(new_noise[i,:])
            self._noise = sp.sparse.coo_array((self._data,(self._row,self._col)),
                                shape=(self.param['N_L2'],len(self.param["T_AXIS"])),
                                dtype=np.complex64).tocsc()

        # Update lop_active so get_noise knows when to call prepare_noise
        self._lop_active = list(set(self._lop_active) | set(new_lop))

        if self.Z2_windowed is not None:
            if self.param["ADAPTIVE"]:
                # Update the temporary noise with new info.
                self.Z2_windowed = self._noise[:, self.list_window_mask]
            else:
                self.Z2_windowed[:, :] = self._noise[:, self.list_window_mask]


    def get_noise(self, t_axis, list_lop=None):
        """
        Gets the noise associated with a given time interval.

        Parameters
        ----------
        1. t_axis : list(float)
                    List of time points.
                    
        2. list_lop : list(int)
                      List of L-operators 
                    
        Returns
        -------
        1. Z2_noise : np.array
                      2D array of noise values, shape (list_lop, t_axis) sampled at the given time points.
        """
        if self._noise is None:
            if self.param["ADAPTIVE"]:
                self._noise = sp.sparse.coo_array( (self.param['N_L2'],
                                                    len(self.param["T_AXIS"])),
                                                   dtype=np.complex64).tocsc()
            else:
                self._noise = np.zeros([self.param["N_L2"],
                                        len(self.param["T_AXIS"])], dtype=np.complex64)

        if list_lop is None:
            list_lop = np.arange(self.param["N_L2"])
        new_lop = list(set(list_lop) - set(self._lop_active))
        #Prepare noise for all L-operators not already prepared.
        if len(new_lop) > 0:
            self._prepare_noise(new_lop)


        #No L-operator removal is implemented yet.
        
        if self.param["INTERPOLATE"]:
            if self.param["NOISE_WINDOW"] is not None:
                print("Warning: noise windowing is not supported while using "
                      "interpolated noise.")
            return self._noise(t_axis)

        else:
            if self.Z2_windowed is not None:
                # If t_axis is out of range of the noise window, create new noise window
                if (np.min(t_axis) < np.min(self.t_ax_windowed) or np.max(t_axis) > np.max(
                        self.t_ax_windowed)):
                    start = np.max((0, np.min(t_axis)))
                    end = np.min((np.max(self.param["T_AXIS"]),
                                  np.max(t_axis) + self.param["NOISE_WINDOW"]))
                    start_index = np.where(self.param["T_AXIS"] >= start)[0][0]
                    end_index = np.where(self.param["T_AXIS"] >= end)[0][0]
                    self.list_window_mask = list(np.arange(start_index, end_index + 1))
                    self.t_ax_windowed = self.param["T_AXIS"][self.list_window_mask]
                    if self.param["ADAPTIVE"]:
                        self.Z2_windowed = self._noise[:, self.list_window_mask]
                    else:
                        self.Z2_windowed = np.zeros([self.param[
                                                    'N_L2'], len(self.t_ax_windowed)],
                                               dtype=np.complex64)
                        self.Z2_windowed[:, :] = self._noise[:, self.list_window_mask]
                # Otherwise the noise window is already created for the given time
                # points.
            else:
                # If we are not using noise windowing, or noise window is larger than
                # the entire t_axis, then set the noise window to be the full time axis.
                if (self.param["NOISE_WINDOW"] is None or self.param["NOISE_WINDOW"]
                        > np.max(self.param["T_AXIS"])):
                    self.list_window_mask = list(np.arange(len(self.param["T_AXIS"])))
                    self.t_ax_windowed = self.param["T_AXIS"]
                    self.Z2_windowed = self._noise
                # Otherwise the noise window is initialized startin' from time 0.
                else:
                    end = np.max([self.param["NOISE_WINDOW"],np.max(t_axis)])
                    end_index = np.where(self.param["T_AXIS"] >= end)[0][0]
                    self.list_window_mask = list(np.arange(end_index+1))
                    self.t_ax_windowed = self.param["T_AXIS"][self.list_window_mask]
                    if self.param["ADAPTIVE"]:
                        self.Z2_windowed = self._noise[:, self.list_window_mask]
                    else:
                        self.Z2_windowed = np.zeros([self.param[
                                                   'N_L2'], len(self.t_ax_windowed)],
                                              dtype=np.complex64)
                        self.Z2_windowed[:, :] = self._noise[:, self.list_window_mask]

            if (np.min(t_axis) < np.min(self.param["T_AXIS"]) or np.max(t_axis) >
                    np.max(self.param["T_AXIS"])):
                raise UnsupportedRequest(
                    "t-samples outside of the defined t-axis",
                    "NoiseModel.get_noise()",
                )

            it_list = []

            for t in t_axis:
                test = np.abs(self.t_ax_windowed - t) < precision
                if np.sum(test) == 1:
                    it_list.append(np.where(test)[0][0])
                else:
                    raise UnsupportedRequest(
                        "Off axis t-samples when INTERPOLATE = False",
                        "NoiseModel.get_noise()",
                    )

            return self._noise_to_array(self.Z2_windowed, it_list, list_lop)

    def _prepare_rand(self,new_lop=None):
        """
        Constructs the uncorrelated complex Gaussian distributions that may be
        converted to a correlated noise trajectory via an FFT filter model. Average
        of this uncorrelated noise trajectory is 0, and average of the absolute
        values of is sqrt(pi)/2 (assuming an arbitrarily long trajectory). The
        uncorrelated noise trajectory may be generated from a Box-Muller distribution or
        a sum of real and imaginary Gaussian distributions, or by using a numpy array
        (or loading a .npy file) to serve as the uncorrelated noise directly.

        Parameters
        ----------
        1. new_lop : list(int)
                     List of L-operators

        Returns
        -------
        1. z_uncorrelated : np.array(np.complex64)
                            The uncorrelated "raw" complex Gaussian random noise
                            trajectory of the proper size to be transformed.  This corresponds
                            to L-operators in "new_lop"

        """
        if new_lop is None:
            new_lop = list(np.arange(self.param['N_L2']))
            self._noise = np.zeros([len(new_lop), len(self.param['T_AXIS'])], dtype=np.complex64)
        # Get the correct size of noise trajectory
        ntaus = len(self.param['T_AXIS'])
        n_lop = len(new_lop)
        # Initialize un-correlated noise
        # ------------------------------
        if (type(self.param['SEED']) is list) or (
                type(self.param['SEED']) is np.ndarray):
            print('Noise Model initialized from input array.')
            # Import a .npy file as a noise trajectory.
            if np.shape(self.param['SEED']) == (self.param['N_L2'], 2 * (len(
                    self.param['T_AXIS']) - 1)):
                return self.param['SEED']
            else:
                raise UnsupportedRequest(
                    'Noise.param[SEED] is an array of the wrong length',
                    'Noise._prepare_rand', True)

        elif type(self.param["SEED"]) is str:
            print("Noise Model intialized from file: {}".format(self.param['SEED']))
            # Import a .npy file as a noise trajectory
            if os.path.isfile(self.param["SEED"]):
                if self.param["SEED"][-4:] == ".npy":
                    self.param["SEED"] = np.load(self.param["SEED"])
                    return self._prepare_rand()
                else:
                    raise UnsupportedRequest(
                        'Noise.param[SEED] of filetype {} is not supported'.format(
                            type(self.param['SEED']))[-4:],
                        'Noise._prepare_rand', True)
            else:
                raise UnsupportedRequest(
                    'Noise.param[SEED] {} is not the address of a valid file '.format(
                        self.param['SEED']), 'Noise._prepare_rand', True)


        elif (type(self.param['SEED']) is int) or (self.param['SEED'] is None):
            
            random_numbers = self._generate_noise_samples(new_lop, ntaus, self.param["RAND_MODEL"])         
            print("Noise Model initialized with SEED = ", self.param["SEED"])
            if self.param["RAND_MODEL"] == "BOX_MULLER":
                # Box-Muller Method: Gaussian Random Number
                # ---------------------------------------------
                # We use special indexing to ensure that changes in the length of the time axis does not change the
                # random numbers assigned to each time point for a fixed seed value.
                # Z_t = (G1 + 1j*G2)/sqrt(2) = sqrt(-ln(g))*exp(2j*pi*phi)
                # G1 = sqrt(-2*ln(g))*Cos(2*pi*phi)
                # G2 = sqrt(-2*ln(g))*Sin(2*pi*phi)
      
                g_index, phi_index = self._construct_indexing(ntaus)
                g_rands = random_numbers[:, np.array(g_index)]
                phi_rands = random_numbers[:, np.array(phi_index)]
                return np.complex64(
                    np.sqrt(-2.0 * np.log(g_rands)) * np.exp(2.0j * np.pi * phi_rands))

            elif self.param["RAND_MODEL"] == "SUM_GAUSSIAN":
                # Sum of two Gaussians Method: Gaussian Random Number
                # ---------------------------------------------
                # This is a slightly more intuitive method of generating random
                # Gaussian complex numbers. Like the Box-Muller Method above,
                # changing the length of the time axis does not change the random
                # numbers assigned to each time point for a fixed seed value.
                
                re_index, im_index = self._construct_indexing(ntaus)
                random_real = random_numbers[:, np.array(re_index)]
                random_imag = random_numbers[:, np.array(im_index)]
                return np.complex64(random_real + 1j*random_imag)

            else:
                raise UnsupportedRequest(
                    'Noise.param[RAND_MODEL] {}'.format(
                        self.param["RAND_MODEL"]),
                    'Noise._prepare_rand')

        else:
            raise UnsupportedRequest(
                'Noise.param[SEED] of type {}'.format(
                    type(self.param['SEED'])),
                'Noise._prepare_rand') 

    def _generate_noise_samples(self,new_lop, n_times, modeltype):
        """
        Generates random numbers for given L-operators.
        
        Parameters
        ----------
        1. new_lop : list(int)
                     List of absolute L-operators for which noise is to be generated
        2. n_times :   list(int)
                       Number of time points 
        3. modeltype : string
                       Type of algorithm used to generate complex Gaussian noise
                       Currently "BOX_MULLER" and "SUM_GAUSSIAN" is implemented
                       
        Returns
        -------
        1. random_numbers : array(complex128)
                            2D noise array of size (num_lop, 4*(n_times-1)) 
        """
        random_numbers = np.zeros((len(new_lop), 4*(n_times-1)))
        for (i,lop) in enumerate(new_lop):
            # Each L-operator is given a unique seed. This seed is generated by
            # jumping the root seed RNG lop times. This ensures that the noise 
            # generated for a given L-operator is consistent regardless of the order
            # in which it is added. The noise is also guaranteed to be unique for
            # each L-operator for any feasible value of n_times.
            bitgenerator = np.random.PCG64(seed=self.param['SEED']).jumped(lop)
            randstate = np.random.default_rng(bitgenerator)
            if modeltype == "BOX_MULLER":
                random_numbers[i,:] = randstate.random(size=4*(n_times-1))
            if modeltype == "SUM_GAUSSIAN":
                random_numbers[i,:] = randstate.normal(loc=0, size=4*(n_times-1))
        return random_numbers
            
    @staticmethod
    def _construct_indexing(ntaus):
        """
        Constructs a self-consistent indexing scheme that controls assignment of
        random numbers to different time points.

        Parameters
        ----------
        1. ntaus : int
                   Length of the noise time-axis (that is, number of time points)

        Returns
        -------
        1. g_index : list(int)
                     List of indices where the Gaussian-determined norm of an
                     uncorrelated random noise point is placed

        2. phi_index : list(int)
                       List of indices where the phase of an uncorrelated random
                       noise point is placed
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

    @staticmethod
    def _construct_correlated_noise(c_t, z_t, model="FFT_FILTER"):
        """
        Calculates a noise trajectory using the bath correlation function (c_t). This
        function is based on the description given by:

        "Exact simulation of complex-valued
        Gaussian stationary Processes via circulant embedding."
        Donald B. Percival Signal Processing 86, p. 1470-1476 (2006)
        [Section 3, p. 5-7]

        Parameters
        ----------
        1. c_t : list(list(complex))
                 Correlation function sampled for each L operator at each time
                 point.

        2. z_t : list(list(complex))
                 List of complex-valued, uncorrelated random noise
                 trajectories with real and imaginary components at
                 each time having mean 0 and variance 1 for each L operator.
                 [Re(z_t) ~ N(0,1), Im(z_t) ~ N(0,1)]

        3. model : str
                   The model of the noise correlation function builder.

        Returns
        -------
        1. corr_noise : list(list(complex))
                        Correlated noise trajectory for each L operator at each time
                        point.
        """
        if model == "FFT_FILTER":
            # Rotate stochastic process such that final entry is real
            # -------------------------------------------------------
            list_angles = np.array(np.angle(c_t), dtype=np.float64)
            list_nu = list_angles[:, -1] / ((len(c_t[0, :]) - 1.0) * 2.0 * np.pi)
            E2_expmatrix = np.exp(
                -1.0j * 2.0 * np.pi * np.outer(list_nu, np.arange(len(c_t[0, :]))))
            tildec_t = np.abs(c_t) * np.exp(1.0j * list_angles) * E2_expmatrix

            # Construct the embedding
            # -----------------------
            
            s_w = np.zeros([len(tildec_t[:,0]),2*len(tildec_t[0,:])-2])
            for lop in range(len(tildec_t[:,0])):
                temp = np.real(np.fft.fft(np.array(
                    np.concatenate([tildec_t[lop,:], np.conj(np.flip(tildec_t[lop, 1:-1]))],
                               )),
                    ))
                s_w[lop,:] = temp

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
            z_w = np.zeros([len(z_t[:,0]),len(z_t[0,:])],dtype=np.complex128)
            for lop in range(len(z_t[:,0])):
                temp = np.fft.fft(z_t[lop,:])
                z_w[lop,:] = temp
            tildey_t = np.zeros_like(E2_expmatrix)
            for lop in range(len(z_w[:,0])):
                
                temp = np.fft.ifft(np.array(np.abs(z_w[lop,:]) * np.exp(1.0j * np.angle(z_w[lop,:]))
                                            * np.sqrt(s_w[lop,:] / 2.0)))[:len(c_t[0, :])]
                tildey_t[lop,:] = temp

            # Undo initial phase rotation
            # ---------------------------
            return (np.abs(tildey_t) * np.exp(1.0j * np.angle(tildey_t)) * np.conj(
                E2_expmatrix))

        elif model == "ZERO":
            return 0*z_t[::2]

        else:
            raise UnsupportedRequest(
                "Noise correlation model {}".format(model),
                "NoiseModel._construct_correlated_noise()",
            )


    def _reset_noise(self):
        """
        Resets the RandomState object responsible for the raw noise, and allows for
        the re-calculation of correlated noise. This is only useful for tests.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if type(self.param["SEED"]) == int or type(self.param["SEED"]) == type(None):
            self.randstate = np.random.RandomState(seed=self.param["SEED"])
        self._unlock()

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
        
    def _noise_to_array(self,Z2_noise_full,t_axis, list_lop=None):
        """
        Auxiliary function which slices the noise to retreive the
        noise for specific times and L-operators.  This is required because
        slicing is not yet fully implemented for sparse arrays.
        
        Parameters
        ----------
        1. Z2_noise_full  : np.array
                            2D noise array
                   
        2. t_axis         : list(int)
                            Time slice to retrieve noise
        3. list_lop       : list(int)
                            L-operator indices for which to retrieve noise
        
        Returns
        -------
        1. Z2_noise : np.array
                      Sliced noise array
        """
        if list_lop is None:
            list_lop = self._lop_active
        num_l2 = len(list_lop)
        num_t = len(t_axis)
        Z2_noise = np.zeros((num_l2,num_t),dtype=np.complex64)
        for (i_l2,l2_ind) in enumerate(list_lop):
            for (i_t,t) in enumerate(t_axis):
                Z2_noise[i_l2][i_t] = Z2_noise_full[l2_ind,t]
        return Z2_noise
