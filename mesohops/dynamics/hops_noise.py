import copy
import os
import numpy as np
from scipy.interpolate import interp1d
from mesohops.util.dynamic_dict import Dict_wDefaults
from mesohops.util.exceptions import LockedException, UnsupportedRequest
from mesohops.util.physical_constants import precision  # constant

__title__ = "Pyhops Noise"
__author__ = "D. I. G. Bennett, J. K. Lynd"
__version__ = "1.4"

# NOISE MODELS:
# =============


NOISE_DICT_DEFAULT = {
    "SEED": None,
    "MODEL": "FFT_FILTER",
    "TLEN": 1000.0,  # Units: fs
    "TAU": 1.0,  # Units: fs,
    "INTERPOLATE": False,
    # "RAND_MODEL": "SUM_GAUSSIAN", # SUM_GAUSSIAN or BOX_MULLER
    "RAND_MODEL": "SUM_GAUSSIAN",  # SUM_GAUSSIAN or BOX_MULLER
    "STORE_RAW_NOISE": False,
}

NOISE_TYPE_DEFAULT = {
    "SEED": [int, type(None), str, np.ndarray],
    "MODEL": [str],
    "TLEN": [float],
    "TAU": [float],
    "INTERPOLATE": [bool],
    "RAND_MODEL": [str],
    "STORE_RAW_NOISE": [type(False)],
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
        if type(self.param["SEED"]) == int or type(self.param["SEED"]) == type(None):
            self.randstate = np.random.RandomState(seed=self.param["SEED"])

    def _corr_func_by_lop_taxis(self, t_axis):
        """
        Calculates the correlation function for each L operator by combining all the
        system-bath components that have the same L-operator.

        Parameters
        ----------
        1. t_axis : list(float)
                    List of time points.

        Returns
        -------
        1. alpha : list(list(complex))
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

    def prepare_noise(self):
        """
        Generates the correlated noise trajectory based on the choice of noise model.
        Options include generating a zero noise trajectory, using an FFT filter
        model to correlate a complex white noise, and using a numpy array (or loading
        a .npy file) to serve as the correlated noise directly.

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

        # Zero noise case:
        if self.param["MODEL"] == "ZERO":
            if self.param['STORE_RAW_NOISE']:
                print("Raw noise is identical to correlated noise in the ZERO noise "
                      "model.")
            z_correlated = np.zeros([len(self.param['T_AXIS']), self.param[
                'N_L2']], dtype=np.complex64).T
            if self.param['INTERPOLATE']:
                self._noise = interp1d(self.param['T_AXIS'], z_correlated, kind='cubic',
                                       axis=1)
            else:
                self._noise = np.complex64(z_correlated)

        # FFTfilter case:
        elif self.param["MODEL"] == "FFT_FILTER":
            # Initialize uncorrelated noise
            # -----------------------------
            # z_uncorrelated = self._prepare_rand(n_lop = self.param["N_L2"],
            #                                     ntaus = len(self.param['T_AXIS']))
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
                    self._noise = self.param['SEED']
                # We should add an interpolation option as well.
                else:
                    raise UnsupportedRequest(
                        'Noise.param[SEED] is an array of the wrong length',
                        'Noise.prepare_noise', True)

            # if seed is a file address
            elif type(self.param["SEED"]) is str:
                print("Noise Model intialized from file: {}".format(self.param['SEED']))
                if os.path.isfile(self.param["SEED"]):
                    if self.param["SEED"][-4:] == ".npy":
                        corr_noise = np.load(self.param["SEED"])
                        if np.shape(corr_noise) == (self.param['N_L2'],
                                                    len(self.param['T_AXIS'])):
                            self._noise = corr_noise
                        # We should add an interpolation option as well.
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

        # Lock Noise Instance
        # -------------------
        # A noise model has been explicitly calculated. All further modifications
        # to this class should be blocked.
        self.__locked__ = True

    def get_noise(self, t_axis):
        """
        Gets the noise associated with a given time interval.

        Parameters
        ----------
        1. t_axis : list(float)
                    List of time points.

        Returns
        -------
        1. noise : list(list(complex))
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

    def _prepare_rand(self):
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
        None

        Returns
        -------
        1. z_uncorrelated : np.array(np.complex64)
                            The uncorrelated "raw" complex Gaussian random noise
                            trajectory of the proper size to be transformed.

        """
        # Get the correct size of noise trajectory
        n_lop = self.param["N_L2"]
        ntaus = len(self.param['T_AXIS'])

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
            print("Noise Model initialized with SEED = ", self.param["SEED"])
            if self.param["RAND_MODEL"] == "BOX_MULLER":
                # Box-Muller Method: Gaussian Random Number
                # ---------------------------------------------
                # We use special indexing to ensure that changes in the length of the time axis does not change the
                # random numbers assigned to each time point for a fixed seed value.
                # Z_t = (G1 + 1j*G2)/sqrt(2) = sqrt(-ln(g))*exp(2j*pi*phi)
                # G1 = sqrt(-2*ln(g))*Cos(2*pi*phi)
                # G2 = sqrt(-2*ln(g))*Sin(2*pi*phi)
                random_numbers = self.randstate.rand(4 * (ntaus - 1), n_lop).T
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
                random_gauss = self.randstate.normal(loc=0, size=(4*(ntaus-1), n_lop)).T
                random_real = random_gauss[:, np.array(re_index)]
                random_imag = random_gauss[:, np.array(im_index)]
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
            for site in range(len(tildec_t[:,0])):
                temp = np.real(np.fft.fft(np.array(
                    np.concatenate([tildec_t[site,:], np.conj(np.flip(tildec_t[site, 1:-1]))],
                               )),
                    ))
                s_w[site,:] = temp

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
            for site in range(len(z_t[:,0])):
                temp = np.fft.fft(z_t[site,:])
                z_w[site,:] = temp
            tildey_t = np.zeros_like(E2_expmatrix)
            for site in range(len(z_w[:,0])):
                
                temp = np.fft.ifft(np.array(np.abs(z_w[site,:]) * np.exp(1.0j * np.angle(z_w[site,:]))
                                            * np.sqrt(s_w[site,:] / 2.0)))[:len(c_t[0, :])]
                tildey_t[site,:] = temp

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
        the re-calculation of correlated noise. This is only useful for testing.

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
