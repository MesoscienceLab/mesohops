import numpy as np
from collections.abc import Mapping
from mesohops.util.exceptions import AuxError
from scipy.special import binom

__title__ = "AuxiliaryVector Class"
__author__ = "D. I. G. Bennett"
__version__ = "1.0"


class AuxiliaryVector(Mapping):
    """
    This is a class that encodes a sparse representation of auxiliary vectors
    with some extra helper functions to simplify some common actions, such as:
    determining the absolute index, adding a unit vector, and calculating the sum.

    The class is not mutable - which is to say, once an auxiliary vector is defined,
    it cannot be changed.

    """

    def __init__(self, aux_array, nmodes):
        """
        INPUTS:
        -------
        1. aux_array : iterable
                      list of (mode, value) pairs for all non-zero indices of the auxiliary
                      vector
        2. nmodes : int
                   the number of modes in the hierarchy which is the length of the dense
                   auxiliary vector.

        RETURNS:
        --------
        None
        """
        self.dict_aux_vec = {
            index_mode: aux_value for (index_mode, aux_value) in aux_array
        }
        self.tuple_aux_vec = tuple(
            [tuple([mode, value]) for (mode, value) in aux_array]
        )
        self.array_aux_vec = np.array(aux_array)
        self.__abs_index = None
        self.__len = nmodes
        self.hash = hash(self.tuple_aux_vec)

    # Dictionary-like methods overwriting Mutable Mapping
    # ===================================================
    def __getitem__(self, key):
        if key in self.dict_aux_vec.keys():
            return self.dict_aux_vec[key]
        elif key < len(self):
            return 0
        else:
            raise AuxError("mode index larger than total number of modes.")

    def __iter__(self):
        return iter(self.dict_aux_vec)

    def __len__(self):
        return self.__len

    def __repr__(self):
        return f"{type(self).__name__}({self.dict_aux_vec})"

    def keys(self):
        """
        This function returns an array of mode indices for the auxiliary vectors

        Parameters
        ----------
        None

        Returns
        -------
        1. keys : array
                  an array of mode indices with nonzero auxiliary index
        """
        if len(self.dict_aux_vec) > 0:
            return self.array_aux_vec[:, 0]
        else:
            return np.array([])

    def values(self):
        """
        This function returns an array of the auxiliary vector values

        Parameters
        ----------
        None

        Returns
        -------
        1. values : array
                    an array of nonzero auxiliary index values
        """
        if len(self.dict_aux_vec) > 0:
            return self.array_aux_vec[:, 1]
        else:
            return np.array([])

    # Comparison Methods
    # ==================
    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash

    def __ne__(self, other):
        return self.hash != other.hash

    def _compare(self, other, comparison_function):
        """
        This function compares two auxiliary vectors

        Parameters
        ----------
        1. other : array
                   the array you want to compare
        2. comparison_function : function
                                 a comparison function

        Returns
        -------
        1. bool_compare : bool
                          a boolean for the comparison
        """
        if isinstance(other, AuxiliaryVector) and len(self) == len(other):
            return comparison_function(self.absolute_index, other.absolute_index)
        else:
            return False

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    # Special Methods
    # ===============
    def dot(self, vec):
        """
        This is a function that performs a sparse dot product between the
        auxiliary index vector and another vector.

        Parameters
        ----------
        1. vec : np.array
                 a vector

        Returns
        -------
        1. product : float
                     the dot product value

        """
        if len(self.dict_aux_vec) == 0:
            return 0
        else:
            return np.dot(self.array_aux_vec[:, 1], vec[self.array_aux_vec[:, 0]])

    def sum(self, **unused_kwargs):
        """
        This function returns the sum of the auxiliary vector values

        Parameters
        ----------
        None

        Returns
        -------
        1. sum : float
                 the sum of the nonzero values of the auxiliary vectors
        """
        return np.sum(self.values())

    def todense(self):
        """
        This function will take a sparse vector and make it dense

        Parameters
        ----------
        None

        Returns
        -------
        1. output : array
                    the dense vector
        """
        output = np.zeros(self.__len)
        if len(self.dict_aux_vec) == 0:
            return output
        output[self.keys()] = self.values()
        return output

    def toarray(self):
        """
        This function converts a dict to an array

        Parameters
        ----------
        None

        Returns
        -------
        1. array : array
                   a dict in an array form
        """
        return self.array_aux_vec

    def get_values(self, index_slice):
        """
        This function gets the dense auxiliary vector values from a sub-indexed list

        Parameters
        ----------
        1. index_slice : list
                         a list of indices

        Returns
        -------
        1. values : array
                    an array of values at the given indices
        """
        return np.array([self.__getitem__(key) for key in index_slice])

    def get_values_nonzero(self, index_slice):
        """
        This function gets the sparse auxiliary vector values from a sub-indexed list

        NOTE: the values are returned in key order not the order
              they are present in index_slice

        Parameters
        ----------
        1. index_slice : list
                         a list of indices

        Returns
        -------
        1. values : array
                    a sparse array of the non-zero auxiliary vector values
        """
        return np.array(
            [self.dict_aux_vec[key] for key in self.keys() if key in index_slice]
        )

    def e_step(self, mode, step):
        """
        This function returns a new Auxiliary Vector with the desired step in the given
        mode

        Parameters
        ----------
        1. mode : int
                  The absolute mode index
        2. step : int
                  The change in the aux value for the given mode

        Returns
        -------
        1. aux_vec : tuple
                     the new sparse auxiliary vector
        """
        return AuxiliaryVector(self.tuple_from_e_step(mode, step), nmodes=self.__len)

    def hash_from_e_step(self, mode, step):
        """
        This function returns the hash of a new Auxiliary Vector with the desired step
        in the given mode

        Parameters
        ----------
        1. mode : int
                  The absolute mode index
        2. step : int
                  The change in the aux value for the given mode

        Returns
        -------
        1. hash :
                  the hash of the tuple sparse auxiliary vector created from e_step
        """
        return hash(self.tuple_from_e_step(mode, step))

    def tuple_from_e_step(self, mode, step):
        """
        Returns the sparse tuple representation of the auxiliary that is the given step
        length along the given absolute mode index away from the current auxiliary.

        Parameters
        ----------
        1. mode : int
                  The absolute mode index
        2. step : int
                  The change in the aux value for the given mode

        Returns
        -------
        1. tuple_aux : tuple
                       The sparse representation of the auxiliary (sorted mode order)
        """
        if step == 0:
            return self.tuple_aux_vec
        elif self.__getitem__(mode) + step < 0:
            return ((0, -1),)
        elif len(self.dict_aux_vec) == 0:
            return tuple([(mode, step)])
        elif mode in self.array_aux_vec[:, 0]:
            if self.__getitem__(mode) + step == 0:
                return tuple(
                    [
                        tuple([mode_i, value_i])
                        for (mode_i, value_i) in self.tuple_aux_vec
                        if mode_i != mode
                    ]
                )
            else:
                return tuple(
                    [
                        tuple([mode_i, value_i + step])
                        if mode_i == mode
                        else tuple([mode_i, value_i])
                        for (mode_i, value_i) in self.tuple_aux_vec
                    ]
                )
        else:
            list_keys = list(self.dict_aux_vec.keys())
            list_keys.append(mode)
            list_keys.sort()
            list_values = [
                step if key == mode else self.dict_aux_vec[key] for key in list_keys
            ]
            return tuple(
                [tuple([mode, value]) for (mode, value) in zip(list_keys, list_values)]
            )

    def index_analytic(self):
        """
         This function provides an absolute index value for an auxiliary
         vector using an analytic function of the indices. The basic idea
         is that the indices are always ordered by increasing hierarchy
         'level' (i.e. the sum of the indices). Within a level they are ordered
         by first comparing the first values, then the second values, etc.


         This gives the indexing a particularly simple form with a level:
         L = sum(i_0,...,i_n)
         (i_0, ... i_N) = sum_n<N sum_L>ln>i_n ((N-n-1 , L-sum(aux[:n])-ln)
         where ((L,K)) denotes a L multichoose K.

         The derivation of the following equations is given on p. 68 of
         Quantum Notebook #1. The sums have been removed by making use of the
         binomial sum property and the binomial symmetry property. The result is
         a equation that only sums over a number of elements equal to the number
         of non-zero terms in aux.


         PARAMETERS
         ----------
         None

         RETURNS
         -------
         1. index : int
                    the absolute index for an auxiliary

         """
        # Constants
        # ---------
        aux = self.toarray()
        n_hmode = self.__len
        L = self.sum()

        if not aux.size:
            return 0
        else:
            # Calculate number of aux at order less than L
            # --------------------------------------------
            n_aux_below_l = int(binom(n_hmode + L - 1, L - 1))

            # Calculate N+ contribution
            # -------------------------
            list_np_boxes = [n_hmode]
            list_np_boxes.extend(n_hmode - aux[:-1, 0] - 1)
            list_np_boxes = np.array(list_np_boxes)

            list_np_balls = [L]
            list_np_balls.extend(L - np.cumsum(aux[:-1, 1]))
            list_np_balls = np.array(list_np_balls)

            n_plus = np.nansum(
                binom(list_np_boxes + list_np_balls - 1, list_np_boxes - 1)
            )

            # Calculate N- contributions
            # --------------------------
            list_nm_boxes = n_hmode - aux[:, 0] - 1
            n_minus = np.nansum(binom(list_nm_boxes + list_np_balls, list_nm_boxes))

            # calculate M contributions
            # -------------------------
            list_m_balls = L - np.cumsum(aux[:, 1]) - 1
            m = np.nansum(binom(list_nm_boxes + list_m_balls, list_m_balls))

            return int(n_aux_below_l + m + n_plus - n_minus)

    @property
    def absolute_index(self):
        if self.__abs_index is None:
            self.__abs_index = self.index_analytic()

        return self.__abs_index
