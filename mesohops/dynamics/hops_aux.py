import numpy as np
from math import factorial
from collections.abc import Mapping
from mesohops.util.exceptions import AuxError
from scipy.special import binom

__title__ = "AuxiliaryVector Class"
__author__ = "D. I. G. Bennett"
__version__ = "1.2"


def binom_integer(n_arr,k_arr):
    """
    A binomial distribution calculator that uses only Python integers to achieve
    arbitrary precision and avoid issues with float overflow. Takes in any iterable
    or single value

    Parameters
    ----------
    1. n_arr : array(int)
               Number of options - must be larger than or equal to k.

    2. k_arr : array(int)
               Number of choices - must be smaller than or equal to n.

    Returns
    -------
    1. dist : int
              Result of the formula n!/(k!(n-k)!).
    """
    if not hasattr(n_arr, '__iter__'):
        return factorial(n_arr)//factorial(k_arr)//factorial(n_arr-k_arr)
    return [factorial(n)//factorial(k)//factorial(n-k) for n,k in zip(n_arr,k_arr)]


def intsum(arr):
    """
    Helper function similar to np.sum that is exclusively for use with integers and
    does not change the data type at any point.

    Parameters
    ----------
    1. arr : array(int)
             Array of integers to be summed over.

    Returns
    -------
    1. sum : int
             Sum of the array of integers.
    """
    sum = 0
    for entry in arr:
        if type(entry) == int or np.isfinite(entry):
            sum += int(entry)
    return sum


class AuxiliaryVector(Mapping):
    """
    This is a class that encodes a sparse representation of auxiliary vectors
    with some extra helper functions to simplify some common actions, such as:
    determining the absolute index, adding a unit vector, and calculating the sum.

    The class is not mutable - which is to say, once an auxiliary vector is defined,
    it cannot be changed.
    """
    __slots__ = ('dict_aux_vec', 'tuple_aux_vec', 'array_aux_vec', '__abs_index', '__len'
                  , '__hash', '_index', '__hash_string','_sum', '_dict_aux_p1', '_dict_aux_m1',
                 '__mode_digits')

    def __init__(self, aux_array, nmodes):
        """
        Initializes the AuxiliaryVector object with a given depth in each of nmodes
        modes.

        Parameters
        ----------
        1. aux_array : iterable
                       List of (mode, value) pairs for all non-zero indices of the
                       auxiliary vector.

        2. nmodes : int
                   Number of modes in the hierarchy which is the length of the dense
                   auxiliary vector.

        RETURNS
        -------
        None
        """
        self.dict_aux_vec = {
            index_mode: aux_value for (index_mode, aux_value) in aux_array
        }
        self.tuple_aux_vec = tuple(
            [tuple([mode, value]) for (mode, value) in aux_array]
        )
        self.array_aux_vec = np.array(aux_array)

        if (len(self.array_aux_vec)>0 and
                not np.all(np.sort(self.array_aux_vec[:,0]) == self.array_aux_vec[:,0])):
                raise AuxError("array_aux_vec not properly ordered")
        self.__abs_index = None
        self.__len = nmodes
        self.__mode_digits = len(str(self.__len))
        self._construct_identity_str()
        self.__hash = hash(self.identity_string)
        self._index = None
        self._sum = np.sum(self.values())

        self._dict_aux_p1 = {}
        self._dict_aux_m1 = {}

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
        Returns an array of mode indices for the auxiliary vectors.

        Parameters
        ----------
        None

        Returns
        -------
        1. keys : array
                  Array of mode indices with nonzero auxiliary index.
        """
        if len(self.dict_aux_vec) > 0:
            return self.array_aux_vec[:, 0]
        else:
            return np.array([])

    def values(self):
        """
        Returns an array of the auxiliary vector values.

        Parameters
        ----------
        None

        Returns
        -------
        1. values : array
                    Array of nonzero auxiliary index values.
        """
        if len(self.dict_aux_vec) > 0:
            return self.array_aux_vec[:, 1]
        else:
            return np.array([])


    # Comparison Methods
    # ==================
    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return self.hash == hash(other)

    def __ne__(self, other):
        return self.hash != hash(other)

    def _compare(self, other, comparison_function):
        """
        Compares two auxiliary vectors.

        Parameters
        ----------
        1. other : array
                   Array you want to compare.

        2. comparison_function : function
                                 Comparison function.

        Returns
        -------
        1. bool_compare : bool
                          Boolean for the comparison.
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
    def difference_by_mode(self, other):
        """
        Compares the current HopsAux object to another HopsAux object. If they differ
        by only 1 step, then it returns the mode along which they differ.

        Parameters
        ----------
        1. other: HopsAux object
                  The HopsAux object to which the current object is compared.

        Returns
        -------
        1. diff_mode : int or False
                       The mode index along which they differ or False if they differ
                       by more than 1 step.
        """
        set_key_self = set(self.keys())
        set_key_other = set(other.keys())

        # Check that the two HopsAux belong to the same hierarchy
        assert self.__len == len(other)

        if np.abs(self._sum - other._sum) == 1:
            if set_key_self == set_key_other:
                values = np.abs(self.array_aux_vec[:,1]- other.array_aux_vec[:,1])
                if np.sum(values) == 1:
                    return self.array_aux_vec[np.where(values)[0][0],0]
            elif (len(set_key_self | set_key_other)
                  - len(set_key_self & set_key_other)) == 1:
                value = 0
                for key in set_key_self | set_key_other:
                    value += np.abs(self[key] - other[key])

                if value == 1:
                    index = list((set_key_self | set_key_other) - (set_key_self &
                                                              set_key_other))[0]
                    return index

        return False

    def dot(self, vec):
        """
        Performs a sparse dot product between the auxiliary index vector and another
        vector.

        Parameters
        ----------
        1. vec : np.array
                 A vector.

        Returns
        -------
        1. product : float
                     Dot product value.

        """
        if len(self.dict_aux_vec) == 0:
            return 0
        else:
            return np.dot(self.array_aux_vec[:, 1], vec[self.array_aux_vec[:, 0]])

    def sum(self, **unused_kwargs):
        """
        Calculates the sum of the auxiliary vector values.

        Parameters
        ----------
        None

        Returns
        -------
        1. sum : float
                 Sum of the nonzero values of the auxiliary vectors.
        """
        try:
            return self._sum
        except:
            return np.sum(self.values())

    def todense(self):
        """
        Convert a sparse vector into a dense vector.

        Parameters
        ----------
        None

        Returns
        -------
        1. output : array
                    Dense vector.
        """
        output = np.zeros(self.__len)
        if len(self.dict_aux_vec) == 0:
            return output
        output[self.keys()] = self.values()
        return output

    def toarray(self):
        """
        Converts a dict to an array.

        Parameters
        ----------
        None

        Returns
        -------
        1. array : array
                   Dict in an array form.
        """
        return self.array_aux_vec

    def get_values(self, index_slice):
        """
        Gets the dense auxiliary vector values from a sub-indexed list.

        Parameters
        ----------
        1. index_slice : list
                         List of indices.

        Returns
        -------
        1. values : array
                    Array of values at the given indices.
        """
        return np.array([self.__getitem__(key) for key in index_slice])

    def get_values_nonzero(self, index_slice):
        """
        Gets the sparse auxiliary vector values from a sub-indexed list.
        NOTE: the values are returned in key order, not the order they are present in
        index_slice.

        Parameters
        ----------
        1. index_slice : list
                         List of indices.

        Returns
        -------
        1. values : array
                    Sparse array of the non-zero auxiliary vector values.
        """
        return np.array(
            [self.dict_aux_vec[key] for key in self.keys() if key in index_slice]
        )

    def e_step(self, mode, step):
        """
        Returns a new Auxiliary Vector with the desired step in the given mode.

        Parameters
        ----------
        1. mode : int
                  Absolute mode index.

        2. step : int
                  Change in the aux value for the given mode.

        Returns
        -------
        1. aux_vec : tuple
                     New sparse auxiliary vector.
        """
        return AuxiliaryVector(self.tuple_from_e_step(mode, step), nmodes=self.__len)

    def tuple_from_e_step(self, mode, step):
        """
        Returns the sparse tuple representation of the auxiliary that is the given step
        length along the given absolute mode index away from the current auxiliary.

        Parameters
        ----------
        1. mode : int
                  Absolute mode index.

        2. step : int
                  Change in the aux value for the given mode.

        Returns
        -------
        1. tuple_aux : tuple
                       Sparse representation of the auxiliary (sorted mode order).
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

         Parameters
         ----------
         None

         RETURNS
         -------
         1. index : int
                    Absolute index for an auxiliary.

         """
        # Constants
        # ---------
        aux = self.toarray()
        n_hmode = self.__len
        L = self.sum()

        if not aux.size:
            return 0
        # Old, faster case: if the total index-depth sum is less than a thousand and
        # ten,
        # then there are fewer auxiliaries than the float limit
        else:
            # Calculate number of aux at order less than L
            # --------------------------------------------
            n_aux_below_l = binom(n_hmode + L - 1, L - 1)

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
            if n_aux_below_l + m + n_plus - n_minus < 1.5e308:
                return int(n_aux_below_l + m + n_plus - n_minus)
            else:
                # Calculate number of aux at order less than L
                # --------------------------------------------
                n_aux_below_l = int(binom_integer(n_hmode + L - 1, L - 1))

                # Calculate N+ contribution
                # -------------------------
                list_np_boxes = [n_hmode]
                list_np_boxes.extend(n_hmode - aux[:-1, 0] - 1)
                list_np_boxes = np.array(list_np_boxes)

                list_np_balls = [L]
                list_np_balls.extend(L - np.cumsum(aux[:-1, 1]))
                list_np_balls = np.array(list_np_balls)

                n_plus = intsum(
                    binom_integer(list_np_boxes + list_np_balls - 1, list_np_boxes - 1)
                )

                # Calculate N- contributions
                # --------------------------
                list_nm_boxes = n_hmode - aux[:, 0] - 1
                n_minus = intsum(binom_integer(list_nm_boxes + list_np_balls,
                                              list_nm_boxes))

                # calculate M contributions
                # -------------------------
                list_m_balls = L - np.cumsum(aux[:, 1]) - 1
                m = intsum(binom(list_nm_boxes + list_m_balls, list_m_balls))

                return int(n_aux_below_l + m + n_plus - n_minus)

    def add_aux_connect(self, index_mode, aux_other, type):
        """
        Updates the HopsAux object to contain a pointer to the
        other HopsAux objects it is connected to.

        Parameters
        ----------
        1. index_mode : int
                        Mode along which the two HopsAux objects are connected.

        2. aux_other : HopsAux
                       HopsAux object that self is connected to.

        3. type : int
                  +1 or -1 depending on if the other aux has a larger or smaller sum.

        Returns
        -------
        None
        """
        if type == 1:
            self._dict_aux_p1.update({index_mode: aux_other})
            aux_other._dict_aux_m1.update({index_mode: self})
        elif type == -1:
            self._dict_aux_m1.update({index_mode: aux_other})
            aux_other._dict_aux_p1.update({index_mode: self})
        else:
            raise AuxError('add_aux_connect does not support type={}'.format(type))

    def remove_aux_connect(self, index_mode, type):
        """
        Removes the connection between the HopsAux object and another
        connected with type (+1/-1) along index mode.

        Parameters
        ----------
        1. index_mode : int
                        Mode along which the two HopsAux objects are connected.

        2. type : int
                  +1 or -1 depending on if the other aux has a larger or smaller sum.

        Returns
        -------
        None
        """
        if type == 1:
            self._dict_aux_p1.pop(index_mode)
        elif type == -1:
            self._dict_aux_m1.pop(index_mode)
        else:
            raise AuxError('add_aux_connect does not support type={}'.format(type))

    def remove_pointers(self):
        """
        Removes all pointers targeting the current HopsAux object
        from the set of HopsAux objects it has connections to.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for (index_mode, aux) in self.dict_aux_p1.items():
            aux.remove_aux_connect(index_mode, -1)

        for (index_mode, aux) in self.dict_aux_m1.items():
            aux.remove_aux_connect(index_mode, 1)

        self._dict_aux_m1 = {}
        self._dict_aux_p1 = {}
        
    def _construct_identity_str(self):
        """
        Constructs a unique string representation for each auxiliary vector.
        The basic construction is that each mode index is recast as an integer string
        and it will appear in the string a number of times equal to the
        [Finish describing the algorithm]

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        mode_digits = len(str(self.__len))
        aux_hash_string = "".join([value * ((mode_digits - len(str(mode))) * "0" + str(mode))
                                   for (mode, value) in self.array_aux_vec])
        self.__hash_string = aux_hash_string

    def get_list_identity_string_down(self):
        """
        Construct the hash values for all auxiliaries that are one step
        down from this auxiliary.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        ref_hash_str = self.identity_string
        list_deleteindex = [0]+ list(np.cumsum([self.__mode_digits*value
                                                for (key,value) in self.array_aux_vec[:-1]]))
        list_value_connects = self.array_aux_vec[:,1]
        list_mode_connects = self.array_aux_vec[:,0]
        list_ident_string_down = [ref_hash_str[0:deleteindex]
                                 + ref_hash_str[(deleteindex+self.__mode_digits):]
                                 for deleteindex in list_deleteindex]
        return list_ident_string_down, list_value_connects, list_mode_connects

    def get_list_identity_string_up(self, modes_in_use):
        """
        Constructing all auxiliaries within one step 'up' from the
        current auxiliary along modes that are currently in the basis.

        [Explain algorithm]

        Parameters
        ----------
        1. modes_in_use : list
                          List of modes along which the new identity string will be
                          calculated

        Returns
        -------
        1. list_hash_string_up : list
                                 List of hash strings of downward auxiliary connections

        2. list_value_connects : list
                                 List of values of the auxiliary corresponding to 1.
        
        3. list_mode_connects : list
                                List of modes of auxiliary connections corresponding
                                to 1.
        """
        keys = self.keys()
        ref_hash_str = self.identity_string
        list_modes = list(set(modes_in_use) | set(keys))
        list_modes.sort()
        list_index_modes_in_use = [list_modes.index(mode) for mode in modes_in_use]

        list_insert_index = np.cumsum([self.__mode_digits * self.dict_aux_vec[mode] if mode in keys
                                       else 0
                                       for mode in list_modes])[list_index_modes_in_use]
        list_value_connects = np.array([self.dict_aux_vec[mode] if mode in keys
                           else 0
                           for mode in list_modes])[list_index_modes_in_use]
        list_ident_string_up = [(ref_hash_str[0:insertindex]
                                  + ((self.__mode_digits - len(str(mode))) * "0" + str(mode))
                                  + ref_hash_str[insertindex:])
                     for (mode, insertindex) in zip(modes_in_use, list_insert_index)]
        list_mode_connects = np.array(list_modes)[list_index_modes_in_use]
        return list_ident_string_up, list_value_connects, list_mode_connects

    @property
    def absolute_index(self):
        if self.__abs_index is None:
            self.__abs_index = self.index_analytic()
        return self.__abs_index

    @property
    def hash(self):
        return self.__hash

    @property 
    def identity_string(self):
        return self.__hash_string

    @property
    def dict_aux_p1(self):
        return self._dict_aux_p1

    @property
    def dict_aux_m1(self):
        return self._dict_aux_m1


