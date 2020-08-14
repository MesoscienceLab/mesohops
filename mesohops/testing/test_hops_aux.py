import numpy as np
from mesohops.dynamics.hops_aux import AuxiliaryVector


def test_keys():
    """
    This function test whether the correct mode indices (keys) are being grabbed by
    the keys function
    """
    aux_1010 = AuxiliaryVector([(0, 1), (2, 1)], 4)
    keys = aux_1010.keys()
    known_keys = np.array([0, 2])
    assert np.array_equal(keys, known_keys)


def test_values():
    """
    This function test whether the correct values are being grabbed by
    the values function
    """
    aux_1010 = AuxiliaryVector([(0, 1), (2, 1)], 4)
    values = aux_1010.values()
    known_values = np.array([1, 1])
    assert np.array_equal(values, known_values)


def test_compare_if():
    """
    This function test whether _compare is properly comparing auxiliary vectors by
    testing whether one vector is less than another
    """
    aux_1010 = AuxiliaryVector([(0, 1), (2, 1)], 4)
    aux_1000 = AuxiliaryVector([(0, 1)], 4)
    flag = aux_1010._compare(aux_1000, lambda s, o: s > o)
    assert flag == True


def test_compare_else():
    """
    This function test to make sure you cannot compare an Auxiliary Vector to another
    type
    """
    aux_1010 = AuxiliaryVector([(0, 1), (2, 1)], 4)
    known_aux = [(1, 0, 1, 0)]
    flag = aux_1010._compare(known_aux, lambda s, o: s > o)
    assert flag == False


def test_dot():
    """
    This function test to make sure the correct dot product value is given when using
    the dot function
    """
    vector = np.array([1, 1, 1, 1])
    aux_1234 = AuxiliaryVector([(0, 1), (1, 2), (2, 3), (3, 4)], 4)
    known_value = 10
    dot_value = aux_1234.dot(vector)
    assert dot_value == known_value


def test_sum():
    """
    This function test to make sure the values are properly being summed
    """
    aux_array = [(0, 1), (2, 1), (4, 2)]
    n_mod = 6
    aux_101020 = AuxiliaryVector(aux_array, n_mod)
    aux_sum = aux_101020.sum()
    known_sum = 4
    assert aux_sum == known_sum
    # np.sum test
    known_np_sum = 4
    aux_np_sum = np.sum(aux_101020)
    assert aux_np_sum == known_np_sum


def test_todense():
    """
    This function test that a sparse vector is properly being made dense
    """
    aux_101010 = AuxiliaryVector([(0, 1), (2, 1), (4, 1)], 6)
    aux_101010 = aux_101010.todense()
    known_aux = (1, 0, 1, 0, 1, 0)
    assert tuple(aux_101010) == known_aux


def test_toarray():
    """
    This function test that a sparse vector is properly being arranged into an array
    """
    aux_010101 = AuxiliaryVector([(1, 1), (3, 1), (5, 1)], 6)
    aux_010101 = aux_010101.toarray()
    known_array = np.array([[1, 1], [3, 1], [5, 1]])
    assert np.array_equal(aux_010101, known_array)


def test_get_values():
    """
    This function test whether the correct sub-indexed values are being grabbed by
    the get_values function
    """
    aux_101010 = AuxiliaryVector([(0, 1), (2, 1), (4, 1)], 6)
    values = aux_101010.get_values([4, 5])
    known_values = np.array([1, 0])
    assert np.array_equal(values, known_values)


def test_get_values_nonzero():
    """
    This function test whether the correct sub-indexed nonzero values are being grabbed
    by the get_values_nonzero function
    """
    aux_101010 = AuxiliaryVector([(0, 1), (2, 1), (4, 1)], 6)
    values = aux_101010.get_values_nonzero([2, 3, 4, 5])
    known_values = np.array([1, 1])
    assert np.array_equal(values, known_values)


def test_hash_from_estep():
    """
    This function test that the returns the hash of a new Auxiliary Vector
    with the desired step in the given mode is the correct hash
    """
    # Define constants
    aux_2000 = AuxiliaryVector([(0, 2)], 4)
    aux_1001 = AuxiliaryVector([(0, 1), (3, 1)], 4)
    aux_1011 = AuxiliaryVector([(0, 1), (2, 1), (3, 1)], 4)
    aux_1000 = AuxiliaryVector([(0, 1)], 4)
    aux_0000 = AuxiliaryVector([], 4)
    hash_m1 = hash(((0, -1),))

    # test when step = 0
    assert aux_1000.hash == aux_1000.hash_from_e_step(3, 0)
    assert aux_0000.hash == aux_0000.hash_from_e_step(0, 0)
    assert aux_1011.hash == aux_1011.hash_from_e_step(2, 0)

    # test when step = 1
    assert aux_1001.hash == aux_1000.hash_from_e_step(3, 1)
    assert aux_2000.hash == aux_1000.hash_from_e_step(0, 1)
    assert aux_1000.hash == aux_0000.hash_from_e_step(0, 1)
    assert aux_1011.hash == aux_1001.hash_from_e_step(2, 1)

    # test when step = -1
    assert hash_m1 == aux_0000.hash_from_e_step(0, -1)
    assert aux_0000.hash == aux_1000.hash_from_e_step(0, -1)
    assert aux_1000.hash == aux_2000.hash_from_e_step(0, -1)
    assert aux_1000.hash == aux_1001.hash_from_e_step(3, -1)
    assert aux_1001.hash == aux_1011.hash_from_e_step(2, -1)


def test_e_step():
    """
    This function test whether e_step returns a new Auxiliary Vector with the desired
     step in the given mode
    """
    # Define constants
    aux_2000 = AuxiliaryVector([(0, 2)], 4)
    aux_1001 = AuxiliaryVector([(0, 1), (3, 1)], 4)
    aux_1000 = AuxiliaryVector([(0, 1)], 4)
    aux_1011 = AuxiliaryVector([(0, 1), (2, 1), (3, 1)], 4)
    aux_0000 = AuxiliaryVector([], 4)
    Aux_m1 = AuxiliaryVector([(0, -1)], 4)

    # test when step = 0
    assert aux_1000 == aux_1000.e_step(3, 0)
    assert aux_0000 == aux_0000.e_step(0, 0)
    assert aux_1011 == aux_1011.e_step(2, 0)

    # test when step = 1
    assert aux_1001 == aux_1000.e_step(3, 1)
    assert aux_2000 == aux_1000.e_step(0, 1)
    assert aux_1000 == aux_0000.e_step(0, 1)
    assert aux_1011 == aux_1001.e_step(2, 1)

    # test when step = -1
    assert Aux_m1 == aux_0000.e_step(0, -1)
    assert aux_0000 == aux_1000.e_step(0, -1)
    assert aux_1000 == aux_2000.e_step(0, -1)
    assert aux_1000 == aux_1001.e_step(3, -1)
    assert aux_1001 == aux_1011.e_step(2, -1)


def test_index_analytic():
    """
    This function provides a test to ensure an absolute index value is returned
    for an auxiliary vector using an analytic function of the indices
    """
    aux = AuxiliaryVector([(0, 1), (2, 1)], 4)
    # known result based on alpha numerical ordering
    known_ind = 7
    assert aux.absolute_index == known_ind


def test_tuple_from_e_step():
    """
    Test whether tuple_from_e_step returns the sparse correct tuple representation of
    the auxiliary
    """
    # Constants
    aux_0101 = AuxiliaryVector([(1, 1), (3, 1)], 4)
    aux_1010 = AuxiliaryVector([(0, 1), (2, 1)], 4)
    aux_1000 = AuxiliaryVector([(0, 1)], 4)
    aux_0000 = AuxiliaryVector([], 4)
    aux_empty = AuxiliaryVector([], 4)

    # test when step = 0
    known_1000 = ((0, 1),)
    known_0000 = ()
    assert known_1000 == aux_1000.tuple_from_e_step(2, 0)
    assert known_0000 == aux_0000.tuple_from_e_step(0, 0)

    # test when mode + step < 0
    known_tuple = ((0, -1),)
    assert known_tuple == aux_0000.tuple_from_e_step(0, -1)

    # test when len(self.dict_aux_vec) == 0
    known_tuple = ((1, 1),)
    assert known_tuple == aux_empty.tuple_from_e_step(1, 1)

    # test when mode is in array_aux_vec[:, 0] and mode + step = 0
    known_tuple = ((3, 1),)
    assert known_tuple == aux_0101.tuple_from_e_step(1, -1)

    # test when mode is in array_aux_vec[:, 0]
    known_tuple = ((1, 2), (3, 1))
    assert known_tuple == aux_0101.tuple_from_e_step(1, 1)

    # test else
    known_tuple = ((0, 1), (2, 1), (3, 1))
    assert known_tuple == aux_1010.tuple_from_e_step(3, 1)
