import pytest
import numpy as np
from pyhops.dynamics.hops_aux import AuxiliaryVector
from pyhops.util.exceptions import AuxError


def test_auxvec_ordering():
    """
    This function test whether an incorrectly ordered array_aux_vex properly raises
    the correct AuxError
    """
    aux_1010 = AuxiliaryVector([(0, 1), (2, 1)], 4)
    assert type(aux_1010) == AuxiliaryVector
    with pytest.raises(AuxError) as excinfo:
        aux_1010 = AuxiliaryVector([(2, 1), (0, 1)], 4)
        assert 'array_aux_vec not properly ordered' in str(excinfo.value)


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


def test_add_aux_connect():
    """
    Test whether add_aux_connect updates the HopsAux object to contain a pointer to the
    other HopsAux objects it is connected to.
    """
    # Define Constants
    aux_1001 = AuxiliaryVector([(0, 1), (3, 1)], 4)
    aux_1011 = AuxiliaryVector([(0, 1), (2, 1), (3, 1)], 4)
    aux_1000 = AuxiliaryVector([(0, 1)], 4)
    aux_1002 = AuxiliaryVector([(0, 1), (3, 2)], 4)

    # Test when type == 1
    aux_1001.add_aux_connect(2, aux_1011, 1)
    assert aux_1001.dict_aux_p1[2] == aux_1011

    # Test when type == -1
    aux_1001.add_aux_connect(3, aux_1000, -1)
    assert aux_1001.dict_aux_m1[3] == aux_1000

    # # Test when type != +/- 1
    with pytest.raises(AuxError) as excinfo:
        aux_1002.add_aux_connect(3, aux_1000, 2)
        assert 'There is a problem in the hierarchy: add_aux_connect does not support ' \
               'type=2' in str(excinfo.value)


def test_remove_aux_connect():
    """
    Test whether the remove_aux_connect function removes the connection between the
    HopsAux object and another connected with type (+1/-1) along index mode.
    """

    # Define Constants
    aux_1001 = AuxiliaryVector([(0, 1), (3, 1)], 4)
    aux_1011 = AuxiliaryVector([(0, 1), (2, 1), (3, 1)], 4)
    aux_1000 = AuxiliaryVector([(0, 1)], 4)

    # Test when type == 1
    aux_1001.add_aux_connect(2, aux_1011, 1)
    aux_1001.remove_aux_connect(2, 1)
    assert aux_1001.dict_aux_p1 == {}

    # Test when type == -1
    aux_1001.add_aux_connect(3, aux_1000, -1)
    aux_1001.remove_aux_connect(3, -1)
    assert aux_1001.dict_aux_m1 == {}

    # Test when type != +/- 1
    with pytest.raises(AuxError) as excinfo:
        aux_1001.remove_aux_connect(3, 2)
        assert 'There is a problem in the hierarchy: remove_aux_connect does not ' \
               'support ' \
               'type=2' in str(excinfo.value)


def test_remove_pointers():
    """
    This will test if the remove_pointers function removes all pointers targeting the
    current HopsAux object from the set of HopsAux objects it has connections to.
    """

    # Define Constants
    aux_1012 = AuxiliaryVector([(0, 1), (2, 1), (3, 2)], 4)
    aux_1011 = AuxiliaryVector([(0, 1), (2, 1), (3, 1)], 4)
    aux_1010 = AuxiliaryVector([(0, 1), (2, 1)], 4)

    # Test with both +/- 1 additions
    aux_1011.add_aux_connect(3, aux_1012, 1)
    aux_1011.add_aux_connect(3, aux_1010, -1)
    aux_1012.add_aux_connect(3,aux_1011,-1)
    aux_1010.add_aux_connect(3,aux_1011,1)
    aux_1011.remove_pointers()
    assert aux_1011.dict_aux_p1 == {}
    assert aux_1011.dict_aux_m1 == {}
    assert aux_1012.dict_aux_m1 == {}
    assert aux_1010.dict_aux_p1 == {}


def test_difference_by_mode():
    """
    This test will ensure that the difference_by_mode function is returning the
    correct mode in which one HopsAux object differs by another HopsAux object, if the
    difference is only 1 step. This function will also test that an error is called
    if the two objects differ by more than 1 step.
    """

    # Define Constants
    aux_1012 = AuxiliaryVector([(0, 1), (2, 1), (3, 2)], 4)
    aux_3012 = AuxiliaryVector([(0, 3), (2, 1), (3, 2)], 4)
    aux_2012 = AuxiliaryVector([(0, 2), (2, 1), (3, 2)], 4)
    aux_1112 = AuxiliaryVector([(0, 1), (1, 1), (2, 1), (3, 2)], 4)
    aux_1022 = AuxiliaryVector([(0, 1), (2, 2), (3, 2)], 4)
    aux_1013 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 4)
    aux_10130 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 5)

    # Test mode 0
    difference_0 = aux_1012.difference_by_mode(aux_3012)
    assert difference_0 is False

    # Test mode 1
    difference_1 = aux_1012.difference_by_mode(aux_1112)
    assert difference_1 == [1]

    # Test mode 2
    difference_2 = aux_1012.difference_by_mode(aux_1022)
    assert difference_2 == [2]

    # Test mode 3
    difference_3 = aux_1012.difference_by_mode(aux_1013)
    assert difference_3 == [3]

    # Test when mode of difference is more than one
    difference_many = aux_2012.difference_by_mode(aux_1112)
    assert difference_many is False

    # Test when the two HopsAux objects don't belong to the same hierarchy
    with pytest.raises(AssertionError):
        aux_10130.difference_by_mode(aux_1013)
