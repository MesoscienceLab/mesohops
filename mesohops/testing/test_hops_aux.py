import pytest
import numpy as np
from mesohops.dynamics.hops_aux import AuxiliaryVector
from mesohops.util.exceptions import AuxError


def test_auxvec_ordering():
    """
    This function test whether an incorrectly ordered array_aux_vex properly raises
    the correct AuxError
    """
    aux_1010 = AuxiliaryVector([(0, 1), (2, 1)], 4)
    assert type(aux_1010) == AuxiliaryVector
    try:
        aux_1010 = AuxiliaryVector([(2, 1), (0, 1)], 4)
    except AuxError as excinfo:
        assert 'array_aux_vec not properly ordered' in str(excinfo)


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
    try:
        aux_1002.add_aux_connect(3, aux_1000, 2)
    except AuxError as excinfo:
        assert 'There is a problem in the hierarchy: add_aux_connect does not support ' \
               'type=2' in str(excinfo)


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
    try:
        aux_1001.remove_aux_connect(3, 2)
    except AuxError as excinfo:
        assert 'There is a problem in the hierarchy: remove_aux_connect does not ' \
               'support ' \
               'type=2' in str(excinfo)

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


def test_construct_id():

    aux = AuxiliaryVector([], 4)
    aux_01 = AuxiliaryVector([(0, 1), (1, 1)], 2)
    aux_1012 = AuxiliaryVector([(0, 1), (2, 1), (3, 2)], 4)
    aux_3012 = AuxiliaryVector([(0, 3), (2, 1), (3, 2)], 4)
    aux_2012 = AuxiliaryVector([(0, 2), (2, 1), (3, 2)], 4)
    aux_1112 = AuxiliaryVector([(0, 1), (1, 1), (2, 1), (3, 2)], 4)
    aux_1022 = AuxiliaryVector([(0, 1), (2, 2), (3, 2)], 4)
    aux_1013 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 4)
    aux_10130 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 5)

    identity_str = aux.id
    identity_str_01 = aux_01.id
    identity_str_1012 = aux_1012.id
    identity_str_3012 = aux_3012.id
    identity_str_2012 = aux_2012.id
    identity_str_1112 = aux_1112.id
    identity_str_1022 = aux_1022.id
    identity_str_1013 = aux_1013.id
    identity_str_10130 = aux_10130.id

    assert identity_str == ''
    assert identity_str_01 == '01'
    assert identity_str_1012 == '0233'
    assert identity_str_3012 == '000233'
    assert identity_str_2012 == '00233'
    assert identity_str_1112 == '01233'
    assert identity_str_1022 == '02233'
    assert identity_str_1013 == '02333'
    assert identity_str_10130 == '02333'

    aux = AuxiliaryVector([], 400)
    aux_01 = AuxiliaryVector([(0, 1), (1, 1)], 200)
    aux_1012 = AuxiliaryVector([(0, 1), (2, 1), (300, 2)], 400)
    aux_3012 = AuxiliaryVector([(0, 3), (3, 2), (200, 1)], 400)
    aux_2012 = AuxiliaryVector([(0, 2), (2, 1), (3, 2)], 400)
    aux_1112 = AuxiliaryVector([(0, 1), (1, 1), (2, 1), (3, 2)], 400)
    aux_1022 = AuxiliaryVector([(0, 1), (2, 2), (300, 2)], 400)
    aux_1013 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 400)
    aux_10130 = AuxiliaryVector([(0, 1), (3, 3), (20, 1)], 500)

    identity_str = aux.id
    identity_str_01 = aux_01.id
    identity_str_1012 = aux_1012.id
    identity_str_3012 = aux_3012.id
    identity_str_2012 = aux_2012.id
    identity_str_1112 = aux_1112.id
    identity_str_1022 = aux_1022.id
    identity_str_1013 = aux_1013.id
    identity_str_10130 = aux_10130.id

    assert identity_str == ''
    assert identity_str_01 == '000001'
    assert identity_str_1012 == '000002300300'
    assert identity_str_3012 == '000000000003003200'
    assert identity_str_2012 == '000000002003003'
    assert identity_str_1112 == '000001002003003'
    assert identity_str_1022 == '000002002300300'
    assert identity_str_1013 == '000002003003003'
    assert identity_str_10130 == '000003003003020'


def test_get_list_id_up():
    aux = AuxiliaryVector([], 4)
    aux_01 = AuxiliaryVector([(0, 1), (1, 1)], 2)
    aux_0233 = AuxiliaryVector([(0, 1), (2, 1), (3, 2)], 4)
    aux_000233 = AuxiliaryVector([(0, 3), (2, 1), (3, 2)], 4)
    aux_00233 = AuxiliaryVector([(0, 2), (2, 1), (3, 2)], 4)
    aux_01233 = AuxiliaryVector([(0, 1), (1, 1), (2, 1), (3, 2)], 4)
    aux_02233 = AuxiliaryVector([(0, 1), (2, 2), (3, 2)], 4)
    aux_02333 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 4)
    aux_02333_2 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 5)

    # First test function using all modes

    identity_str_up = aux.get_list_id_up([0, 1])
    identity_str_up_01 = aux_01.get_list_id_up([0, 1, 2, 3])
    identity_str_up_0233 = aux_0233.get_list_id_up([0, 1, 2, 3])
    identity_str_up_000233 = aux_000233.get_list_id_up([0, 1, 2, 3])
    identity_str_up_00233 = aux_00233.get_list_id_up([0, 1, 2, 3])
    identity_str_up_01233 = aux_01233.get_list_id_up([0, 1, 2, 3])
    identity_str_up_02233 = aux_02233.get_list_id_up([0, 1, 2, 3])
    identity_str_up_02333 = aux_02333.get_list_id_up([0, 1, 2, 3])
    identity_str_up_02333_2 = aux_02333_2.get_list_id_up([0, 1, 2, 3, 4])

    assert np.all(identity_str_up[0] == ['0', '1'])
    assert np.all(identity_str_up[1] == [0, 0])

    assert np.all(identity_str_up_01[0] == ['001', '011', '012', '013'])
    assert np.all(identity_str_up_01[1] == [1, 1, 0, 0])

    assert np.all(identity_str_up_0233[0] == ['00233', '01233', '02233', '02333'])
    assert np.all(identity_str_up_0233[1] == [1, 0, 1, 2])

    assert np.all(identity_str_up_000233[0] == ['0000233', '0001233', '0002233', '0002333'])
    assert np.all(identity_str_up_000233[1] == [3, 0, 1, 2])

    assert np.all(identity_str_up_00233[0] == ['000233', '001233', '002233', '002333'])
    assert np.all(identity_str_up_00233[1] == [2, 0, 1, 2])

    assert np.all(identity_str_up_01233[0] == ['001233', '011233', '012233', '012333'])
    assert np.all(identity_str_up_01233[1] == [1, 1, 1, 2])

    assert np.all(identity_str_up_02233[0] == ['002233', '012233', '022233', '022333'])
    assert np.all(identity_str_up_02233[1] == [1, 0, 2, 2])

    assert np.all(identity_str_up_02333[0] == ['002333', '012333', '022333', '023333'])
    assert np.all(identity_str_up_02333[1] == [1, 0, 1, 3])

    assert np.all(
        identity_str_up_02333_2[0] == ['002333', '012333', '022333', '023333', '023334'])
    assert np.all(identity_str_up_02333_2[1] == [1, 0, 1, 3, 0])

    # Now test function using some or only unoccupied modes

    identity_str_up = aux.get_list_id_up([1])
    identity_str_up_01 = aux_01.get_list_id_up([1, 2])
    identity_str_up_0233 = aux_0233.get_list_id_up([1])
    identity_str_up_000233 = aux_000233.get_list_id_up([1, 2, 3])
    identity_str_up_00233 = aux_00233.get_list_id_up([2])
    identity_str_up_01233 = aux_01233.get_list_id_up([0, 1])
    identity_str_up_02233 = aux_02233.get_list_id_up([0, 1])
    identity_str_up_02333 = aux_02333.get_list_id_up([4])
    identity_str_up_02333_2 = aux_02333_2.get_list_id_up([5, 6])

    assert np.all(identity_str_up[0] == ['1'])
    assert np.all(identity_str_up[1] == [0])

    assert np.all(identity_str_up_01[0] == ['011', '012'])
    assert np.all(identity_str_up_01[1] == [1, 0])

    assert np.all(identity_str_up_0233[0] == ['01233'])
    assert np.all(identity_str_up_0233[1] == [0])

    assert np.all(identity_str_up_000233[0] == ['0001233', '0002233', '0002333'])
    assert np.all(identity_str_up_000233[1] == [0, 1, 2])

    assert np.all(identity_str_up_00233[0] == ['002233'])
    assert np.all(identity_str_up_00233[1] == [1])

    assert np.all(identity_str_up_01233[0] == ['001233', '011233'])
    assert np.all(identity_str_up_01233[1] == [1, 1])

    assert np.all(identity_str_up_02233[0] == ['002233', '012233'])
    assert np.all(identity_str_up_02233[1] == [1, 0])

    assert np.all(identity_str_up_02333[0] == ['023334'])
    assert np.all(identity_str_up_02333[1] == [0])

    assert np.all(identity_str_up_02333_2[0] == ['023335', '023336'])
    assert np.all(identity_str_up_02333_2[1] == [0, 0])


def test_get_list_id_down():
    aux = AuxiliaryVector([], 4)
    aux_01 = AuxiliaryVector([(0, 1), (1, 1)], 2)
    aux_1012 = AuxiliaryVector([(0, 1), (2, 1), (3, 2)], 4)
    aux_3012 = AuxiliaryVector([(0, 3), (2, 1), (3, 2)], 4)
    aux_2012 = AuxiliaryVector([(0, 2), (2, 1), (3, 2)], 4)
    aux_1112 = AuxiliaryVector([(0, 1), (1, 1), (2, 1), (3, 2)], 4)
    aux_1022 = AuxiliaryVector([(0, 1), (2, 2), (3, 2)], 4)
    aux_1013 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 4)
    aux_10130 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 5)


    identity_str_down_01 = aux_01.get_list_id_down()
    identity_str_down_1012 = aux_1012.get_list_id_down()
    identity_str_down_3012 = aux_3012.get_list_id_down()
    identity_str_down_2012 = aux_2012.get_list_id_down()
    identity_str_down_1112 = aux_1112.get_list_id_down()
    identity_str_down_1022 = aux_1022.get_list_id_down()
    identity_str_down_1013 = aux_1013.get_list_id_down()
    identity_str_down_10130 = aux_10130.get_list_id_down()


    assert np.all(identity_str_down_01[0] == ['1', '0'])
    assert np.all(identity_str_down_01[1] == [1, 1])
    assert np.all(identity_str_down_01[2] == [0, 1])

    assert np.all(identity_str_down_1012[0] == ['233', '033', '023'])
    assert np.all(identity_str_down_1012[1] == [1, 1, 2])
    assert np.all(identity_str_down_1012[2] == [0, 2, 3])

    assert np.all(identity_str_down_3012[0] == ['00233', '00033', '00023'])
    assert np.all(identity_str_down_3012[1] == [3, 1, 2])
    assert np.all(identity_str_down_3012[2] == [0, 2, 3])

    assert np.all(identity_str_down_2012[0] == ['0233', '0033', '0023'])
    assert np.all(identity_str_down_2012[1] == [2, 1, 2])
    assert np.all(identity_str_down_2012[2] == [0, 2, 3])

    assert np.all(identity_str_down_1112[0] == ['1233', '0233', '0133', '0123'])
    assert np.all(identity_str_down_1112[1] == [1, 1, 1, 2])
    assert np.all(identity_str_down_1112[2] == [0, 1, 2, 3])

    assert np.all(identity_str_down_1022[0] == ['2233', '0233', '0223'])
    assert np.all(identity_str_down_1022[1] == [1, 2, 2])
    assert np.all(identity_str_down_1022[2] == [0, 2, 3])

    assert np.all(identity_str_down_1013[0] == ['2333', '0333', '0233'])
    assert np.all(identity_str_down_1013[1] == [1, 1, 3])
    assert np.all(identity_str_down_1013[2] == [0, 2, 3])

    assert np.all(identity_str_down_10130[0] == ['2333', '0333', '0233'])
    assert np.all(identity_str_down_10130[1] == [1, 1, 3])
    assert np.all(identity_str_down_10130[2] == [0, 2, 3])

    aux = AuxiliaryVector([], 400)
    aux_000001 = AuxiliaryVector([(0, 1), (1, 1)], 200)
    aux_000002300300 = AuxiliaryVector([(0, 1), (2, 1), (300, 2)], 400)
    aux_000000000003003200 = AuxiliaryVector([(0, 3), (3, 2), (200, 1)], 400)
    aux_000000002003003 = AuxiliaryVector([(0, 2), (2, 1), (3, 2)], 400)
    aux_000001002003003 = AuxiliaryVector([(0, 1), (1, 1), (2, 1), (3, 2)], 400)
    aux_000002002300300 = AuxiliaryVector([(0, 1), (2, 2), (300, 2)], 400)
    aux_000002003003003 = AuxiliaryVector([(0, 1), (2, 1), (3, 3)], 400)
    aux_0003030320 = AuxiliaryVector([(0, 1), (3, 3), (20, 1)], 50)


    identity_str_down_000001 = aux_000001.get_list_id_down()
    identity_str_down_000002300300 = aux_000002300300.get_list_id_down()
    identity_str_down_000000000003003200 = aux_000000000003003200.get_list_id_down()
    identity_str_down_000000002003003 = aux_000000002003003.get_list_id_down()
    identity_str_down_000001002003003 = aux_000001002003003.get_list_id_down()
    identity_str_down_000002002300300 = aux_000002002300300.get_list_id_down()
    identity_str_down_000002003003003 = aux_000002003003003.get_list_id_down()
    identity_str_down_0003030320 = aux_0003030320.get_list_id_down()



    assert np.all(identity_str_down_000001[0] == ['001', '000'])
    assert np.all(identity_str_down_000001[1] == [1, 1])
    assert np.all(identity_str_down_000001[2] == [0, 1])

    assert np.all(
        identity_str_down_000002300300[0] == ['002300300', '000300300', '000002300'])
    assert np.all(identity_str_down_000002300300[1] == [1, 1, 2])
    assert np.all(identity_str_down_000002300300[2] == [0, 2, 300])

    assert np.all(
        identity_str_down_000000000003003200[0] == ['000000003003200', '000000000003200',
                                                '000000000003003'])
    assert np.all(identity_str_down_000000000003003200[1] == [3, 2, 1])
    assert np.all(identity_str_down_000000000003003200[2] == [0, 3, 200])

    assert np.all(identity_str_down_000000002003003[0] == ['000002003003', '000000003003',
                                                       '000000002003'])
    assert np.all(identity_str_down_000000002003003[1] == [2, 1, 2])
    assert np.all(identity_str_down_000000002003003[2] == [0, 2, 3])

    assert np.all(identity_str_down_000001002003003[0] == ['001002003003', '000002003003',
                                                       '000001003003', '000001002003'])
    assert np.all(identity_str_down_000001002003003[1] == [1, 1, 1, 2])
    assert np.all(identity_str_down_000001002003003[2] == [0, 1, 2, 3])

    assert np.all(identity_str_down_000002002300300[0] == ['002002300300', '000002300300',
                                                       '000002002300'])
    assert np.all(identity_str_down_000002002300300[1] == [1, 2, 2])
    assert np.all(identity_str_down_000002002300300[2] == [0, 2, 300])

    assert np.all(identity_str_down_000002003003003[0] == ['002003003003', '000003003003',
                                                       '000002003003'])
    assert np.all(identity_str_down_000002003003003[1] == [1, 1, 3])
    assert np.all(identity_str_down_000002003003003[2] == [0, 2, 3])

    assert np.all(identity_str_down_0003030320[0] == ['03030320', '00030320', '00030303'])
    assert np.all(identity_str_down_0003030320[1] == [1, 3, 1])
    assert np.all(identity_str_down_0003030320[2] == [0, 3, 20])
