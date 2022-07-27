# Title: QArchitect HOPS Library -- Base Class
# Authors: D. I. G. Bennett
# Date: Feb. 11, 2019

"""
This is the base class for all HOPS classes. It implements 
convience functions for initializing and working with classes. 
"""

import copy


from pyhops.util.exceptions import LockedException, UnsupportedRequest


class Dict_wDefaults(object):
    """
    This is a 'Dictionary with Defaults' class that implements a initialization 
    routine for a dictionary that makes use of: 
    1. use input dictionary
    2. default paramter dicitionary
    3. a dictionary that deinfes the types of each user input parameter
    
    The result is a convenient class to inheret from for a number of classes whose 
    primary purpose is to contain user defined inputs. It allows, among other things, 
    for us to check the types of the user input. 
    
    Static Method:
    -------------
    1. _initialize_dictionary
    
    Properties:
    -----------
    1. param
    2. update_param
    """

    def __init__(self):
        pass

    @staticmethod
    def _initialize_dictionary(param, default_params, default_param_types, name):
        """
        This is a utility function for initializing the central
        dictionary of the class. It checks the user inputs against both
        default values and the allowed types. 
        """
        param_new = copy.deepcopy(param)

        # Check that all user keys are suppported
        for key in param_new.keys():
            if not key in default_params.keys():
                raise UnsupportedRequest(key, name)

        # Loop over all possible keys
        for key in default_params.keys():
            if key in param_new.keys():
                # If user specified the wrong type --> raise error
                if not type(param_new[key]) in default_param_types[key]:
                    raise TypeError(
                        str(key)
                        + "-->"
                        + str(param_new[key])
                        + " is of type "
                        + str(type(param_new[key]))
                        + " not the supported types "
                        + str(default_param_types[key])
                    )

            # If user did not specify, then insert default value
            else:
                param_new[key] = copy.deepcopy(default_params[key])

        return param_new

    @property
    def param(self):
        return self.__param

    @param.setter
    def param(self, param_usr):
        self.__param = self._initialize_dictionary(
            param_usr, self._default_param, self._param_types, type(self).__name__
        )

    def update_param(self, param_usr):
        self.__param.update(param_usr)
