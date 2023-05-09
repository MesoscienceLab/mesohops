# Title: QArchitect HOPS Library - Exceptions and Errors
# Author: Doran I. G. Bennett
# Date: Feb. 8, 2019


class UnsupportedRequest(Exception):
    """
    An exception to be raised for unsupported requests.
    """
    def __init__(self, request, function, override=False):
        """
        Parameters
        ----------
        1. request : str
                     The request by the user.

        2. function : str
                      The function in which an unsupported request is raised.

        3. override : boolean
                      Flag that simplifies the exception message.
        """
        self.request = request
        self.function = function
        self.override = override

    def __str__(self):

        if not self.override:
            msg = "The current code does not support {} in the {} function.".format(
            self.request, self.function
            )
        if self.override:
            msg = "{} in the {} function.".format(self.request, self.function)
        return str(msg)

class LockedException(Exception):
    """
    Exception called when a user is trying to access a restricted function.
    """
    def __init__(self, func_name, class_name):
        """
        Parameters
        ----------
        1. func_name : str
                       The restricted function called by the user.

        2. class_name : str
                        The class to which the restricted function belongs.
        """
        self.func_name = func_name
        self.class_name = class_name

    def __str__(self):

        msg = (
            "The "
            + str(self.class_name)
            + "is currently locked - the attempt to access "
            + str(self.func)
            + " has been canceled."
        )
        return str(msg)

class AuxError(Exception):
    """
    Catch-all exceptions for misconfiguration of an AuxiliaryVector.
    """
    def __init__(self, sub_msg):
        """
        Parameters
        ----------
        1. sub_msg : str
                     Error message.
        """
        self.sub_msg = sub_msg

    def __str__(self):
        msg = "There is a problem in the hierarchy: " + self.sub_msg
        return str(msg)


class TrajectoryError(Exception):
    """
    Catch-all exceptions for misconfiguration of a HopsTrajectory.
    """
    def __init__(self, sub_msg):
        """
        Parameters
        ----------
        1. sub_msg : str
                     Error message.
        """
        self.sub_msg = sub_msg

    def __str__(self):
        msg = (
            self.sub_msg + " not supported in current implementation of HOPSTrajectory."
        )
        return str(msg)
