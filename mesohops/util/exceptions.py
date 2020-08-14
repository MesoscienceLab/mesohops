# Title: QArchitect HOPS Library - Exceptions and Errors
# Author: Doran I. G. Bennett
# Date: Feb. 8, 2019


class UnsupportedRequest(Exception):
    def __init__(self, request, function):
        self.request = request
        self.function = function

    def __str__(self):
        msg = "The current code does not support {} in the {} function.".format(
            self.request, self.function
        )
        return str(msg)


class FlagError(Exception):
    """Error to catch corrupted files"""

    def __init__(self, expression, message):
        self.message = message
        self.expression = expression
        pass

    def __str__(self):
        msg = message % expression
        return str(msg)


class LockedException(Exception):
    def __init__(self, func_name, class_name):
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


class WrongNumberAux(Exception):
    def __init__(self, naux_teor, calc_aux, n, maxhier):
        self.n = n
        self.hier = maxhier
        self.calc_aux = calc_aux
        self.naux_teor = calc_aux

    def __str__(self):
        msg = (
            " %d number of aux vec doesn't agree with expected %d for %d sites and %d maxhier"
            % (self.calc_aux, self.naux_teor, self.n, self.hier)
        )
        return str(msg)


class AuxError(Exception):
    def __init__(self, sub_msg):
        self.sub_msg = sub_msg

    def __str__(self):
        msg = "There is a problem in the hierarchy: " + self.sub_msg
        return str(msg)


class TrajectoryError(Exception):
    def __init__(self, sub_msg):
        self.sub_msg = sub_msg

    def __str__(self):
        msg = (
            self.sub_msg + " not supported in current implementation of HOPSTrajectory."
        )
        return str(msg)
