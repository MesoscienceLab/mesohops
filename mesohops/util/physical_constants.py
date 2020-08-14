# Function Title: hysical Constants
# Project: Exciton Dynamics
# Author: Doran I. G. Bennett
# Date:January 7, 2016

# Description:
# This is a list of physical constants with the "correct" units.


from __future__ import division, print_function
from numpy import pi


def main():
    pass


if __name__ == "__main__":
    main()

# Calculation Constants
# ---------------------
precision = 1e-8


# Physical Constants
# ------------------
h = 33357.179  # Units: cm^-1*fs
hbar = h / (2 * pi)  # Units: cm^-1*fs/rad
kB = 0.69503  # Units: cm^-1/K
c = 299.79  # Units: nm/fs

# Unit Conversion
# Unit conversion[A][B] converts A<--B
_ev = {"cm-1": 1 / 8065.6, "J": 1 / 1.60218e-19, "meV": 1000.0, "eV": 1.0}
_cm_m1 = {"eV": 8065.6, "J": 8065.6 / 1.60218e-19, "meV": 8.0656, "cm-1": 1}
_j = {
    "cm-1": (1.60218e-19 / 8065.6),
    "eV": (1.60218e-19),
    "meV": ((1.60218e-19) * 1000),
    "J": 1.0,
}
convert_energy_units = {"eV": _ev, "cm-1": _cm_m1, "J": _j}
