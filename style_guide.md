# MesoScience Lab Code Style Guide
DISCLAIMER: This style guide is a living document. While we continue aligning the codebase with it, please view the guide as aspirational rather than fully implemented.

NOTE: If a coding situation arises that is not explicitly covered here, it is best to default to the PEP 8 style guide as the baseline standard. For the official PEP 8 guide, see: https://peps.python.org/pep-0008/.

# Formatting Rules
## Line Length
* Our group currently uses 88 characters for our maximum line length (both for code and documentation). When overflow occurs, the statement should be broken out over multiple lines for readability as determined in code review.

## Indentation
* Indentations should be done with 4 spaces.

## Line Breaks
* Two blank lines after:
    * Imports (section)
    * Classes
    * Functions
* One blank line after:
    * Docstrings
    * Methods

## Strings
* Strings should use single quotes, `'string'`, when possible.
* Exceptions:
    *  Docstrings
    *  If your string must contain apostrophes/single quotes

## Spacing
* There should generally be a single space after, but not before, commas and colons.
    * For commas, no space should be included if there is a trailing comma in, say, a tuple initialization.
```python
# GOOD
x, y = (0, 1)

# GOOD
x = (0,)

# BAD
x = (0, )
```
* For colons, don't include spacing before or after the colon if using for slicing (with exception of spaces added due to commas).
```python
# GOOD
X2_example_array = array_things[:, ::2]  

# BAD
X2_example_array = array_things[: ,: : 2]
```
* Binary operators should generally have a space on either side.
    * Exceptions to this may be made to indicate order of operations.
```python
# GOOD
result = a + b**2   

# ACCEPTABLE (but less clear)
result = a + b ** 2 

# BAD
result = a+b**2  
```
* Assignment and comparison operators (e.g. `=`, `+=`, `==`, `>`) should always have a space on either side, except in the case of default value assignment.
```python
# GOOD
def calc_energy(temp=300, pressure=1.0): 
    
# BAD
def calc_energy(temp = 300):            
```

# Naming Conventions
## Allowed Characters
* Any printable ASCII character is valid. Non-printable ASCII characters (e.g. Greek letters) are forbidden.

```python
# GOOD
phi     

# BAD
Φ     
```

## Standard Abbreviations
* Always use the following abbreviated forms outside of documentation inside your code:

| Word/Phrase                | Abbreviation |
| -------------------------- | ------------ |
| Absolute                   | `abs`        |
| Auxiliary                  | `aux`        |
| Bath Correlation Function  | `bcf`        |
| Calculate                  | `calc`       |
| Correlation                | `corr`       |
| Deletion                   | `del`        |
| Derivative                 | `deriv`      |
| Destination                | `dest`       |
| Diagonal                   | `diag`       |
| Dictionary                 | `dict`       |
| Dimension                  | `dim`        |
| Drude Lorentz              | `dl`         |
| Dyadic                     | `dyad`       |
| Expectation                | `expect`     |
| Exponential                | `exp`        |
| Fast Fourier Transform     | `fft`        |
| Function                   | `func`       |
| Hierarchy                  | `hier`       |
| Hilbert                    | `hilb`       |
| Index                      | `idx`        |
| L-Operator                 | `lop`        |
| Low-Temperature Correction | `ltc`        |
| Memory                     | `mem`        |
| Normalize                  | `norm`       |
| Parameters                 | `param`      |
| Physical                   | `phys`       |
| Random                     | `rand`       |
| Relative                   | `rel`        |
| Squared                    | `sq`         |
| System                     | `sys`        |
| Temporary                  | `tmp`        |
| Temperature                | `temp`       |
| Threshold                  | `thresh`     |
| Underdamped                | `ud`         |

Note: any grammatical version version of these words use the same abbreviation (e.g. normalize/normalized/normalizing/norm)
```python
# GOOD
def calc_energy_deriv(temp_phys): 
    
# BAD
def calculate_energy_derivative(temperature_physical):
```

## Capitalization
* Classes should be named in CapWords (CamelCase) style.
* Methods, functions, and most variables should be named in lowercase.
* Global constants should be in all caps (e.g., `HBAR`)

## Lists & N-D Arrays
* The name of a list should begin with `list_` (e.g., `list_pigments`).
* The name of N-D Arrays should have a capital letter describing the variable followed by a number describing the dimensionality of the variable. Lastly, a descriptive name follows separated by an underscore (e.g., `H2_sys_hamiltonian`).
* If an array's primary use is to be iterated over in the fashion of a list, the list naming convention is acceptable.

```python
# GOOD
H2_sys_hamiltonian
H1_energy_levels    

# BAD
hamiltonian_2d 

# BAD (missing dimensionality) 
sys_hamiltonian     

# BAD (lowercase variable letter) 
h2_sys_hamiltonian  
```

## Mathematical Variables
* For variables representing mathematical objects (matrices, vectors, scalars) that use established notation, separate the mathematical symbol from descriptive suffixes with underscores.

```python
# GOOD
K_tmp       
theta_initial  
mu_hat      

# BAD
Ktmp        
thetainitial
muhat     
```

# Documentation
## Docstrings
### Structure
* For classes:
```python
"""
Brief description of class.
"""
```

* For methods/functions:
    - Triple quotes directly under `def`.
    - Order: *Brief description → Parameters → Returns*.
    - Headings: `Parameters`, `Returns` (title case, underlined with `----------`).
    - One space after colons, aligned indentation.  
    - One blank line between sections.  

    **Description**
    - Descriptions are concise.

    **Parameters**
    - Numbered list:  
  `1. name: type1 | type2 [units: cm^-1]`  
        - Note: we use brackets, not parens, for units   
    - Description on the next indented line, starting aligned with the first character after the colon.


    **Returns**
    - Same numbered format as parameters.
    -  When a method/function has no returns, do not number returns in docstring. Instead, use the following format:
    ```python
    Returns
    -------
    None
    ```



**Example**
```python
# GOOD
"""
Brief description of what the function does.

Parameters
----------
1. generic_var: float | np.ndarray [units: cm^-1]
                Brief description.
2. rizzler: str [options: 'rizz_yes', 'rizz_no']
            Brief description.

Returns
-------
1. fojangle: float [units: fs^-1]
             Brief description.
"""

# BAD
"""
Brief description of what the function does.

Parameters
----------
1. generic_var: float | np.ndarray (units: cm^-1)
                Brief description.
2. rizzler: str [options: 'rizz_yes', 'rizz_no']
      Brief description.

Returns
-------
1. fojangle: float [fs^-1]
             Brief description.
"""
```
### Rules of Thumb
* Always document the nesting structure of list variables explicitly.

```python
# GOOD 
"""
Parameters
----------
1. list_states: list[list[int]]
                Nested list where outer list contains trajectories
                and inner lists contain state indices.
"""

# BAD (unclear nesting) 
"""
Parameters
----------
1. list_states: list
                List of states.
"""
```

### Referencing
* For **published articles**, cite the relevant section (if applicable) and format references in **abbreviated journal style**, including:
  - Full title  
  - Author list  
  - Abbreviated journal name  
  - Volume, page, year  
  - DOI (if available)

  **Example:**
  * This equation corresponds to Eq. 23 in Section II of "MesoHOPS: Size-invariant scaling calculations of multi-excitation open quantum systems." Brian Citty, Jacob K. Lynd, et al. J. Chem. Phys. 160, 144118 (2024). (DOI: 10.1063/5.0197825)

* For **preprints**, cite the relevant section and provide the full title and link. If a DOI is assigned, include it as well.

  **Example:**
  * This function corresponds to Section S2.B from the SI of "Characterizing the Role of Peierls Vibrations in Singlet Fission with the Adaptive Hierarchy of Pure States," available at https://arxiv.org/abs/2505.02292 (DOI: 10.48550/arXiv.2505.02292)

## Comments
### Structure
* Comments will be placed directly above the piece of code they describe and will begin with a space followed by a capital letter.
* Exception: when assigning `__slots__`, inline comments should be used.

```python
# GOOD 
# Calculate the trace of the density matrix
trace_value = np.trace(rho2_density)

# GOOD (slots)
__slots__ = (
        'basis',           # Basis management (HopsBasis)
        'storage',         # Storage management (HopsStorage)
        'dsystem_dt',      # System derivative function
        '_phi',            # Full hierarchy vector (current state)
)

# BAD (inline comment after code) 
trace_value = np.trace(rho2_density)  # Calculate trace
 
# BAD (lowercase start) 
# calculate the trace of the density matrix
trace_value = np.trace(rho2_density)

# BAD (no space)
#Calculate the trace
trace_value = np.trace(rho2_density)
```

### When to Comment
* We require clear comments for any of the following logic:
    * Array Manipulation
    * Indexing Schemes
    * Memory Management Tricks
    * Physical Justifications
    * Loop Logic
    * Complicated Mathematical Expressions
    * Ambiguous Code (if the code is not easily understood, it should be commented)

# Code Structure
## Classes
### Ordering
* Classes should be structured in the following order:
    1. Docstring
    2. `__slots__`
    3. `__init__`
    4. Other Magic Methods (e.g. `__repr__`, `__new__`, `__str__`, etc.)
    5. Public Methods (`public_method`)
    6. Protected Methods (`_protected_method`)
    7. Private Methods (`__private_method`)
    8. Properties (`@property`)
    9. Setters / Getters (`@property.setter` / `@property.getter`)

```python
# GOOD 
class HamiltonianSystem:
    """
    Represents a quantum Hamiltonian system.
    """
    __slots__ = ['_energy', '_states']
    
    def __init__(self, energy, states):
        self._energy = energy
        self._states = states
    
    def __repr__(self):
        return f'HamiltonianSystem(energy={self._energy})'
    
    def calc_eigenvalues(self):
        # Public method implementation
        pass
    
    def _validate_states(self):
        # Protected method implementation
        pass
    
    @property
    def energy(self):
        return self._energy
```
    
## Methods / Functions
### Return Logic
* Return logic should be as clear as possible. This means two things:
    * Returned variables should be descriptively named.
    * Where possible, simple logic should be returned without first declaring a variable.

```python
# GOOD (complex logic with descriptive variable) 
# Parameters have complicated defaults that aren't similar, so we
# need to build a block of logic for each one:
# thus, we construct the dictionary they live in before returning it.
def define_param_dict(a=None, b=None):
    dict_params = {}
    if a in params:
        dict_params['a'] = a
    else:
        dict_params['a'] = 0
    if b in params:
        dict_params['b'] = b
    else:
        dict_params['b'] = np.sin(a)
    return dict_params

# GOOD (simple logic returned directly)
# Simple definition of parameters can be done in few lines,
# no need to pre-define the return dictionary
def define_param_dict_simple(a, b):
    return {'a': a,
            'b': b}
    
# BAD (unnecessary variable for simple logic) 
def calc_sq_magnitude(vec):
    sq_mag = np.dot(vec, vec)
    return sq_mag

# BAD (unclear variable name) 
def calc_energy_spectrum_max_gap(H2_hamiltonian):
    # Ignore that this is a dumb way of doing this, it's an example.
    result = np.max(np.linalg.eigvalsh(H2_hamiltonian))-np.min(np.linalg.eigvalsh(H2_hamiltonian))
    return result
```

# Imports
## Structure
* Always import the specific submodule required, and reference its members explicitly (e.g., `submodule.function`).
* If importing from external modules, the from statement should be used sparingly for cases in which a specific object or set of objects will exclusively be used. If you are not sure, import the whole module or relevant submodule.
* If importing from internal modules, the from statement should generally be used unless there are namespace conflicts.
* NEVER use `from module import *`. List all objects explicitly.

```python
# GOOD 
import numpy as np
import scipy.sparse as sparse

from our_package.hops import HopsTrajectory, HopsBasis

# BAD 
from numpy import *
from our_package.hops import *
```

## Ordering
* Imports should be divided into 3 sections, divided by a single line of whitespace. The sections are:
    1. Python Standard Library Modules
    2. External Modules
    3. User-Defined Modules
* Within each section, imports are sorted alphabetically after ordering by category:
    1. `import module`
    2. `import module.submodule as sub`
    3. `from module import object`
* Imports should also generally include the specific submodule required (e.g., `submodule.function` over `module.submodule.function`).
```python
# GOOD 
import os
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt

from mesohops.dynamics import HopsTrajectory
from mesohops.util.physical_constants import HBAR

# BAD (sections not separated, not alphabetical) 
import numpy as np
import os
from mesohops.dynamics import HopsTrajectory
import sys
```

# Unit Tests
## Some Definitions
* We will use the following terms when describing our unit tests:
    * Test suite: a cluster of unit test functions that checks all functionalities of a method / function 
    * Test: a single unit test function that checks a single functionality of a method / function
    * Case: a portion of a unit test function that checks a specific scenario of a single functionality of a method / function

## Sectioning Cases / Subcases
**General Rules**
- Each unit test file contains *all tests* for its corresponding source file.  
  Example: `test_hops_trajectory.py` → `hops_trajectory.py`.
- Each **test suite** (function group) tests a single function from the module.
- Each **test** consists of:
  1. A **comment header** describing the functionality being tested.  
  2. A **single test function** that includes all case checks for that functionality.
- Each **case** within the test function has its own inline comment describing the specific scenario.

**Formatting Requirements**
- Use three consistent comment levels to visually organize files:
  - **Suite header:**  
    Placed **above each test suite**, describing which function is being tested.  
    ```python
    # ============================================================
    # TEST SUITE: <function_name>()
    # ============================================================
    ```
  - **Test header:**  
    Placed **above each test function**, describing the functionality being tested.  
    ```python
    # ------------------------------------------------------------
    # TEST: <short description of functionality being tested>
    # ------------------------------------------------------------
    ```
  - **Case comments:**  
    Placed **inside the test function**, before each logical block of assertions.  
    ```python
    # This case tests <specific scenario under this functionality>
    ```

- Maintain consistent spacing and comment alignment.
- Group related assertions under each case.

**Example Structure**
```python
# test_<module>.py
# This file contains all tests for <module>.py

import pytest
from <module> import <func_a>, <func_b>

# ============================================================
# TEST SUITE: <func_a>()
# ============================================================

# ------------------------------------------------------------
# TEST: <short description of functionality being tested>
# ------------------------------------------------------------
def test_<func_a>_<functionality>():
    # This case tests <scenario under this functionality>
    # <assertions>

    # This case tests <scenario under this functionality>
    # <assertions>

    # This case tests <scenario under this functionality>
    # <assertions>
```
## To Test or Not To Test?
* In general, you should test anything more complex than simple arithmetic operations or variable/object initializations.
* The only occasions in which unit tests are not necessary are:
    * If using a method/function you've imported from somewhere else.
    * If it is already tested in a parent class.

# Authorship
## File Header Structure
* We maintain authorship history within our code in each file header. We will use two authorship distinctions: `__author__` and `__maintainer__`.
* The structure of the file header is as follows:

```python
__title__ = "ToadObliterator Class"
__author__ = "Itsam Emario, Wall U. Ouija"
__maintainer__ = "Wall U. Ouija, Toe Debt"
```

### `__author__`
* These are developers responsible for a significant portion of code in a file. Lavish them with praise for anything that goes right.

### `__maintainer__`
* These are active members of the lab responsible for maintaining the relevant section of the code. Execute them if anything is wrong.