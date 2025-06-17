import os
import pytest
import importlib.resources as resources
from mesohops.timing.timing_models import *

# Hardcoded seed and nstate for testing purposes (will not save data)
seed = 1010101010101010
nstate = 1

@pytest.mark.level(3)
def test_absorption():
    """
    Tests the absorption.py script in the mesohops.timing package.
    """
    # Locate the absorption.py file within the mesohops.timing package
    with resources.path("mesohops.timing.timing_models", "absorption.py") as absorption_path:
        # Tests that the absorption file runs without error
        output = os.system(f"python3 {absorption_path} {seed} {nstate}")
        assert output == 0

@pytest.mark.level(3)
def test_fluorescence():
    """
    Tests the fluorescence.py script in the mesohops.timing package.
    """
    # Locate the fluorescence.py file within the mesohops.timing package
    with resources.path("mesohops.timing.timing_models",
                        "fluorescence.py") as fluorescence_path:
        # Tests that the fluorescence file runs without error
        output = os.system(f"python3 {fluorescence_path} {seed} {nstate}")
        assert output == 0

@pytest.mark.level(3)
def test_holstein_1_particle():
    """
    Tests the holstein_1_particle.py script in the mesohops.timing package.
    """
    # Locate the holstein_1_particle.py file within the mesohops.timing package
    with resources.path("mesohops.timing.timing_models",
                        "holstein_1_particle.py") as holstein_1_particle_path:
        # Tests that the holstein_1_particle file runs without error
        output = os.system(f"python3 {holstein_1_particle_path} {seed} {nstate}")
        assert output == 0

@pytest.mark.level(3)
def test_holstein_2_particle():
    """
    Tests the holstein_2_particle.py script in the mesohops.timing package.
    """
    # Locate the holstein_2_particle.py file within the mesohops.timing package
    with resources.path("mesohops.timing.timing_models",
                        "holstein_2_particle.py") as holstein_2_particle_path:
        # Tests that the holstein_2_particle file runs without error
        output = os.system(f"python3 {holstein_2_particle_path} {seed} {nstate}")
        assert output == 0

@pytest.mark.level(3)
def test_markovian_filter():
    """
    Tests the markovian_filter.py script in the mesohops.timing package.
    """
    # Locate the markovian_filter.py file within the mesohops.timing package
    with resources.path("mesohops.timing.timing_models",
                        "markovian_filter.py") as markovian_filter_path:
        # Tests that the markovian_filter file runs without error
        output = os.system(f"python3 {markovian_filter_path} {seed} {nstate}")
        assert output == 0

@pytest.mark.level(3)
def test_longedge_filter():
    """
    Tests the longedge_filter.py script in the mesohops.timing package.
    """
    # Locate the longedge_filter.py file within the mesohops.timing package
    with resources.path("mesohops.timing.timing_models",
                        "longedge_filter.py") as longedge_filter_path:
        # Tests that the longedge_filter file runs without error
        output = os.system(f"python3 {longedge_filter_path} {seed} {nstate}")
        assert output == 0

@pytest.mark.level(3)
def test_triangular_filter():
    """
    Tests the triangular_filter.py script in the mesohops.timing package.
    """
    # Locate the triangular_filter.py file within the mesohops.timing package
    with resources.path("mesohops.timing.timing_models",
                        "triangular_filter.py") as triangular_filter_path:
        # Tests that the triangular_filter file runs without error
        output = os.system(f"python3 {triangular_filter_path} {seed} {nstate}")
        assert output == 0

@pytest.mark.level(3)
def test_peierls():
    """
    Tests the peierls.py script in the mesohops.timing package.
    """
    # Locate the peierls.py file within the mesohops.timing package
    with resources.path("mesohops.timing.timing_models",
                        "peierls.py") as peierls_path:
        # Tests that the peierls file runs without error
        output = os.system(f"python3 {peierls_path} {seed} {nstate}")
        assert output == 0

