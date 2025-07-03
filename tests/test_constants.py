import numpy as np
import pytest
from gym_so100.constants import unnormalize


def test_unnormalize():
    """Test the unnormalize function."""
    # Test case 1: num = -1, should return min_val
    assert unnormalize(-1, -10, 10) == -10

    # Test case 2: num = 1, should return max_val
    assert unnormalize(1, -10, 10) == 10

    # Test case 3: num = 0, should return the middle value
    assert unnormalize(0, -10, 10) == 0

    # Test case 4: num = 0.5, should return the scaled value
    assert unnormalize(0.5, -10, 10) == 5

    # Test case 5: num = -0.5, should return the scaled value
    assert unnormalize(-0.5, -10, 10) == -5

    # Test case 6: Clipping below min_val
    assert unnormalize(-2, -10, 10) == -10

    # Test case 7: Clipping above max_val
    assert unnormalize(2, -10, 10) == 10

    # Test with different min/max
    assert unnormalize(0, 0, 20) == 10
    assert unnormalize(-1, 0, 20) == 0
    assert unnormalize(1, 0, 20) == 20

    # Test with floating point values
    assert np.isclose(unnormalize(0.25, -5.0, 5.0), 1.25)
