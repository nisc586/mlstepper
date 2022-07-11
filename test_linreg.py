from linreg import LinearRegression
import pytest
import math


# Test using Anscombe's quartet
@pytest.mark.parametrize(
    "x, y",
    [
        (
            [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        ),
        (
            [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74]
        ),
        (
            [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
        ),
        (
            [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
            [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89]
        )
    ]
)
def test_linear_regression(x, y):
    model = LinearRegression(x, y)
    model.fit()
    alpha, beta = model.get_coeffs()
    assert math.isclose(alpha, 3, abs_tol = 0.01)
    assert math.isclose(beta, 0.5, abs_tol = 0.01)
    assert math.isclose(model.r_square, 0.67, abs_tol=0.01)