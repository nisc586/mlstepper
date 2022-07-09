import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

# x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
# y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

class LinearRegression:
    def __init__(self, x, y, regressors=None):
        # linear model:
        # y_i = xT_i * beta + eps_i
        # Create two column vectors with the observed data
        self.independent_var = np.array(x).reshape(-1, 1)
        self.dependent_var = np.array(y).reshape(-1, 1)

        if regressors is None:
            self.regressors = [lambda x: 1, lambda x: x]
        
        self.x = np.hstack([
            np.vectorize(f)(self.independent_var)
            for f in self.regressors
        ])


    def solve(self):
        # solve ordinary least squares
        y = self.dependent_var

        self.coeffs = linalg.inv(self.x.T @ self.x) @ self.x.T @ y
        self.estimates = self.x @ self.coeffs

        self.rss = np.sum((self.estimates - y.mean())**2)  # "explained" variance
        self.tss = np.sum((y - y.mean())**2)  # "total" variance
        self.r_square = 1 - self.rss/self.tss  # coefficient of determination
    

    def show(self, ax=None):
        print("Coefficients:", self.coeffs)
        print("R^2:", self.r_square)

        if ax is None:
            fig, ax = plt.subplots()
        plt.scatter(self.independent_var, self.dependent_var)
        plt.scatter(self.independent_var, self.estimates, color="red")

        plt.show()
    

    def predict(self, x):
        return sum(
            beta * f(x)
            for beta, f in zip(self.coeffs, self.regressors)
        )