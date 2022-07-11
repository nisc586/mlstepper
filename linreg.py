import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, x, y, regressors=None):
        # linear model:
        # y_i = xT_i * beta + eps_i
        # Create two column vectors with the observed data
        self.independent_var = np.array(x).reshape(-1, 1)
        self.dependent_var = np.array(y).reshape(-1, 1)

        self.regressors = [lambda x: 1, lambda x: x] if regressors is None else regressors
        
        self.x = np.hstack([
            np.vectorize(f)(self.independent_var)
            for f in self.regressors
        ])


    def fit(self):
        # solve ordinary least squares
        y = self.dependent_var

        self.coeffs = linalg.inv(self.x.T @ self.x) @ self.x.T @ y
        self.estimates = self.x @ self.coeffs

        # R^2 is defined as the ratio between "explained" variance and "total" variance
        self.r_square = np.sum((self.estimates - y.mean())**2) /np.sum((y - y.mean())**2)
    

    def show(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.independent_var, self.dependent_var)

        p = np.linspace(min(self.independent_var), max(self.independent_var), 100)
        q = sum(
            beta * np.vectorize(f)(p)
            for beta, f in zip(self.coeffs, self.regressors)
        )

        ax.plot(p, q, "-r")
        plt.show()
    

    def predict(self, x):
        return sum(
            beta * f(x)
            for beta, f in zip(self.coeffs, self.regressors)
        )

    def get_coeffs(self):
        return self.coeffs.flatten().tolist()