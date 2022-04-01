import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class KMeans:
    def __init__(self, x, y, *, k=2, means=None, labels=("", "")):
        self.x = x
        self.y = y
        self.k = k

        if means is None:
            means = self._random_means()
        self.x_means = np.array([m[0] for m in means])
        self.y_means = np.array([m[1] for m in means])
        self.k = len(means)

        self.xlabel, self.ylabel = labels


    def _random_means(self, seed=None):
        """Initialize random means"""
        if seed is not None:
            np.random.seed(seed)
        ixs = np.random.choice(range(len(self.x)), size=self.k, replace=False)
        return np.array([(self.x[i], self.y[i]) for i in ixs])


    def step(self):
        self.expectation_step()
        self.minimization_step()


    def expectation_step(self):
        dist_x = np.subtract.outer(self.x, self.x_means)
        dist_y = np.subtract.outer(self.y, self.y_means)

        dist = np.sqrt(dist_x**2 + dist_y**2)
        self.cluster_ixs = dist.argmin(1)


    def minimization_step(self):
        for i in range(self.k):
            self.x_means[i] = np.mean(self.x[self.cluster_ixs == i])
            self.y_means[i] = np.mean(self.y[self.cluster_ixs == i])


    def show_step(self, ax=None):
        """Perform a step, then return a scatter plot with old and new means marked."""
        last_x_means = self.x_means.copy()
        last_y_means = self.y_means.copy()

        self.step()
        if ax is None:
            _, ax = plt.subplots()

        scttr = ax.scatter(self.x, self.y, c=self.cluster_ixs)
        ax.legend(*scttr.legend_elements())
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        
        ax.plot(last_x_means, last_y_means, "rx")
        ax.plot(self.x_means, self.y_means, "rx")

        # Add some arrows to see, how the means moved
        xyAs = zip(last_x_means, last_y_means)
        xyBs = zip(self.x_means, self.y_means)
        for xyA, xyB in zip(xyAs, xyBs):
            con = mpatches.ConnectionPatch(xyA, xyB, coordsA="data", coordsB="data", arrowstyle="->")
            ax.add_artist(con)
