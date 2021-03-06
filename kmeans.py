import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

class KMeans:
    """Class that models the k-means algorithm.
    
    Available attributes are:
    - k: The number of clusters
    - x_means, y_means: x, and y coordinates of the cluster centers
    - xlabel, ylabel: axis labels for x, and y axis
    - legend_labels: labels corresponding to the clusters
    """

    def __init__(self, x, y, *, k=2, means=None, labels=("", ""), legend_labels=None):
        """Create a new model.
        
        Params:
            x, y: x, and y coordinates of the data points
            k: number of clusters
            means: initial center points of the clusters, if None use random points
            labels: labels for x, and y axis
            legend_labels: names for the clusters
        """
        self.x = x
        self.y = y
        self.k = k

        if means is None:
            means = self._random_means()
        self.x_means = np.array([m[0] for m in means])
        self.y_means = np.array([m[1] for m in means])
        self.k = len(means)

        self.xlabel, self.ylabel = labels
        self.legend_labels = legend_labels

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
        """Assign all data points to a cluster."""
        dist_x = np.subtract.outer(self.x, self.x_means)
        dist_y = np.subtract.outer(self.y, self.y_means)

        dist = np.sqrt(dist_x**2 + dist_y**2)
        self.cluster_ixs = dist.argmin(1)


    def minimization_step(self):
        """Calculate new cluster center points."""
        for i in range(self.k):
            self.x_means[i] = np.mean(self.x[self.cluster_ixs == i])
            self.y_means[i] = np.mean(self.y[self.cluster_ixs == i])


    def show_step(self, ax=None):
        """Perform a step, then draws a scatter plot with old and new means marked.
        
        Creates a new subplot, if ax is None."""
        last_x_means = self.x_means.copy()
        last_y_means = self.y_means.copy()

        self.step()
        if ax is None:
            _, ax = plt.subplots()

        scttr = ax.scatter(self.x, self.y, c=self.cluster_ixs)
        handles, labels = scttr.legend_elements()
        if self.legend_labels is not None:
            labels = self.legend_labels
        ax.legend(handles, labels)
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

    def predict(self, x, y):
        """Predict a cluster for a point x, y."""
        dist = np.sqrt((x - self.x_means)**2 + (y - self.y_means)**2)
        min_idx = np.where(dist == dist.min())[0]
        return self.legend_labels[min_idx] if self.legend_labels is not None else min_idx