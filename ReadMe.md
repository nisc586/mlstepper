# ML-Stepper
A collection of algorithm implementations that allows you to step through the algorithms and visualize what they do.

## KMeans
The KMeans class expects some x and y values as well as the initial mean values, or a k. With the `step` or `show_step` methods you can step through the fitting algorithm. The `x_means` and `y_means` attributes will hold the x and y coordinates of the cluster means. `show_step` will return a scatter plot which visualizes how the cluster means moved.
![A scatter plot of the first step of the kmeans-algorithm on the iris dataset](/imgs/kmeans_step.png)
