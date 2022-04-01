from kmeans import KMeans
import numpy as np

def test_step():
    xs = np.array([-4, -2, -4, -2, 2, 4, 2, 4])
    ys = np.array([1, 1, -1, -1, 2, 2, 4, 4])

    start_means = [(-3, 3), (1, 3)]

    km = KMeans(xs, ys, means=start_means)
    km.step()
    # expected means are (-3, 0) and (3, 3)
    assert np.equal(km.x_means, np.array([-3, 3])).all()
    assert np.equal(km.y_means, np.array([0, 3])).all()
