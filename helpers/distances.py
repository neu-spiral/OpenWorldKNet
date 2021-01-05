import numpy as np
import sklearn.metrics


def median_of_pairwise_distance(U):
	vv = np.median(sklearn.metrics.pairwise.pairwise_distances(U))
	return vv