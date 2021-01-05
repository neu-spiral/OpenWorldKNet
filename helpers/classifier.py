from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans

def kmeans(k, U, Y=None):
	# U = ensure_matrix_is_numpy(U)
	allocation = KMeans(k, n_init=10).fit_predict(U)

	if Y is None:
		return allocation
	else:
		nmi = normalized_mutual_info_score(allocation, Y)
		return [allocation, nmi]