import sklearn.metrics
import numpy as np
from sklearn.preprocessing import normalize

def getLaplacian(data, σ, H=None):
	[L, Dinv] = normalized_rbk_sklearn(data, σ)

	if H is not None:
		L = H.dot(L).dot(H)	

	return [L, Dinv]

def normalized_rbk_sklearn(X, σ):
	# X = ensure_matrix_is_numpy(X)
	Kx = rbk_sklearn(X, σ)       	
	np.fill_diagonal(Kx, 0)			#	Set diagonal of adjacency matrix to 0
	Dinv = compute_inverted_Degree_matrix(Kx)

	#KD = Kx - Dinv
	#return [KD, Dinv]

	DKD = Dinv.dot(Kx).dot(Dinv)
	return [DKD, Dinv]

def rbk_sklearn(data, σ):
	γ = 1.0/(2*σ*σ)
	rbk = sklearn.metrics.pairwise.rbf_kernel(data, gamma=γ)
	return rbk

def compute_inverted_Degree_matrix(M):
	return np.diag(1.0/np.sqrt(M.sum(axis=1)))

def L_to_U(num_cluster, L, return_eig_val=False):
	# L = ensure_matrix_is_numpy(L)
	eigenValues,eigenVectors = np.linalg.eigh(L)

	n2 = len(eigenValues)
	n1 = n2 - num_cluster
	U = eigenVectors[:, n1:n2]
	# print("U is here:")
	# np.set_printoptions(threshold=np.inf)
	# print(U)
	U_lambda = eigenValues[n1:n2]
	U_normalized = normalize(U, norm='l2', axis=1)
	
	if return_eig_val: return [U, U_normalized, U_lambda]
	else: return [U, U_normalized]

def center_matrix(M):
	N = M.shape[0]
	H = np.eye(N) - np.ones((N,N))/float(N)
	return H.dot(M).dot(H)

