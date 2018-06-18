import numpy as np
from statistics import mean
import warnings

def pairwise_mahalanobis(X, labels, centroids, n_jobs=1, squared=True):
	warnings.filterwarnings('error')

	"""
	Distance between eliptical clusters:
		dist[i, j] = distance between clusters i and j
	"""
	unique_labels = np.unique(labels)
	l_uql = len(unique_labels)
	dist = np.ones((l_uql, l_uql))
	if n_jobs != 1:
		dist = pairwise_mahalanobis_parallel(X, labels, 
					unique_labels, centroids=centroids, pre_dist=dist, n_jobs=n_jobs)
		if squared == True:	
			return dist
		elif squared == False:
			return np.sqrt(dist)
	hashed_clusters = {}
	global hashed_cluster_covs
	hashed_cluster_covs = {}

	for i, i_center in zip(unique_labels, centroids):
		I = hashed_clusters.get(i, False)
		if I is False:
			I = X[np.where(labels == i)]
			hashed_clusters.setdefault(i, I)
		for j, j_center in zip(unique_labels, centroids):
			if i == j:
				dist[i, j] = 0
				continue
			if dist[j, i] != 1:
				dist[i, j] = dist[j, i]

			J = hashed_clusters.get(j, False)
			if J is False:
				J = X[np.where(labels == j)]
				hashed_clusters.setdefault(j, J)

			#dist[i, j] = dist_clusters(I, J, i_center, j_center, i, j)
			#dist[i, j] = mean([dist_one_cluster(I, i_center, j_center, i),
			#	dist_one_cluster(J, i_center, j_center, j)])
			dist[i, j] = min(dist_one_cluster(I, i_center, j_center, i),
				dist_one_cluster(J, i_center, j_center, j))

	warnings.resetwarnings(); del hashed_cluster_covs;#clean up
	if squared == True:	
		return dist
	elif squared == False:
		return np.sqrt(dist)


def dist_one_cluster(C, i_center, j_center, c):

	global hashed_cluster_covs
	inv_cov = hashed_cluster_covs.get(c, False)
	if inv_cov is False:
		try:
			inv_cov = np.linalg.inv(np.cov(C.T))
		except (Warning, np.linalg.linalg.LinAlgError):
			# these errors occur when C consists of few observations
			# which means that the identity is a suitable replacement to avoid errors
			inv_cov = np.eye(N=C.shape[1])

		hashed_cluster_covs.setdefault(c, inv_cov)

	diff = i_center - j_center
	dist1 = diff.T @ inv_cov @ diff
	return dist1

def dist_clusters(I, J, i_center, j_center, i, j):

	global hashed_cluster_covs
	inv_covs = []
	for C, c_name in zip([I, J], [i, j]):
		inv_cov = hashed_cluster_covs.get(c_name, False)
		if inv_cov is False:
			try:
				inv_cov = np.linalg.inv(np.cov(C.T))
			except (Warning, np.linalg.linalg.LinAlgError):
				# these errors occur when C consists of few observations
				# which means that the identity is a suitable replacement to avoid errors
				inv_cov = np.eye(N=C.shape[1])
			
			hashed_cluster_covs.setdefault(c_name, inv_cov)
		inv_covs.append(inv_cov)

	diff = i_center - j_center
	dist = diff.T @ inv_covs[0] @ inv_covs[1] @ diff
	return dist

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	from sklearn.datasets import make_blobs
	from sklearn.cluster import MiniBatchKMeans as KMeans
	from sklearn.cluster import Birch, DBSCAN
	from sklearn.mixture import GaussianMixture as GMM
	from time import time

	X = make_blobs(n_samples=10000, centers=3, n_features=2, cluster_std=1.5, center_box=(-10.0, 10.0), shuffle=True, random_state=42)[0]

	start = time()
	# Birch

	km = KMeans(n_clusters=100)
	km.fit(X=X)
	print('KMeans took: ', time() - start)
	print(len(np.unique(km.cluster_centers_)))

	start = time()
	print(pairwise_mahalanobis(X, km.subcluster_labels_, centroids=km.subcluster_centers_, squared=False))

	print('distance took: ', time() - start)
	plt.scatter(X[:, 0], X[:, 1])
	plt.show()

	#query_size = 2000
	#d, i = k_tree.query(X[
	#    np.random.randint(len(df), size=query_size), :],
	#                    k=K, dualtree=True)