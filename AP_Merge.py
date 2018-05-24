import numpy as np
from AffinityPropagation import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from dists import pairwise_mahalanobis
import pandas as pd
from pandas import Series, DataFrame
from cytof_io import load_df
from time import time
from scipy.cluster import hierarchy
from Bio import Phylo as ph
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
from plot_tSNE import random_subset, get_random_colors
from sklearn.manifold import TSNE
from plot_trees import change_names, getNewick, get_color_f


def vec_translate(a, my_dict):
	"""
	Vectorized translation of a numpy array with a Python dictionary.
	"""
	return np.vectorize(my_dict.__getitem__)(a)

class _APMerge_classifier(object):
	def __init__(self, km, _dict, exemplars, AP):
		self.km = km
		self._dict = _dict
		self.exemplars = exemplars
		self.AP = AP
		self.APlabels_ = self.AP.labels_
		self.n_clusters = len(np.unique(self.APlabels_))

	def predict(self, X):
		return vec_translate(self.km.predict(X), self._dict)

	def set_labels(self, X):
		self.labels_ = self.predict(X)

	def __repr__(self):
		return str((self.km, self.AP))

	def plot(self, df=None, markers=None, individual="individual", 
					H=1, D=2, colors_f=get_color_f,
					show=False, save=True, fname="APMergeHeatmap"):

		val_c = df[individual].value_counts()
		total_h, total_d  = val_c[1], val_c[2]
		self.labels_ = self.predict(df[markers].values)
		labels_H = Series(self.labels_[df[individual] == H])
		labels_D = Series(self.labels_[df[individual] == D])
#		labels_H = Series(self.predict(df[df[individual] == H][markers]))
#		labels_D = Series(self.predict(df[df[individual] == D][markers]))

		c_expr = lambda center, name: Series({m: v for m, v in zip(markers, center)}, name=name)

		H = Series(labels_H.value_counts(), name='H', dtype = int)
		D_nt = Series(np.round(labels_D.value_counts(), 0), dtype=int)
		D = Series(np.round(D_nt * (total_h/total_d),
		             0), name='D', dtype=int)
		H_over_D = np.round(Series(H/D, name='H/D'), 3)
		H_plus_D = (H + D_nt).fillna(H).fillna(D_nt)
		differences = pd.concat([H, D, H_over_D], axis=1)
		# for each cluster I need to have P, Count(AML), Count(H) and markers as rows in the dataframe
		cluster_matrix = pd.concat(
								(c_expr(center, i) for i, center in enumerate(self.exemplars)),
							axis=1)

		for_heat_map = cluster_matrix.T
		cluster_matrix = pd.concat([differences.transpose(), cluster_matrix], axis=0)

		differential_D = cluster_matrix.loc[:, (cluster_matrix.loc['H/D'] < 0.5)]
		D_names = list(differential_D)
		normal = cluster_matrix.loc[:, (cluster_matrix.loc['H/D'] < 2) & (cluster_matrix.loc['H/D'] > 0.5)]
		N_names = list(normal)
		differential_H = cluster_matrix.loc[:, (cluster_matrix.loc['H/D'] > 2)]
		H_names = list(differential_H)

		D_N_H = pd.concat([differential_D, normal, differential_H], axis=1)
		##################################################################################################
		################## SPLIT UP THE FUNCTION HERE.												   ###
		################## if it is split here I could reuse: cluster IDs, ratios and counts,for plots ###
		################## saving the dataframes would also allow for consensus clustering and		   ###
		################## assessment of clustering stability. 										   ###
		################## stable clusters will be flagged as real clusters and brought to attention!  ###
		##################################################################################################
		Z = hierarchy.linkage(D_N_H.iloc[3:].T, method='average', metric='euclidean')
		tree = hierarchy.to_tree(Z, False)
		leaf_names = D_names + N_names + H_names
		nwc = getNewick(tree, "", tree.dist, leaf_names)
		handle = StringIO(nwc)
		tree = ph.read(handle, 'newick')
		
		get_pop_representation = lambda n: str(round(H_over_D.loc[n], 3))
		condition = {str(n): get_pop_representation(n) for n in D_names}
		condition_D = condition.copy()
		condition_normal = {str(n): get_pop_representation(n) for n in N_names}
		condition_healthy = {str(n): get_pop_representation(n) for n in H_names}
		condition.update(condition_normal.copy())
		condition.update(condition_healthy.copy())
		tree, re_order = change_names(tree, condition)

		matplotlib.rc('font', size=8)

		fig, (ax_tree, ax_imshow) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
		colors = colors_f(condition_D, condition_healthy, condition_normal)
		ph.draw(tree, do_show = False, label_colors=colors, axes=ax_tree)

		for_heat_map = for_heat_map.iloc[re_order]
		ax_imshow.imshow(for_heat_map, aspect='auto', cmap='hot')
		ax_tree.axis('off')
		# scale fontsizes depending on cluster number (which would be len(H_plus_D))
		ax_imshow.set_yticks(range(len(for_heat_map)))
		ax_imshow.set_yticklabels(H_plus_D.iloc[re_order], fontsize=8)

		ax_imshow.tick_params(axis='x', bottom=True, top=True, labelbottom=True, labeltop=True)
		ax_imshow.set_xticks(range(len(markers)))
		ax_imshow.set_xticklabels(list(for_heat_map), rotation=90, fontsize=8)
		fig.tight_layout()
		if save:
			plt.savefig(fname=fname, format='jpg')
		if show:
			plt.show()
		plt.close()
		self.heat_map = for_heat_map
		self.order = list(map(int, re_order))
		return None

class APMerge(object):

	def __init__(self, n_clusters=50, random_state=1, init='k-means++', 
		init_size=50000, n_init=300, batch_size=2000, reassignment_ratio=0.4,
		percentile=95, convergence_iter=15, damping=0.5, max_iter=1000,
		verbose=0, df=None, markers=None):

		self.km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, init=init, 
									init_size=init_size, n_init=n_init, batch_size=batch_size,
									reassignment_ratio=reassignment_ratio, verbose=verbose,)
		self.percentile = percentile
		self.convergence_iter = convergence_iter
		self.damping = damping
		self.max_iter = max_iter
		self.dm_fit = False
		self.tsne_fit = False
		self.models = {}
		self.verbose = verbose
		if df is not None and markers is not None:
			# do it here so that original df can be deleted if too big
			self.df = df.copy()
			self.markers = markers

	def fit(self, X=None, percentile=None, set_labels=True):
		if X is None:
			X = self.df[self.markers]
		self.set_dm(X) # K-Means partitioning and distance matrix construction, does not compute again if already done

		if percentile is None:
			percentile = self.percentile

		preference = np.percentile(self.dm, percentile)
		if self.verbose:
			start = time()
			print('Starting AffinityPropagation with preference:{}'.format(preference))
		### to do: catch Convergence warning and add more iterations if caught.
		AP = AffinityPropagation(convergence_iter=self.convergence_iter, 
									damping=self.damping, max_iter=self.max_iter,
									preference=preference, copy=False, verbose=self.verbose).fit(
									X=self.km.cluster_centers_, S=self.dm) # set verbose to self.verbose
		if self.verbose:
			finished_in = time() - start
			print('AffinityPropagation found {} clusters'.format(len(np.unique(AP.labels_))))
		# dictionary to convert from KMeans partitioning to AP clustering
		PRE_to_AP = {pre_p: ap_p for pre_p, ap_p in enumerate(AP.labels_, 0)}
		current_level_model = _APMerge_classifier(
							km=self.km, _dict=PRE_to_AP, exemplars=AP.cluster_centers_, AP=AP)
		current_level_model.set_labels(X)
		self.models.update({round(preference, 1): current_level_model})
		self.set_levels()
		if set_labels == True:
			self.labels_ = self.models[self.levels[-1]].predict(X)

		return self.models[round(preference, 1)]

	def set_dm(self, X):
		if self.verbose:
			start = time()
			print('Started KMeans, computing partition distance matrix.')
		if self.dm_fit == False:
			self.km.fit(X)
			self.dm = -pairwise_mahalanobis(X=X, labels=self.km.labels_,
									centroids=self.km.cluster_centers_, squared=True)
			self.dm_fit = True
		if self.verbose:
			finished_in = time() - start
			print('Partition distance matrix computed in {}'.format(finished_in))

	def hierarchical_fit(self, X=None, percentiles=[50, 85, 95]):
		if X is None:
			X = self.df[self.markers].values

		self.set_dm(X) # does not set_dm if it is already fit
		for p in percentiles:
			self.fit(X, percentile=p, set_labels=False)

		self.set_levels()
		self.labels_ = self.models[self.levels[-1]].predict(X)
		return self.models

	def predict(self, X, level=-1):
		return self.get_level(level).predict(X)

	def set_levels(self):
		self.levels = sorted(self.models.keys())


	def get_level(self, K):
		return self.models[round(self.levels[K], 1)]

	def plot_hierarchies(self, df=None, markers=None, fname="APMergeHeatmap", **kwargs):
		for i, level in enumerate(self.levels):
			self.models[round(level, 1)].plot(fname=fname+":"+str(i), df=df, markers=markers, **kwargs)

	def plot_level(self, level,  df=None, markers=None, fname="APMergeHeatmap", **kwargs):
		if df is None and markers is None:
			df, markers = self.df, self.markers
		self.models[round(self.levels[level], 1)]\
					.plot(fname=fname, df=df, markers=markers, **kwargs)
	def plot_tsne_maps(self, df=None, markers=None, save=True, show=False,
							label="AP", fname="APMerge-tSNEmap", **kwargs):
		self.set_tsne_map(df=df, markers=markers, fname=fname, **kwargs)
		# Fit t-SNE on data, subsetting from the lowest hierarchy (highest n_clusters)
		# get colors from models[round(level, 1)].labels_
		# plot those using my plot_tSNE library
		if markers is None: markers = self.markers

		for preference in sorted(self.models.keys()):
			model = self.models[preference]
			predictions = model.predict(self.df_subset[markers].values)
			for cluster, color in zip(np.unique(model.APlabels_), get_random_colors()):
				points = self.embedding[predictions == cluster]
				plt.scatter(points[:, 0], points[:, 1], color=color)
				# add as label, cluster ID (in the heatmap) and cluster representation
			if save:
				plt.savefig(fname+str(model.n_clusters))
			if show:
				plt.show()
			plt.close()

	def set_tsne_map(self, df=None, markers=None, label=None, fname=None, **kwargs):
		if not self.tsne_fit:
			if df is None and markers is None:
				df, markers = self.df, self.markers
			df[label] = self.labels_ # the labels of the hierarchy with the highest cluster number.
			self.df_subset, indices, proportions = random_subset(df=df, label_name=label, **kwargs)
			self.embedding = TSNE(n_components=2, verbose=0, perplexity=50, 
								n_iter=300, n_iter_without_progress=100, 
								method="barnes_hut").fit_transform(self.df_subset[self.markers].values)

			self.tsne_fit = True
		

if __name__ == "__main__":
	df, markers = load_df()
	for i in range(1, 6):
		start = time()
		NC = 150
		n_init = 1
		level = 1
		model = APMerge(n_clusters=NC, n_init=n_init, random_state=i, verbose=0, df=df, markers=markers)
		model.hierarchical_fit(percentiles=[50, 85, 95])
		level_nc = model.get_level(level).n_clusters

		model.plot_level(level=level, fname="NC={};N_init={};lvl:{},nc:{};i:{}".format(
							NC, n_init,level+1, level_nc,i), show=False, save=False)
		model.plot_tsne_maps(fname="APMerge-tSNEmap;i:"+str(i)+";NC:", desired_length=100, show=False, save=False)
		print('Preferences: ', sorted(list(model.models.keys())))
		print('Took:{}seconds'.format(round(time()-start, 1)))

	# Create a consensus APMerge class, which run kmeans with different random states
	# then it finds similar clusters (in markers) with similar expression (ratio H/D)
