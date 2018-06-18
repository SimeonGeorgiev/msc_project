import numpy as np
from AffinityPropagation import AffinityPropagation
from sklearn.cluster import MiniBatchKMeans
from dists import pairwise_mahalanobis
import pandas as pd
from time import time
import matplotlib
import matplotlib.pyplot as plt
from vis_correction import plot_cluster_medians
from collections import OrderedDict as od
from math import ceil
from umap import UMAP
from cytof_io import shuffle

def get_random_colors(n):
    return np.random.uniform(low=0.2, high=0.8, size=(n, 3))


def similar(color, colors, l1_sum=0.1):
    for used_color in colors:
        if abs(color - used_color).sum() > l1_sum:
            return False
    else:
        return True

def vec_translate(a, my_dict):
    """
    Vectorized translation of a numpy array with a Python dictionary.
    """
    return np.vectorize(my_dict.__getitem__)(a)

class _APMerge_classifier(object):
    def __init__(self, km, _dict, exemplars, AP, X):
        self.km = km
        self._dict = _dict
        self.exemplars = exemplars
        self.AP = AP
        self.APlabels_ = self.AP.labels_
        self.n_clusters = len(np.unique(self.APlabels_))
        self.set_labels(X)

    def predict(self, X):
        return vec_translate(self.km.predict(X), self._dict)

    def set_labels(self, X):
        self.labels_ = self.predict(X)

    def __repr__(self):
        return str((self.km, self.AP))

    def plot(self, *args, **kwargs):
        self.heat_map, _ = plot_cluster_medians(*args, **kwargs)


class APMerge(object):

    def __init__(self, n_clusters=50, random_state=1, init='k-means++', 
        init_size=50000, n_init=300, batch_size=2000, reassignment_ratio=0.4,
        percentile=95, convergence_iter=15, damping=0.5, max_iter=1000,
        verbose=0, df=None, cols=None):

        self.km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, init=init, 
                                    init_size=init_size, n_init=n_init, batch_size=batch_size,
                                    reassignment_ratio=reassignment_ratio, verbose=verbose,)
        self.percentile = percentile
        self.convergence_iter = convergence_iter
        self.damping = damping
        self.max_iter = max_iter
        self.dm_fit = False
        self.umap_fit = False
        self.models = {}
        self.verbose = verbose

        if df is not None and cols is not None:
            # do it here so that original df can be deleted if too big
            self.df = df
            self.cols = cols

    def fit(self, X=None, percentile=None):
        if X is None:
            X = self.df[self.cols]
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
                            km=self.km, _dict=PRE_to_AP, X = X, 
                            exemplars=AP.cluster_centers_, AP=AP)
        self.models.update({round(preference, 1): current_level_model})
        self.set_levels()

        self.labels_ = self.models[self.levels[-1]].predict(X)

        return self.models[round(preference, 1)]

    def set_dm(self, X):
        start = time()
        self.if_verbose('Started KMeans, computing partition distance matrix.')
        
        if self.dm_fit == False:
            self.km.fit(X)
            self.dm = -pairwise_mahalanobis(X=X, labels=self.km.labels_,
                                    centroids=self.km.cluster_centers_, squared=True)
            self.dm_fit = True

        finished_in = time() - start
        self.if_verbose('Partition distance matrix computed in {}'.format(finished_in))


    def hierarchical_fit(self, X=None, percentiles=[50, 85, 95]):
        if X is None:
            X = self.df[self.cols].values

        self.set_dm(X) # does not set_dm if it is already fit

        for p in percentiles:
            self.fit(X, percentile=p)

        self.set_levels()
        self.labels_ = self.models[self.levels[-1]].predict(X)
        return self

    def predict(self, X, level=-1):
        return self.get_level(level).predict(X)

    def set_levels(self):
        self.levels = sorted(self.models.keys())

    def iter_levels(self, range_=None):

        if range_ is None: 
            range_= (0, len(self.models))

        for K in range(*range_):
            yield self.get_level(K)

    def get_level(self, K):
        return self.models[round(self.levels[K], 1)]

    def plot_hierarchies(self, df=None, cols=None, 
                            title="APMergeHeatmap", **kwargs):
        for i, level in enumerate(self.levels):
            col = self.get_level(i).labels_
            self.models[round(level, 1)].plot(title=title+":"+str(i),
                df=df, col=col, cols=cols, **kwargs)

    def plot_level(self, level,  df=None, cols=None,
                            title="APMergeHeatmap", **kwargs):
        if df is None and cols is None:
            df, cols = self.df, self.cols
        model = self.models[round(self.levels[level], 1)]
        col = model.labels_
        model.plot(title=title, df=df, col=col, cols=cols, **kwargs)

    def plot_maps(self, *args, **kwargs):
        for K in range(len(self.models)):
            self.plot_map(K=K, *args, **kwargs)

    def plot_map(self, df=None, cols=None, save=False, show=True, K=-1,
                figsize=(10, 10), title="AP_Merge_UMAP_plot"):

        self.if_verbose("Setting Embedding.")
        np.random.seed(42)
        start = time()
        self.set_embedding(df=df, cols=cols)
        self.if_verbose("Embedding took: {:.3}".format(time() - start))


        labels = self.get_level(K).labels_
        labels = labels[self.sub_sample_indices]

        plt.figure(figsize=figsize)
        self.if_verbose(len(np.unique(labels)), np.unique(labels))
        unique_labels = np.unique(labels); len_uq = len(unique_labels)
        colors = get_random_colors(len_uq)

        for i, C in enumerate(unique_labels):

            indices = labels == C
            plt.scatter(self.embedding[indices, 0], 
                        self.embedding[indices, 1],
                        alpha=0.2, s=1, c=colors[i])

        plt.title(title)
        if show:
            plt.show()
        if save:
            plt.savefig(title + "K{}".format(K) + ".png")

    def set_embedding(self, df=None, cols=None, verbose=0):
        if self.umap_fit == False:
            if df is None and cols is None:
                df, cols = self.df, self.cols

            if verbose != self.verbose:
                self.verbose = verbose
                revert = True
            else: revert = False

            MAX = 50000
            to_sample = len(df) if len(df) < MAX else MAX
            embed_this = shuffle(df[cols]).iloc[:to_sample]

            self.if_verbose("Sampling {} cells".format(to_sample))
            self.sub_sample_indices = embed_this.index.values
            self.label_subsample = self.labels_[self.sub_sample_indices]

            self.if_verbose("embed_this.shape: {}".format(embed_this.shape))

            self.embedding = UMAP(n_neighbors=5, min_dist=0.2,
                                metric='manhattan').fit_transform(embed_this)


            self.umap_fit = True
            
            if revert:
                self.verbose = not self.verbose
    def if_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs, flush=True)

test_algo = False
test_similar = True
if __name__ == "__main__" and test_algo == True:
    from cytof_io import load_df
    df, cols = load_df()

    start = time()
    NC = 5
    n_init = 1
    level = 1
    model = APMerge(n_clusters=NC, n_init=n_init, random_state=1, verbose=0, df=df, cols=cols)
    model.hierarchical_fit(percentiles=[50, 85, 95])
    level_nc = model.get_level(level).n_clusters

    model.plot_level(level=level, title="NC={};N_init={};lvl:{},nc:{}".format(
                        NC, n_init,level+1, level_nc, 1), show=False, save=False)
    model.plot_tsne_maps(title="APMerge-tSNEmap;"+";NC:", desired_length=100, show=False, save=False)
    print('Preferences: ', sorted(list(model.models.keys())))
    print('Took:{}seconds'.format(round(time()-start, 1)))

# Create a consensus APMerge class, which run kmeans with different random states
    # then it finds similar clusters (in cols) with similar expression (ratio H/D)
if __name__ == "__main__" and test_similar==True:
    assert (similar(np.array([0, 0, 0]), np.array([0, 0, 0])))
    assert (similar(np.array([0, 0, 0]), np.array([0, 1, 0]))) == False
    assert (similar(np.array([0, 0, 0]), np.array([0, .01, 0])))
    assert (similar(np.array([0, 0, 0]), np.array([0, .1, 0]))) == False