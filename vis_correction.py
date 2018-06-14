from cytof_io import load_sample
import os
import pandas as pd
from collections import OrderedDict as od
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from cytof_io import cd

def load_samples(f_calib, f_target, ID_non_cal='C1', cut_at=-1,
				cor_dir="MMDnet_Results", raw_dir='PBMC_150518'):
	with cd(cor_dir):
		calib = pd.read_csv(f_calib, index_col=0)
		target = pd.read_csv(f_target, index_col=0)
		metals = list(calib)[:cut_at]
		calib['T'] = False # class labels for distinguishing between samples of the same condition
		target['T'] = True

	#################### LOAD THE NON-CALIBRATED DATASET
	with cd(raw_dir):
		non_calib = load_sample(ID_non_cal)
		non_calib['T'] = False
	return (target, calib, non_calib), metals

def mod_names(metal_name):
	splitted = metal_name.split("_")
	if len(splitted) < 3: # if len(splitted) == 2: return splitted[1]
		return splitted[len(splitted)-1] # last element
	else:
		return "_".join(splitted[1:])

def in_ipynb():
    ### use this instead of explicit arguments
    try:
        cfg = get_ipython().config 
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False

def box_plot_samples(dfs, metals, titles, ipython):
	if ipython:
		t = 2
		t2 = 1.5
	else:
		t = 1
		t2 = 1
	fig, axes = plt.subplots(ncols=1, nrows=len(dfs), figsize=(2*len(dfs)*t, 2*len(dfs)*t), sharey=True)
	flierprops = dict(markerfacecolor='0.75', markersize=1*t,
              linestyle='none', alpha=0.2)
	for i, df in enumerate(dfs):
		axes[i] = sns.boxplot(data=df[metals], palette="hls", ax=axes[i], flierprops=flierprops)
		axes[i].set_xticklabels(labels=map(mod_names, metals), rotation=75)
		axes[i].tick_params(labelsize=8*t2)
		axes[i].set_title(titles[i], fontsize=12*t2)
	plt.tight_layout()
	plt.show()

ordered = lambda d: list(od(sorted(d.items())).values())

def vis_correction(f_calib, f_target, sample_names, ipython=False, **kwargs):
	"""
	f_calib -> csv with the calibrated sample
	f_target -> csv with the target sample
	sample_names -> names
	ipython -> True or False for whether this is ran in a notebook.
	if True, will make the plot bigger
	"""

	s_names_ord = {i: n for i, n in enumerate(sample_names)}
	samples, metals = load_samples(f_calib=f_calib,
									f_target=f_target,**kwargs)

	box_plot_samples(samples, metals, ipython=ipython, titles=ordered(s_names_ord))
	return samples, metals

def TSNE_vis(df, cols, title, show=True, size=7500, **kwargs):
    tsne = TSNE(n_components=2, perplexity=50)
    trans = tsne.fit_transform(df.iloc[:size][cols].values)
    plt.scatter(trans[:, 0], trans[:, 1], alpha=0.1, **kwargs)
    title += " KL div:" + str(np.round(tsne.kl_divergence_, 2)) + "."
    plt.title(title)
    if show:plt.show()
    else:plt.savefig(title, kwargs["format"])
    return tsne, trans

def PCA_vis(df, cols, title, show=True, **kwargs):
    pca = PCA(n_components=2)
    trans = pca.fit_transform(df[cols].values)
    plt.scatter(trans[:, 0], trans[:, 1], alpha=0.1,**kwargs)
    title += " Exp. var:" + str(np.round(pca.explained_variance_ratio_, 2))[1: -1] + "."
    plt.title(title)
    if show:plt.show()
    else:plt.savefig(title, kwargs["format"])
    return pca, trans

def PCA_vis_clusters(df, cols, title, c_marker, show=True, **kwargs):
    pca = PCA(n_components=2)
    trans = pca.fit_transform(df[cols].values)
    df[c_marker]
    for c in df[c_marker].unique():
        idx = df[df[c_marker] == c].index.values
        plt.scatter(trans[idx, 0], trans[idx, 1], alpha=0.3,**kwargs)
    title += " Exp. var:" + str(np.round(pca.explained_variance_ratio_, 2))[1: -1] + "."
    plt.title(title)
    if show:plt.show()
    else:plt.savefig(title, kwargs["format"])
    return pca, trans

def clust_medians(df, col, cols, factors):
    
    unique = df[col].unique()
    if -1 in unique:
      outliers = df[df[col] == -1][cols]
    else: outliers = None

    if factors != None:
      medians = np.empty((len(unique), df[cols].shape[1] + len(factors)))
      temp_factors = np.empty((len(unique), len(factors)))
    else:
      medians = np.empty((len(unique), df[cols].shape[1]))

    temp_medians = np.empty((len(unique), df[cols].shape[1]))
    indices = []; total = len(df)

    for c in unique:
      this_cluster = df[df[col] == c]
      ratio = len(this_cluster)
      indices.append(ratio)
      temp_medians[c] = np.asarray(this_cluster[cols].median())
      if factors != None:
        this_factors = this_cluster[factors]
        temp_factors[c] = np.asarray(this_factors.mean())
          
    medians = pd.DataFrame(temp_medians, columns=cols, index=indices)
    if factors != None:
      temp_factors *= temp_medians.max()
      for i, factor in enumerate(factors):
        medians[factor] = temp_factors[:, i]
    return medians, outliers

def plot_cluster_medians(df, col, cols, title, factors=None, col_cluster=True,
    plot_xor_save=True):
    
    means_df, outliers = clust_medians(df, col, cols, factors)
    sns.clustermap(means_df, col_cluster=col_cluster)
    plt.title(title)
    if plot_xor_save == True:
      plt.show()
    elif plot_xor_save == False:
      plt.savefig(title+".png")
    return means_df, outliers

if __name__ == "__main__":
	s_names = ["CA, PBMCs frozen 6 months ago.", "C1, calibrated to CA.",
				"C1, PBMCs frozen 4 years ago."]
	samples, metals = vis_correction(f_calib="C1toCA2018-06-03_15:34:35_calibrated_impMMD.csv",
							f_target="CA_phenograph_clustered;Q:0.908286;_target.csv",
							sample_names=s_names, ipython=False)
	print(metals)
