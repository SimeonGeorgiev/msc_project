import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from collections import OrderedDict as OD
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from collections import namedtuple

SimulationResults = namedtuple('SimulationResults', 'average_purity ratios_list cluster_sizes labels')

class BarcodingError(object):
    """
        Contains the results from a random forest trained to distinguish between two samples.
        The samples are defined by their barcoding antibodies: barcode1 and barcode2.
        df - the dataframe used for the analysis, a clustered and transformed CyTOF sample
        barcode{1, 2} - the barcoding antibody for condition 1 and 2
        cluster_col_name - the cluster column where label assignments are stored
        threshold- a list of thresholds used to label a cluster as positive or negative.
        
    """
    
    def __init__(self, df, barcode1, barcode2,
                 labels, class_, renaming=False, 
                 cluster_col_name='C', thresholds=[0.9, 0.1],
                 ratio=0.1, rs=42, other_barcodes = None,
                 verbose=0, CORES=-1, n_trees=50):
        """
        Runs all analysis on the sample.
        First, a Random Forest is trained to tell apart the two de-barcoded groups of cells.
                Then the sample is split into missclassified and correctly classified.
        Second, cells in the wrong cluster are found by looking for clusters with 90% purity for one sample
                and taking the cells from the other sample.
        Third, the intersection between the missclassified cells and those in the wrong cluster is taken.
        Finally, a summary is produced in a dictionary format. This can be used as an entry into a dataframe.
        """
        # Train
        self.set_fields(df, barcode1, barcode2, class_,
                        labels, cluster_col_name, other_barcodes)

        self.train_RFC(labels=labels, ratio=ratio, rs=rs, verbose=verbose, CORES=CORES,
                      n_trees=n_trees, class_=class_)
        # Analyse
        self.analyse(thresholds)

    def set_fields(self, df, barcode1, barcode2, class_, labels, cluster_col_name, other_barcodes):
        self.df = df.copy()
        df = self.df
        self.N = len(df)

        pd.options.mode.chained_assignment = None
        df["Ind"] = range(df.shape[0])
        pd.options.mode.chained_assignment = "warn"

        self.barcode1=barcode1; self.barcode2=barcode2;
        self.class_ = class_
        self.labels = labels
        self.C = cluster_col_name
        self.all_labels = self.labels + [self.C, self.class_, self.barcode1, self.barcode2, "Ind"]
        if other_barcodes != None:
            self.all_labels += other_barcodes
    
    def analyse(self, thresholds, *args, **kwargs):
        self.set_total_score()
        self.thresholds = thresholds
        self.set_wrong_clusters(*args, **kwargs) # this is the only different method in BarcodingErrorWithControl
        self.set_swapped()
        self.set_summary()

    def train_RFC(self, labels, ratio, rs, verbose, CORES, n_trees, class_):
        df = self.df
        X, Y = df[labels].values, df[class_].astype(int).values

        self.forest = RFC(n_estimators=n_trees, criterion="gini", 
                     bootstrap=True, min_samples_split=100,
                     min_samples_leaf=100, oob_score=True,
                     n_jobs=CORES, verbose=verbose,
                     warm_start=True, random_state=rs)

        self.forest.fit(df[labels].values, df[class_].astype(int).values)
        self.score = self.forest.oob_score_
        predicted = self.forest.predict(X)

        correct_index = (predicted == Y)
        self.correct = df[correct_index][self.all_labels]
        self.wrong = df[~correct_index][self.all_labels]
        self.feature_importances_ = self.forest.feature_importances_
        self.set_top_labels()

    def set_top_labels(self):
        labels_features = OD()
        for fi, lab in zip(self.feature_importances_, self.labels):
            labels_features[fi] = lab

        filtered_labels_features = OD()
        for fi in filter(lambda x: x > self.feature_importances_.mean(), labels_features.keys()):
           filtered_labels_features[fi] = labels_features[fi]

        self.top_labels = list(filtered_labels_features.values())

    def set_total_score(self):
        self.total_score = len(self.correct) / (
            len(self.correct) + len(self.wrong))

    def plot(self, name, barcode1, barcode2, plot_xor_save=True, simple=True, *args, **kwargs):
        assert (barcode1 != None) and (barcode2 != None), "Provide names for the barcoding antibodies."

        plt.scatter(self.wrong[barcode1], self.wrong[barcode2], c='r', alpha=1, s=12)
        plt.scatter(self.correct[barcode1], self.correct[barcode2], c='b', alpha=0.1, s=12)
        plt.xlabel(barcode1); plt.ylabel(barcode2)
        dataset_size = "Sample Size: " + str(len(self.wrong) + len(self.correct))
        ratio_correct="AC2: {:.3}".format(self.total_score)

        barcoding_dispersion = "Halfway_points_R:{:.3}".format(self.r_halfway_wrong)
        
        miss_clustered = "WCN:{}".format(self.WCN)
        additional = ""
        if simple == False: additional += ". AC1: {:.3};{};{};{};".format(
            self.score, ratio_correct, miss_clustered, barcoding_dispersion)

        plt.title(name + additional + ";{};Len(Union):{}".format(
            dataset_size, self.r_union()))
        if plot_xor_save: plt.show()
        else: plt.savefig(name+barcode1+barcode2+".png")
    
    def get_sample(self, class_val):
        one_cond = lambda df: df[df[self.class_] == class_val]
        return one_cond(self.df)

    def get_sample_barcode(self, sample): 
        if sample[self.barcode1].mean() > sample[self.barcode2].mean():
            return self.barcode1, self.barcode2
        else: return self.barcode2, self.barcode1
    
    def set_wrong_clusters(self, *args, **kwargs):
        if self.C != False:
            ###do stuff
            self.wrong_clusters = []
            sample = self.df
            class_ = self.class_
            cluster_ids = sample[self.C].unique()
            self.n_clusters_ = len(cluster_ids)
            clust_medians = pd.DataFrame(np.empty((self.n_clusters_, len(self.labels) + 1)),
                                      columns=self.labels + ["%class_"], index=cluster_ids)
            for C in cluster_ids:
                this_cluster = sample[sample[self.C] == C]
                means_C = this_cluster[class_].mean()
                try:
                    clust_medians.loc[C, self.labels] = this_cluster[self.labels].median()
                    clust_medians.loc[C, "%class_"] = means_C
                except Exception as e:
                    print(C, self.n_clusters_, cluster_ids, clust_medians.shape)
                    raise Exception(e)

                if means_C   > self.thresholds[0]: # default = 0.9
                    self.wrong_clusters.append(
                        this_cluster[this_cluster[class_] == 0])
                
                elif means_C < self.thresholds[1]: # default = 0.1
                    self.wrong_clusters.append(
                        this_cluster[this_cluster[class_] == 1])
            else:
                try:
                    self.wrong_clusters = pd.concat(self.wrong_clusters)[self.all_labels]# DataFrame
                    self.WCN = len(self.wrong_clusters) # Wrong cluster number
                except ValueError:
                    self.wrong_clusters = None
                    self.WCN = 0
                self.clust_medians_and_class_mean = clust_medians

        else:
            self.WCN = None
            self.wrong_clusters = None
            
    def set_swapped(self):
        if type(self.wrong_clusters) == type(None):
            self.swapped = None
        else:
            stacked = self.wrong_clusters.append(
                self.wrong, ignore_index=True)

            self.swapped = stacked[stacked.duplicated("Ind")]

    def r_union(self):
        if type(self.swapped) == type(None):
            return "NaN"
        else:
            return len(self.swapped)
    
    def set_summary(self):
        self.summary = OD((
            ("%Correct", self.total_score),
            ("Score:", self.score),
            ("#Wrong", len(self.wrong)),
            ("#Wrong_cluster", self.WCN),
            ("#Swapped?", len(self.swapped)),
            ("Size", self.N),
            ("N_clusters", self.n_clusters_)
        ))

    
def mahalanobis_dist(point, mean, invcov):
    return cdist(point, mean, metric="mahalanobis", VI=invcov)

def mirror(sample, ab):
    mirrored = sample.copy()
    mirrored[ab] *= -1
    return mirrored

def bootstrap_df(df, i):
    return df.sample(frac=1.0, replace=True, random_state=i)

class BarcodingErrorKMeans(BarcodingError):
    def __init__(self, df, barcode1, barcode2,
                 labels, class_, renaming=False, 
                 cluster_col_name='C', thresholds=[0.9, 0.1],
                 ratio=0.1, rs=42, other_barcodes=None,
                 verbose=0, CORES=-1, n_trees=50):
        """
            trains_RFC.
            finds the best separating features for distinguishing between the 2 classes.
            clusters the dataset.
            then calculate the expected number of cells (from control)
            and the actual number of cells (from sample)
        """
        # Train
        self.set_fields(df, barcode1, barcode2, class_,
                        labels, cluster_col_name, other_barcodes)
    
        self.train_RFC(labels=labels, ratio=ratio, rs=rs, verbose=verbose, CORES=CORES,
                      n_trees=n_trees, class_=class_)
        self.thresholds = thresholds
        self.simulation_results = {}

    def cluster(self, K=5, init="k-means++", bootstrap=True, i=None, **kwargs):
        
        max_iter, n_init, df = self.get_KMeans_params(type(init), bootstrap, K, i)
        cluster_markers = self.cluster_markers

        self.clstr = KMeans(n_clusters=K, 
                            max_iter=max_iter, 
                            n_init=n_init, init=init).fit(df[cluster_markers])

        self.KMeans_labels = self.clstr.predict(self.df[cluster_markers])
        self.df[self.KMeans] = self.KMeans_labels

        self.cluster_centers_ = self.clstr.cluster_centers_
        
        self.value_counts = {0: self.get_sample(class_val = 0)[self.KMeans].value_counts(),
                             1: self.get_sample(class_val = 1)[self.KMeans].value_counts(),
                            "all": self.df[self.KMeans].value_counts()}
        
        self.ratios = {0: self.value_counts[0]/self.value_counts["all"],
                       1: self.value_counts[1]/self.value_counts["all"]
                      }

    def get_KMeans_params(self, type_, bootstrap, K, i):
        if type_ != str:
            max_iter = 1; n_init = 1
        else:
            max_iter = 500; n_init = 200
        
        if bootstrap == True:
            df = bootstrap_df(self.df, i)
        else:
            df = self.df
        self.KMeans = "{}clusters_debarcoding".format(K)
        return max_iter, n_init, df

    def set_cluster_counts(self):
        self.minority_cells = {}
        self.counts_per_condition = []
        self.classes = {}
        self.cluster_sizes = {}
        cluster_labels = sorted(self.df[self.KMeans].unique())
        for C in cluster_labels:
            this_cluster = self.df[self.df[self.KMeans] == C]
            majority = int(round(this_cluster[self.class_].mean(), 0))
            minority = int(not majority)

            self.minority_cells[C] = len(this_cluster[
                this_cluster[self.class_] != majority])
            self.classes[C] = majority
            self.cluster_sizes[C] = len(this_cluster)
            
            self.counts_per_condition.append(
            {majority: self.minority_cells[C], 
             minority: self.cluster_sizes[C] - self.minority_cells[C]})
        
        self.counts_per_condition = pd.DataFrame(self.counts_per_condition)

    def analyse(self, K=5, *args, **kwargs):
        self.cluster(K, **kwargs)
        self.set_cluster_counts()

        self.set_total_score()
        self.set_wrong_clusters(*args, **kwargs) # this is the only different method in BarcodingErrorWithControl
        self.set_swapped()
        self.set_summary()

    def intersect_labels(self, other):
        intersect = lambda other_labels: list(set(self.top_labels).intersection(*other_labels))
        if type(other) == list:
            list_labels = [i.top_labels for i in other]
            self.cluster_markers = intersect(list_labels)
            for sample in other:
                sample.cluster_markers = self.cluster_markers
        else:
            self.cluster_markers = intersect(other.top_labels)
            other.cluster_markers = self.cluster_markers
            # find the right number of clusters here and set self.K and other.K
    
    def set_simulation_results(self, K, average_purity, 
                                ratios_list, cluster_sizes, labels):
        self.simulation_results[K] = \
            SimulationResults(average_purity, ratios_list, cluster_sizes, labels)


class BarcodingErrorSample(BarcodingErrorKMeans):    
    def set_summary(self):
        self.summary = OD((
            ("%Correct", self.total_score),
            ("Score:", self.score),
            ("#Wrong", len(self.wrong)),
            ("#Wrong_cluster", self.WCN),
            ("#Swapped?", len(self.swapped)),
            ("Size", self.N),
            ("N_clusters", self.n_clusters_)
        ))