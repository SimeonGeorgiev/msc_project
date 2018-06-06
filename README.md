The title of this project is: Automated gating (clustering) and batch effect correction of CyTOF data.

This project is working towards the goal of reproducible CyTOF data analysis. 
Clustering of cytometry data has traditionally been done by manual gating (drawing boxes in scatter plots). 

This type of analysis is not suitable for 30+ markers, as the number of plots exhibits a combinatorial explosion.
Another potential problem with mass cytometry is the batch effect. This project will aim to investigate how severe it is and whether current methods: bead normalization and MMD-ResNET, can be used to help integrate datasets taken from different batches.

This repository is under active development and when finished will include code for clustering models, normalization algorithms, visualizations and benchmarking. I have not included my experimental data as it is the property of Newcastle University. It will be shared as soon as possible, if possible.

A short review of used methods:

Batch effect correction:

Bead normalization is the correction of instrument fluctuation throughout the time course of the experiment, it is well established and is offered by several MATLAB and R packages as well as standalone programs.

MMDResNet: A neural network optimization approach that minimizes the maximum mean discrepancy between two samples. The idea is that by fitting statistical distribution of two samples, allowing for some error, the technical variation will be removed, with limited loss of biological variation.

SAUCIE: A similar approach to MMDResNet, but also accomplishes clustering using a multi-tasking network.


Clustering:

I am applying NMF- non-negative matrix factorization. This approach has not beend used in mass cytometry as far as I am aware.
https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
I have tested it using the purity metric: https://en.wikipedia.org/wiki/Cluster_analysis#External_evaluation
And it has performed well, with the right sparsity constraints it shows 87% purity on the Levine dataset.

Phenograph- a well established approach for clustering CyTOF and scRNA data. This algorithm is based on 
optimising the Louvain modularity of a k-nearest neghbour graph, constructed from the data using any distance metric. 

Over-clustering, then merging clusters- this is not a specific algorithm, it is more of a general approach. I have implemented this using K-Means clustering and then merging the clusters using AffinityPropagation on mahalanobis distance matrix between clusters. Other  implementations of this approach can be based on other algorithms.
