The title of this project is: Automated gating (clustering) and batch effect correction of CyTOF data.

This project is working towards the goal of reproducible CyTOF data analysis. 
Clustering of cytometry data has traditionally been done by manual gating (drawing boxes in scatter plots). This type of analysis is not suitable for 30+ markers, as the 
Another potential problem with mass cytometry is the batch effect. This project will aim to investigate how severe it is and whether current methods: bead normalization and MMD-ResNET, can be used to help integrate datasets taken from different batches.

This repository is under active development and when finished will include code for clustering models, normalization algorithms, visualizations and benchmarking.

I am applying NMF- non-negative matrix factorization. This approach has not beend used in mass cytometry as far as I am aware.
https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
I have tested it using the purity metric: https://en.wikipedia.org/wiki/Cluster_analysis#External_evaluation
And it has performed well, with the right sparsity constraints it shows 87% purity on the Levine dataset.

