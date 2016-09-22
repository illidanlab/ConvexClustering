# Robust Convex Clustering

Clustering is an unsupervised learning approach that explores data and seeks
groups of similar objects. Many classical clustering models such as $k$-means
and DBSCAN are based on heuristics algorithms and suffer from local optimal
solutions and numerical instability. Recently convex clustering has
received increasing attentions, which leverages the sparsity inducing norms
and enjoys many attractive theoretical properties. However, convex clustering
is based on Euclidean distance and is thus not robust against outlier
features. Since the outlier features are very common especially when
dimensionality is high, the vulnerability has greatly limited the applicability
of convex clustering to analyze many real-world datasets. 
We address the challenge by proposing a novel robust convex clustering method
that simultaneously performs convex clustering and identifies outlier
features. Specifically, the proposed method learns to decompose the data
matrix into a clustering structure component and a group sparse component that
captures feature outliers. We develop a block coordinate descent algorithm
which iteratively performs convex clustering after outliers features are
identified and eliminated. We also propose an efficient algorithm for solving
the convex clustering by exploiting the structures on its dual problem. This is the 
code for the proposed method. Folder examples shows the efficiency comparision of our
method and AMA on solving convex clustering problem and how robust convex clustering detect outlier features.

