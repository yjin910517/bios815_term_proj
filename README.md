## BIOSTAT815 Term Project

#### Implementations of two recommendation algorithms in Rcpp

Given user by item rating matrix, this package provide predictions of unknown ratings through UBCF or Funk SVD algorithm. Missing ratings should be represented as 0. Non-missing ratings should be re-scale to exclude zero.

Major user interface functions are:

- ubcf_predict(): Predict ratings using UBCF algorithm

- top_n_list(): Recommend N items for specified user

- funk_svd(): Decompose ratings matrix

- funk_predict(): Predict ratings using the output fron funk_svd()

Other available functions are: normalize(), cos_similarity(), get_neighbors(), and predict_u(). See function documentation for detail.