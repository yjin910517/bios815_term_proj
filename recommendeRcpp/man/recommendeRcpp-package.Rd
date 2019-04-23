\name{recommendeRcpp-package}
\alias{recommendeRcpp-package}
\alias{recommendeRcpp}
\docType{package}
\title{
\packageTitle{recommendeRcpp}
}
\description{
\packageDescription{recommendeRcpp}
}
\details{

Given user by item rating matrix, this package provide predictions of unknown ratings through UBCF or Funk SVD algorithm. Missing ratings should be represented as 0. Non-missing ratings should be re-scale to exclude zero.

Major user interface functions are:

- ubcf_predict(): Predict ratings using UBCF algorithm

- top_n_list(): Recommend N items for specified user

- funk_svd(): Decompose ratings matrix

- funk_predict(): Predict ratings using the output fron funk_svd()

Other available functions are: normalize(), cos_similarity(), get_neighbors(), and predict_u(). See function documentation for detail.

}
\author{
\packageAuthor{recommendeRcpp}

Maintainer: \packageMaintainer{recommendeRcpp}
}
\references{
[1]  Michael Hahsler (2018)recommenderlab: Lab for Developing and Testing RecommenderAlgorithms. R package version 0.2-3. http://lyle.smu.edu/IDA/recommenderlab

[2]  Simon Funk (2006) the original blog post of Funk SVD.https://sifter.org/simon/journal/20061211.html

}
\keyword{ package }