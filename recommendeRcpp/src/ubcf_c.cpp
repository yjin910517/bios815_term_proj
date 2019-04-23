#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;


//' A function to remove individual rating bias from each user
//' 
//' @param train A (n*p) user*item rating matrix from training users
//' @param test A (m*p) user*item rating matrix from testing users
//' @return A list of following outputs:
//' \item{train_norm}{A (n*p) normalized rating matrix for training users}
//' \item{test_norm}{A (m*p) normalized rating matrix for testing users}
//' \item{test_Center}{A m length vector with the mean rating of each testing user}
// [[Rcpp::export]]
List normalize(arma::mat& train,arma::mat& test) {
  int p=train.n_cols;
  arma::rowvec ones(p);
  ones.fill(1);
  arma::colvec train_center=sum(train,1)/sum(train>0,1);
  arma::colvec test_center=sum(test,1)/sum(test>0,1);
  arma::mat train_norm=(train-train_center*ones)%(train>0);
  arma::mat test_norm=(test-test_center*ones)%(test>0);
  return List::create(Named("train_norm")=train_norm,
                      Named("test_norm")=test_norm,
                      Named("test_center")=test_center);
}


//' A function to calculate the pairwise cosine similarity between two matrices
//' 
//' @param A A (n*p) user*item rating matrix 
//' @param B A (m*p) user*item rating matrix
//' @return A (m*n) matrix, where (i,j) element representing the cosine similarity 
//' between row vector A_j and row vector B_i  
// [[Rcpp::export]]
arma::mat cos_similarity(arma::mat& A,arma::mat& B) {
  arma::vec normA=pow(sum(A%A,1),0.5);
  arma::vec normB=pow(sum(B%B,1),0.5);
  return B*A.t()/(normB*normA.t());
}


//' A function to predict (normalized) ratings given k nearest neighbors of testing users
//' 
//' @param sim A (m*n) matrix that contains the pairwise similarity 
//' between m testing users and n training users, output of function cos_similarity()
//' @param neighbors A (m*k) matrix, each row contains the index of 
//' k nearest neighbors of testing users, output of function get_neighbors()
//' @param train A (n*p) rating matrix from n users of training set 
//' @return A (m*p) matrix of predicted ratings for m testing users on all items
// [[Rcpp::export]]
arma::mat predict_u(arma::mat& sim, arma::mat& neighbors, 
                            arma::mat& train) {
  int m=neighbors.n_rows;
  int k=neighbors.n_cols;
  int p=train.n_cols;
  arma::mat ratings(k,p);
  arma::mat weights(1,k);
  arma::mat pred(1,p);
  arma::mat res(m,p);
  int i;
  
  for (i=0;i<m;++i) {
    arma::uvec neighbor_idx=arma::conv_to<arma::uvec>::from(neighbors.row(i))-1;
    arma::uvec uid(1);
    uid.fill(i);
    ratings=train.rows(neighbor_idx);
    weights=sim.submat(uid,neighbor_idx);
    pred=weights*ratings;
    res.row(i)=pred.row(0)/accu(weights);
  }
  return res;
}

