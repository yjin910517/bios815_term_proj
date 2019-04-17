#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

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

// [[Rcpp::export]]
arma::mat cos_similarity(arma::mat& A,arma::mat& B) {
  arma::vec normA=pow(sum(A%A,1),0.5);
  arma::vec normB=pow(sum(B%B,1),0.5);
  return B*A.t()/(normB*normA.t());
}


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

