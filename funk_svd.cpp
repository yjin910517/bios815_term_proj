#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <math.h> 
#include <limits>
using namespace Rcpp;

// [[Rcpp::export]]
List funk_svd(arma::mat& X, int n_features=10,
              double alpha=0.001,double gamma=0.015,
              double tol=1.0e-6,int max_iter=200) {
  
  int n_users=(int)X.n_rows;
  int n_items=(int)X.n_cols;
  arma::mat U(n_users,n_features,arma::fill::randu);
  arma::mat V(n_items,n_features,arma::fill::randu);
  
  arma::mat UV(n_users,n_items);
  arma::mat error_mat(n_users,n_items);
  
  int iter,i,j,f;
  double error_sum=0;
  double prev_err=std::numeric_limits<double>::max();
  
  for (iter=0;iter<max_iter;++iter) {
    //Calcuate estimated X and error matrix from factors
    UV=U*V.t();
    error_mat=X-UV;
    //0 represents missing ratings, which does not count in the error calculation
    error_mat=error_mat%(X!=0.0);
    error_sum=arma::accu(error_mat%error_mat);
    
    for (f=0;f<n_features;++f) {
      //make a copy of U so that the update of V still use U from previous iteration
      arma::mat temp_U(U);
      
      for (j=0;j<n_items;++j) {
        U.col(f)=U.col(f)+alpha*(V(j,f)*error_mat.col(j)-gamma*U.col(f));
      }
      
      for (i=0;i<n_users;++i) {
        V.col(f)=V.col(f)+alpha*(temp_U(i,f)*error_mat.row(i).t()-gamma*V.col(f));
      }
      
    }
    
    //Check convergence
    if (prev_err-error_sum>0) {
      if ((prev_err-error_sum)/error_sum<tol) {
        break;
      }
    } else {
      alpha/=2;
    }
    
    prev_err=error_sum;
    
  }//end of each iteration
  
  return List::create(Named("U") = U,
                      Named("V") = V,
                      Named("Iterations")=iter,
                      Named("Errors")=error_sum/n_users/n_items);
}

