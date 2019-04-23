#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <math.h> 
#include <limits>
using namespace Rcpp;


//' A function to decompose rating matrix with missing values through stochastic gradient descent
//' 
//' @param X A (n*p) user by item rating matrix. Missing ratings should be represented
//'  as 0. Non-missing ratings should be re-scale to exclude zero.
//' @param n_features Dimension of each factor vector
//' @param alpha Learning rate
//' @param gamma Regularization parameter
//' @param tol Tolerance for convergence checking
//' @param max_iter Maximum iterations 
//' @return A list of following outputs:
//' \item{U}{A (n*k) matrix of n user factors}
//' \item{V}{A (p*k) matrix of p item factors}
//' \item{Iterations}{Numer of iterations operated}  
//' \item{Errors}{Root mean square error (RMSE) of the last iteration}
//' @examples
//' # Example code for Funk SVD
//' data("MovieLenseSub") # A subset of MovieLense Data
//' train <- MovieLenseSub[1:200,1:500]
//' train[is.na(train)]<-0
//' 
//' res<-funk_svd(train,max_iter=2000)
//' 
//' users<-c(1,2,3)
//' items<-c(4,5,6,7)
//' 
//' funk_predict(res$U,res$V,users-1,items-1)
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
    
    for (f=0;f<n_features;++f) {
      
      //Update error matrix for user factors update
      UV=U*V.t();
      error_mat=X-UV;
      //0 represents missing ratings, which does not count in the error calculation
      error_mat=error_mat%(X!=0.0);
      
      //Update user factors
      for (j=0;j<n_items;++j) {
        U.col(f)=U.col(f)+alpha*(V(j,f)*error_mat.col(j)-gamma*U.col(f));
      }
      
      //Update error matrix for item factors update
      UV=U*V.t();
      error_mat=X-UV;
      //0 represents missing ratings, which does not count in the error calculation
      error_mat=error_mat%(X!=0.0);
      
      //Update item factors
      for (i=0;i<n_users;++i) {
        V.col(f)=V.col(f)+alpha*(U(i,f)*error_mat.row(i).t()-gamma*V.col(f));
      }
      
    }
    
    // Calculate error terms after a complete iteration
    UV=U*V.t();
    error_mat=X-UV;
    error_mat=error_mat%(X!=0.0);
    error_sum=arma::accu(error_mat%error_mat);
    
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
                      Named("Errors")=pow(error_sum/arma::accu(X>0),0.5) // RMSE
                      );
}



//' A function to predict ratings of given users and items from Funk SVD factors
//' 
//' @param U A (n*k) matrix of n user factors, output from funk_svd()
//' @param V A (p*k) matrix of p item factors, output from funk_svd()
//' @param users A u length vector that contains zero-based indices of selected users
//' @param items A i length vector taht contains zero-based indices of selected items
//' @return A (u*i) matrix with predicted ratings of i items from u users 
//' @examples
//' # Example code for Funk SVD
//' data("MovieLenseSub") # A subset of MovieLense Data
//' train <- MovieLenseSub[1:200,1:500]
//' train[is.na(train)]<-0
//' 
//' res<-funk_svd(train,max_iter=2000)
//' 
//' users<-c(1,2,3)
//' items<-c(4,5,6,7)
//' 
//' funk_predict(res$U,res$V,users-1,items-1)
// [[Rcpp::export]]
arma::mat funk_predict(arma::mat U,arma::mat V,arma::uvec users,arma::uvec items) {
  return U.rows(users)*V.rows(items).t();
}

