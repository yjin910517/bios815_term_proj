#' A function to get the index of the k nearest neighbors
#' 
#' @param sim A (m*n) matrix that contains the pairwise similarity 
#' between m testing users and n training users, output of function cos_similarity()
#' @param k An integer, the number of nearest neighbors to be returned
#' @return A (m*k) matrix, each row contains the index of 
#' k nearest neighbors of testing users
#' @export
get_neighbors<-function(sim,k){
  neighbors<-t(apply(sim,1,function(x) order(x,decreasing=TRUE,na.last=TRUE)))[,1:k]
  return(neighbors)
}


#' A function to predict ratings for testing users, given training user rating matrix.
#' 
#' 
#' @param train A (n*p) user*item rating matrix from training users. 
#' Missing ratings should be represented as 0. 
#' Non-missing ratings should be re-scale to exclude zero.
#' @param test A (m*p) user*item rating matrix from testing users. 
#' Representation should be the same as training data.
#' @param k_neighbors An integer, the number of nearest neighbors to be used
#' @return A (m*p) matrix of predicted ratings for m testing users on all items
#' @examples 
#' # Example code for UBCF
#' data("MovieLenseSub") # A subset of MovieLense Data
#' train <- MovieLenseSub[1:200,]
#' test <- MovieLenseSub[201:250,]
#' train[is.na(train)]<-0
#' test[is.na(test)]<-0
#' 
#' predicted<-ubcf_predict(train,test,k_neighbors=25)
#' topN.idx<-top_n_list(test,predicted,N=10)
#' @export
ubcf_predict<-function(train, test,k_neighbors) {
  normalized<-normalize(train,test)
  sim<-cos_similarity(normalized$train_norm,normalized$test_norm)
  sim[is.na(sim)]<-0
  neighbors<-get_neighbors(sim,k_neighbors)
  pred_norm<-predict_u(sim,neighbors,normalized$train_norm) # m*p
  return(pred_norm+normalized$test_center[,1])
}


#' A function to return Top N recommendations for testing users
#' 
#' @param test A (m*p) original rating matrix from testing users
#' @param pred A (m*p) predicted rating matrix from testing users, 
#' output of function ubcf_predict()
#' @param N An integer, top N items to be recommended
#' @return A (m*N) matrix, each row is the index of recommended N items for the user
#' @examples 
#' # Example code for UBCF
#' data("MovieLenseSub") # A subset of MovieLense Data
#' train <- MovieLenseSub[1:200,]
#' test <- MovieLenseSub[201:250,]
#' train[is.na(train)]<-0
#' test[is.na(test)]<-0
#' 
#' predicted<-ubcf_predict(train,test,k_neighbors=25)
#' topN.idx<-top_n_list(test,predicted,N=10)
#' @export
top_n_list<-function(test,pred,N=10) {
  pred[test>0]<-0
  recom_id<-t(apply(pred,1,function(x) order(x,decreasing=TRUE,na.last=TRUE)))[,1:N]
  return(recom_id)
}