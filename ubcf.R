#' @export
get_neighbors<-function(sim,k){
  neighbors<-t(apply(sim,1,function(x) order(x,decreasing=TRUE,na.last=TRUE)))[,1:k]
  return(neighbors)
}

#' @param train n*p matrix of all ratings from n training users
#' @param test m*p matrix of all ratings from m testing users
#' @param k_neighbors integer that integer the number of nearest neighbors to be used
#' @export
ubcf_predict<-function(train, test,k_neighbors) {
  # train_center<-apply(train,1,function(x) mean(x[x>0])) # n vector
  # test_center<-apply(test,1,function(x) mean(x[x>0])) # m vector
  # train_norm<-train-train_center
  # train_norm[train==0]<-0
  # test_norm<-test-test_center[,1]
  # test_norm[test==0]<-0
  normalized<-normalize(train,test)
  sim<-cos_similarity(normalized$train_norm,normalized$test_norm)
  #sim<-cos_similarity(train,test)
  sim[is.na(sim)]<-0
  neighbors<-get_neighbors(sim,k_neighbors)
  pred_norm<-predict_u(sim,neighbors,normalized$train_norm) # m*p
  return(pred_norm+normalized$test_center[,1])
  #return(predict_u(sim,neighbors,train))
}