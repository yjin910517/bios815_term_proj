// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// funk_svd
List funk_svd(arma::mat& X, int n_features, double alpha, double gamma, double tol, int max_iter);
RcppExport SEXP _recommendeRcpp_funk_svd(SEXP XSEXP, SEXP n_featuresSEXP, SEXP alphaSEXP, SEXP gammaSEXP, SEXP tolSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type n_features(n_featuresSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(funk_svd(X, n_features, alpha, gamma, tol, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// funk_predict
arma::mat funk_predict(arma::mat U, arma::mat V, arma::uvec users, arma::uvec items);
RcppExport SEXP _recommendeRcpp_funk_predict(SEXP USEXP, SEXP VSEXP, SEXP usersSEXP, SEXP itemsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type U(USEXP);
    Rcpp::traits::input_parameter< arma::mat >::type V(VSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type users(usersSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type items(itemsSEXP);
    rcpp_result_gen = Rcpp::wrap(funk_predict(U, V, users, items));
    return rcpp_result_gen;
END_RCPP
}
// normalize
List normalize(arma::mat& train, arma::mat& test);
RcppExport SEXP _recommendeRcpp_normalize(SEXP trainSEXP, SEXP testSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type train(trainSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type test(testSEXP);
    rcpp_result_gen = Rcpp::wrap(normalize(train, test));
    return rcpp_result_gen;
END_RCPP
}
// cos_similarity
arma::mat cos_similarity(arma::mat& A, arma::mat& B);
RcppExport SEXP _recommendeRcpp_cos_similarity(SEXP ASEXP, SEXP BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type B(BSEXP);
    rcpp_result_gen = Rcpp::wrap(cos_similarity(A, B));
    return rcpp_result_gen;
END_RCPP
}
// predict_u
arma::mat predict_u(arma::mat& sim, arma::mat& neighbors, arma::mat& train);
RcppExport SEXP _recommendeRcpp_predict_u(SEXP simSEXP, SEXP neighborsSEXP, SEXP trainSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type sim(simSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type neighbors(neighborsSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type train(trainSEXP);
    rcpp_result_gen = Rcpp::wrap(predict_u(sim, neighbors, train));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_recommendeRcpp_funk_svd", (DL_FUNC) &_recommendeRcpp_funk_svd, 6},
    {"_recommendeRcpp_funk_predict", (DL_FUNC) &_recommendeRcpp_funk_predict, 4},
    {"_recommendeRcpp_normalize", (DL_FUNC) &_recommendeRcpp_normalize, 2},
    {"_recommendeRcpp_cos_similarity", (DL_FUNC) &_recommendeRcpp_cos_similarity, 2},
    {"_recommendeRcpp_predict_u", (DL_FUNC) &_recommendeRcpp_predict_u, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_recommendeRcpp(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}