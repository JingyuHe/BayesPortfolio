// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// gibbs_smapler
arma::mat gibbs_smapler(arma::mat R, arma::mat F, arma::mat X, arma::mat Z, size_t T, size_t N, size_t K, size_t M, size_t nsamp, size_t burnin, double tau);
RcppExport SEXP _BayesPortfolio_gibbs_smapler(SEXP RSEXP, SEXP FSEXP, SEXP XSEXP, SEXP ZSEXP, SEXP TSEXP, SEXP NSEXP, SEXP KSEXP, SEXP MSEXP, SEXP nsampSEXP, SEXP burninSEXP, SEXP tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type F(FSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< size_t >::type T(TSEXP);
    Rcpp::traits::input_parameter< size_t >::type N(NSEXP);
    Rcpp::traits::input_parameter< size_t >::type K(KSEXP);
    Rcpp::traits::input_parameter< size_t >::type M(MSEXP);
    Rcpp::traits::input_parameter< size_t >::type nsamp(nsampSEXP);
    Rcpp::traits::input_parameter< size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    rcpp_result_gen = Rcpp::wrap(gibbs_smapler(R, F, X, Z, T, N, K, M, nsamp, burnin, tau));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_BayesPortfolio_gibbs_smapler", (DL_FUNC) &_BayesPortfolio_gibbs_smapler, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_BayesPortfolio(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
