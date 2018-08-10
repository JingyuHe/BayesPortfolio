#include "RcppArmadillo.h"

using namespace arma;
using namespace Rcpp;

arma::mat rwishart(size_t df, const arma::mat& S);


arma::mat riwishart(size_t df, const arma::mat& S);


arma::mat hat_matrix(arma::mat& W);


double squared_error(arma::mat& Y, arma::mat& X);

arma::mat rmatNorm(arma::mat& M, arma::mat& U, arma::mat& V);


Rcpp::List rwishart_bayesm(double nu, arma::mat const& V);
Rcpp::List runiregGibbs_rcpp_loop(vec const& y, mat const& X, vec const& betabar, mat const& A, double nu, double ssq, 
                      double sigmasq, int R, int keep, int nprint);
