#include "RcppArmadillo.h"

using namespace arma;
using namespace Rcpp;

arma::mat rwishart(size_t df, const arma::mat& S);


arma::mat riwishart(size_t df, const arma::mat& S);


arma::mat hat_matrix(arma::mat& W);


double squared_error(arma::mat& Y, arma::mat& X);


arma::mat rmatNorm(arma::mat& M, arma::mat& U, arma::mat& V);


Rcpp::List rwishart_bayesm(double nu, arma::mat const& V);


Rcpp::List runiregGibbs(vec const& y, mat const& X, vec const& betabar, mat const& A, double nu, double ssq, double sigmasq, size_t R, size_t keep, size_t nprint);


void rmultireg_IG_singlerun(arma::mat const& Y, arma::mat const& X, arma::mat const& betabar_all, arma::mat const& A, double nu, arma::mat& beta_mat, arma::vec& sigmasq_vec);


List runireg(arma::vec const& y, arma::mat const& X, arma::vec const& betabar, arma::mat const& A, double nu, double ssq, size_t R, size_t keep, size_t nprint);


void runireg_singlerun(arma::vec const& y, arma::mat const& X, arma::vec const& betabar, arma::mat const& A, double nu, double ssq, arma::vec& beta, double& sigmasq);