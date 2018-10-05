#include "RcppArmadillo.h"

using namespace arma;
using namespace Rcpp;

arma::mat rwishart(size_t df, const arma::mat& S);


arma::mat riwishart(size_t df, const arma::mat& S);


arma::mat hat_matrix(arma::mat& W);


double squared_error(arma::mat& Y, arma::mat& X);


arma::mat rmatNorm(arma::mat& M, arma::mat& U, arma::mat& V);


void rwishart_bayesm(double nu, arma::mat const& V, arma::mat& CI, arma::mat& C);


Rcpp::List runiregGibbs(vec const& y, mat const& X, vec const& betabar, mat const& A, double nu, double ssq, double sigmasq, size_t R, size_t keep, size_t nprint);


void rmultireg_IG_singlerun(arma::mat const& Y, arma::mat const& X, arma::mat const& betabar_all, arma::mat const& A, double nu, arma::mat& beta_mat, arma::vec& sigmasq_vec);


void rmultireg_IW_singlerun(arma::mat const& Y, arma::mat const& X, arma::mat const& Bbar, arma::mat const& A, double nu, arma::mat const& V, arma::mat& B, arma::mat& Sigma);


List runireg(arma::vec const& y, arma::mat const& X, arma::vec const& betabar, arma::mat const& A, double nu, double ssq, size_t R, size_t keep, size_t nprint);


void runireg_singlerun(arma::vec const& y, arma::mat const& X, arma::vec const& betabar, arma::mat const& A, double nu, double ssq, arma::vec& beta, double& sigmasq);


void rmultireg_IW_multirun(arma::mat const& Y, arma::mat const& X, arma::mat const& Bbar, arma::mat const& A, double nu, arma::mat const& V, arma::mat& B_output, arma::mat& Sigma_output, size_t nsamps);



void rmultireg_IG_multirun(arma::mat const& Y, arma::mat const& X, arma::mat const& betabar_all, arma::mat const& A, double nu, arma::mat& beta_output, arma::mat& sigmasq_vec_output, size_t nsamps);
