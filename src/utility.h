#include "RcppArmadillo.h"

arma::mat rwishart(size_t df, const arma::mat& S);


arma::mat riwishart(size_t df, const arma::mat& S);


arma::mat hat_matrix(arma::mat& W);


double squared_error(arma::mat& Y, arma::mat& X);