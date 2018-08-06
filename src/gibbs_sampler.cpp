// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include "utility.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R
//
// [[Rcpp::export]]
arma::mat gibbs_smapler(arma::mat R, arma::mat F, arma::mat X, arma::mat Z, size_t T, size_t N, size_t K, size_t M, size_t nsamp, size_t burnin, double tau){

    arma::mat results(1,1);

    arma::mat A_hat = inv(X.t() * X) * X.t() * F;



    arma::mat sigma_ff; 
    for(size_t i = 0; i < nsamp + burnin; i ++ ){
        // sigma_ff = 


    }

    return results;
}

