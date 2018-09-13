// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include "../inst/include/utility.h"

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
arma::mat gibbs_2(arma::mat R, arma::mat F, arma::mat Z, arma::mat X, size_t T, size_t N, size_t K, size_t M, size_t nsamp, size_t burnin, double tau){

    // X is one period lagged Z

    // R: T * N returns
    // F: T * K factors
    // Z: T * M other variables
    // X: T * M, one period lagged Z

    arma::mat Xi;

    arma::cube results(1 + 2 * M + K * (M + 1), N, nsamp);

    arma::mat XXinv = inv(X.t() * X);

    arma::mat A_hat = XXinv * X.t() * F;

    arma::mat Hat_X = hat_matrix(X);

    arma::mat sigma_ff; 

    arma::mat A_f;

    arma::mat Res_f;

    arma::mat W2;

    arma::mat sigma_zz;

    arma::mat gamma_hat;

    arma::mat W2W2inv;

    arma::mat A_z;

    arma::mat z_all_coef;

    arma::mat Res_z;

    arma::mat V_z;

    arma::mat W1;

    arma::mat W1W1inv;

    arma::mat phi_hat;

    arma::mat r_all_coef;

    arma::mat sigma_nn;
    
    arma::mat Bf_z;

    for(size_t i = 0; i < nsamp + burnin; i ++ ){

        // first regression
        // for each i, regress r_i on factors F
        



        if(i > burnin - 1){
            results.slice(i - burnin) = r_all_coef;
        }

    }

    arma::cube alpha(X.n_rows, 1, nsamp);
    arma::cube beta0(K, 1, nsamp);
    arma::cube beta1(Xi.n_rows, 1, nsamp);
    arma::mat temp;
    arma::mat temp2;
    for(size_t j = 0; j < nsamp; j ++ ){
        temp = results.slice(j);
        temp2 = temp.col(74);
        alpha.slice(j) = X * temp.rows(0, M);
        // alpha.slice(j) = X * results.subcube(0, 0, j, M, 0, j).slice(j);

        beta0.slice(j) = results.subcube(1 + M, 0, j, M + K, 0, j);

        beta1.slice(j) = Xi * temp2.rows(M + K + 1, 1 + 2 * M + K * (M + 1) - M - 1);
        
    }

    // arma::mat beta0

    return results;
}

