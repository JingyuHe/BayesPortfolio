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
arma::mat gibbs(arma::mat R, arma::mat F, arma::mat X, arma::mat Xi, arma::mat Z, size_t T, size_t N, size_t K, size_t M, size_t nsamp, size_t burnin, double tau)
{

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

    for (size_t i = 0; i < nsamp + burnin; i++)
    {
        // sigma_ff =
        sigma_ff = riwishart(T - 2 * M + N - 1, F.t() * Hat_X * F);

        A_f = rmatNorm(A_hat, XXinv, sigma_ff);

        Res_f = F - X * A_f;

        W2 = arma::join_rows(X, Res_f);

        sigma_zz = riwishart(T - N - M - 1, Z.t() * hat_matrix(W2) * Z);

        W2W2inv = inv(W2.t() * W2);

        gamma_hat = W2W2inv * W2.t() * Z;

        z_all_coef = rmatNorm(gamma_hat, W2W2inv, sigma_zz);

        Bf_z = z_all_coef.rows(M + 1, M + K);

        A_z = z_all_coef.rows(0, M);

        Res_z = Z - X * A_z;

        V_z = Res_z - Res_f * Bf_z;

        W1 = arma::join_rows(arma::join_rows(arma::join_rows(X, F), Xi), V_z);

        W1W1inv = inv(W1.t() * W1);

        phi_hat = W1W1inv * R;

        sigma_nn = riwishart(T - M * (K + 1) - 1, R.t() * hat_matrix(W1) * R);

        r_all_coef = rmatNorm(phi_hat, W1W1inv, sigma_nn);

        if (i > burnin - 1)
        {
            results.slice(i - burnin) = r_all_coef;
        }
    }

    arma::cube alpha(X.n_rows, 1, nsamp);
    arma::cube beta0(K, 1, nsamp);
    arma::cube beta1(Xi.n_rows, 1, nsamp);
    arma::mat temp;
    arma::mat temp2;
    for (size_t j = 0; j < nsamp; j++)
    {
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
