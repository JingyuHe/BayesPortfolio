// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include "../inst/include/utility.h"


// [[Rcpp::export]]
Rcpp::List log_posterior(arma::mat R, arma::mat F, arma::mat Z, arma::mat X){
    double output = 0.0;    
    size_t T = R.n_rows;
    size_t N = R.n_cols;
    size_t K = F.n_cols;
    size_t M = Z.n_cols;

    // augement F with one column of 1
    arma::mat H = arma::ones<arma::mat>(T, 1);
    H = join_rows(H, F);

    // first regression, R ~ F
    arma::mat Gamma_i_hat;
    arma::mat r;
    arma::mat HTH = inv(trans(H) * H) * trans(H);
    arma::mat res;
    // arma::mat ressq;
    double ressq_sum;
    double sigmasq_i;
    for(size_t i = 0; i < N; i ++ ){
        r = R.col(i);
        Gamma_i_hat = HTH * r;
        res = r - H * Gamma_i_hat;
        ressq_sum = as_scalar(sum(pow(res, 2)));
        sigmasq_i = ressq_sum / T;
        output = output - T / 2.0 * log(sigmasq_i) - 0.5 / sigmasq_i * ressq_sum;
    }

    // second regression, F ~ X
    arma::mat XTX = inv(trans(X) * X) * trans(X);
    arma::mat Omega_F_hat = XTX * F;
    arma::mat res_F = F - X * Omega_F_hat;
    arma::mat Sigma_u = trans(res_F) * res_F / T;
    arma::mat Sigma_u_inv = inv(Sigma_u);
    double Sigma_u_det = det(Sigma_u);
    output = output - T / 2.0 * log(Sigma_u_det);// - 0.5 * trace(Sigma_u_inv * Sigma_u)

    // third regression, Z ~ X + res_F + res_R
    arma::mat res_R = R - H * HTH * R;

    arma::mat U = join_rows(join_rows(X, res_F), res_R);

    arma::mat UTU = inv(trans(U) * U) * trans(U);
    arma::mat Delta = UTU * Z;
    arma::mat res_U = Z - U * Delta;
    arma::mat Sigma_v = trans(res_U) * res_U / T;
    arma::mat Sigma_v_inv = inv(Sigma_v);
    double Sigma_v_det = det(Sigma_u);
    output = output - T / 2.0 * log(Sigma_v_det);





    return Rcpp::List::create(
        Named("posterior") = output
    );
}


