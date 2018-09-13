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
Rcpp::List gibbs_2(arma::mat R, arma::mat F, arma::mat Z, arma::mat X, size_t nsamp, size_t burnin){

    // X is one period lagged Z

    // R: T * N returns
    // F: T * K factors
    // Z: T * M other variables
    // X: T * M, one period lagged Z

    size_t T = R.n_rows;
    size_t N = R.n_cols;
    size_t K = F.n_cols;
    size_t M = Z.n_cols;

    // augement F with one column of 1
    arma::mat H = arma::ones<arma::mat>(T, 1);
    H = join_rows(H, F);

    arma::vec Psi(N);
    arma::mat Sigma_u(K, K);
    arma::mat Omega_F(M + 1, K);
    arma::mat Gamma_R(K + 1, N);
    arma::mat Delta(M + 1 + K + N, M);


    // cout << Delta.n_cols << endl;

    arma::mat res_R;
    arma::mat res_F;
    arma::mat W_Z;


    // set priors
    arma::mat A_r_prior_mean = arma::zeros<arma::mat>(K + 1, N);
    arma::mat A_r_prior_cov = arma::eye<arma::mat>(1 + K, 1 + K) * 1000;

    arma::mat A_f_prior_mean = arma::zeros<arma::mat>(M + 1, K);
    arma::mat A_f_prior_cov = arma::eye<arma::mat>(M + 1, M + 1) * 1000;

    arma::mat A_z_prior_mean = arma::zeros<arma::mat>(1 + M + N + K, M);
    arma::mat A_z_prior_cov = arma::eye<arma::mat>(1 + M + N + K, 1 + M + N + K) * 1000;

    // priors of inverse wishart, flat, so all zeros
    arma::mat V_F = arma::zeros<arma::mat>(K, K);
    arma::mat V_Z = arma::zeros<arma::mat>(M, M);
    double nu = 1.0;


    arma::mat sigma_zz_vec;


    // other intermediate variables
    arma::mat Sigma_zz_condition(M, M);
    arma::mat Sigma_v;
    arma::mat Sigma_vu_Sigma_u_inv;
    arma::mat Sigma_ve_Psi_inv;
    arma::mat Sigma_vu;
    arma::mat Sigma_ve;
// cout << "ie " << endl;

// cout << H.n_cols << endl;
    // return/
    // arma::mat results(1, 1);

    // for(size_t i = 0; i < nsamp + burnin; i ++ ){

        // first regression
        // for each i, regress r_i on factors F

        rmultireg_IG_singlerun(R, H, A_r_prior_mean, A_r_prior_cov, nu, Gamma_R, Psi);

    // cout << Gamma_R << endl;

    // cout << "--------- " << endl;

    //     cout << H * Gamma_R << endl;

        // second regression
        // regress F on X

        rmultireg_IW_singlerun(F, X, A_f_prior_mean, A_f_prior_cov, nu, V_F, Omega_F, Sigma_u);


        // cout << H.n_cols << " " << Gamma_R.n_cols << " " << Gamma_R.n_rows << endl;
        // compute residuals of first two regressions
        res_R = R - H * Gamma_R;
        res_F = F - X * Omega_F;

        // create regressors for the third regression
        W_Z = join_rows(join_rows(X, res_R), res_F);


        // cout << Z.n_cols << " " << Z.n_rows << endl;
        //         cout << W_Z.n_cols << " " << W_Z.n_rows << endl;
        // cout << A_z_prior_mean.n_cols << " " << A_z_prior_mean.n_rows << endl;
        // cout << A_z_prior_cov.n_cols << " " << A_z_prior_cov.n_rows << endl;
        // cout << V_Z.n_cols << " " << V_Z.n_rows << endl;
        // cout << Delta.n_cols << " " << Delta.n_rows << endl;
        // cout << Sigma_zz_condition.n_cols << " " << Sigma_zz_condition.n_rows << endl;
        // cout << inv(trans(W_Z) * W_Z) << endl;
        // cout<< res_F << endl;



        // third regression
        rmultireg_IW_singlerun(Z, W_Z, A_z_prior_mean, A_z_prior_cov, nu, V_Z, Delta, Sigma_zz_condition);


        // cout << "ok" << endl;
        // cout << Delta << endl;

        // cout << Delta.n_cols << endl;

        // cout << M << " " << K << " " << N << endl;

        // recover correct unconditional covariance of Z residual
        // sigma_zz_mat = ();
    // cout << "ok " << endl;

        // Delta has 1 + M + K + N columns
        // 0 ~ M are Omega_z
        // M + 1 ~ M + K are Sigma_vu_Sigma_u_inv
        // M + K + 1 ~ M + K + N are Sigma_ve_Psi_inv
        Sigma_vu_Sigma_u_inv = trans(Delta.rows(M+1, M + K));
        Sigma_ve_Psi_inv = trans(Delta.rows(M + K + 1, M + K + N));

        Sigma_vu = Sigma_vu_Sigma_u_inv * Sigma_u;
        Sigma_ve = Sigma_ve_Psi_inv * Psi;

        Sigma_v = Sigma_zz_condition + Sigma_ve_Psi_inv * trans(Sigma_ve) + Sigma_vu_Sigma_u_inv * trans(Sigma_vu);

        // if(i > burnin - 1){
        //     results.slice(i - burnin) = r_all_coef;
        // }

    // }

    // arma::cube alpha(X.n_rows, 1, nsamp);
    // arma::cube beta0(K, 1, nsamp);
    // arma::cube beta1(Xi.n_rows, 1, nsamp);
    // arma::mat temp;
    // arma::mat temp2;
    // for(size_t j = 0; j < nsamp; j ++ ){
    //     temp = results.slice(j);
    //     temp2 = temp.col(74);
    //     alpha.slice(j) = X * temp.rows(0, M);
    //     // alpha.slice(j) = X * results.subcube(0, 0, j, M, 0, j).slice(j);

    //     beta0.slice(j) = results.subcube(1 + M, 0, j, M + K, 0, j);

    //     beta1.slice(j) = Xi * temp2.rows(M + K + 1, 1 + 2 * M + K * (M + 1) - M - 1);
        
    // }

    // arma::mat beta0

    return Rcpp::List::create(
        Named("H") = H,
        Named("Gamma_R") = Gamma_R
    );
}

