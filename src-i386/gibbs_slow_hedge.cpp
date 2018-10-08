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
Rcpp::List gibbs_slow_hedge(arma::mat R, arma::mat F, arma::mat Z, arma::mat X, double risk, double r_f, size_t nsamps, arma::mat A_r_prior_mean, arma::mat A_r_prior_precision, arma::mat A_f_prior_mean, arma::mat A_f_prior_precision, arma::mat A_z_prior_mean, arma::mat A_z_prior_precision, double nu, arma::mat V_F, arma::mat V_Z){

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
    // arma::mat A_r_prior_mean = arma::zeros<arma::mat>(K + 1, N);
    // arma::mat A_r_prior_cov = arma::eye<arma::mat>(1 + K, 1 + K) * 1000;

    // arma::mat A_f_prior_mean = arma::zeros<arma::mat>(M + 1, K);
    // arma::mat A_f_prior_cov = arma::eye<arma::mat>(M + 1, M + 1) * 1000;

    // arma::mat A_z_prior_mean = arma::zeros<arma::mat>(1 + M + N + K, M);
    // arma::mat A_z_prior_cov = arma::eye<arma::mat>(1 + M + N + K, 1 + M + N + K) * 1000;

    // // priors of inverse wishart, flat, so all zeros
    // double nu = 3.0;
    // arma::mat V_F = arma::eye<arma::mat>(K, K) * nu;
    // arma::mat V_Z = arma::eye<arma::mat>(M, M) * nu;


    arma::mat sigma_zz_vec;


    // other intermediate variables
    arma::mat Sigma_zz_condition(M, M);
    arma::mat Sigma_v(M, M);
    arma::mat Sigma_vu_Sigma_u_inv;
    arma::mat Sigma_ve_Psi_inv;
    arma::mat Sigma_vu;
    arma::mat Sigma_ve;
    arma::mat Sigma_z;

    // compute weights
    arma::mat mu_assets;
    arma::mat cov_assets;
    arma::mat cov_assets_inv;
    arma::mat A;
    arma::mat B;
    arma::mat alpha;
    arma::mat beta;
    arma::mat theta;
    arma::mat gamma;
    arma::mat weight;
    arma::mat Sigma_f;


    // initialize outputs
    arma::mat Gamma_R_output(nsamps, Gamma_R.n_elem);
    arma::mat Psi_output(nsamps, Psi.n_elem);
    arma::mat Omega_F_output(nsamps, Omega_F.n_elem);
    arma::mat Sigma_u_output(nsamps, Sigma_u.n_elem);
    arma::mat Delta_output(nsamps, Delta.n_elem);
    arma::mat Sigma_zz_condition_output(nsamps, Sigma_zz_condition.n_elem);
    arma::mat Sigma_v_output(nsamps, Sigma_v.n_elem);
    arma::mat mu_output(nsamps, N);
    arma::mat cov_output(nsamps, pow(N, 2));
    arma::mat Sigma_z_output(nsamps, pow(M, 2));
    arma::mat weight_output(nsamps, N);
    arma::mat Sigma_f_output(nsamps, pow(K, 2));


    // MLE estimators for 

    res_R = R - H * inv(trans(H) * H) * trans(H) * R;
    res_F = F - X * inv(trans(X) * X) * trans(X) * F;

    
    for(size_t i = 0; i < nsamps; i ++ ){

        // first regression
        // for each i, regress r_i on factors F

        rmultireg_IG_singlerun(R, H, A_r_prior_mean, A_r_prior_cov, nu, Gamma_R, Psi);

        // second regression
        // regress F on X

        rmultireg_IW_singlerun(F, X, A_f_prior_mean, A_f_prior_cov, nu, V_F, Omega_F, Sigma_u);


        // compute residuals of first two regressions
        // res_R = R - H * Gamma_R;
        // res_F = F - X * Omega_F;

        // create regressors for the third regression
        W_Z = join_rows(join_rows(X, res_R), res_F);

        // third regression
        rmultireg_IW_singlerun(Z, W_Z, A_z_prior_mean, A_z_prior_cov, nu, V_Z, Delta, Sigma_zz_condition);



        // recover unconditonal covariance matrix
        // Delta has 1 + M + K + N columns
        // 0 ~ M are Omega_z
        // M + 1 ~ M + K are Sigma_vu_Sigma_u_inv
        // M + K + 1 ~ M + K + N are Sigma_ve_Psi_inv
        Sigma_vu_Sigma_u_inv = trans(Delta.rows(M+1, M + K));
        Sigma_ve_Psi_inv = trans(Delta.rows(M + K + 1, M + K + N));

        Sigma_vu = Sigma_vu_Sigma_u_inv * Sigma_u;
        Sigma_ve = Sigma_ve_Psi_inv * diagmat(Psi);
        Sigma_v = Sigma_zz_condition + Sigma_ve_Psi_inv * trans(Sigma_ve) + Sigma_vu_Sigma_u_inv * trans(Sigma_vu);


        // compute weights
        A = trans(Delta.row(0));
        B = trans(Delta.rows(1, M));
        alpha = trans(Gamma_R.row(0));
        beta = trans(Gamma_R.rows(1, K));
        theta = trans(Omega_F.row(0));
        gamma = trans(Omega_F.rows(1, M));

        mu_assets = alpha + beta * (theta + gamma * (inv(B + eye(B.n_cols, B.n_cols)) * A));

        Sigma_z = inv(eye(pow(M,2), pow(M,2)) - kron(B, B)) * vectorise(Sigma_v);
        Sigma_z.reshape(M, M);

        Sigma_f = gamma * Sigma_z * trans(gamma) + Sigma_u;

        cov_assets = beta * Sigma_f * trans(beta) + diagmat(Psi);


        cov_assets_inv = inv(cov_assets);



        // cout << cov_assets_inv * mu_assets / as_scalar(1 + trans(mu_assets) * cov_assets_inv * mu_assets) << endl;

        // cout << (1.0 / risk - r_f)  << endl;
        // weight = (1.0 / risk - r_f) * cov_assets_inv * mu_assets / as_scalar(1 + trans(mu_assets) * cov_assets_inv * mu_assets);

        weight = 1.0 / risk * cov_assets_inv * mu_assets;
    // cout << weight << endl;
        // cout << mu_assets << endl;

        // save samples
        cov_output.row(i) = trans(vectorise(cov_assets));
        Sigma_z_output.row(i) = trans(vectorise(Sigma_z));
        mu_output.row(i) = trans(mu_assets);
        Gamma_R_output.row(i) = trans(vectorise(Gamma_R));
        Psi_output.row(i) = trans(Psi);
        Omega_F_output.row(i) = trans(vectorise(Omega_F));
        Sigma_u_output.row(i) = trans(vectorise(Sigma_u));
        Delta_output.row(i) = trans(vectorise(Delta));
        Sigma_zz_condition_output.row(i) = trans(vectorise(Sigma_zz_condition));
        Sigma_v_output.row(i) = trans(vectorise(Sigma_v));
        weight_output.row(i) = trans(weight);
        Sigma_f_output.row(i) = trans(vectorise(Sigma_f));

    }

    return Rcpp::List::create(
        Named("Gamma_R") = Gamma_R_output,
        Named("Psi") = Psi_output,
        Named("Omega_F") = Omega_F_output,
        Named("Sigma_u") = Sigma_u_output,
        Named("Delta") = Delta_output,
        Named("Sigma_zz_condition") = Sigma_zz_condition_output,
        Named("Sigma_v") = Sigma_v_output,
        Named("mu_assets") = mu_output,
        Named("cov_assets") = cov_output,
        Named("Sigma_z") = Sigma_z_output,
        Named("weights") = weight_output,
        Named("Sigma_f") = Sigma_f_output   
    );
}

