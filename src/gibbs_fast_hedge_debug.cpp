// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include "../inst/include/utility.h"

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//
// Main function, all other functions are useless
//
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


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
Rcpp::List gibbs_fast_hedge_debug(arma::mat R, arma::mat F, arma::mat Z, arma::mat X, double risk, double r_f, size_t nsamps, arma::mat A_r_prior_mean, arma::mat A_r_prior_precision, arma::mat A_f_prior_mean, arma::mat A_f_prior_precision, arma::mat A_z_prior_mean, arma::mat A_z_prior_precision, double nu, arma::mat V_F, arma::mat V_Z, size_t n_hedge = 1)
{

    // X is one period lagged Z

    // R: T * N returns
    // F: T * K factors
    // Z: T * M other variables
    // X: T * M, one period lagged Z

    size_t T = R.n_rows;
    size_t N = R.n_cols;
    size_t K = F.n_cols;
    size_t M = Z.n_cols;

    assert(n_hedge < K);

    // augement F with one column of 1
    // H is intercept + factors
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
    // arma::mat A_r_prior_precision = arma::eye<arma::mat>(1 + K, 1 + K) * 1000;

    // arma::mat A_f_prior_mean = arma::zeros<arma::mat>(M + 1, K);
    // arma::mat A_f_prior_precision = arma::eye<arma::mat>(M + 1, M + 1) * 1000;

    // arma::mat A_z_prior_mean = arma::zeros<arma::mat>(1 + M + N + K, M);
    // arma::mat A_z_prior_precision = arma::eye<arma::mat>(1 + M + N + K, 1 + M + N + K) * 1000;

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
    arma::mat beta_f;
    arma::mat beta_g;
    arma::mat theta_0;
    arma::mat theta_1;
    arma::mat weight;
    arma::mat Sigma_f;
    arma::mat mu_factors;
    arma::mat mu_factor_f;
    arma::mat mu_factor_g;
    arma::mat mu_r_tilde;
    arma::mat cov_factor_f;
    arma::mat cov_factor_g;
    arma::mat cov_factor_fg;
    arma::mat cov_new_assets;
    arma::mat cov_factor_f_tilde_r;
    arma::mat cov_tilde_r;
    arma::mat theta_factor_f;
    arma::mat theta_factor_g;
    arma::mat Simga_z;
    arma::mat xi_0;
    arma::mat xi_1;
    arma::mat mu_new_assets;
    arma::mat mu_z;

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
    arma::mat mu_new_assets_output(nsamps, N + n_hedge);
    arma::mat cov_new_assets_output(nsamps, pow(N + n_hedge, 2));

    // MLE estimators for

    // res_R = R - H * inv(trans(H) * H) * trans(H) * R;
    // res_F = F - X * inv(trans(X) * X) * trans(X) * F;
    rmultireg_IW_multirun(F, X, A_f_prior_mean, A_f_prior_precision, nu, V_F, Omega_F_output, Sigma_u_output, nsamps);


    rmultireg_IG_multirun(R, H, A_r_prior_mean, A_r_prior_precision, nu, Gamma_R_output, Psi_output, nsamps);

    // rmultire

    for (size_t i = 0; i < nsamps; i++)
    {

        // first regression
        // for each i, regress r_i on factors F

        // rmultireg_IG_singlerun(R, H, A_r_prior_mean, A_r_prior_precision, nu, Gamma_R, Psi);

        // second regression
        // regress F on X

        // compute residuals of first two regressions

        Gamma_R = Gamma_R_output.row(i);
        Omega_F = Omega_F_output.row(i);
        Gamma_R.reshape(K + 1, N);
        Omega_F.reshape(M + 1, K);
        Psi = vectorise(Psi_output.row(i));
        res_R = R - H * Gamma_R;
        res_F = F - X * Omega_F;
        Sigma_u = Sigma_u_output.row(i);
        Sigma_u.reshape(K, K);

        // cout << Sigma_u << endl;

        // cout << Sigma_u << endl;
        // inv(Sigma_u) ;

        // create regressors for the third regression
        W_Z = join_rows(join_rows(X, res_R), res_F);

        // third regression
        rmultireg_IW_singlerun(Z, W_Z, A_z_prior_mean, A_z_prior_precision, nu, V_Z, Delta, Sigma_zz_condition);

        // recover unconditonal covariance matrix
        // Delta has 1 + M + K + N columns
        // 0 ~ M are Omega_z
        // M + 1 ~ M + K are Sigma_vu_Sigma_u_inv
        // M + K + 1 ~ M + K + N are Sigma_ve_Psi_inv
        Sigma_z = trans(Delta.rows(0, M));
        xi_0 = trans(Delta.row(0));
        xi_1 = trans(Delta.rows(1, M));
        Sigma_vu_Sigma_u_inv = trans(Delta.rows(M + 1, M + K));
        Sigma_ve_Psi_inv = trans(Delta.rows(M + K + 1, M + K + N));

        Sigma_vu = Sigma_vu_Sigma_u_inv * Sigma_u;
        Sigma_ve = Sigma_ve_Psi_inv * diagmat(Psi);
        Sigma_v = Sigma_zz_condition + Sigma_ve_Psi_inv * trans(Sigma_ve) + Sigma_vu_Sigma_u_inv * trans(Sigma_vu);

        // compute weights

        alpha = trans(Gamma_R.row(0));
        beta = trans(Gamma_R.rows(1, K));
        theta_0 = trans(Omega_F.row(0));
        theta_1 = trans(Omega_F.rows(1, M));


        mu_z = inv(eye(M, M) - xi_1) * xi_0;

        // column vector, mean of each factor
        mu_factors = theta_0 + theta_1 * mu_z;

        // column vector, mean of each asset
        mu_assets = alpha + beta * mu_factors;

        // covariance of all macro variables
        Sigma_z = inv(eye(pow(M, 2), pow(M, 2)) - kron(xi_1, xi_1)) * vectorise(Sigma_v);
        Sigma_z.reshape(M, M);

        // covariance of all factors
        Sigma_f = theta_1 * Sigma_z * trans(theta_1) + Sigma_u;


        // This is weight on asset R_t
        cov_assets = beta * Sigma_f * trans(beta) + diagmat(Psi);
        cov_assets_inv = inv(cov_assets);
        weight = 1.0 / risk * cov_assets_inv * mu_assets;


        // cout << mu_assets.n_rows << " " << mu_assets.n_cols << endl;


        // weight on asset (ft, tilde{R}_t)
        if(n_hedge == 1){
            beta_f = beta.row(0);
            beta_g = beta.rows(1, K - 1);
            mu_factor_f = mu_factors.row(0);
            mu_factor_g = mu_factors.rows(1, K - 1);
            cov_factor_f = Sigma_f.submat(0, 0, 0, 0);
            cov_factor_g = Sigma_f.submat(1, 1, K - 1, K - 1);
            cov_factor_fg = Sigma_f.submat(0, 1, 0, K - 1);
            // theta_factor_f = Omega_F.submat(1, 0, M, 0);
            // theta_factor_g = Omega_F.submat(1, 1, M, K - 1);
            theta_factor_f = theta_1.row(0);
            theta_factor_g = theta_1.rows(1, K - 1);

        }else{
            beta_f = beta.rows(0, n_hedge - 1);
            beta_g = beta.rows(n_hedge, K - 1);
            mu_factor_f = mu_factors.rows(0, n_hedge - 1);
            mu_factor_g = mu_factors.rows(n_hedge, K - 1);
            cov_factor_f = Sigma_f.submat(0, 0, n_hedge - 1, n_hedge - 1);
            cov_factor_g = Sigma_f.submat(n_hedge, n_hedge, K - 1, K - 1);
            cov_factor_fg = Sigma_f.submat(0, n_hedge, n_hedge - 1, K - 1);

            theta_factor_f = theta_1.rows(0, n_hedge - 1);
            theta_factor_g = theta_1.rows(n_hedge, K - 1);
            // theta_factor_f = Omega_F.submat(1, 0, M, n_hedge - 1);
            // theta_factor_g = Omega_F.submat(1, n_hedge, M, K - 1);
        }
        
        // cov_factor_f_tilde_r = (trans(theta_factor_f) * Sigma_z * theta_factor_g + cov_factor_fg) * (beta_g);

        cov_factor_f_tilde_r = cov_factor_fg * trans(beta_g);

        // cov_tilde_r = trans(beta_g) * (trans(theta_factor_g) * Sigma_z * (theta_factor_g) + Sigma_u.submat(n_hedge, n_hedge, K - 1, K - 1)) * beta_g + diagmat(Psi);

        cov_tilde_r = beta_g * cov_factor_g * trans(beta_g) + diagmat(Psi);


        cov_new_assets = join_cols(join_rows(cov_factor_f, cov_factor_f_tilde_r), join_rows(trans(cov_factor_f_tilde_r), cov_tilde_r));

        mu_r_tilde = alpha + beta_g * mu_factor_g; 

        mu_new_assets = join_cols(mu_factor_f, mu_r_tilde);

        // save samples
        cov_output.row(i) = trans(vectorise(cov_assets));
        Sigma_z_output.row(i) = trans(vectorise(Sigma_z));
        mu_output.row(i) = trans(mu_assets);
        // Gamma_R_output.row(i) = trans(vectorise(Gamma_R));
        // Psi_output.row(i) = trans(Psi);
        // Omega_F_output.row(i) = trans(vectorise(Omega_F));
        // Sigma_u_output.row(i) = trans(vectorise(Sigma_u));
        Delta_output.row(i) = trans(vectorise(Delta));
        Sigma_zz_condition_output.row(i) = trans(vectorise(Sigma_zz_condition));
        Sigma_v_output.row(i) = trans(vectorise(Sigma_v));
        weight_output.row(i) = trans(weight);
        Sigma_f_output.row(i) = trans(vectorise(Sigma_f));
        mu_new_assets_output.row(i) = trans(mu_new_assets);
        cov_new_assets_output.row(i) = trans(vectorise(cov_new_assets));

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
        Named("Sigma_f") = Sigma_f_output,
        Named("mu_new_assets") = mu_new_assets_output,
        Named("cov_new_assets") = cov_new_assets_output
    );
}