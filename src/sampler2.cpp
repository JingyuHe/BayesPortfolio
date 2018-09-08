#include "../inst/include/utility.h"

// [[Rcpp::export]]
Rcpp::List sampler2(arma::mat& R, arma::mat& F, arma::mat Z, size_t nsamps, size_t burnin){

    size_t n = R.n_rows;
    size_t p_r = R.n_cols;
    size_t p_f = F.n_cols;
    size_t p_z = Z.n_cols;

    // create a column of 1 as intercept
    arma::mat icept = arma::ones<mat>(n, 1);

    arma::mat A_z(1 + p_z, p_f); // coefficient of F ~ Z + intercept
    arma::mat sigma_ff(p_f, p_f); // residual covariance matrix of F ~ Z + intercept

    arma::mat U; // temp matrix for (F, F - ZA_z)

    arma::mat A_f(1 + 2 * p_f, p_r); // coefficient of R ~ F + (F - ZA_z) + intercept
    arma::vec sigma_rr(p_r); // residual variance of R ~ F + (F - ZA_z) + intercept, assume diagonal structure
    sigma_rr.fill(1.0);

    arma::mat A_f_prior_mean = arma::zeros<arma::mat>(1 + 2 * p_f, p_r);
    arma::mat A_f_prior_cov = arma::eye<arma::mat>(1 + 2 * p_f, 1 + 2 * p_f);
    arma::mat A_z_prior_cov = arma::eye<arma::mat>(1 + p_z, 1 + p_z);
    arma::mat A_z_prior_mean = arma::zeros<arma::mat>(1 + p_z, p_f);
    arma::mat V = arma::eye<arma::mat>(p_f, p_f) * 0.01;

    arma::mat A_z_out(nsamps, A_z_prior_mean.n_elem);
    arma::mat sigma_ff_out(nsamps, sigma_ff.n_elem);
    arma::mat A_f_out(nsamps, A_f_prior_mean.n_elem);
    arma::mat sigma_rr_out(nsamps, sigma_rr.n_elem);


    arma::mat Z_aug = arma::join_rows(icept, Z);
    for(size_t i = 0; i < nsamps + burnin; i ++ ){

        // first regression F ~ Z + intercept
        rmultireg_IW_singlerun(F, Z_aug, A_z_prior_mean, A_z_prior_cov, 1, V, A_z, sigma_ff);

        U = join_rows(icept, join_rows(F, F - Z_aug * A_z));

        rmultireg_IG_singlerun(R, U, A_f_prior_mean, A_f_prior_cov, 1, A_f, sigma_rr);

        if(i > burnin - 1){
            A_z_out.row(i - burnin) = trans(vectorise(A_z));
            sigma_ff_out.row(i - burnin) = trans(vectorise(sigma_ff));
            A_f_out.row(i - burnin) = trans(vectorise(A_f));
            sigma_rr_out.row(i - burnin) = trans(vectorise(sigma_rr));
        }
    }

    return Rcpp::List::create(
        Named("A_z") = A_z_out,
        Named("A_f") = A_f_out,
        Named("sigma_ff") = sigma_ff_out,
        Named("sigma_rr") = sigma_rr_out);

}