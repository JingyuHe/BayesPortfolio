#include "../inst/include/utility.h"

// [[Rcpp::export]]
void rmultireg_IG_multirun(arma::mat const &Y, arma::mat const &X, arma::mat const &betabar_all, arma::mat const &A, double nu, arma::mat &B_output, arma::vec &Sigma_output, size_t nsamps)
{

    double sigmasq = 1.0; // initialize

    // Keunwoo Kim 09/09/2014

    // Purpose: perform iid draws from posterior of regression model using conjugate prior

    // Arguments:
    //  y,X
    //  betabar,A      prior mean, prior precision
    //  nu, ssq        prior on sigmasq
    //  R number of draws
    //  keep thinning parameter

    // Output: list of beta, sigmasq

    // Model:
    //  y = Xbeta + e  e ~N(0,sigmasq)
    //  y is n x 1
    //  X is n x k
    //  beta is k x 1 vector of coefficients

    // Prior:
    //  beta ~ N(betabar,sigmasq*A^-1)
    //  sigmasq ~ (nu*ssq)/chisq_nu
    //

    arma::vec y;

    size_t n = Y.n_rows;
    size_t m = Y.n_cols; // number of response
    size_t k = X.n_cols;

    //   arma::cube betaoutput(R/keep, k, m);
    //   arma::mat sigmaoutput(R/keep, m);

    arma::mat btilde;

    arma::mat betabar;
    arma::mat beta;
    // mat XpX = trans(X)*X;
    arma::mat IR;
    double s;

    arma::mat res;

    arma::mat RA = chol(A);

    arma::mat W = join_cols(X, RA);

    arma::mat z;

    IR = solve(trimatu(chol(trans(W) * W)), eye(k, k));

    arma::mat temp = (IR * trans(IR)) * trans(W);

    arma::mat beta_mat;

    arma::vec sigmasq_vec;

    for (size_t j = 0; j < nsamps; j++)
    {
        for (size_t i = 0; i < m; i++)
        {
            // loop over each y variable
            // assume diagonal covariance matrix, we can run regressions separately

            y = Y.col(i);

            betabar = betabar_all.col(i);

            z = join_cols(y, RA * betabar);

            btilde = temp * z;

            res = z - W * btilde;

            s = as_scalar(trans(res) * res);

            sigmasq = (s) / rchisq(1, n)[0];

            beta = btilde + sqrt(sigmasq) * (IR * vec(rnorm(k)));

            beta_mat.col(i) = beta;
            sigmasq_vec(i) = sigmasq;
        }
        B_output.row(j) = trans(vectorise(beta_mat));
        Sigma_output.row(j) = sigmasq_vec;
    }

    return;
}
