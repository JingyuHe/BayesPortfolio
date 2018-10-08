#include "../inst/include/utility.h"

// [[Rcpp::export]]
List runireg_BP(arma::vec const &y, arma::mat const &X, arma::vec const &betabar, arma::mat const &A, double nu, double ssq,
                size_t R, size_t keep, size_t nprint)
{

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

    size_t mkeep;
    double s, sigmasq;
    mat RA, W, IR;
    vec z, btilde, res, beta;

    size_t nvar = X.n_cols;
    size_t nobs = y.size();

    vec sigmasqdraw(R / keep);
    mat betadraw(R / keep, nvar);

    for (size_t rep = 0; rep < R; rep++)
    {
        RA = chol(A);
        W = join_cols(X, RA); //analogous to rbind() in R
        z = join_cols(y, RA * betabar);
        // W'W=R'R ;  (W'W)^-1 = IR IR'  -- this is UL decomp
        IR = solve(trimatu(chol(trans(W) * W)), eye(nvar, nvar)); //trimatu interprets the matrix as upper triangular and makes solve more efficient
        btilde = (IR * trans(IR)) * (trans(W) * z);
        res = z - W * btilde;
        s = as_scalar(trans(res) * res); //converts the matrix to a scalar

        // first draw Sigma
        sigmasq = (nu * ssq + s) / rchisq(1, nu + nobs)[0]; //rchisq returns a vectorized object, so using [0] allows for the conversion to double

        // now draw beta given Sigma
        beta = btilde + sqrt(sigmasq) * (IR * vec(rnorm(nvar)));

        if ((rep + 1) % keep == 0)
        {
            mkeep = (rep + 1) / keep;
            betadraw(mkeep - 1, span::all) = trans(beta);
            sigmasqdraw[mkeep - 1] = sigmasq;
        }
    }

    return List::create(
        Named("betadraw") = betadraw,
        Named("sigmasqdraw") = NumericVector(sigmasqdraw.begin(), sigmasqdraw.end()));
}
