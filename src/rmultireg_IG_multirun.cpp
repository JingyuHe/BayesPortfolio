#include "../inst/include/utility.h"


// [[Rcpp::export]]
void rmultireg_IG_multirun(arma::mat const& Y, arma::mat const& X, arma::mat const& betabar_all, arma::mat const& A, double nu, arma::mat& beta_output, arma::mat& sigmasq_vec_output, size_t nsamps){

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

  // arma::mat beta_mat;
  arma::vec sigma;

  size_t n = Y.n_rows;
  size_t m = Y.n_cols; // number of response
  size_t k = X.n_cols;

//   arma::cube betaoutput(R/keep, k, m);
//   arma::mat sigmaoutput(R/keep, m);

  arma::mat beta_hat;
  arma::mat betabar;
  arma::mat beta;
    // mat XpX = trans(X)*X;
  arma::mat IR;
  double s;
  arma::mat res;

  IR = solve(trimatu(chol(trans(X) * X)), eye(k, k));

  arma::mat beta_mat(k, m);
  arma::vec sigmasq_vec(m);

  // arma::cube beta_hat_all(k, 1, );
  arma::mat beta_hat_all(k, m);
  arma::mat res_all(n, m);
  arma::vec s_all(m);

  for(size_t i = 0; i < m; i ++ ){
    y = Y.col(i);
     beta_hat_all.col(i) = (IR * trans(IR)) * trans(X) * y;
     res_all.col(i) = y - X * beta_hat_all.col(i);
     s_all(i) = as_scalar(trans(res_all.col(i)) * res_all.col(i));
  }


  for(size_t j = 0; j < nsamps; j ++ ){
    for(size_t i = 0; i < m; i ++ ){
      betabar = betabar_all.col(i);
      sigmasq = s / rchisq(1, n)[0];
      beta = beta_hat_all.col(i) + sqrt(sigmasq) * (IR * vec(rnorm(k)));
      beta_mat.col(i) = beta;
      sigmasq_vec(i) = sigmasq;
    }
    beta_output.row(j) = trans(vectorise(beta_mat));
    sigmasq_vec_output.row(j) = trans(sigmasq_vec);
  }



  // for(size_t i = 0; i < m; i ++ ){
  //     // loop over each y variable
  //     // assume diagonal covariance matrix, we can run regressions separately

  //   y = Y.col(i);

  //   betabar = betabar_all.col(i);


  //   beta_hat = (IR * trans(IR)) * trans(X) * y;

  //   res = y - X * beta_hat;

  //   s = as_scalar(trans(res) * res);

  //   sigmasq = (s) / rchisq(1, n)[0];

  //   beta = beta_hat + sqrt(sigmasq) * (IR * vec(rnorm(k)));

  //   beta_mat.col(i) = beta;
  //   sigmasq_vec(i) = sigmasq;

  // }
  
  
  return;
}