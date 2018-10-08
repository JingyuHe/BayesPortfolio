#include "../inst/include/utility.h"

// [[Rcpp::export]]
void rmultireg_IW_singlerun(arma::mat const& Y, arma::mat const& X, arma::mat const& Bbar, arma::mat const& A, double nu, arma::mat const& V, arma::mat& B, arma::mat& Sigma) {

// Keunwoo Kim 09/09/2014

// Purpose: draw from posterior for Multivariate Regression Model with natural conjugate prior

// Arguments:
//  Y is n x m matrix
//  X is n x k
//  Bbar is the prior mean of regression coefficients  (k x m)
//  A is prior precision matrix
//  nu, V are parameters for prior on Sigma

// Output: list of B, Sigma draws of matrix of coefficients and Sigma matrix
 
// Model: 
//  Y=XB+U  cov(u_i) = Sigma
//  B is k x m matrix of coefficients

// Prior:  
//  beta|Sigma  ~ N(betabar,Sigma (x) A^-1)
//  betabar=vec(Bbar)
//  beta = vec(B) 
//  Sigma ~ IW(nu,V) or Sigma^-1 ~ W(nu, V^-1)

  size_t n = Y.n_rows;
  size_t m = Y.n_cols;
  size_t k = X.n_cols;

  // mat IR = solve(trimatu(chol(trans(X)*X)), eye(k,k));
  // mat Btilde = (IR*trans(IR)) * (trans(X)*Y);
  // mat E = Y-X*Btilde;
  // mat S = trans(E)*E;
  // mat ucholinv = solve(trimatu(chol(S)), eye(m,m));
  // mat VSinv = ucholinv*trans(ucholinv);
  // mat CI;
  // mat C;
  // rwishart_bayesm(nu+n, VSinv, CI, C);


  mat RA = chol(A);
  mat W = join_cols(X, RA);
  mat Z = join_cols(Y, RA * Bbar);
  mat IR = solve(trimatu(chol(trans(W) * W)), eye(k, k));
  mat Btilde = (IR*trans(IR)) * (trans(W)*Z);
  mat E = Z-W*Btilde;
  mat S = trans(E)*E;
  mat ucholinv = solve(trimatu(chol(V+S)), eye(m,m));
  mat VSinv = ucholinv*trans(ucholinv);
  mat CI;
  mat C;
  rwishart_bayesm(nu+n, VSinv, CI, C);

  

  mat draw = mat(rnorm(k*m));
  draw.reshape(k,m);
  B = Btilde + IR*draw*trans(CI);
  Sigma = CI * trans(CI);

  return;
}
