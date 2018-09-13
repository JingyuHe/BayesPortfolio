#include "../inst/include/utility.h"


// [[Rcpp::export]]
void rmultireg_IG_singlerun(arma::mat const& Y, arma::mat const& X, arma::mat const& betabar_all, arma::mat const& A, double nu, arma::mat& beta_mat, arma::vec& sigmasq_vec) {

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

  arma::mat beta_hat;

  arma::mat betabar;
  arma::mat beta;
    // mat XpX = trans(X)*X;
  arma::mat IR;
  double s;

  arma::mat res;

  for(size_t i = 0; i < m; i ++ ){
      // loop over each y variable
      // assume diagonal covariance matrix, we can run regressions separately

    y = Y.col(i);

    betabar = betabar_all.col(i);


    IR = solve(trimatu(chol(trans(X) * X)), eye(k, k));

    beta_hat = (IR * trans(IR)) * trans(X) * y;

    res = y - X * beta_hat;

    s = as_scalar(trans(res) * res);

    sigmasq = (s) / rchisq(1, n)[0];

    beta = beta_hat + sqrt(sigmasq) * (IR * vec(rnorm(k)));

    // sigmasq = sigmasq_vec(i);

    // beta = beta_mat.col(i);

    // double ssq = 1.0;
    // size_t mkeep;
    // double s;
    // mat RA, W, IR;
    // vec z, btilde, beta;
    
    // size_t nvar = X.n_cols;
    // size_t nobs = y.size();
    
    // // vec sigmasqdraw(R/keep);
    // // mat betadraw(R/keep, nvar);
    

    // vec Xpy = trans(X)*y;
    
    // vec Abetabar = A*betabar;
    

    // // for (size_t rep=0; rep<R; rep++){   
      
    //   //first draw beta | sigmasq
    //   IR = solve(trimatu(chol(XpX/sigmasq+A)), eye(nvar,nvar)); //trimatu interprets the matrix as upper triangular and makes solve more efficient
    //   btilde = (IR*trans(IR)) * (Xpy/sigmasq+Abetabar);
    //   beta = btilde + IR*vec(rnorm(nvar));
      
    //   //now draw sigmasq | beta
    //   s = sum(square(y-X*beta));
    //   sigmasq = (nu*ssq+s) / rchisq(1,nu+nobs)[0]; //rchisq returns a vectorized object, so using [0] allows for the conversion to double
          
    // //   if((rep+1)%keep==0){
    //     // mkeep = (rep+1)/keep;
    //     // betadraw(mkeep-1, span::all) = trans(beta);
    //     // sigmasqdraw[mkeep-1] = sigmasq;
    // //   }   
    // // }  

    beta_mat.col(i) = beta;
    sigmasq_vec(i) = sigmasq;

  }
  
  
  return;
}