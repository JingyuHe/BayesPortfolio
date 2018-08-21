#include "../inst/include/utility.h"

arma::mat rwishart(size_t df, const arma::mat& S){
  // Dimension of returned wishart
  size_t m = S.n_rows;
  
  // Z composition:
  // sqrt chisqs on diagonal
  // random normals below diagonal
  // misc above diagonal
  arma::mat Z(m,m);
  
  // Fill the diagonal
  for(size_t i = 0; i < m; i++){
    Z(i,i) = sqrt(R::rchisq(df-i));
  }
  
  // Fill the lower matrix with random guesses
  for(size_t j = 0; j < m; j++){  
    for(size_t i = j+1; i < m; i++){    
      Z(i,j) = R::rnorm(0,1);
    }
  }
  
  // Lower triangle * chol decomp
  arma::mat C = arma::trimatl(Z).t() * arma::chol(S);
  
  // Return random wishart
  return C.t()*C;
}


arma::mat riwishart(size_t df, const arma::mat& S){
  return rwishart(df,S.i()).i();
}


arma::mat hat_matrix(arma::mat& W){
  size_t N = W.n_cols;
  arma::mat output(N, N);
  output.eye();

  output = output - W * inv(W.t() * W) * W.t();
  return output;
}


double squared_error(arma::mat& Y, arma::mat& X){
  arma::mat beta = inv(X.t() * X) * X.t() * Y;
  arma::mat res = Y - X * beta;
  double output = arma::as_scalar(res.t() * res);
  return output;
}

// [[Rcpp::export]]
arma::mat rmatNorm(arma::mat& M, arma::mat& U, arma::mat& V){
  // M mean
  // U between row covariance
  // V between column covariance
  size_t n = M.n_rows;
  size_t p = M.n_cols;
  arma::mat X = arma::randn(n, p);

  arma::mat A = arma::chol(U, "lower");

  arma::mat B = arma::chol(V, "lower");

  arma::mat output = M + A * X * B;

  return output;
}



void rwishart_bayesm(double nu, arma::mat const& V, arma::mat& CI, arma::mat& C){

// Wayne Taylor 4/7/2015

// Function to draw from Wishart (nu,V) and IW
 
// W ~ W(nu,V)
// E[W]=nuV

// WI=W^-1
// E[WI]=V^-1/(nu-m-1)
  
  // T has sqrt chisqs on diagonal and normals below diagonal
  size_t m = V.n_rows;
  arma::mat T = arma::zeros(m,m);
  
  for(size_t i = 0; i < m; i++) {
    T(i,i) = sqrt(Rcpp::rchisq(1,nu-i)[0]); //rchisq returns a vectorized object, so using [0] allows for the conversion to double
  }
  
  for(size_t j = 0; j < m; j++) {  
    for(size_t i = j+1; i < m; i++) {    
      T(i,j) = Rcpp::rnorm(1)[0]; //rnorm returns a NumericVector, so using [0] allows for conversion to double
  }}
  
  C = trans(T)*arma::chol(V);
  CI = solve(trimatu(C),arma::eye(m,m)); //trimatu interprets the matrix as upper triangular and makes solve more efficient
  
  // C is the upper triangular root of Wishart therefore, W=C'C
  // this is the LU decomposition Inv(W) = CICI' Note: this is
  // the UL decomp not LU!
  
  // W is Wishart draw, IW is W^-1
  
  // return Rcpp::List::create(
    // Rcpp::Named("W") = trans(C) * C,
    // Rcpp::Named("IW") = CI * trans(CI),
    // Rcpp::Named("C") = C,
    // Rcpp::Named("CI") = CI);
  return;
}

