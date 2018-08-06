#include "utility.h"

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