#include "../inst/include/utility.h"
  
// [[Rcpp::export]]
Rcpp::List sampler1(arma::mat& Y, arma::mat& X, arma::mat Z, size_t nsamps, size_t burnin){

    return Rcpp::List::create(
        Named("nsamps") = nsamps);
    
}