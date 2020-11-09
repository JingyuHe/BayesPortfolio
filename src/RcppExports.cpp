// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "BayesPortfolio.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// rsurGibbs_rcpp_loop
List HER_cpp(List const& regdata, vec const& indreg, vec const& cumnk, vec const& nk, mat const& XspXs, mat Sigmainv, mat & A, vec & Abetabar, double nu, mat const& V, int nvar, mat E, mat const& Y, int R, int keep, int nprint);
RcppExport SEXP BayesPortfolio_HER_cpp(SEXP regdataSEXP, SEXP indregSEXP, SEXP cumnkSEXP, SEXP nkSEXP, SEXP XspXsSEXP, SEXP SigmainvSEXP, SEXP ASEXP, SEXP AbetabarSEXP, SEXP nuSEXP, SEXP VSEXP, SEXP nvarSEXP, SEXP ESEXP, SEXP YSEXP, SEXP RSEXP, SEXP keepSEXP, SEXP nprintSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List const& >::type regdata(regdataSEXP);
    Rcpp::traits::input_parameter< vec const& >::type indreg(indregSEXP);
    Rcpp::traits::input_parameter< vec const& >::type cumnk(cumnkSEXP);
    Rcpp::traits::input_parameter< vec const& >::type nk(nkSEXP);
    Rcpp::traits::input_parameter< mat const& >::type XspXs(XspXsSEXP);
    Rcpp::traits::input_parameter< mat >::type Sigmainv(SigmainvSEXP);
    Rcpp::traits::input_parameter< mat & >::type A(ASEXP);
    Rcpp::traits::input_parameter< vec & >::type Abetabar(AbetabarSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< mat const& >::type V(VSEXP);
    Rcpp::traits::input_parameter< int >::type nvar(nvarSEXP);
    Rcpp::traits::input_parameter< mat >::type E(ESEXP);
    Rcpp::traits::input_parameter< mat const& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< int >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type keep(keepSEXP);
    Rcpp::traits::input_parameter< int >::type nprint(nprintSEXP);
    rcpp_result_gen = Rcpp::wrap(HER_cpp(regdata, indreg, cumnk, nk, XspXs, Sigmainv, A, Abetabar, nu, V, nvar, E, Y, R, keep, nprint));
    return rcpp_result_gen;
END_RCPP
}


static const R_CallMethodDef CallEntries[] = {
    {"BayesPortfolio_HER_cpp", (DL_FUNC) &BayesPortfolio_HER_cpp, 16},
    {NULL, NULL, 0}
};

RcppExport void R_init_BayesPortfolio(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
