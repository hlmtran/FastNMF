#' Run Fast NMF
#' 
#' See RcppML::nmf for more details
#' 
#' @param A sparse input matrix giving data to be factorized
#' @param k rank of the factorization
#' @param tol Pearson correlation distance between model across consecutive iterations at which to call convergence and stop updates
#' @param maxit maximum number of updates, if \code{tol} remains unsatisfied
#' @param L1 L1/LASSO-like penalty to increase sparsity of the model
#' @param verbose print fitting tolerances with each iteration
#' @param threads number of threads to use (0 = let OpenMP decide)
#' @returns List of model matrices \code{w}, \code{d}, and \code{h}
#' @export
#' @useDynLib FastNMF, .registration = TRUE
#' @import Matrix RcppML ggplot2 microbenchmark
#' @importFrom methods as
#' @importFrom stats runif
#' @author Zach DeBruine
#' @examples \dontrun{
#' data(iris)
#' model <- FastNMF(as.matrix(iris[,1:4]), 3)
#' str(model)
#' heatmap(model$w)
#' }
FastNMF <- function(A, k, tol = 1e-4, maxit = 100, verbose = FALSE, L1 = 0, threads = 0){
  w <- matrix(runif(nrow(A) * k), k, nrow(A))   # randomly initialize 'w'
  A <- as(A, "dgCMatrix") # coerce to sparse, if not already
  return(c_nmf(A, t(A), tol, maxit, verbose, L1, threads, w))
}
