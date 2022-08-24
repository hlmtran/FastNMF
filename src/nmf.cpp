/* This file contains a very simple NMF implementation
 * 
 * Lots of functionality is stripped to make the core
 * update procedures and performance bottlenecks obvious
 * 
 * This file will guide efforts to introduce graph 
 * regularization, GPU parallelization, and distributed
 * computing.
 * 
 * The file is provided within an R package to make testing 
 * and benchmarking easy.
 * 
 */

#define EIGEN_NO_DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

// load the Eigen library headers
//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// currently parallelization is done with OpenMP
//[[Rcpp::plugins(openmp)]]
#ifdef _OPENMP
#include <omp.h>
#endif

// RCPP SPARSE MATRIX CLASS
// Motivation: https://gallery.rcpp.org/articles/sparse-matrix-class/
// The `InnerIterator` class allows for interconversion with `Eigen::SparseMatrix<double>` if templating
class spmat {
public:
  // see the R Matrix::dgCMatrix class documentation
  Rcpp::NumericVector x;
  Rcpp::IntegerVector i, p, Dim;
  
  // constructors
  spmat(Rcpp::NumericVector x, Rcpp::IntegerVector i, Rcpp::IntegerVector p, Rcpp::IntegerVector Dim) : x(x), i(i), p(p), Dim(Dim) {}
  spmat(const Rcpp::S4& s) {
    x = s.slot("x");
    i = s.slot("i");
    p = s.slot("p");
    Dim = s.slot("Dim");
  }
  spmat() {}
  
  size_t rows() { return Dim[0]; }
  size_t cols() { return Dim[1]; }
  
  // const column iterator
  class InnerIterator {
  public:
    InnerIterator(spmat& ptr, int col) : ptr(ptr), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1]) {}
    operator bool() const { return (index < max_index); }
    InnerIterator& operator++() {
      ++index;
      return *this;
    }
    double& value() { return ptr.x[index]; }
    int row() const { return ptr.i[index]; }
    
  private:
    spmat& ptr;
    int col_, index, max_index;
  };
};

// NMF HELPER FUNCTIONS 
// Pearson correlation between two matrices (used for determining convergence)
inline double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
  double x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
  const size_t n = x.size();
  for (size_t i = 0; i < n; ++i) {
    x_i = (*(x.data() + i));
    y_i = (*(y.data() + i));
    sum_x += x_i;
    sum_y += y_i;
    sum_xy += x_i * y_i;
    sum_x2 += x_i * x_i;
    sum_y2 += y_i * y_i;
  }
  return 1 - (n * sum_xy - sum_x * sum_y) / std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
}

// fast symmetric matrix multiplication, A * A.transpose()
// see https://stackoverflow.com/questions/72100483/matrix-multiplication-of-an-eigen-matrix-for-a-subset-of-columns
Eigen::MatrixXd AAt(const Eigen::MatrixXd& A) {
  Eigen::MatrixXd AAt = Eigen::MatrixXd::Zero(A.rows(), A.rows());
  AAt.selfadjointView<Eigen::Lower>().rankUpdate(A);
  AAt.triangularView<Eigen::Upper>() = AAt.transpose();
  AAt.diagonal().array() += 1e-15; // for numerical stability during coordinate descent NNLS
  return AAt;
}

// scale rows in w (or h) to sum to 1 and put previous rowsums in d
void scale(Eigen::MatrixXd& w, Eigen::VectorXd& d) {
  d = w.rowwise().sum();
  d.array() += 1e-15;
  for (size_t i = 0; i < w.rows(); ++i)
    for (size_t j = 0; j < w.cols(); ++j)
      w(i, j) /= d(i);
};

// NNLS SOLVER FOR SYSTEMS IN THE FORM OF ax=b
// optimized and modified from github.com/linxihui/NNLM "c_nnls" function
inline void nnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& h, const size_t sample) {
  double tol = 1;
  for (uint8_t it = 0; it < 100 && (tol / b.size()) > 1e-8; ++it) {
    tol = 0;
    for (size_t i = 0; i < h.rows(); ++i) {
      double diff = b(i) / a(i, i);
      if (-diff > h(i, sample)) {
        if (h(i, sample) != 0) {
          b -= a.col(i) * -h(i, sample);
          tol = 1;
          h(i, sample) = 0;
        }
      } else if (diff != 0) {
        h(i, sample) += diff;
        b -= a.col(i) * diff;
        tol += std::abs(diff / (h(i, sample) + 1e-15));
      }
    }
  }
}

// NMF UPDATE FUNCTIONS
// update h given A and w
void predict(spmat A, const Eigen::MatrixXd& w, Eigen::MatrixXd& h, const double L1, const int threads) {
  Eigen::MatrixXd a = AAt(w);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
  for (size_t i = 0; i < h.cols(); ++i) {
    if (A.p[i] == A.p[i + 1]) continue; // no values in this column
    Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
    
    // BOTTLENECK OPERATION
    for (spmat::InnerIterator it(A, i); it; ++it)
      b += it.value() * w.col(it.row());
    // END BOTTLENECK OPERATION
    b.array() -= L1;
    nnls(a, b, h, i);
  }
}

// NMF FUNCTION
// "w" must be a matrix of dimensions nrow(A)
//[[Rcpp::export]]
Rcpp::List c_nmf(const Rcpp::S4& A_, const Rcpp::S4& At_, const double tol, const uint16_t maxit, const bool verbose,
                   const double L1, const int threads, Eigen::MatrixXd w) {
  // note that "Eigen::MatrixXd w" is passed into this function as a deep copy from Rcpp::NumericMatrix 
  // and handled automatically by Rcpp in "RcppExports.cpp"
  spmat A(A_), At(At_);
  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
  if(w.rows() == A.rows()) w = w.transpose();
  if(w.cols() != A.rows()) Rcpp::stop("dimensions of A and w are incompatible");
  Eigen::MatrixXd h(w.rows(), A.cols());
  Eigen::VectorXd d(w.rows());
  double tol_ = 1;
  for (size_t iter_ = 0; iter_ < maxit && tol_ > tol; ++iter_) {
    Eigen::MatrixXd w_it = w;
    
    // update h, scale h, update w, scale w
    predict(A, w, h, L1, threads);
    scale(h, d);
    predict(At, h, w, L1, threads);
    scale(w, d);
    
    // calculate tolerance of the model fit to detect convergence
    tol_ = cor(w, w_it);  // correlation between "w" across consecutive iterations
    if (verbose) Rprintf("%4d | %8.2e\n", iter_ + 1, tol_);
    Rcpp::checkUserInterrupt();
  }
  
  return Rcpp::List::create(Rcpp::Named("w") = w, Rcpp::Named("d") = d, Rcpp::Named("h") = h);
}
