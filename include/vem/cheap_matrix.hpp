#pragma once
#include "vem/flux_moment_indexer3.hpp"
#include <Eigen/Sparse>


// This is not matrix free, but also a full matrix. It's just cheap

namespace vem {
class CheapMatrix;
}
using Eigen::SparseMatrix;
 
namespace Eigen {
namespace internal {
  // CheapMatrix looks-like a SparseMatrix, so let's inherits its traits:
  template<>
  struct traits<vem::CheapMatrix> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
  {};
}
}
 
namespace vem {
// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
class CheapMatrix : public Eigen::EigenBase<CheapMatrix> {
public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
 
  Index rows() const { return mp_mat->rows(); }
  Index cols() const { return mp_mat->cols(); }
 
  template<typename Rhs>
  Eigen::Product<CheapMatrix,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<CheapMatrix,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
  }
 
  // Custom API:
  CheapMatrix(const FluxMomentIndexer3& indexer) : mp_mat(0) {}
 
  void attachMyMatrix(const SparseMatrix<double> &mat) {
    mp_mat = &mat;
  }
  const SparseMatrix<double> my_matrix() const { return *mp_mat; }
 
private:
  const SparseMatrix<double> nontrivial_section;
  const FluxMomentIndexer3& indexer;
};
}
 
 
// Implementation of CheapMatrix * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
 
  template<typename Rhs>
  struct generic_product_impl<vem::CheapMatrix, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<vem::CheapMatrix,Rhs,generic_product_impl<vem::CheapMatrix,Rhs> >
  {
    typedef typename Product<vem::CheapMatrix,Rhs>::Scalar Scalar;
 
    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const CheapMatrix& lhs, const Rhs& rhs, const Scalar& alpha)
    {
      // This method should implement "dst += alpha * lhs * rhs" inplace,
      // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
      assert(alpha==Scalar(1) && "scaling is not implemented");
      EIGEN_ONLY_USED_FOR_DEBUG(alpha);
 
      // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
      // but let's do something fancier (and less efficient):
      for(Index i=0; i<lhs.cols(); ++i)
        dst += rhs(i) * lhs.my_matrix().col(i);
    }
  };
 
}
}
