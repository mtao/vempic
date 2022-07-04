#pragma once
#include <Eigen/Sparse>
#include <mtao/types.hpp>

namespace vem {
// enum class MatrixStructure {
//    Generic,            // do a SVD to solve it
//    Invertible,         // do a householder QR (which isn't supposed to be too
//                        // accurate)
//    SymmetricDefinite,  // do a LLT decomposition
//    SymmetricSemiDefinite  // do a LDLT decomposition
//};
// Given a block Gram matrix of the form
// [ A B ]
// [ C D ]
// Compute the Schur projection matrix
// A - B D^{-1} C
// Eigen::SparseMatrix<double> schur_projector(
//    const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>&
//    B, const Eigen::SparseMatrix<double>& C, const
//    Eigen::SparseMatrix<double>& D);

// mtao::MatXd schur_projector(const mtao::MatXd& A, const mtao::MatXd& B,
//                            const mtao::MatXd& C, const mtao::MatXd& D);
//
// This variant of schur projector assumes that D is a semidefinite matrix
mtao::MatXd semidefinite_schur_projector(const mtao::MatXd &B,
                                         const mtao::MatXd &C,
                                         const mtao::MatXd &D);

}// namespace vem
