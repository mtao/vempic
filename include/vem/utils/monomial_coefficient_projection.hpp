#pragma once
#include "vem/monomial_basis_indexer.hpp"

namespace vem::utils {

Eigen::SparseMatrix<double> monomial_coefficient_projection(
    const MonomialBasisIndexer& from, const MonomialBasisIndexer& to);
}

