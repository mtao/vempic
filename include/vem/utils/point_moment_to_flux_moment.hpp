#pragma once
#include "vem/flux_moment_indexer.hpp"
#include "vem/point_moment_indexer.hpp"

namespace vem::utils {
Eigen::SparseMatrix<double> point_moment_to_flux_moment(
    const PointMomentIndexer& pmi, const FluxMomentIndexer& fmi);

Eigen::SparseMatrix<double> point_moment_to_flux_moment(
    const PointMomentCell& pmi, const FluxMomentCell& fmi);
}  // namespace vem::utils
