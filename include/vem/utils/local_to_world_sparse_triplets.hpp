#pragma once
#include <Eigen/Sparse>
#include <mtao/iterator/enumerate.hpp>

namespace vem::utils {
template <typename RowContainer, typename ColContainer, typename Derived,
          typename TripletContainerType>
void local_to_world_sparse_triplets(const RowContainer &RC,
                                    const ColContainer &CC,
                                    const Eigen::MatrixBase<Derived> &M,
                                    TripletContainerType &TC) {
    for (auto &&[ri, gr] : mtao::iterator::enumerate(RC)) {
        for (auto &&[ci, gc] : mtao::iterator::enumerate(CC)) {
            double v = M(ri, ci);
            if (std::abs(v) > 1e-20) {
                TC.emplace_back(gr, gc, v);
            }
        }
    }
}
template <typename RowContainer, typename ColContainer, typename Derived>
std::vector<Eigen::Triplet<typename Derived::Scalar>>
local_to_world_sparse_triplets(const RowContainer &RC, const ColContainer &CC,
                               const Eigen::MatrixBase<Derived> &M) {
    std::vector<Eigen::Triplet<typename Derived::Scalar>> ret;
    ret.reserve(M.size());
    local_to_world_sparse_triplets(RC, CC, M, ret);
    return ret;
}

}  // namespace vem::utils
