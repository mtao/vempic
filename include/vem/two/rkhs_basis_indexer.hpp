#pragma once
#include <spdlog/spdlog.h>

#include <iostream>
#include <mtao/iterator/enumerate.hpp>
#include <mtao/iterator/interval.hpp>
#include <mtao/iterator/range.hpp>
#include <mtao/quadrature/gauss_lobatto.hpp>
#include <set>

#include "vem/mesh.hpp"
#include "vem/partitioned_coefficient_indexer.hpp"

namespace vem {
// a fancy name for the point-sample indexer used in this VEM implementation
// it is pecular in that it is indexed along edges, meaning that there are
// certain operations over edges vs cells that need to be taken with care
class RKHSBasisIndexer : public PartitionedCoefficientIndexer {
   public:
    // pass in the number of samples are held on the interior of each edge
    RKHSBasisIndexer(const VEMMesh2 &mesh, size_t num_internal_indices);
    explicit RKHSBasisIndexer(const VEMMesh2 &mesh,
                              const std::vector<size_t> &internal_edge_sizes);
    RKHSBasisIndexer(const RKHSBasisIndexer &) = default;
    RKHSBasisIndexer(RKHSBasisIndexer &&) = default;
    RKHSBasisIndexer &operator=(const RKHSBasisIndexer &) = default;
    RKHSBasisIndexer &operator=(RKHSBasisIndexer &&) = default;

    size_t edge_offset(size_t index) const;
    size_t num_internal_edge_indices(size_t index) const;
    size_t num_edge_indices(size_t index) const;
    size_t num_coefficients() const;
    std::array<size_t, 2> edge_internal_index_range(
        size_t boundary_index) const;
    // should be the same as the number of edges
    size_t size() const;
    bool is_vertex_index(size_t index) const {
        return index < partition_offsets().front();
    }
    std::tuple<std::strong_ordering, size_t> get_edge(size_t index) const;
    mtao::Vec2d get_position(size_t index) const;
    mtao::ColVecs2d get_positions() const;

    // interior denotes whether we should only use the interior or not
    mtao::ColVecs2d edge_sample_points(size_t index,
                                       bool interior_only = false) const;

    std::set<size_t> cell_indices(size_t cell_index) const;
    std::set<size_t> edge_interior_indices(size_t edge_index) const;
    std::set<size_t> edge_indices(size_t edge_index) const;
    std::vector<size_t> ordered_edge_indices(size_t edge_index) const;

    // evaluates a scalar function at each basis element
    template <typename Func>
    mtao::VecXd evaluate_coefficients(Func &&f) const;

    template <typename Func>
    mtao::ColVecs2d evaluate_vector_field(Func &&f) const;

    template <typename Func>
    mtao::VecXd evaluate_coefficients(size_t cell_index, Func &&f) const;

    template <typename Func>
    mtao::ColVecs2d evaluate_vector_field(size_t cell_index, Func &&f) const;

    // integrates the coefficients along each edge and returns a per-edge
    // integral
    mtao::VecXd integrate_edges(const mtao::VecXd &coefficients) const;
    mtao::VecXd integrate_edges(size_t cell,
                                const mtao::VecXd &coefficients) const;
    mtao::VecXd integrate_edges(
        size_t cell, const std::function<double(const mtao::Vec2d &)> &f) const;

    // given a vector field V, returns N \cdot V where the orientation is by
    // each edge's default orientation
    mtao::VecXd boundary_fluxes(const mtao::ColVecs2d &coefficients) const;

    // constructs edge offsets assuming each edge has the same number of
    // internal nodes
    void construct_edge_offsets(size_t num_internal_edge_indices);

    const std::vector<size_t> &edge_offsets() const;

    const VEMMesh2 &mesh() const { return _mesh; }

    std::map<size_t, std::set<size_t>> sample_faces() const;
    std::map<size_t, std::set<size_t>> sample_edges() const;

   private:
    const VEMMesh2 &_mesh;
};
template <typename Func>
mtao::VecXd RKHSBasisIndexer::evaluate_coefficients(Func &&f) const {
    mtao::VecXd R(num_coefficients());
    for (int j = 0; j < _mesh.V.cols(); ++j) {
        R(j) = f(_mesh.V.col(j));
    }
    for (auto &&[edge_index, pr] : mtao::iterator::enumerate(
             mtao::iterator::interval<2>(edge_offsets()))) {
        auto &&[start, end] = pr;

        auto P = edge_sample_points(edge_index, /*interior=*/true);
        for (size_t j = start; j < end; ++j) {
            R(j) = f(P.col(j - start));
        }
    }
    return R;
}
template <typename Func>
mtao::ColVecs2d RKHSBasisIndexer::evaluate_vector_field(Func &&f) const {
    mtao::ColVecs2d R(2, num_coefficients());
    for (int j = 0; j < _mesh.V.cols(); ++j) {
        R.col(j) = f(_mesh.V.col(j));
    }
    for (auto &&[edge_index, pr] : mtao::iterator::enumerate(
             mtao::iterator::interval<2>(edge_offsets()))) {
        auto &&[start, end] = pr;

        auto P = edge_sample_points(edge_index, /*interior=*/true);
        for (size_t j = start; j < end; ++j) {
            R.col(j) = f(P.col(j - start));
        }
    }
    return R;
}
template <typename Func>
mtao::VecXd RKHSBasisIndexer::evaluate_coefficients(size_t cell_index,
                                                    Func &&f) const {
    auto inds = cell_indices(cell_index);

    mtao::VecXd R(inds.size());
    for (auto &&[j, ind] : mtao::iterator::enumerate(inds)) {
        R(j) = f(get_position(ind));
    }
    return R;
}
template <typename Func>
mtao::ColVecs2d RKHSBasisIndexer::evaluate_vector_field(size_t cell_index,
                                                        Func &&f) const {
    mtao::ColVecs2d R(2, num_coefficients());
    for (int j = 0; j < _mesh.V.cols(); ++j) {
        R.col(j) = f(_mesh.V.col(j));
    }
    for (auto &&[edge_index, pr] : mtao::iterator::enumerate(
             mtao::iterator::interval<2>(edge_offsets()))) {
        auto &&[start, end] = pr;

        auto P = edge_sample_points(edge_index, /*interior=*/true);
        for (size_t j = start; j < end; ++j) {
            R.col(j) = f(P.col(j - start));
        }
    }
    return R;
}

}  // namespace vem
