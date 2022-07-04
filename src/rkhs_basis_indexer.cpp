#include "vem/rkhs_basis_indexer.hpp"

#include <spdlog/spdlog.h>

#include <iostream>
#include <mtao/quadrature/gauss_lobatto.hpp>
#include <numeric>

#include "vem/utils/parent_maps.hpp"

namespace vem {
// the vertex_count offsets are so that the internal edges are offset by the
// boundary ones
RKHSBasisIndexer::RKHSBasisIndexer(const VEMMesh2 &mesh,
                                   size_t num_internal_indices)
    : PartitionedCoefficientIndexer(mesh.edge_count(), num_internal_indices,
                                    mesh.vertex_count()),
      _mesh(mesh) {}
RKHSBasisIndexer::RKHSBasisIndexer(
    const VEMMesh2 &mesh, const std::vector<size_t> &internal_edge_sizes)
    : vem::PartitionedCoefficientIndexer(internal_edge_sizes,
                                         mesh.vertex_count()),
      _mesh(mesh) {
}

size_t RKHSBasisIndexer::edge_offset(size_t index) const {
    return PartitionedCoefficientIndexer::partition_offset(index);
}
size_t RKHSBasisIndexer::num_internal_edge_indices(size_t index) const {
    return PartitionedCoefficientIndexer::num_coefficients(index);
}
size_t RKHSBasisIndexer::num_coefficients() const {
    return PartitionedCoefficientIndexer::num_coefficients();
}
size_t RKHSBasisIndexer::num_edge_indices(size_t index) const {
    return num_internal_edge_indices(index) + 2;
}

size_t RKHSBasisIndexer::size() const {
    return PartitionedCoefficientIndexer::num_partitions();
}

std::array<size_t, 2> RKHSBasisIndexer::edge_internal_index_range(
    size_t index) const {
    return PartitionedCoefficientIndexer::coefficient_range(index);
}

const std::vector<size_t> &RKHSBasisIndexer::edge_offsets() const {
    return partition_offsets();
}
mtao::ColVecs2d RKHSBasisIndexer::edge_sample_points(size_t index,
                                                     bool interior) const {
    size_t sample_count = num_edge_indices(index);
    auto p = mtao::quadrature::gauss_lobatto_sample_points<double>(sample_count,
                                                                   0, 1);
    if (index >= _mesh.edge_count()) {
        spdlog::warn("Tried to access an invalid edge index {} not in [0,{})",
                     index, _mesh.edge_count());
        return {};
    }
    auto e = _mesh.E.col(index);

    mtao::ColVecs2d P(2, sample_count - (interior ? 2 : 0));
    auto a = _mesh.V.col(e(0));
    auto b = _mesh.V.col(e(1));
    // hardcode the interior ones and lerp the interior
    if (!interior) {
        P.col(0) = a;
        P.col(sample_count - 1) = b;
    }
    for (int j = 1; j < sample_count - 1; ++j) {
        const double &t = p(j);
        int col = j - (interior ? 1 : 0);
        P.col(col) = (1 - t) * a + t * b;
    }
    return P;
}

mtao::VecXd RKHSBasisIndexer::integrate_edges(
    const mtao::VecXd &coefficients) const {
    mtao::VecXd R(_mesh.edge_count());

    for (auto &&[edge_index, pr] : mtao::iterator::enumerate(
             mtao::iterator::interval<2>(edge_offsets()))) {
        auto &&[start, end] = pr;
        auto e = mesh().E.col(edge_index);
        auto a = mesh().V.col(e(0));
        auto b = mesh().V.col(e(1));
        mtao::VecXd D(end - start + 2);
        D(0) = coefficients(e(0));
        D(D.size() - 1) = coefficients(e(1));
        D.segment(1, end - start) = coefficients.segment(start, end - start);
        R(edge_index) = mtao::quadrature::gauss_lobatto(D, (b - a).norm());
    }
    return R;
}

mtao::VecXd RKHSBasisIndexer::integrate_edges(
    size_t cell, const mtao::VecXd &coefficients) const {
    const auto &fbm = _mesh.face_boundary_map.at(cell);
    mtao::VecXd R(fbm.size());

    auto c = _mesh.C.col(cell);
    for (auto &&[local_index, pr] : mtao::iterator::enumerate(fbm)) {
        auto &&[edge_index, sign] = pr;
        auto &&[start, end] = edge_internal_index_range(edge_index);
        auto e = mesh().E.col(edge_index);
        auto a = mesh().V.col(e(0));
        auto b = mesh().V.col(e(1));
        mtao::VecXd D(end - start + 2);
        D(0) = coefficients(e(0));
        D(D.size() - 1) = coefficients(e(1));
        D.segment(1, end - start) = coefficients.segment(start, end - start);
        R(local_index) = mtao::quadrature::gauss_lobatto(D, (b - a).norm());
    }
    return R;
}

mtao::VecXd RKHSBasisIndexer::integrate_edges(
    size_t cell, const std::function<double(const mtao::Vec2d &)> &f) const {
    const auto &fbm = _mesh.face_boundary_map.at(cell);
    mtao::VecXd R(fbm.size());

    auto c = _mesh.C.col(cell);
    for (auto &&[local_index, pr] : mtao::iterator::enumerate(fbm)) {
        auto &&[edge_index, sign] = pr;
        auto &&[start, end] = edge_internal_index_range(edge_index);
        auto e = mesh().E.col(edge_index);
        auto a = mesh().V.col(e(0));
        auto b = mesh().V.col(e(1));
        auto pts = edge_sample_points(edge_index);
        // std::cout << a.transpose() << " =====> " << b.transpose() <<
        // std::endl; std::cout << pts << std::endl;
        mtao::VecXd D(pts.cols());
        for (size_t i = 0; i < pts.cols(); ++i) {
            D(i) = f(pts.col(i));
        }
        // std::cout << D.transpose() << std::endl;
        // std::cout << D.transpose() << std::endl;
        R(local_index) = mtao::quadrature::gauss_lobatto(D, (b - a).norm());
        // spdlog::trace("Got integrated coeff {}", R(local_index));
    }
    // std::cout << R.transpose() << " => " << R.sum() << std::endl;
    return R;
}
// given a vector field V, returns N \cdot V where the orientation is by each
// edge's default orientation
mtao::VecXd RKHSBasisIndexer::boundary_fluxes(const mtao::ColVecs2d &V) const {
    mtao::VecXd R(_mesh.edge_count());

    for (auto &&[edge_index, pr] : mtao::iterator::enumerate(
             mtao::iterator::interval<2>(edge_offsets()))) {
        auto &&[start, end] = pr;
        auto e = mesh().E.col(edge_index);
        auto a = mesh().V.col(e(0));
        auto b = mesh().V.col(e(1));
        auto ba = (a - b);
        mtao::Vec2d N(-ba.y(), ba.x());
        mtao::VecXd D(end - start + 2);
        D(0) = V.col(e(0)).dot(N);
        D(D.size() - 1) = V.col(e(1)).dot(N);
        D.segment(1, end - start) =
            V.block(0, start, 2, end - start).transpose() * N;

        R(edge_index) = mtao::quadrature::gauss_lobatto(D, (b - a).norm());
    }
    return R;
}
std::set<size_t> RKHSBasisIndexer::cell_indices(size_t cell_index) const {
    std::set<size_t> ret;
    for (auto &&[eidx, sgn] : _mesh.face_boundary_map.at(cell_index)) {
        ret.merge(edge_interior_indices(eidx));
        auto e = _mesh.E.col(eidx);
        ret.emplace(e(0));
        ret.emplace(e(1));
    }
    return ret;
}
std::set<size_t> RKHSBasisIndexer::edge_interior_indices(
    size_t bound_index) const {
    // auto e = _mesh.col(bound_index);
    // auto ee = e.cast<size_t>();

    std::set<size_t> ret;
    //({ee(0), ee(1)});
    auto [start, end] = edge_internal_index_range(bound_index);
    for (size_t i = start; i < end; ++i) {
        ret.emplace(i);
    }
    return ret;
}
std::set<size_t> RKHSBasisIndexer::edge_indices(size_t boundary_index) const {
    auto e = _mesh.E.col(boundary_index);
    auto ret = edge_interior_indices(boundary_index);
    ret.emplace(e(0));
    ret.emplace(e(1));
    return ret;
}

std::vector<size_t> RKHSBasisIndexer::ordered_edge_indices(
    size_t boundary_index) const {
    std::vector<size_t> ret(num_edge_indices(boundary_index));
    auto e = _mesh.E.col(boundary_index);

    ret.front() = e(0);
    ret.back() = e(1);

    auto [start, end] = edge_internal_index_range(boundary_index);
    if (ret.size() > 2) {
        auto startit = ret.begin() + 1;
        auto endit = ret.begin() + ret.size() - 1;

        std::iota(startit, endit, start);
    }

    return ret;
}

std::tuple<std::strong_ordering, size_t> RKHSBasisIndexer::get_edge(
    size_t index) const {
    return get_partition(index);
}
mtao::ColVecs2d RKHSBasisIndexer::get_positions() const {
    mtao::ColVecs2d P(2, num_coefficients());
    for (int j = 0; j < num_coefficients(); ++j) {
        P.col(j) = get_position(j);
    }
    return P;
}
mtao::Vec2d RKHSBasisIndexer::get_position(size_t index) const {
    spdlog::trace("Index: {}", index);
    auto [comp, edge_index] = get_partition(index);

    if (comp == std::strong_ordering::less) {
        spdlog::trace("Getting mesh vertex {} / {}", index,
                      _mesh.vertex_count());
        return _mesh.V.col(index);
    } else if (comp == std::strong_ordering::equivalent) {
        spdlog::trace("Getting vertex in edge {} / {}", edge_index,
                      _mesh.edge_count());
        auto e = _mesh.E.col(edge_index);
        auto a = _mesh.V.col(e(0));
        auto b = _mesh.V.col(e(1));
        size_t sample_count = num_edge_indices(edge_index);
        size_t poly_index = index + 1 - edge_offset(edge_index);
        spdlog::trace("Quadrature entry location {} {}", sample_count,
                      poly_index);
        double t = mtao::quadrature::gauss_lobatto_sample_point<double>(
            sample_count, 0, 1, poly_index);
        return (1 - t) * a + t * b;

    } else {
        spdlog::warn("Tried to get the position of an index too large");
        return {};
    }
}
std::map<size_t, std::set<size_t>> RKHSBasisIndexer::sample_faces() const {
    auto ret = vem::utils::vertex_faces(_mesh);
    auto edge_face_map = vem::utils::edge_faces(_mesh);
    for (size_t eidx = 0; eidx < size(); ++eidx) {
        if (!edge_face_map.contains(eidx)) {
            continue;
        } else {
            const auto &faces = edge_face_map.at(eidx);
            auto vidxs = edge_indices(eidx);
            for (auto &&[fidx, sgn] : faces) {
                for (auto &&vidx : vidxs) {
                    ret[vidx].emplace(fidx);
                }
            }
        }
    }
    return ret;
}
std::map<size_t, std::set<size_t>> RKHSBasisIndexer::sample_edges() const {
    std::map<size_t, std::set<size_t>> ret;
    for (size_t eidx = 0; eidx < size(); ++eidx) {
        auto vidxs = edge_indices(eidx);
        for (auto &&vidx : vidxs) {
            ret[vidx].emplace(eidx);
        }
    }
    return ret;
}

// std::vector<size_t> RKHSBasisIndexer::cell_sample_indices_vec(
//    size_t cell_index) const {
//    auto indset = cell_sample_indices(cell_index);
//    std::vector<size_t> indices(indset.begin(), indset.end());
//    return indices;
//}
}  // namespace vem
