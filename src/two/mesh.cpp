#include "vem/mesh.hpp"


#include <mtao/geometry/mesh/edges_to_polygons.hpp>
#include <mtao/geometry/volume.hpp>
#include <numeric>

#include "vem/two/edge_lengths.hpp"
namespace vem::two {

Eigen::AlignedBox<double, 2> VEMMesh2::bounding_box() const {
    return Eigen::AlignedBox<double, 2>(V.rowwise().minCoeff(),
                                        V.rowwise().maxCoeff());
}
std::set<int> VEMMesh2::boundary_edge_indices() const { return {}; }
std::vector<std::set<int>> VEMMesh2::cell_regions() const {
    return {};
    // currently all of the code assumes that empty set = all, so this following
    // bit isn't necessary

    // std::vector<std::set<int>> a(1);
    // for (int j = 0; j < cell_count(); ++j) {
    //    a[0].emplace(j);
    //}
    // return a;
}
double VEMMesh2::dx() const { return edge_lengths(*this).mean(); }

std::vector<mtao::ColVecs3i> VEMMesh2::triangulated_faces() const {
    std::vector<mtao::ColVecs3i> faces;
    auto boundary_loops = face_loops();
    faces.reserve(boundary_loops.size());
    std::transform(boundary_loops.begin(), boundary_loops.end(),
                   std::back_inserter(faces),
                   [&](auto &&loop) { return loop.triangulate(V); });
    return faces;
}
auto VEMMesh2::face_loops() const -> std::vector<PolygonBoundaryIndices> {
    std::vector<PolygonBoundaryIndices> ret(cell_count());

    std::transform(face_boundary_map.begin(), face_boundary_map.end(),
                   ret.begin(), [&](const std::map<int, bool> &map) {
                       mtao::ColVecs2i E(2, map.size());
                       for (auto &&[idx, pr] : mtao::iterator::enumerate(map)) {
                           auto e = E.col(idx);
                           auto me = this->E.col(pr.first);
                           if (pr.second) {
                               e = me.reverse();
                           } else {
                               e = me;
                           }
                       }
                       auto loops =
                           mtao::geometry::mesh::edges_to_polygons(V, E);

                       auto it = loops.begin();
                       double vol = std::abs(mtao::geometry::curve_volume(
                           V, it->begin(), it->end()));
                       for (auto it2 = it + 1; it2 != loops.end(); ++it2) {
                           double nvol = std::abs(mtao::geometry::curve_volume(
                               V, it2->begin(), it2->end()));
                           if (nvol > vol) {
                               it = it2;
                               vol = nvol;
                           }
                       }
                       return *it;
                   });
    return ret;
}


double VEMMesh2::diameter(size_t cell_index) const {
    if (debug_diameter) {
        return 1;
    }
    // std::cout << cell_count() << std::endl;
    // std::cout << E << std::endl;
    // std::cout << V.cols() << std::endl;
    // first compute the radii - i.e the furthest distance from teh center
    const auto &fbm = face_boundary_map.at(cell_index);
    std::set<size_t> verts;
    for (auto &&[eidx, sgn] : fbm) {
        auto e = E.col(eidx);
        verts.emplace(e(0));
        verts.emplace(e(1));
    }
    double d = 0;
    auto c = C.col(cell_index);
    // we're doing things twice, which is stupid. but who cares
    for (auto &&v : verts) {
        for (auto &&u : verts) {
            d = std::max(d, (V.col(v) - V.col(u)).norm());
        }
    }
    return d;
}
double VEMMesh2::boundary_diameter(size_t face_index) const {
    auto e = E.col(face_index);
    auto a = V.col(e(0));
    auto b = V.col(e(1));
    return (a - b).norm();
}
mtao::Vec2d VEMMesh2::boundary_center(size_t edge_index) const {
    auto e = E.col(edge_index);
    auto a = V.col(e(0));
    auto b = V.col(e(1));
    return (a + b) / 2.0;
}

}  // namespace vem
