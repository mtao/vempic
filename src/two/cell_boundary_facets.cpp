#include "vem/two/cell_boundary_facets.hpp"

namespace vem::two {

/*
std::set<size_t> cell_boundary_edge_indices(const VEMMesh3& mesh,
                                        int cell_index) {
std::set<size_t> edges;
for (auto&& [fidx, sgn] : mesh.cell_boundary_map.at(cell_index)) {
    for (auto&& [eidx, sgn] : mesh.face_boundary_map.at(fidx)) {
        edges.emplace(eidx);
    }
}
return edges;
}
*/

std::set<size_t> cell_boundary_vertices(const VEMMesh2& mesh, int cell_index) {
    std::set<size_t> ret;
    for (auto&& [eidx, sgn] : mesh.face_boundary_map.at(cell_index)) {
        auto e = mesh.E.col(eidx);

        ret.emplace(e(0));
        ret.emplace(e(1));
    }
    return ret;
}
}  // namespace vem::two
