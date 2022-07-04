#include "vem/utils/cell_boundary_facets.hpp"

namespace vem::utils {

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
std::set<size_t> cell_boundary_vertices(const VEMMesh3& mesh, int cell_index) {
    std::set<size_t> ret;
    for (auto&& [fidx, sgn] : mesh.cell_boundary_map.at(cell_index)) {
        auto l = mesh.face_loops(fidx);
        for (auto&& v : l) {
            ret.emplace(v);
        }
        for (auto&& l : l.holes) {
            for (auto&& v : l) {
                ret.emplace(v);
            }
        }
    }
    return ret;
}

std::set<size_t> cell_boundary_vertices(const VEMMesh2& mesh, int cell_index) {
    std::set<size_t> ret;
    for (auto&& [eidx, sgn] : mesh.face_boundary_map.at(cell_index)) {
        auto e = mesh.E.col(eidx);

        ret.emplace(e(0));
        ret.emplace(e(1));
    }
    return ret;
}
}  // namespace vem::utils
