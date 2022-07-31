
#include "vem/utils/boundary_facets.hpp"
#include "vem/three/boundary_facets.hpp"
namespace vem::two {

std::map<size_t, size_t> boundary_face_map(const VEMMesh3 &mesh,
                                           const std::set<int> &active_cells) {
    return utils::boundary_facet_indices(mesh.cell_boundary_map, mesh.face_count(),
                                  active_cells);
}
std::set<size_t> boundary_vertices(const VEMMesh3 &mesh,
                                   const std::set<int> &active_cells) {
    std::set<size_t> ret;
    utils::loop_over_active_indices(
        mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto s = cell_boundary_vertices(mesh, cell_index);
            ret.merge(std::move(s));
        });
    return ret;
}
std::vector<std::set<size_t>> face_coboundary_map(
    const VEMMesh3 &mesh, const std::set<int> &active_cells) {
    std::vector<std::set<size_t>> ret(mesh.face_count());

    utils::loop_over_active_indices(
        mesh.cell_count(), active_cells, [&](size_t cell_index) {
            for (auto &&[f, s] : mesh.cell_boundary_map.at(cell_index)) {
                ret.at(f).emplace(cell_index);
            }
        });
    return ret;
}
}
