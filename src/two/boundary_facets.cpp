#include "vem/two/boundary_facets.hpp"
#include "vem/utils/boundary_facets.hpp"
#include "vem/utils/loop_over_active.hpp"

#include "vem/two/cell_boundary_facets.hpp"
namespace vem::two {

// 2d version
std::map<size_t, size_t> boundary_edge_map(const VEMMesh2 &mesh,
                                           const std::set<int> &active_cells) {
    auto r = utils::boundary_facet_indices(mesh.face_boundary_map, mesh.edge_count(),
                                    active_cells);
    return r;
}

std::set<size_t> boundary_vertices(const VEMMesh2 &mesh,
                                   const std::set<int> &active_cells) {
    std::set<size_t> ret;
    for (auto &&[eidx, parent] : boundary_edge_map(mesh, active_cells)) {
        auto e = mesh.E.col(eidx);
        ret.emplace(e(0));
        ret.emplace(e(1));
    }
    return ret;
}

std::vector<std::set<size_t>> edge_coboundary_map(
    const VEMMesh2 &mesh, const std::set<int> &active_cells) {
    std::vector<std::set<size_t>> ret(mesh.edge_count());

    utils::loop_over_active_indices(
        mesh.cell_count(), active_cells, [&](size_t cell_index) {
            for (auto &&[e, s] : mesh.face_boundary_map.at(cell_index)) {
                ret.at(e).emplace(cell_index);
            }
        });
    return ret;
}

}  // namespace vem::utils
