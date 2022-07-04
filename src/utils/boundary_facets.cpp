
#include "vem/utils/boundary_facets.hpp"

#include "vem/utils/cell_boundary_facets.hpp"
#include "vem/utils/loop_over_active.hpp"
namespace vem::utils {

std::map<size_t, size_t> boundary_facet_indices(
    const std::vector<std::map<int, bool>> &cells, size_t facet_count,
    const std::set<int> &active_cells = {}) {
    std::vector<int> sizes(facet_count, 0);

    std::map<size_t, size_t> ret;
    loop_over_active_indices(cells.size(), active_cells,
                             [&](size_t cell_index) {
                                 auto &c = cells.at(cell_index);
                                 for (auto &&[eidx, sgn] : c) {
                                     sizes.at(eidx) += sgn ? -1 : 1;
                                     // sizes.at(eidx)++;
                                     ret[eidx] = cell_index;
                                 }
                             });

    for (auto it = ret.begin(); it != ret.end();) {
        // if (sizes.at(it->first) != 0) {
        if (sizes.at(it->first) % 2 != 0) {
            ++it;
        } else {
            it = ret.erase(it);
        }
    }
    return ret;
}

// 2d version
std::map<size_t, size_t> boundary_edge_map(const VEMMesh2 &mesh,
                                           const std::set<int> &active_cells) {
    auto r = boundary_facet_indices(mesh.face_boundary_map, mesh.edge_count(),
                                    active_cells);
    return r;
}
std::map<size_t, size_t> boundary_face_map(const VEMMesh3 &mesh,
                                           const std::set<int> &active_cells) {
    return boundary_facet_indices(mesh.cell_boundary_map, mesh.face_count(),
                                  active_cells);
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
std::set<size_t> boundary_vertices(const VEMMesh3 &mesh,
                                   const std::set<int> &active_cells) {
    std::set<size_t> ret;
    loop_over_active_indices(
        mesh.cell_count(), active_cells, [&](size_t cell_index) {
            auto s = cell_boundary_vertices(mesh, cell_index);
            ret.merge(std::move(s));
        });
    return ret;
}
std::vector<std::set<size_t>> edge_coboundary_map(
    const VEMMesh2 &mesh, const std::set<int> &active_cells) {
    std::vector<std::set<size_t>> ret(mesh.edge_count());

    loop_over_active_indices(
        mesh.cell_count(), active_cells, [&](size_t cell_index) {
            for (auto &&[e, s] : mesh.face_boundary_map.at(cell_index)) {
                ret.at(e).emplace(cell_index);
            }
        });
    return ret;
}

std::vector<std::set<size_t>> face_coboundary_map(
    const VEMMesh3 &mesh, const std::set<int> &active_cells) {
    std::vector<std::set<size_t>> ret(mesh.face_count());

    loop_over_active_indices(
        mesh.cell_count(), active_cells, [&](size_t cell_index) {
            for (auto &&[f, s] : mesh.cell_boundary_map.at(cell_index)) {
                ret.at(f).emplace(cell_index);
            }
        });
    return ret;
}
}  // namespace vem::utils
