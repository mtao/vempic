#include "vem/two/cells_adjacent_to_edge.hpp"

#include "vem/utils/loop_over_active.hpp"

namespace vem::utils {

std::vector<std::set<int>> cells_adjacent_to_edge(
    const two::VEMMesh2& mesh, const std::set<int>& active_cells) {
    std::vector<std::set<int>> ret(mesh.edge_count());
    loop_over_active_indices(
        mesh.cell_count(), active_cells, [&](size_t cell_index) {
            const auto& cnbrs = mesh.neighboring_cells[cell_index];
            for (auto&& [e, sgn] : mesh.face_boundary_map[cell_index]) {
                ret[e].emplace(cell_index);
                continue;
                auto& r = ret[e];
                if (r.empty()) {
                    r = cnbrs;
                } else {
                    // do a suboptimal inplace set intersection
                    for (auto it = r.begin(); it != r.end();) {
                        if (cnbrs.contains(*it)) {
                            it = r.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }
        });
    return ret;
}
}  // namespace vem::utils
