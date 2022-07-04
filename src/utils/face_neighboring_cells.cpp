#include "vem/utils/face_neighboring_cells.hpp"

#include <spdlog/spdlog.h>

#include <mtao/iterator/enumerate.hpp>
namespace vem::utils {

std::vector<std::set<int>> face_neighboring_cells(
    const std::vector<std::map<int, bool>>& cells, size_t face_count) {
    std::vector<std::set<int>> ret(cells.size());

    std::vector<std::set<int>> faces(face_count);

    for (auto&& [cidx, c] : mtao::iterator::enumerate(cells)) {
        for (auto&& [f, sgn] : c) {
            faces[f].emplace(cidx);
        }
    }
    for (auto&& c : faces) {
        for (auto&& a : c) {
            for (auto&& b : c) {
                if (a != b) {
                    ret[a].emplace(b);
                }
            }
        }
    }
    return ret;
}
std::vector<std::set<int>> face_neighboring_cells(
    const std::vector<std::map<int, bool>>& cells) {
    size_t face_count = 0;
    for (auto&& [cidx, c] : mtao::iterator::enumerate(cells)) {
        for (auto&& [f, sgn] : c) {
            face_count = std::max<int>(face_count, f);
        }
    }
    return face_neighboring_cells(cells, face_count + 1);
}
std::vector<std::set<int>> face_neighboring_cells(const VEMMesh3& mesh) {
    return face_neighboring_cells(mesh.cell_boundary_map, mesh.face_count());
}
std::vector<std::set<int>> face_neighboring_cells(const VEMMesh2& mesh) {
    return face_neighboring_cells(mesh.face_boundary_map, mesh.edge_count());
}
}  // namespace vem::utils
