
#pragma once
#include "vem/mesh.hpp"

namespace vem::utils {

std::vector<std::set<int>> cells_adjacent_to_face(
    const VEMMesh3& mesh, const std::set<int>& active_cells = {});
}  // namespace vem::utils
