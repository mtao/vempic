
#pragma once
#include "mesh.hpp"

namespace vem::two {

std::vector<std::set<int>> cells_adjacent_to_edge(
    const VEMMesh2& mesh, const std::set<int>& active_cells = {});
}  // namespace vem::utils
   //
namespace vem::utils {

std::vector<std::set<int>> cells_adjacent_to_edge(
    const two::VEMMesh2& mesh, const std::set<int>& active_cells = {});
}  // namespace vem::utils
