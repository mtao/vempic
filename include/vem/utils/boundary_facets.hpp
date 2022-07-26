#pragma once

#include <vector>
#include <map>
namespace vem::utils {

// generic call where we pass in teh boundary map data directly
std::map<size_t, size_t> boundary_facet_map(
    const std::vector<std::map<int, bool>> &cells, size_t facet_count);
}
