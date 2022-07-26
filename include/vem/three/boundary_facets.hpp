#pragma once
#include "../utils/boundary_facets.hpp"
#include "mesh.hpp"

namespace vem::utils {

// generic call where we pass in teh boundary map data directly
std::map<size_t, size_t> boundary_facet_map(
    const std::vector<std::map<int, bool>> &cells, size_t facet_count);

// 3d version
std::map<size_t, size_t> boundary_face_map(
    const VEMMesh3 &mesh, const std::set<int> &active_cells = {});

std::set<size_t> boundary_vertices(const VEMMesh3 &mesh,
                                   const std::set<int> &active_cells = {});


std::vector<std::set<size_t>> face_coboundary_map(const VEMMesh3& mesh, const std::set<int>& active_cells = {});

}  // namespace vem::utils
