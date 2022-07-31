#pragma once

#include "polygon_mesh.hpp"

namespace vem::two {

// generates VEMMeshes. by default the centers chosen are centroidal
PolygonVEMMesh2 from_polygons(const mtao::ColVecs2d &V,
                              const std::vector<std::vector<int>> &polygons);
// VEMTopology2 from_polygons(const std::vector<std::vector<int>>& polygons);
PolygonVEMMesh2 from_polygons(const mtao::vector<mtao::ColVecs2d> &polygons);

}  // namespace vem
