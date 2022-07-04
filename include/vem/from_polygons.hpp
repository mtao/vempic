#pragma once

#include <vem/mesh.hpp>

namespace vem {

class PolygonVEMMesh2 : public VEMMesh2 {
   public:
    PolygonVEMMesh2(const mtao::ColVecs2d &V,
                    const std::vector<std::vector<int>> &polygons);
    int get_cell(const mtao::Vec2d &p, int last_known = -1) const override;
    bool in_cell(const mtao::Vec2d &p, int cell_index) const override;

   private:
    std::vector<std::vector<int>> _polygons;
};
// generates VEMMeshes. by default the centers chosen are centroidal
PolygonVEMMesh2 from_polygons(const mtao::ColVecs2d &V,
                              const std::vector<std::vector<int>> &polygons);
// VEMTopology2 from_polygons(const std::vector<std::vector<int>>& polygons);
PolygonVEMMesh2 from_polygons(const mtao::vector<mtao::ColVecs2d> &polygons);

}  // namespace vem
