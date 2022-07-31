

#pragma once

#include "mesh.hpp"

namespace vem::two {

class PolygonVEMMesh2 : public VEMMesh2 {
   public:
    PolygonVEMMesh2(const mtao::ColVecs2d &V,
                    const std::vector<std::vector<int>> &polygons);
    int get_cell(const mtao::Vec2d &p, int last_known = -1) const override;
    bool in_cell(const mtao::Vec2d &p, int cell_index) const override;

   private:
    std::vector<std::vector<int>> _polygons;
};
}  // namespace vem
