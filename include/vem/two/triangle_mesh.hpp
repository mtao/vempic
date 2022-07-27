#pragma once

#include "mesh.hpp"

namespace vem::two {

class TriangleVEMMesh2 : public VEMMesh2 {
   public:
    TriangleVEMMesh2(const mtao::ColVecs2d &V, const mtao::ColVecs3i &F);
    std::vector<mtao::ColVecs3i> triangulated_faces() const override;
    int get_cell(const mtao::Vec2d &p, int last_known = -1) const override;
    bool in_cell(const mtao::Vec2d &p, int cell_index) const override;

   private:
    // triangle vertices are stored in the input
    mtao::ColVecs3i F;
};


}  // namespace vem
