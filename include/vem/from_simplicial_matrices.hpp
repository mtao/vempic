#pragma once

#include "vem/mesh.hpp"

namespace vem {

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

TriangleVEMMesh2 from_triangle_mesh(const mtao::ColVecs2d &V,
                                    const mtao::ColVecs3i &F);
// VEMMesh3 from_tetrahedral_mesh(const mtao::ColVecs3d& V,
//                               const mtao::ColVecs4i& F);
// VEMTopology3 from_tetrahedral_mesh(const mtao::ColVecs4i& F);

}  // namespace vem
