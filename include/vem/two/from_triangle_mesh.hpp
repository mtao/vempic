#pragma once

#include "triangle_mesh.hpp"

namespace vem::two {


TriangleVEMMesh2 from_triangle_mesh(const mtao::ColVecs2d &V,
                                    const mtao::ColVecs3i &F);
// VEMMesh3 from_tetrahedral_mesh(const mtao::ColVecs3d& V,
//                               const mtao::ColVecs4i& F);
// VEMTopology3 from_tetrahedral_mesh(const mtao::ColVecs4i& F);

}  // namespace vem
