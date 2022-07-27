#pragma once


#include "mandoline_mesh.hpp"

namespace vem::two {


MandolineVEMMesh2 from_mandoline(const mandoline::CutCellMesh<2> &ccm,
                                 bool delaminate = false);
// VEMMesh3 from_mandoline(const mandoline::CutCellMesh<3>& ccm);

MandolineVEMMesh2 from_mandoline(const Eigen::AlignedBox<double, 2> &bb, int nx,
                                 int ny, const mtao::ColVecs2d &V,
                                 const mtao::ColVecs2i &E,
                                 bool delaminate = false);

}  // namespace vem
