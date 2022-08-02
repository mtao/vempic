#pragma once
#include <mandoline/mesh3.hpp>
#include <mandoline/operators/nearest_facet.hpp>
#include "mandoline_mesh.hpp"

#include "mesh.hpp"

namespace vem::three {


MandolineVEMMesh3 from_mandoline(const mandoline::CutCellMesh<3> &ccm);
// VEMMesh3 from_mandoline(const mandoline::CutCellMesh<3>& ccm);

MandolineVEMMesh3 from_mandoline(const Eigen::AlignedBox<double, 3> &bb, int nx, int ny, int nz, const mtao::ColVecs3d &V, const mtao::ColVecs3i &F, int adaptive_grid = 0);

}// namespace vem
