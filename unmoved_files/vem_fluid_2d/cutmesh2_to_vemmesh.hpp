#pragma once
#include <mandoline/mesh2.hpp>
#include "vem_mesh2.hpp"


void cutmesh2_to_vemmesh(const mandoline::CutCellMesh<2>& ccm, VEMMesh2& vem);
