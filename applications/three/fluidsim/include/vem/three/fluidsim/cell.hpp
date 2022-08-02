#pragma once
#include <Eigen/Sparse>

#include "vem/three/flux_moment_cell.hpp"
#include "vem/three/mesh.hpp"

namespace vem::three::fluidsim {

class FluidVEM3Cell : public FluxMomentVEM3Cell {
    using FluxMomentVEM3Cell::FluxMomentVEM3Cell;
};
}  // namespace vem::fluidsim_3d
