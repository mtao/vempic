#pragma once
#include <Eigen/Sparse>

#include "vem/flux_moment_cell3.hpp"
#include "vem/mesh.hpp"
#include "vem/flux_moment_cell3.hpp"

namespace vem::fluidsim_3d {

class FluidVEM3Cell : public FluxMomentVEM3Cell {
    using FluxMomentVEM3Cell::FluxMomentVEM3Cell;
};
}  // namespace vem::fluidsim_3d
