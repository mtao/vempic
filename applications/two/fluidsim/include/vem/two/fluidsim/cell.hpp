#pragma once
#if defined(VEM_FLUX_MOMENT_FLUID)
#include "vem/two/flux_moment_cell.hpp"
#else
#include "vem/two/point_moment_cell.hpp"
#endif
#include "vem/two/flux_moment_cell.hpp"
#include "vem/two/point_moment_cell.hpp"
namespace vem::two::fluidsim {
//#if defined(VEM_FLUX_MOMENT_FLUID)
//class FluidVEM2Cell : public FluxMomentVEM2Cell {
//    using FluxMomentVEM2Cell::FluxMomentVEM2Cell;
//};
//#else
//class FluidVEM2Cell : public PointMomentVEM2Cell {
//    using PointMomentVEM2Cell::PointMomentVEM2Cell;
//};
//#endif

}  // namespace vem::two::fluidsim
