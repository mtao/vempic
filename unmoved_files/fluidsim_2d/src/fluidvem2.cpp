
#include "vem/fluidsim_2d/fluidvem2.hpp"

#include <mtao/eigen/mat_to_triplets.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <vem/fluidsim_2d/fluidvem2_cell.hpp>
#include <vem/polynomial_utils.hpp>
#include <vem/utils/cell_identifier.hpp>
#include <vem/utils/local_to_world_sparse_triplets.hpp>
#include <vem/utils/loop_over_active.hpp>
#include <vem/utils/parent_maps.hpp>
#include <vem/utils/volumes.hpp>

using namespace vem::polynomials::two;
namespace vem::fluidsim_2d {

FluidVEM2Base_noT::FluidVEM2Base_noT(const VEMMesh2 &_mesh) : _mesh(_mesh) {}
size_t FluidVEM2Base_noT::cell_count() const { return mesh().cell_count(); }

const std::set<int> &FluidVEM2Base_noT::active_cells() const {
    return _active_cells;
}
void FluidVEM2Base_noT::set_active_cells(std::set<int> c) {
    _active_cells = std::move(c);
}

bool FluidVEM2Base_noT::is_active_cell(int index) const {
    if (_active_cells.empty()) {
        return index >= 0 && index < cell_count();
    } else {
        return _active_cells.contains(index);
    }
}

size_t FluidVEM2Base_noT::active_cell_count() const {
    if (_active_cells.empty()) {
        return cell_count();
    } else {
        return _active_cells.size();
    }
}


}  // namespace vem::fluidsim_2d
