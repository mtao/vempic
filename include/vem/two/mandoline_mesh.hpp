
#pragma once

#include <mandoline/mesh2.hpp>

#include "mesh.hpp"

namespace vem::two {

class MandolineVEMMesh2 : public VEMMesh2 {
   public:
    // delaminate_boundaries makes sure that any mesh boundary is pslit into two
    // merge_cell_volume merges all cells of size < volume * dx().prod() in a
    // way that preserves cell regions (only using axial cut-faces)
    MandolineVEMMesh2(const mandoline::CutCellMesh<2> &ccm,
                      const std::optional<double> &merge_cell_volume = {},
                      bool delaminate_boundaries = false);
    bool in_cell(const mtao::Vec2d &p, int cell_index) const override;
    int get_cell(const mtao::Vec2d &p, int last_known = -1) const override;
    std::vector<PolygonBoundaryIndices> face_loops() const override;
    std::set<int> boundary_edge_indices() const override;
    std::vector<std::set<int>> cell_regions() const override;

    double dx() const override;

   private:
    mandoline::CutCellMesh<2> _ccm;
#if defined(MERGE_2D_CELLS)
    std::vector<size_t> _ccm_cell_to_cell;
    std::vector<std::set<size_t>> _cell_to_ccm_cells;
#endif
    std::set<int> boundary_edges;
};


}  // namespace vem
