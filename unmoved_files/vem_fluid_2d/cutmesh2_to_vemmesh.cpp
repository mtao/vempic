#include "cutmesh2_to_vemmesh.hpp"

#include <spdlog/spdlog.h>

void cutmesh2_to_vemmesh(const mandoline::CutCellMesh<2>& ccm, VEMMesh2& vem) {
    {
        vem.cells.resize(ccm.num_cells());
        vem.centers.resize(2, ccm.num_cells());
        const auto& ccmB = ccm.m_face_boundary_map;
        spdlog::warn("CCM face boundary map size: {}", ccmB.size());
        for (auto&& [cidx, mp] : ccmB) {
            auto& a = vem.cells[cidx];
            std::transform(mp.begin(), mp.end(), std::inserter(a, a.end()),
                           [](const std::pair<const int, bool>& pr)
                               -> std::pair<const size_t, bool> {
                               auto [a, b] = pr;
                               return {size_t(a), b};
                           });
        }
        for (auto&& [fidx, face] : mtao::iterator::enumerate(ccm.cut_faces())) {
            if (face.external_boundary) {
                auto [bound, sgn] = *face.external_boundary;
                auto cell = ccm.cell_grid().unindex(bound);
                int cell_index =
                    ccm.exterior_grid.cell_indices()(cell) + ccm.num_cutcells();
                vem.cells[cell_index][fidx] = sgn;
            }
        }
        for (auto&& [fidx_noff, pr] : mtao::iterator::enumerate(
                 ccm.exterior_grid.boundary_facet_pairs())) {
            int fidx = fidx_noff + ccm.num_cutfaces();
            int axis = ccm.exterior_grid.get_face_axis(fidx);
            bool flip = axis == 1;
            auto [a, b] = pr;
            if (a >= 0) {
                vem.cells[a + ccm.num_cutcells()][fidx] = false ^ flip;
            }
            if (b >= 0) {
                vem.cells[b + ccm.num_cutcells()][fidx] = true ^ flip;
            }
        }
        // make the centers the grid cell centers
        for (auto&& [cidx, ccindx] : ccm.cell_grid_ownership) {
            auto center = ccm.cell_grid().vertex(cidx);
            for (auto&& c : ccindx) {
                vem.centers.col(c) = center;
            }
        }

        for (auto&& [ind, cell] :
             mtao::iterator::enumerate(ccm.exterior_grid.cell_coords())) {
            vem.centers.col(ind + ccm.num_cutcells()) =
                ccm.cell_grid().vertex(cell);
        }
    }
    vem.edges = ccm.edges();
    vem.vertices = ccm.vertices();
}
