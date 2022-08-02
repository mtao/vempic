
#pragma once

#include <mtao/geometry/grid/staggered_grid.hpp>

#include "mesh.hpp"

namespace vem::three {

class GridVEMMesh3 : public VEMMesh3,
                     public mtao::geometry::grid::StaggeredGrid3d {
   public:
    using GridType = mtao::geometry::grid::StaggeredGrid3d;
    GridVEMMesh3(const GridType &grid);
    int get_cell(const mtao::Vec3d &p, int last_known = -1) const override;
    bool in_cell(const mtao::Vec3d &p, int cell_index) const override;
    PolygonBoundaryIndices face_loops(size_t cell_index) const override;
    mtao::ColVecs3i triangulated_face(size_t face_index) const override;
    std::string type_string() const override;

    double dx() const override;
    double diameter(size_t cell_index) const override;
    double face_diameter(size_t face_index) const override;
    mtao::Vec3d normal(int face_index) const override;
    int face_count() const override;
    bool collision_free(size_t cell_index) const override;
    std::optional<int> cell_category(size_t cell_index) const override;
};

}  // namespace vem::three
