#pragma once

#include "vem/mesh.hpp"

namespace vem {

class TetrahedronVEMMesh3 : public VEMMesh3 {
   public:
    using GridType = mtao::geometry::grid::StaggeredGrid3d;
    TetrahedronVEMMesh3(const mtao::ColVecs3d& V);
    int get_cell(const mtao::Vec3d& p, int last_known = -1) const override;
    bool in_cell(const mtao::Vec3d& p, int cell_index) const override;
    PolygonBoundaryIndices face_loops(size_t cell_index) const override;
    mtao::ColVecs3i triangulated_face(size_t face_index) const override;
    double dx() const override;
    double diameter(size_t cell_index) const override;
    mtao::Vec3d normal(int face_index) const override;
    int face_count() const override;
};

// generates VEMMeshes. by default the centers chosen are centroidal
TetrahedronVEMMesh3 from_tetrahedron(const mtao::ColVecs3d& V);

// generates VEMMeshes. by default the centers chosen are centroidal
TetrahedronVEMMesh3 from_tetrahedron();

}  // namespace vem
