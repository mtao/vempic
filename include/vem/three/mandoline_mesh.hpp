#pragma once
#include <mandoline/mesh3.hpp>
#include <mandoline/operators/nearest_facet.hpp>

#include "mesh.hpp"

namespace vem::three {

class MandolineVEMMesh3 : public VEMMesh3 {
  public:
    //MandolineVEMMesh3(mandoline::CutCellMesh<3> ccm);
    MandolineVEMMesh3(const mandoline::CutCellMesh<3>& ccm);
    bool in_cell(const mtao::Vec3d &p, int cell_index) const override;
    // bool in_domain(const mtao::Vec3d &p, int cell_index) const override;
    int get_cell(const mtao::Vec3d &p, int last_known = -1) const override;
    mtao::VecXi get_cells(const mtao::ColVecs3d &p,
                          const mtao::VecXi &last_known = {}) const override;
    PolygonBoundaryIndices face_loops(size_t cell_index) const override;
    std::vector<std::set<int>> cell_regions() const override;
    mtao::ColVecs3i triangulated_face(size_t face_index) const override;
    int face_count() const override;
    std::string type_string() const override;

    double dx() const override;
    mtao::Vec3d normal(int face_index) const override;
    int grade(size_t cell_index) const override;
    bool collision_free(size_t cell_index) const override;
    std::optional<int> cell_category(size_t cell_index) const override;

    // private:
    mandoline::CutCellMesh<3> _ccm;
    std::vector<size_t> face_to_ccm_face_map;
    mandoline::operators::CellParentMaps3 _ccm_cell_parents;

    // std::vector<size_t> _ccm_cell_to_cell;
    // std::vector<std::set<size_t>> _cell_to_ccm_cells;
    // std::vector<PolygonBoundaryIndices> _face_loops;
};


}// namespace vem
