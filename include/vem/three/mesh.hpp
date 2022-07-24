#pragma once
#include <map>
#include <optional>
#include <set>
#include <vector>
#include "../polygon_boundary_indices.hpp"

namespace vem {

struct VEMMesh3 : public std::enable_shared_from_this<VEMMesh3> {
    // a cached copy of the VEMTopology2's face_loops
    // it may be faster to use face_boundary_map to do operations on the edges?
    // std::vector<std::vector<int>> F;
    std::vector<std::map<int, bool>> cell_boundary_map;
    virtual int face_count() const = 0;
    // this overwrites the 2d variant
    int cell_count() const { return cell_boundary_map.size(); }
    // vertex positions
    mtao::ColVecs3d V;
    // centers
    mtao::ColVecs3d C;
    // face centers
    mtao::ColVecs3d FC;
    std::vector<mtao::Matrix<double, 3, 2>> face_frames;

    std::vector<mtao::ColVecs3i> triangulated_faces;

    VEMMesh3() = default;
    VEMMesh3(const VEMMesh3 &) = default;
    VEMMesh3(VEMMesh3 &&) = default;
    VEMMesh3 &operator=(const VEMMesh3 &) = default;
    VEMMesh3 &operator=(VEMMesh3 &&) = default;
    ~VEMMesh3();

    std::shared_ptr<VEMMesh3> handle() { return shared_from_this(); }
    std::shared_ptr<VEMMesh3 const> handle() const {
        return shared_from_this();
    }

    int vertex_count() const { return V.cols(); }
    virtual std::string type_string() const = 0;
    virtual PolygonBoundaryIndices face_loops(size_t face_index) const = 0;
    virtual mtao::ColVecs3i triangulated_face(size_t face_index) const = 0;
    virtual int get_cell(const mtao::Vec3d &p, int last_known = -1) const = 0;
    virtual mtao::VecXi get_cells(const mtao::ColVecs3d &P,
                                  const mtao::VecXi &last_known = {}) const;
    virtual bool in_cell(const mtao::Vec3d &p, int cell_index) const = 0;
    virtual std::vector<std::set<int>> cell_regions() const;
    virtual Eigen::AlignedBox<double, 3> bounding_box() const;
    // some grid spacing concept, pick mean edge length by default
    virtual double dx() const = 0;
    virtual mtao::Vec3d normal(int face_index) const = 0;
    virtual mtao::Vec4d plane_equation(int face_index) const;
    virtual double surface_area(int face_index) const;
    virtual double diameter(size_t cell_index) const;
    virtual double face_diameter(size_t cell_index) const;

    // grading determines how much the degree of a cell has to be boosted
    virtual bool collision_free(size_t cell_index) const = 0;
    virtual int grade(size_t cell_index) const;
    virtual std::optional<int> cell_category(size_t cell_index) const;

    std::vector<std::set<int>> neighboring_cells;

    // pass in active regions and get a boundary collision mesh, and finally a
    // map from collision triangles to mesh boundary faces
    //
    virtual std::tuple<mtao::ColVecs3d, mtao::ColVecs3i,
                       std::map<size_t, std::set<size_t>>>
    collision_mesh(const std::set<int> &active_cells = {}) const;
};

template <int D>
using VEMMesh = std::conditional_t<D == 2, VEMMesh2, VEMMesh3>;

}  // namespace vem
