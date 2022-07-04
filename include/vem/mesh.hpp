#pragma once
#include <map>
#include <mtao/geometry/mesh/edges_to_polygons.hpp>
#include <mtao/types.hpp>
#include <optional>
#include <set>
#include <vector>

namespace vem {

struct PolygonBoundaryIndices
    : public mtao::geometry::mesh::PolygonBoundaryIndices {
    using Base = mtao::geometry::mesh::PolygonBoundaryIndices;
    PolygonBoundaryIndices &operator=(const PolygonBoundaryIndices &) = default;
    PolygonBoundaryIndices &operator=(const Base &b) {
        static_cast<Base *>(this)->operator=(b);
        return *this;
    }
    PolygonBoundaryIndices() = default;
    PolygonBoundaryIndices(std::vector<int> b,
                           std::set<std::vector<int>> h = {})
        : Base(b, h) {}
    mtao::ColVecs3i triangulate(const mtao::ColVecs2d &V) const;
    bool is_inside(const mtao::ColVecs2d &V, const mtao::Vec2d &p) const;
};
struct VEMTopology2 {
    mtao::ColVecs2i E;
    std::vector<std::map<int, bool>> face_boundary_map;

    // 3d overwrites this function
    int cell_count() const { return face_boundary_map.size(); }
    int edge_count() const { return E.cols(); }
};

struct VEMMesh2 : public VEMTopology2, std::enable_shared_from_this<VEMMesh2> {
    // vertex positions
    mtao::ColVecs2d V;
    // centers
    mtao::ColVecs2d C;

    VEMMesh2() = default;
    VEMMesh2(const VEMMesh2 &) = default;
    VEMMesh2(VEMMesh2 &&) = default;
    VEMMesh2(VEMTopology2 &&top) : VEMTopology2(top) {}
    VEMMesh2 &operator=(const VEMMesh2 &) = default;
    VEMMesh2 &operator=(VEMMesh2 &&) = default;
    VEMMesh2(VEMTopology2 &&T, mtao::ColVecs2d V);
    std::shared_ptr<VEMMesh2> handle() { return shared_from_this(); }
    std::shared_ptr<VEMMesh2 const> handle() const {
        return shared_from_this();
    }

    std::vector<std::set<int>> neighboring_cells;

    int vertex_count() const { return V.cols(); }
    virtual std::vector<PolygonBoundaryIndices> face_loops() const;
    virtual std::vector<mtao::ColVecs3i> triangulated_faces() const;
    virtual int get_cell(const mtao::Vec2d &p, int last_known = -1) const = 0;
    virtual bool in_cell(const mtao::Vec2d &p, int cell_index) const = 0;
    virtual std::set<int> boundary_edge_indices() const;
    virtual std::vector<std::set<int>> cell_regions() const;
    virtual Eigen::AlignedBox<double, 2> bounding_box() const;
    // some grid spacing concept, pick mean edge length by default
    virtual double dx() const;
    virtual double diameter(size_t cell_index) const;
    virtual double boundary_diameter(size_t cell_index) const;
    virtual mtao::Vec2d boundary_center(size_t cell_index) const;
    bool debug_diameter = false;
};
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
