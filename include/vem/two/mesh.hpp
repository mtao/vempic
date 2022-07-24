#include <map>
#include <optional>
#include <set>
#include <vector>
#include "../polygon_boundary_indices.hpp"

namespace vem {
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
}
