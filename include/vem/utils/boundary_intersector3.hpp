#include <map>

#include "vem/mesh.hpp"

namespace igl {
template <typename DerivedV, int DIM>
class AABB;
namespace embree {
class EmbreeIntersector;
}
}  // namespace igl

namespace vem::utils {

struct BoundaryIntersectionDetector3 {
   public:
    BoundaryIntersectionDetector3(const VEMMesh3& mesh,
                                  const std::set<int>& active_cells);
    ~BoundaryIntersectionDetector3();

    bool is_valid_cell(size_t cell_index) const;
    int raw_get_cell(const mtao::Vec3d& p, int last_known = -1) const;
    int get_cell(const mtao::Vec3d& p, int last_known = -1) const;
    int get_projected_cell(const mtao::Vec3d& p, int last_known = -1) const;
    mtao::Vec3d get_projected_position(const mtao::Vec3d& p,
                                       int last_known = -1) const;

    std::tuple<mtao::Vec3d, int> get_projected_position_with_face(
        const mtao::Vec3d& p, int last_known = -1) const;
    int get_nearest_face(const mtao::Vec3d& p, int last_known = -1) const;

    // returns a plane equation if the points isect
    //
    std::optional<mtao::Vec4d> raycast(const mtao::Vec3d& p,
                                       const mtao::Vec3d& q) const;

    void make_boundaries();

    mtao::Vec3d closest_boundary_point(const mtao::Vec3d& p) const;

    // returns the closest face as well as the point itself
    std::tuple<mtao::Vec3d, int> closest_boundary_point_with_face(
        const mtao::Vec3d& p) const;

    auto face_normal(int index) const {
        return _intersector_face_plane_matrix.block<1, 3>(index, 0);
    }

    // amount should lie in [0,1] where 0 implies the vector will end up
    // parallel 1 implies an elastic reflection
    template <typename Derived>
    void reflect_vector(int face_index, Derived& vec, double amount) const {
        auto fn = face_normal(face_index);
        auto proj = vec.dot(fn.transpose().cast<double>());
        vec -= fn.transpose().cast<double>() * (1 + amount);
    }

    template <typename Derived>
    void parallelize_vector(int face_index, Derived& vec) const {
        return reflect_vector(face_index, vec, 0.0);
    }

    std::set<int> boundary_edge_indices() const;
    mtao::Vec4d line_equation(int face_index) const;
    bool is_active_cell(int index) const;

   private:
    const VEMMesh3& _mesh;
    const std::set<int>& _active_cells;
    std::map<size_t, size_t> _boundary_cell_map;
    std::map<size_t, std::set<size_t>> _cell_boundary_map;
    std::unique_ptr<igl::embree::EmbreeIntersector> _intersector;
    std::unique_ptr<igl::AABB<mtao::RowVecs3f, 3>> _aabb;
    mtao::RowVecs3f _intersector_point_matrix;
    // the first 3 columns form the surface normals of the faces (orthogonal)
    mtao::RowVectors<double, 4> _intersector_face_plane_matrix;
    mtao::RowVecs3i _intersector_face_matrix;
    // makes collision triangles to boundary faces
    std::map<size_t, std::set<size_t>> _collision_boundary_map;
};

}  // namespace vem::utils
