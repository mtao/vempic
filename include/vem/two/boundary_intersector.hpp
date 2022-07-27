#include <map>

#include "mesh.hpp"

namespace vem::two {

struct BoundaryIntersectionDetector {
   public:
    BoundaryIntersectionDetector(const VEMMesh2& mesh,
                                 const std::set<int>& active_cells);

    bool is_valid_cell(size_t cell_index) const;
    int raw_get_cell(const mtao::Vec2d& p, int last_known = -1) const;
    int get_cell(const mtao::Vec2d& p, int last_known = -1) const;
    int get_projected_cell(const mtao::Vec2d& p, int last_known = -1) const;

    // casts a ray between p and q. can return a bad cell index if both cels
    // were outside
    std::optional<std::tuple<int, double>> raycast(const mtao::Vec2d& p,
                                                   const mtao::Vec2d& q,
                                                   double radius,
                                                   int last_known = -1) const;

    std::optional<std::tuple<int, double>> raycast(const mtao::Vec2d& p,
                                                   const mtao::Vec2d& q,
                                                   int pcell, int qcell, double radius) const;

    std::optional<double> line_intersection_with_edge(const mtao::Vec3d& line,
                                                      const mtao::Vec3d& normal,
                                                      int edge_index) const;

    std::optional<double> line_intersection_with_capsule(
            const mtao::Vec2d& p,
            const mtao::Vec3d& line,
                                                      const mtao::Vec3d& normal,
                                                      int edge_index, double radius) const;
    std::optional<double> line_intersection_with_capsule(
            const mtao::Vec3d& line,
                                                      const mtao::Vec3d& normal,
                                                      int edge_index, double radius) const;

    std::optional<double> line_intersection_with_ball(const mtao::Vec3d& line, const mtao::Vec3d& normal,
                                                      int vertex_index, double radius) const;
    // two line equations (vec3d-like)
    template <typename A, typename B>
    static mtao::Vec2d line_intersection(const Eigen::MatrixBase<A>& a,
                                         const Eigen::MatrixBase<B>& b);
    template <typename A, typename B>
    static mtao::Vec3d line_equation(const Eigen::MatrixBase<A>& a,
                                     const Eigen::MatrixBase<B>& b);

    template <typename A, typename B>
    static mtao::Vec3d unnormalized_line_equation(const Eigen::MatrixBase<A>& a,
                                     const Eigen::MatrixBase<B>& b);
    template <typename A, typename B>
    static mtao::Vec3d normal_equation(const Eigen::MatrixBase<A>& a,
                                       const Eigen::MatrixBase<B>& b);
    void make_boundaries();

    mtao::Vec3d edge_line_equation(int edge_index) const;
    mtao::Vec3d edge_normal_equation(int edge_index) const;

    std::tuple<int, double> get_closest_boundary_edge(
        const mtao::Vec2d& p) const;
    mtao::Vec2d closest_boundary_point(const mtao::Vec2d& p) const;

    std::set<int> boundary_edge_indices() const;

   private:
    const VEMMesh2& _mesh;
    const std::set<int>& _active_cells;
    // maps face -> cell for each face on teh boundary
    std::map<size_t, size_t> _boundary_cell_map;
    std::map<size_t, std::set<size_t>> _cell_boundary_map;
};

    template <typename A, typename B>
mtao::Vec3d BoundaryIntersectionDetector::normal_equation(
    const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
    mtao::Vec3d R;
    auto T = R.head<2>();
    auto ba = b - a;
    T.x() = -ba.y();
    T.y() = ba.x();

    T /= T.squaredNorm();
    R(2) = -T.dot(a);
    return R;
}
    template <typename A, typename B>
mtao::Vec3d BoundaryIntersectionDetector::line_equation(
    const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
    mtao::Vec3d R;
    auto T = R.head<2>();
    T = b - a;

    T /= T.squaredNorm();
    R(2) = -T.dot(a);
    return R;
}

    template <typename A, typename B>
mtao::Vec3d BoundaryIntersectionDetector::unnormalized_line_equation(
    const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
    mtao::Vec3d R;
    auto T = R.head<2>();
    T = b - a;

    T.normalize();
    R(2) = -T.dot(a);
    return R;
}
}  // namespace vem::utils
