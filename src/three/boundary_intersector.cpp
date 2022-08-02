#include "vem/three/boundary_intersector.hpp"

#include <igl/AABB.h>
#include <igl/embree/EmbreeIntersector.h>
#include <tbb/parallel_for.h>

#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/mesh/write_obj.hpp>

#include "vem/three/boundary_facets.hpp"
namespace vem::three {
void BoundaryIntersectionDetector3::make_boundaries() {
    _boundary_cell_map.clear();
    _cell_boundary_map.clear();
    // for any boundary face return the cell attached
    _boundary_cell_map = boundary_face_map(_mesh, _active_cells);

    // the inverse map of the above (from a cell acceess which faces are on the
    // boundary
    for (auto&& [a, b] : _boundary_cell_map) {
        _cell_boundary_map[b].emplace(a);
    }
    spdlog::info(
        "{} cells contact a boundary. there are {} boundaries total of {} "
        "active / {} cells",
        _cell_boundary_map.size(), _boundary_cell_map.size(),
        _active_cells.size(), _mesh.cell_count());

    // get the list of cells connected ot hte boundary
    std::set<int> active_cells;
    for (auto&& [b, c] : _boundary_cell_map) {
        active_cells.emplace(c);
    }

    auto [V, F, cbm] = _mesh.collision_mesh(_active_cells);
    _collision_boundary_map = std::move(cbm);
    mtao::geometry::mesh::write_objD(V, F, "/tmp/vem_collision_mesh.obj");

    _intersector_point_matrix = V.transpose().cast<float>();
    _intersector_face_matrix = F.transpose();

    _intersector_face_plane_matrix.resize(_intersector_face_matrix.rows(), 4);
    tbb::parallel_for(
        int(0), int(_intersector_face_plane_matrix.rows()),
        [&](int face_index) {
            auto f = _intersector_face_matrix.row(face_index);
            auto i = _intersector_point_matrix.row(f(0)).transpose();
            auto j = _intersector_point_matrix.row(f(1)).transpose();
            auto k = _intersector_point_matrix.row(f(2)).transpose();
            auto R = _intersector_face_plane_matrix.row(face_index);
            R.head<3>() =
                (j - i).cross(k - i).cast<double>().normalized().transpose();
            R(3) = -R.head<3>().dot(i.cast<double>());
        });
    if (!_intersector) {
        _intersector = std::make_unique<igl::embree::EmbreeIntersector>();
    }

    if (!_aabb) {
        _aabb = std::make_unique<igl::AABB<mtao::RowVecs3f, 3>>();
        ;
    }
    _intersector->init(_intersector_point_matrix, _intersector_face_matrix,
                       true);
    _aabb->init(_intersector_point_matrix, _intersector_face_matrix);
    // TODO: make raycasting use the intersctor and distance computations use
    // the AABB
}
BoundaryIntersectionDetector3::BoundaryIntersectionDetector3(
    const VEMMesh3& mesh, const std::set<int>& active_cells)
    : _mesh(mesh), _active_cells(active_cells) {
    make_boundaries();
}

BoundaryIntersectionDetector3::~BoundaryIntersectionDetector3() {}

int BoundaryIntersectionDetector3::raw_get_cell(const mtao::Vec3d& p,
                                                int last_known) const {
    return _mesh.get_cell(p, last_known);
}
int BoundaryIntersectionDetector3::get_cell(const mtao::Vec3d& p,
                                            int last_known) const {
    int cell = raw_get_cell(p, last_known);
    if (!is_valid_cell(cell)) {
        return -1;
    } else {
        return cell;
    }
}
int BoundaryIntersectionDetector3::get_projected_cell(const mtao::Vec3d& p,
                                                      int last_known) const {
    int cell = get_cell(p, last_known);
    if (cell < 0) {
        if (_boundary_cell_map.empty()) {
            return 0;
        } else {
            int face_index = get_nearest_face(p, last_known);
            if (_cell_boundary_map.contains(face_index)) {
                return _boundary_cell_map.at(face_index);
            }
        }
    }
    return cell;
}
std::tuple<mtao::Vec3d, int>
BoundaryIntersectionDetector3::closest_boundary_point_with_face(
    const mtao::Vec3d& p) const {
    int cidx;
    mtao::RowVector<float, 3> point;
    _aabb->squared_distance(_intersector_point_matrix, _intersector_face_matrix,
                            p.transpose().cast<float>(), cidx, point);
    return {point.cast<double>().transpose(), cidx};
}
mtao::Vec3d BoundaryIntersectionDetector3::closest_boundary_point(
    const mtao::Vec3d& p) const {
    return std::get<0>(closest_boundary_point_with_face(p));
}

bool BoundaryIntersectionDetector3::is_valid_cell(size_t cell) const {
    if (_active_cells.empty()) {
        return cell >= 0 && cell < _mesh.cell_count();
    } else {
        return _active_cells.contains(cell);
    }
}

std::optional<mtao::Vec4d> BoundaryIntersectionDetector3::raycast(
    const mtao::Vec3d& p, const mtao::Vec3d& q) const {
    igl::Hit hit;

    if (_intersector->intersectRay(p.cast<float>(), (q - p).cast<float>(), hit,
                                   0, 1)) {
        return line_equation(hit.id);
    } else {
        return {};
    }
}
std::tuple<mtao::Vec3d, int>
BoundaryIntersectionDetector3::get_projected_position_with_face(
    const mtao::Vec3d& p, int last_known) const {
    if (!is_active_cell(get_cell(p, last_known))) {
        return closest_boundary_point_with_face(p);
    }
    return {p, -1};
}

mtao::Vec3d BoundaryIntersectionDetector3::get_projected_position(
    const mtao::Vec3d& p, int last_known) const {
    return std::get<0>(get_projected_position_with_face(p, last_known));
}
bool BoundaryIntersectionDetector3::is_active_cell(int index) const {
    if (_active_cells.empty()) {
        return index >= 0 && index < _mesh.cell_count();
    } else {
        return _active_cells.contains(index);
    }
}
int BoundaryIntersectionDetector3::get_nearest_face(const mtao::Vec3d& p,
                                                    int last_known) const {
    mtao::RowVector<float, 3> point;
    int cidx;

    _aabb->squared_distance(_intersector_point_matrix, _intersector_face_matrix,
                            p.transpose().cast<float>(), cidx, point);
    if (auto it = _collision_boundary_map.find(cidx);
        it != _collision_boundary_map.end()) {
        const auto& cells = it->second;
        if (cells.size() == 1) {
            return *cells.begin();
        } else {
            spdlog::info(
                "Currently we only supprt many to one collision geometries, as "
                "i the case when the collision faces are triangulated "
                "polyhedra");
        }
    }
    return -1;
}
std::set<int> BoundaryIntersectionDetector3::boundary_edge_indices() const {
    std::set<int> ret;
    for (auto&& [a, b] : _boundary_cell_map) {
        ret.emplace(a);
    }
    return ret;
}

mtao::Vec4d BoundaryIntersectionDetector3::line_equation(int face_index) const {
    return _intersector_face_plane_matrix.row(face_index).transpose();
}
}  // namespace vem::utils
