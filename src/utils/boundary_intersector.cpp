#include "vem/utils/boundary_intersector.hpp"

#include "vem/utils/boundary_facets.hpp"
namespace vem::utils {
void BoundaryIntersectionDetector::make_boundaries() {
    _boundary_cell_map.clear();
    _cell_boundary_map.clear();
    _boundary_cell_map = boundary_edge_map(_mesh, _active_cells);
    for (auto&& [a, b] : _boundary_cell_map) {
        _cell_boundary_map[b].emplace(a);
    }
    spdlog::info(
        "{} cells contact a boundary. there are {} boundaries total of {} "
        "active / {} cells",
        _cell_boundary_map.size(), _boundary_cell_map.size(),
        _active_cells.size(), _mesh.cell_count());
}
BoundaryIntersectionDetector::BoundaryIntersectionDetector(
    const VEMMesh2& mesh, const std::set<int>& active_cells)
    : _mesh(mesh), _active_cells(active_cells) {
    make_boundaries();
}

int BoundaryIntersectionDetector::raw_get_cell(const mtao::Vec2d& p,
                                               int last_known) const {
    return _mesh.get_cell(p, last_known);
}
int BoundaryIntersectionDetector::get_cell(const mtao::Vec2d& p,
                                           int last_known) const {
    int cell = raw_get_cell(p, last_known);
    if (!is_valid_cell(cell)) {
        return -1;
    } else {
        return cell;
    }
}
int BoundaryIntersectionDetector::get_projected_cell(const mtao::Vec2d& p,
                                                     int last_known) const {
    int cell = get_cell(p, last_known);
    if (cell < 0) {
        if (_boundary_cell_map.empty()) {
            return 0;
        } else {
            auto [eidx, _] = get_closest_boundary_edge(p);
            if (_cell_boundary_map.contains(eidx)) {
                return _boundary_cell_map.at(eidx);
            }
        }
    }
    return cell;
}
mtao::Vec2d BoundaryIntersectionDetector::closest_boundary_point(
    const mtao::Vec2d& p) const {
    auto [eidx, t] = get_closest_boundary_edge(p);
    auto e = _mesh.E.col(eidx);
    auto va = _mesh.V.col(e(0));
    auto vb = _mesh.V.col(e(1));
    return (1 - t) * va + t * vb;
}
std::tuple<int, double> BoundaryIntersectionDetector::get_closest_boundary_edge(
    const mtao::Vec2d& p) const {
    std::tuple<int, double> ret{-1, -1};
    auto& [closest, closest_t] = ret;
    double nearest_distance = std::numeric_limits<double>::max();
    for (auto&& [eidx, c] : _boundary_cell_map) {
        mtao::Vec3d equation = edge_line_equation(eidx);

        double t = equation.dot(p.homogeneous());
        t = std::clamp<double>(0, 1, t);
        auto e = _mesh.E.col(eidx);
        auto va = _mesh.V.col(e(0));
        auto vb = _mesh.V.col(e(1));
        double dist = ((1 - t) * va + t * vb - p).norm();
        if (nearest_distance > dist) {
            nearest_distance = dist;
            closest = eidx;
            closest_t = t;
        }
    }

    return ret;
}
bool BoundaryIntersectionDetector::is_valid_cell(size_t cell) const {
    if (_active_cells.empty()) {
        return cell >= 0 && cell < _mesh.cell_count();
    } else {
        return _active_cells.contains(cell);
    }
}

std::optional<std::tuple<int, double>> BoundaryIntersectionDetector::raycast(
    const mtao::Vec2d& p, const mtao::Vec2d& q, double radius,
    int last_known) const {
    int pc = get_cell(p, last_known);
    int qc = get_cell(q, last_known);
    return raycast(p, q, pc, qc, radius);
}
std::optional<std::tuple<int, double>> BoundaryIntersectionDetector::raycast(
    const mtao::Vec2d& p, const mtao::Vec2d& q, int pc, int qc,
    double radius) const {
    // spdlog::info("Raycasting from cell {} to cell {}", pc, qc);
    if (pc >= 0 && qc >= 0) {
        // spdlog::info("Was inside on both cells");
        // return {};
    } else if (qc < 0 && pc < 0) {
        return {};
    }

    mtao::Vec3d my_line;
    mtao::Vec3d my_normal;
    bool inverted = qc < 0;
    if (inverted) {
        my_line = line_equation(q, p);
        my_normal = normal_equation(q, p);
    } else {
        my_line = line_equation(p, q);
        my_normal = normal_equation(p, q);
    }
    // std::cout << "Line equation: " << my_line.transpose() << std::endl;
    int best_edge = -1;
    double best_isect = 2;

    auto try_isect_eidx = [&](int eidx) {
        // spdlog::info("Trying to isect with edge {}", eidx);
        auto isect_opt =
            // line_intersection_with_capsule(p,my_line, my_normal, eidx,
            // radius);
            line_intersection_with_edge(my_line, my_normal, eidx);
        if (!isect_opt) {
            return;
        }
        // std::cout << "Got an isect at t=" << isect << std::endl;
        const double& isect = *isect_opt;
        if (std::isfinite(isect)) {
            if (isect >= 0) {
                if (isect < best_isect) {
                    // spdlog::info("Got a better isect: {} on edge {}", isect,
                    // eidx);
                    best_isect = isect;
                    best_edge = eidx;
                }
            }
        }
    };
    // try to intersect with edges in this cell
    if (auto it = _cell_boundary_map.find(pc); it != _cell_boundary_map.end()) {
        // spdlog::info("Trying my own edges, tehre should be {} of them",
        // it->second.size());
        for (auto&& eidx : it->second) {
            try_isect_eidx(eidx);
        }
    } else {
        // spdlog::info("I dont have my own edges");
    }
    // spdlog::info("Trying other edges");
    // try intersecting with other edges
    if (best_edge < 0 || best_isect > 1) {
        for (auto&& [eidx, cidx] : _boundary_cell_map) {
            if (cidx == qc || cidx == pc) {
                // spdlog::info(
                //    "Checking edge {} because its cell {} is close to {} or
                //    {}", eidx, cidx, qc, pc);
                continue;
            } else {
                try_isect_eidx(eidx);
            }
        }
    }
    if (best_edge >= 0 && best_isect <= 1) {
        return std::make_tuple(best_edge,
                               inverted ? (1 - best_isect) : best_isect);
    } else {
        return {};
    }
}

mtao::Vec3d BoundaryIntersectionDetector::edge_line_equation(
    int edge_index) const {
    auto e = _mesh.E.col(edge_index);
    auto va = _mesh.V.col(e(0));
    auto vb = _mesh.V.col(e(1));
    return line_equation(va, vb);
}
mtao::Vec3d BoundaryIntersectionDetector::edge_normal_equation(
    int edge_index) const {
    auto e = _mesh.E.col(edge_index);
    auto va = _mesh.V.col(e(0));
    auto vb = _mesh.V.col(e(1));
    return normal_equation(va, vb);
}

std::optional<double> BoundaryIntersectionDetector::line_intersection_with_edge(
    const mtao::Vec3d& line, const mtao::Vec3d& line_no, int edge_index) const {
    auto eline = edge_line_equation(edge_index);
    auto eline_no = edge_normal_equation(edge_index);

    // auto line = line_equation(p, q);
    // auto line_no = normal_equation(p, q);

    // std::cout << edge_index << " ) " << e.transpose() << "edge line: " <<
    // eline.transpose() << std::endl; std::cout << va.transpose() << " ==> " <<
    // vb.transpose() << std::endl;
    mtao::Vec2d p = line_intersection(line_no, eline_no);
    if (double v = eline.dot(p.homogeneous()); std::abs(v - .5) > .5 + 1e-5) {
        return {};
    }
    if (double v = line.dot(p.homogeneous()); std::abs(v - .5) > .5 + 1e-5) {
        return {};
    }
    // std::cout << "Line equation distances: "
    //          << line_eq.dot(isect_pt.homogeneous()) << " || "
    //          << edge_line_eq.dot(isect_pt.homogeneous()) << std::endl;
    // std::cout << "Line equation normal compat : "
    //          << normal_eq.dot(isect_pt.homogeneous()) << " || "
    //          << edge_normal_eq.dot(isect_pt.homogeneous()) << std::endl;
    // std::cout << "Line isect: " << p.transpose() << std::endl;
    return p.homogeneous().dot(line);
}

std::optional<double>
BoundaryIntersectionDetector::line_intersection_with_capsule(
    const mtao::Vec3d& line, const mtao::Vec3d& normal, int edge_index,
    double radius) const {
    auto p = line.cross(normal);
    return line_intersection_with_capsule(p.head<2>() / p(2), line, normal,
                                          edge_index, radius);
}

std::optional<double>
BoundaryIntersectionDetector::line_intersection_with_capsule(
    const mtao::Vec2d& start, const mtao::Vec3d& line,
    const mtao::Vec3d& line_no, int edge_index, double radius) const {
    auto eline = edge_line_equation(edge_index);
    auto eline_no = edge_normal_equation(edge_index);

    const bool below_edge = start.homogeneous().dot(eline_no) < 0;
    if (below_edge) {
        eline_no(2) -= radius / eline_no.head<2>().norm();
    } else {
        eline_no(2) += radius / eline_no.head<2>().norm();
    }
    // shifts so that we capture
    //
    // (_______)
    //
    //     .
    //
    //
    // const bool now_below_edge = start.homogeneous().dot(eline_no) < 0;

    mtao::Vec2d p = line_intersection(line_no, eline_no);
    // if we're outside the range, check for
    // (       )
    if (double v = eline.dot(p.homogeneous()); std::abs(v - .5) > .5) {
        auto e = _mesh.E.col(edge_index);
        if (auto opt = line_intersection_with_ball(line, line_no, e(0), radius);
            opt) {
            return opt;
        } else {
            return line_intersection_with_ball(line, line_no, e(1), radius);
        }
    }

    // ok so we would hit the offset line, make sure that we hit the point
    // _______
    if (double v = line.dot(p.homogeneous()); std::abs(v - .5) > .5) {
        return {};
    }

    return p.homogeneous().dot(line);
}

std::optional<double> BoundaryIntersectionDetector::line_intersection_with_ball(
    const mtao::Vec3d& line, const mtao::Vec3d& normal, int vertex_index,
    double radius) const {
    auto p = _mesh.V.col(vertex_index);
    double t = line.dot(p.homogeneous());

    double norm = line.head<2>().norm();

    mtao::Vec2d D = line.head<2>().normalized();

    // normalized vector needs to be double unnormalized
    t = t / norm;

    auto lp = line.cross(normal);
    mtao::Vec2d start = lp.head<2>() / lp(2);

    mtao::Vec2d np = start + t * D;
    if (double dist = (np - p).norm(); dist < radius) {
        double sub = std::sqrt(radius * radius - dist * dist);
        t = t - sub;
        if (t >= 0 && t < 1) {
            return t;
        }
    }
    return {};
}

template <typename A, typename B>
mtao::Vec2d BoundaryIntersectionDetector::line_intersection(
    const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
    mtao::Vec3d c = a.cross(b);

    return c.head<2>() / c(2);
}

std::set<int> BoundaryIntersectionDetector::boundary_edge_indices() const {
    std::set<int> ret;
    for (auto&& [a, b] : _boundary_cell_map) {
        ret.emplace(a);
    }
    return ret;
}
}  // namespace vem::utils
