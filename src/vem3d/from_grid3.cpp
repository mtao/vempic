#include <mtao/geometry/grid/triangulation.hpp>
#include <vem/from_grid3.hpp>

namespace vem {

double GridVEMMesh3::dx() const {
    auto dx = GridType::dx();
    return *std::min_element(dx.begin(), dx.end());
}
std::string GridVEMMesh3::type_string() const { return "grid"; }
double GridVEMMesh3::diameter(size_t cell_index) const {
    auto dx = GridType::dx();
    auto m = mtao::eigen::stl2eigen(dx);
    return m.norm();
}
int GridVEMMesh3::face_count() const { return GridType::form_size<2>(); }
GridVEMMesh3::GridVEMMesh3(const mtao::geometry::grid::StaggeredGrid3d &grid)
    : mtao::geometry::grid::StaggeredGrid3d(grid) {
    this->V = GridType::vertices();
    this->C = GridType::cell_vertices();
    auto bb = grid.bbox();
    std::cout << "Bounding box: " << bb.min().transpose() << " => "
              << bb.max().transpose() << std::endl;
    this->FC.resize(3, GridType::form_size<2>());
    this->face_frames.resize(GridType::form_size<2>());

    for (auto &&[face_idx, frame] : mtao::iterator::enumerate(face_frames)) {
        int index = GridType::form_type<2>(face_idx);
        frame.col(0).setUnit((index + 1) % 3);
        frame.col(1).setUnit((index + 2) % 3);
        if ((frame.col(0).cross(frame.col(1)) - normal(face_idx)).norm() >
            1e-5) {
            spdlog::error(
                "Grid faces of axis {} are not oriented with their planes "
                "properly",
                index);
        }
    }

    for (int o = 0; o < GridType::form_size<2>(); ++o) {
        int dim = GridType::form_type<2>(o);
        FC.col(o) = GridType::staggered_vertex<2>(o, dim);
    }

    auto B3 = GridType::boundary<3>();
    this->cell_boundary_map.resize(GridType::cell_size());

    for (int o = 0; o < B3.outerSize(); ++o) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(B3, o); it; ++it) {
            bool value = it.value() < 0;
            int edge_dim = GridType::form_type<2>(it.row());

            this->cell_boundary_map[it.col()][it.row()] = value;
        }
    }
    neighboring_cells.resize(cell_count());

    GridType::cell_grid().loop_parallel([&](const std::array<int, 3> &a) {
        int cell_index = GridType::cell_index(a);
        auto &pool = neighboring_cells.at(cell_index);
        std::array<int, 3> min = a;
        std::array<int, 3> max = a;
        for (auto &&v : min) {
            --v;
        }
        for (auto &&v : max) {
            v += 3;
        }
        mtao::geometry::grid::utils::multi_loop(
            min, max, [&](const std::array<int, 3> &c) {
                if (GridType::cell_grid().valid_index(c)) {
                    int cell_index = GridType::cell_index(c);
                    pool.emplace(cell_index);
                }
            });
    });

    triangulated_faces.resize(face_count());
    for (auto &&[idx, tf] : mtao::iterator::enumerate(triangulated_faces)) {
        tf = triangulated_face(idx);
    }
}
bool GridVEMMesh3::collision_free(size_t cell_index) const {
    auto c = GridType::cell_grid().unindex(cell_index);

    auto &&s = GridType::cell_grid().shape();
    for (auto &&[c, s] : mtao::iterator::zip(c, s)) {
        if (c == 0 || c == s - 1) {
            return true;
        }
    }
    return false;
}

mtao::Vec3d GridVEMMesh3::normal(int face_index) const {
    return mtao::Vec3d::Unit(GridType::form_type<2>(face_index));
}
mtao::ColVecs3i GridVEMMesh3::triangulated_face(size_t face_index) const {
    mtao::geometry::grid::GridTriangulator<GridType> gt(*this);
    auto [a, b, c, d] = gt.face_loop(face_index);
    mtao::ColVecs3i F(3, 2);
    F.col(0) << a, b, c;
    F.col(1) << a, c, d;
    return F;
}

PolygonBoundaryIndices GridVEMMesh3::face_loops(size_t face_index) const {
    mtao::geometry::grid::GridTriangulator<GridType> gt(*this);
    auto arr = gt.face_loop(face_index);
    int index = GridType::form_type<2>(face_index);

    PolygonBoundaryIndices ret;
    std::copy(arr.begin(), arr.end(), std::back_inserter(ret));
    return ret;
}

double GridVEMMesh3::face_diameter(size_t face_index) const {
    int dim = GridType::form_type<2>(face_index);
    auto &&dx = GridType::dx();

    double a = dx((face_index + 1) % 3);
    double b = dx((face_index + 2) % 3);
    return std::sqrt(a * a + b * b);
}

bool GridVEMMesh3::in_cell(const mtao::Vec3d &p, int cell_index) const {
    return get_cell(p) == cell_index;
}
int GridVEMMesh3::get_cell(const mtao::Vec3d &p, int last_known) const {
    // auto c = std::get<0>(GridType::vertex_grid().coord(p));
    auto [c, q] = GridType::vertex_grid().coord(p);
    // spdlog::info("{} {} => Grid cell: {} {}", p.x(), p.y(), fmt::join(c,","),
    // fmt::join(q,","));
    if (!GridType::cell_grid().valid_index(c) ||
        (mtao::eigen::stl2eigen(q).array() < 0).any()) {
        return -1;
    } else {
        return GridType::cell_grid().index(c);
    }
}

GridVEMMesh3 from_grid(const mtao::geometry::grid::StaggeredGrid3d &g) {
    return GridVEMMesh3(g);
}

// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh3 from_grid(const Eigen::AlignedBox<double, 3> &bb, int nx, int ny,
                       int nz) {
    mtao::geometry::grid::StaggeredGrid3d g =
        mtao::geometry::grid::StaggeredGrid3d::from_bbox(
            bb, std::array<int, 3>{{nx, ny, nz}});
    std::cout << "Input bbox: " << bb.min().transpose() << ","
              << bb.max().transpose() << " with shape " << nx << "," << ny
              << "," << nz << std::endl;
    return from_grid(g);
}

std::optional<int> GridVEMMesh3::cell_category(size_t cell_index) const {
    return 0;
}

}  // namespace vem
