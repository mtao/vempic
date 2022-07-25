#include <vem/from_grid.hpp>

namespace vem {

double GridVEMMesh2::dx() const {
    auto dx = GridType::dx();
    return *std::min_element(dx.begin(), dx.end());
}
double GridVEMMesh2::diameter(size_t cell_index) const {
    auto dx = GridType::dx();
    auto m = mtao::eigen::stl2eigen(dx);
    return m.norm();
}
GridVEMMesh2::GridVEMMesh2(const mtao::geometry::grid::StaggeredGrid2d &grid)
    : mtao::geometry::grid::StaggeredGrid2d(grid) {
    this->V = GridType::vertices();
    this->C = GridType::cell_vertices();
    this->E.resize(2, GridType::form_size<1>());

    auto B1 = GridType::boundary<1>();

    for (int o = 0; o < B1.outerSize(); ++o) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(B1, o); it; ++it) {
            auto e = this->E.col(it.col());
            int &val = e((it.value() > 0) ? 1 : 0);
            val = it.row();
        }
    }
    auto B2 = GridType::boundary<2>();
    this->face_boundary_map.resize(GridType::cell_size());

    neighboring_cells.resize(cell_count());

    GridType::cell_grid().loop_parallel([&](const std::array<int, 2> &a) {
        int cell_index = GridType::cell_index(a);
        auto &pool = neighboring_cells.at(cell_index);
        std::array<int, 2> min = a;
        std::array<int, 2> max = a;
        for (auto &&v : min) {
            --v;
        }
        for (auto &&v : max) {
            v += 2;
        }
        mtao::geometry::grid::utils::multi_loop(
            min, max, [&](const std::array<int, 2> &c) {
                if (GridType::cell_grid().valid_index(c)) {
                    int cell_index = GridType::cell_index(c);
                    pool.emplace(cell_index);
                }
            });
    });

    for (int o = 0; o < B2.outerSize(); ++o) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(B2, o); it; ++it) {
            bool value = it.value() < 0;
            int edge_dim = GridType::form_type<1>(it.row());
            if (edge_dim == 0) {
                value = !value;
            }

            this->face_boundary_map[it.col()][it.row()] = value;
        }
    }
}

std::set<int> GridVEMMesh2::boundary_edge_indices() const {
    std::set<int> boundary;
    auto u_grid = GridType::grid<1, 0>();
    for (int j = 0; j < u_grid.shape()[0]; ++j) {
        boundary.emplace(
            GridType::staggered_index<1, 0>(std::array<int, 2>{{j, 0}}));
        boundary.emplace(GridType::staggered_index<1, 0>(
            std::array<int, 2>{{j, u_grid.shape()[1] - 1}}));
    }
    auto v_grid = GridType::grid<1, 1>();
    for (int j = 0; j < v_grid.shape()[1]; ++j) {
        boundary.emplace(
            GridType::staggered_index<1, 1>(std::array<int, 2>{{0, j}}));
        boundary.emplace(GridType::staggered_index<1, 1>(
            std::array<int, 2>{{v_grid.shape()[0] - 1, j}}));
    }
    return boundary;
}
bool GridVEMMesh2::in_cell(const mtao::Vec2d &p, int cell_index) const {
    return get_cell(p) == cell_index;
}
int GridVEMMesh2::get_cell(const mtao::Vec2d &p, int last_known) const {
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

GridVEMMesh2 from_grid(const mtao::geometry::grid::StaggeredGrid2d &g) {
    return GridVEMMesh2(g);
}

// generates VEMMeshes. by default the centers chosen are centroidal
GridVEMMesh2 from_grid(const Eigen::AlignedBox<double, 2> &bb, int nx, int ny) {
    mtao::geometry::grid::StaggeredGrid2d g =
        mtao::geometry::grid::StaggeredGrid2d::from_bbox(
            bb, std::array<int, 2>{{nx, ny}});
    return from_grid(g);
}

}  // namespace vem
