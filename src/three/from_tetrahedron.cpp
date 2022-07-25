#include <mtao/geometry/barycentric.hpp>
#include <vem/from_tetrahedron3.hpp>

namespace vem {

double TetrahedronVEMMesh3::dx() const {
    return (V.colwise() - C.col(0)).colwise().norm().maxCoeff();
}
double TetrahedronVEMMesh3::diameter(size_t cell_index) const {
    return 2 * dx();
}
int TetrahedronVEMMesh3::face_count() const { return 4; }
TetrahedronVEMMesh3::TetrahedronVEMMesh3(
    const mtao::geometry::grid::StaggeredGrid3d &grid)
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

mtao::Vec3d TetrahedronVEMMesh3::normal(int face_index) const {
    return mtao::Vec3d::Unit(GridType::form_type<2>(face_index));
}
mtao::ColVecs3i TetrahedronVEMMesh3::triangulated_face(
    size_t face_index) const {
    mtao::geometry::grid::GridTriangulator<GridType> gt(*this);
    auto [a, b, c, d] = gt.face_loop(face_index);
    mtao::ColVecs3i F(3, 2);
    F.col(0) << a, b, c;
    F.col(1) << a, c, d;
    return F;
}

PolygonBoundaryIndices TetrahedronVEMMesh3::face_loops(
    size_t face_index) const {
    switch (face_index) {
        case 0:
            return {1, 2, 3};
    }
    mtao::geometry::grid::GridTriangulator<GridType> gt(*this);
    auto arr = gt.face_loop(face_index);
    int index = GridType::form_type<2>(face_index);

    PolygonBoundaryIndices ret;
    std::copy(arr.begin(), arr.end(), std::back_inserter(ret));
    return ret;
}

bool TetrahedronVEMMesh3::in_cell(const mtao::Vec3d &p, int cell_index) const {
    auto b = mtao::eigen::barycentric_simplicial<3>(V.leftCols<4>(), p);
    if (b.minCoeff() >= 1e-5 && b.maxCoeff() <= 1 + 1e-5) {
        return true;
    } else {
        return false;
    }
}
int TetrahedronVEMMesh3::get_cell(const mtao::Vec3d &p, int last_known) const {
    if (in_cell(p)) {
        return 0;
    } else {
        return -1;
    }
}

TetrahedronVEMMesh3 from_tetrahedron(const mtao::ColVecs3d &V) {
    return TetrahedronVEMMesh3(V);
}

// generates VEMMeshes. by default the centers chosen are centroidal
TetrahedronVEMMesh3 from_tetrahedron() {
    mtao::ColVecs3d V(3, 4);
    V.setZero();
    V.col(1)(0) = 1;
    V.col(1)(1) = 1;
    V.col(2)(2) = 1;
    return from_tetrahedron(V);
}

}  // namespace vem
