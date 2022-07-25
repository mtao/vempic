#include <igl/parula.h>
#include <mtao/geometry/mesh/boundary_facets.h>
#include <mtao/geometry/mesh/boundary_matrix.h>
#include <spdlog/spdlog.h>

#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <iostream>
#include <mtao/eigen/stl2eigen.hpp>
#include <mtao/geometry/mesh/read_obj.hpp>
#include <mtao/geometry/mesh/write_obj.hpp>
#include <mtao/geometry/mesh/write_ply.hpp>
#include <mtao/iterator/enumerate.hpp>

#include "dirichlet_triangle_laplacian.h"
#include "diwlevin_vem_mesh2.hpp"
#include "plcurve2.hpp"
#include "pointwise_vem_mesh2.hpp"

void mesh_boundary(const std::string& filename) {
    auto [V, F] = mtao::geometry::mesh::read_objD(filename);
    V.row(0).array() -= V.row(0).mean();
    V.row(1).array() -= V.row(1).mean();
    V.array() /= V.maxCoeff();

    auto E = mtao::geometry::mesh::boundary_facets(F);
    auto B = mtao::geometry::mesh::boundary_matrix<double>(F, E);
    mtao::VecXd bounds = B * mtao::VecXd::Ones(B.cols());
    PointwiseVEMMesh2 vem;
    //DIWLevinVEMMesh2 vem;
    //vem.integrated_edges = true;
    vem.order = 3;
    vem.cells.emplace_back();
    auto& cell = vem.cells.front();
    vem.centers.resize(2, 1);
    vem.centers.col(0) << 0., 0.;

    vem.vertices = V.topRows<2>();
    vem.edges = E;

    std::set<int> bvs;
    vem.initialize_interior_offsets(0);

    std::vector<std::array<int, 2>> edges_vec;
    for (size_t i = 0; i < bounds.size(); ++i) {
        if (bounds(i) != 0) {
            auto e = E.col(i);
            cell[edges_vec.size()] = bounds(i);
            edges_vec.emplace_back(std::array<int, 2>{{e(0), e(1)}});
            bvs.insert(e(0));
            bvs.insert(e(1));
        }
    }
    vem.edges = mtao::eigen::stl2eigen(edges_vec);
    std::map<size_t, double> dirichlet_vertices;
    std::cout << "Dirichlet constraint size: " << dirichlet_vertices.size()
              << std::endl;

    for (auto&& i : bvs) {
        auto v = V.col(i);
        double ang = std::atan2(v.y(), v.x());
        dirichlet_vertices[i] = v.z();
        double val = 1 - 5 * v.head<2>().norm();
        val += std::cos(3 * ang);
        // val = 1 - 5 * v.x() * v.x() + v.y() * v.y();
        // val = std::pow(1e-2 * v.head<2>().norm(),3.);
        // val = std::pow(1e-1 * v.y(),3.);

        dirichlet_vertices[i] = val;
    }
    {
        std::vector<int> I(bvs.begin(), bvs.end());
        size_t N = I.size();
        dirichlet_vertices.clear();
        dirichlet_vertices[I[0]] = .5;
        dirichlet_vertices[I[N / 2]] = -.5;
        // dirichlet_vertices[I[N / 4]] = .1;
        // dirichlet_vertices[I[3 * N / 4]] = -.1;
    }

//    {
//        Eigen::SparseMatrix<double> cell_to_world = vem.cell_to_world(0);
//        mtao::MatXd P = vem.poly_projection(0);
//        mtao::VecXd x;
//
//        x = vem.laplace_problem(dirichlet_vertices);
//        mtao::VecXd px = cell_to_world.transpose() * x;
//
//        mtao::VecXd y = P * px;
//        // std::cout << x << std::endl;
//        std::cout << "CoeffS: " << y.transpose() << std::endl;
//
//        for (auto&& [idx, val] : dirichlet_vertices) {
//            mtao::Vec2d p = vem.sample_position(idx);
//            std::cout << idx << ") " << val << " == "
//                      << " | " << x(idx) << " | "
//                      << vem.polynomial_eval(0, p, y) << std::endl;
//        }
//
//        for (int i = 0; i < V.cols(); ++i) {
//            auto v = V.col(i);
//            v.z() = vem.polynomial_eval(0, v.head<2>(), y)(0);
//        }
//        std::map<size_t, double> bvd;
//        for (auto&& v : bvs) {
//            V.col(v).z() = x(v);
//            bvd[v] = x(v);
//        }
//        auto zvals = dirichlet_laplacian(V.topRows<2>(), F, bvd);
//        V.row(2) = zvals.transpose();
//        Eigen::MatrixXd colors;
//        igl::parula(zvals, true, colors);
//        mtao::geometry::mesh::write_obj(V, F, "output.obj");
//        mtao::geometry::mesh::write_plyD(V, colors.transpose(), F,
//                                         "output.ply");
//        V.row(2).setZero();
//        for (auto&& v : bvs) {
//            V.col(v).z() = x(v);
//        }
//        mtao::geometry::mesh::write_obj(V, F, "output_boundary.obj");
//        // for(auto&& [i,v]: dirichlet_vertices) {
//        //    V.col(i).z() = v;
//        //}
//
//        // std::cout << "Desired rhs: " << (A * x ).transpose() << std::endl;
//    }
}

void test(int N) {
    // PointwiseVEMMesh2 vem;
    DIWLevinVEMMesh2 vem;
    vem.cells.emplace_back();
    auto& cell = vem.cells.front();
    vem.edges.resize(2, N);
    vem.vertices.resize(2, N);
    double dt = 2. * M_PI / (N);
    for (int j = 0; j < N; ++j) {
        cell.emplace(j, 0);
        vem.edges.col(j) << j, (j + 1) % N;
        double t = dt * j + M_PI / 4.;
        vem.vertices.col(j) << std::cos(t), std::sin(t);
        // vem.vertices.col(j) *= 1. + .1 * std::cos( 4 * t);
    }
    // vem.vertices.resize(2,3);
    // vem.vertices.col(0) << -0.5,-0.5;
    // vem.vertices.col(1) << .5,-0.5;
    // vem.vertices.col(2) << -0.5,.5;

    // vem.edges.resize(2,2);
    // vem.edges.col(0) << 0,1;
    // vem.edges.col(1) << 0,2;
    //
    // cell.clear();
    // cell[0] = false;
    // cell[1] = false;

    vem.vertices *= 1. / std::sqrt(2.);
    std::cerr << "Vertices: " << std::endl;
    std::cerr << vem.vertices << std::endl;

    vem.centers.resize(2, 1);
    vem.centers.col(0) << 0., 0.;

    vem.order = 2;
    vem.initialize_interior_offsets(0);
    /*
    for (auto&& [e, c] : vem.cells[0]) {
        auto es = vem.E(e);
        auto a = vem.V(es(0));
        auto b = vem.V(es(1));
        double len = (b - a).norm();
        std::cout << vem.V(vem.E(e)(0)).transpose() << " => ";
        std::cout << vem.V(vem.E(e)(1)).transpose() << std::endl;
        auto E = vem.per_boundary_per_monomial_product(0, e);
        auto rhs = vem.per_edge_per_monomial_linear_integral(0, e);
        std::cout << "Edge product" << std::endl;
        std::cout << E << std::endl << std::endl;
        std::cout << "Rhs\n" << rhs.transpose() << std::endl;
        std::cout << std::endl;
        for (size_t l = 0; l < vem.coefficient_size(); ++l) {
            mtao::VecXd f = mtao::VecXd::Unit(vem.coefficient_size(), l);
        for (size_t k = 0; k < vem.coefficient_size(); ++k) {
            mtao::VecXd e = mtao::VecXd::Unit(vem.coefficient_size(), k);
            double v =0;
            auto eval = [&](double t) -> double {
                mtao::Vec2d p = (1 - t) * a + t * b;
                double v = vem.polynomial_eval(0, p, e)(0);
                double v2 = vem.polynomial_eval(0, p, f)(0);
                return  v * v2;
            };
            size_t count = 100;
            for (double j = 0; j < 1.; j += 1. / count) {
                v += eval(j);
            }
            v /= count;
            v *= len;
            std::cout << v<<" ";
        } std::cout << std::endl;
        }
            return;
    }
    */

    // std::cout << " monomial products" << std::endl;
    // std::cout << vem.per_cell_per_monomial_products(0) << std::endl;

    // std::cout << "linear integrals" << std::endl;
    // std::cout << vem.per_cell_per_monomial_linear_integral(0) << std::endl;
    // std::cout << vem.boundary_weighted_poly_projection_sample2sample(0) <<
    // std::endl; auto M = vem.poly_coefficient_matrix(0); for(size_t i = 0; i <
    // vem.coefficient_size(); ++i) {
    //    std::cout << "Data: " << M.row(i)<< std::endl;
    //    std::cout << (vem.boundary_weighted_poly_projection_sample2sample(0) *
    //    M.row(i).transpose()).transpose() << std::endl;;

    //}
    auto V = vem.cell_sample_positions(0);

    mtao::VecXd fv(vem.num_samples());
    // fv.array() += 5 * V.row(0).array().pow(2.0).transpose();
    // fv.array() += 2 * (V.row(0).array() * V.row(1).array()).transpose();

    // auto P = vem.poly_projection_sample2poly(0);
    // auto PLM = vem.per_cell_laplacian(0);
    // std::cerr << (P * fv).transpose() << std::endl;
    // std::cerr << "Dirichlet energy: " << fv.transpose() * PLM * fv  <<
    // std::endl;

    int midpoint = N / 2;
    std::map<size_t, double> dirichlet_vertices;
    dirichlet_vertices[0] = 10;
    dirichlet_vertices[N / 2] = -10;
    // dirichlet_vertices[N / 4] = 10;
    // dirichlet_vertices[3 * N / 4] = -10;

    // fmt::print("Boundary laplace\n");
    mtao::VecXd x = vem.laplace_problem(dirichlet_vertices);
    std::cout << "Solution: " << x.transpose() << std::endl;
    ////mtao::VecXd y = P * x;
    ////std::cout << "CoeffS: " << y.transpose() << std::endl;
    //// std::cout << "energy: " << x.transpose() * PLM * x << std::endl;

    // for (auto&& [idx, val] : dirichlet_vertices) {
    //    mtao::Vec2d p = vem.sample_position(idx);
    //    std::cout << idx << ") " << val
    //        << " == " << vem.polynomial_eval(0, p, y) << std::endl;
    //}

    // std::cout << "Edge quadrature weights: " <<
    // vem.per_cell_edge_quadrature_weights(0).transpose() << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        mesh_boundary(argv[1]);
    } else {
        test(8);
    }

#ifdef UNUSED
    PointwiseVEMMesh2 vem;
    vem.cells.emplace_back();
    auto& cell = vem.cells.front();
    cell.emplace(0, 0);
    cell.emplace(1, 0);
    cell.emplace(2, 0);
    cell.emplace(3, 0);
    // cell.emplace(4, 0);
    // cell.emplace(5, 0);
    vem.edges.resize(2, 4);
    vem.edges.col(0) << 0, 1;
    vem.edges.col(1) << 1, 2;
    vem.edges.col(2) << 2, 3;
    vem.edges.col(3) << 3, 0;
    // vem.edges.col(3) << 3, 4;
    // vem.edges.col(4) << 4, 5;
    // vem.edges.col(5) << 5, 0;

    vem.vertices.resize(2, 4);
    vem.vertices.col(0) << 0., 0.;
    vem.vertices.col(1) << 0., 1.;
    vem.vertices.col(2) << 1., 1.;
    vem.vertices.col(3) << 1., 0.;
    // vem.vertices.col(4) << 1., .2;
    // vem.vertices.col(5) << 5., 1.;

    vem.centers.resize(2, 1);
    vem.centers.col(0) << 0., 0.;

    vem.order = 3;

    mtao::VecXd coeffs(vem.coefficient_size());
    coeffs.setZero();

    coeffs(0) = 1.;

    std::cerr
        << vem.polynomial_eval(0, mtao::Vec2d(0.0, 2.0), coeffs) << ": "
        << vem.polynomial_grad(0, mtao::Vec2d(0.0, 2.0), coeffs).transpose()
        << std::endl;
    std::cerr
        << vem.polynomial_eval(0, mtao::Vec2d(3.0, 0.0), coeffs) << ": "
        << vem.polynomial_grad(0, mtao::Vec2d(3.0, 0.0), coeffs).transpose()
        << std::endl;
    coeffs.setZero();
    coeffs(1 + 0) = 1.;
    coeffs(1 + 1) = 1.;
    std::cerr
        << vem.polynomial_eval(0, mtao::Vec2d(0.0, 2.0), coeffs) << ": "
        << vem.polynomial_grad(0, mtao::Vec2d(0.0, 2.0), coeffs).transpose()
        << std::endl;
    std::cerr
        << vem.polynomial_eval(0, mtao::Vec2d(3.0, 0.0), coeffs) << ": "
        << vem.polynomial_grad(0, mtao::Vec2d(3.0, 0.0), coeffs).transpose()
        << std::endl;

    coeffs.setZero();
    coeffs(3 + 0) = 1.;
    coeffs(3 + 2) = 1.;
    std::cerr
        << vem.polynomial_eval(0, mtao::Vec2d(0.0, 2.0), coeffs) << ": "
        << vem.polynomial_grad(0, mtao::Vec2d(0.0, 2.0), coeffs).transpose()
        << std::endl;
    std::cerr
        << vem.polynomial_eval(0, mtao::Vec2d(3.0, 0.0), coeffs) << ": "
        << vem.polynomial_grad(0, mtao::Vec2d(3.0, 0.0), coeffs).transpose()
        << std::endl;

    coeffs.setZero();
    coeffs(6 + 0) = 1.;
    coeffs(6 + 3) = 1.;
    std::cerr
        << vem.polynomial_eval(0, mtao::Vec2d(0.0, 2.0), coeffs) << ": "
        << vem.polynomial_grad(0, mtao::Vec2d(0.0, 2.0), coeffs).transpose()
        << std::endl;
    std::cerr
        << vem.polynomial_eval(0, mtao::Vec2d(3.0, 0.0), coeffs) << ": "
        << vem.polynomial_grad(0, mtao::Vec2d(3.0, 0.0), coeffs).transpose()
        << std::endl;

    std::cerr << "Some random finite diff checks" << std::endl;
    for (int k = 0; k < 10; ++k) {
        coeffs.setRandom();
        double eps = 1e-5;
        mtao::Vec2d xy;
        xy.setRandom();
        double x = xy.x();
        double y = xy.y();
        std::cerr << xy << ": ";
        std::cerr
            << ((vem.polynomial_eval(0, mtao::Vec2d(x + eps, y), coeffs) -
                 vem.polynomial_eval(0, mtao::Vec2d(x - eps, y), coeffs)) /
                (2 * eps))
            << " "
            << ((vem.polynomial_eval(0, mtao::Vec2d(x, y + eps), coeffs) -
                 vem.polynomial_eval(0, mtao::Vec2d(x, y - eps), coeffs)) /
                (2 * eps))
            << ": "
            << vem.polynomial_grad(0, mtao::Vec2d(x, y), coeffs).transpose()

            << "     "
            << ((vem.polynomial_eval(0, mtao::Vec2d(x + eps, y), coeffs) -
                 2 * vem.polynomial_eval(0, mtao::Vec2d(x, y), coeffs) +
                 vem.polynomial_eval(0, mtao::Vec2d(x - eps, y), coeffs)) /
                (eps * eps)) +
                   ((vem.polynomial_eval(0, mtao::Vec2d(x, y + eps), coeffs) -
                     2 * vem.polynomial_eval(0, mtao::Vec2d(x, y), coeffs) +
                     vem.polynomial_eval(0, mtao::Vec2d(x, y - eps), coeffs)) /
                    (eps * eps))
            << ": " << vem.polynomial_laplacian(0, mtao::Vec2d(x, y), coeffs)
            << std::endl;
    }

    std::cerr << "dirichichlet energies" << std::endl;
    std::cerr << vem.per_monomial_dirichlet_energy(0) << std::endl;

    vem.order = 4;
    vem.edge_order = 15;
    // vem.vertices.setRandom();
    vem.centers = vem.vertices.rowwise().mean();
    vem.centers.setZero();
    auto [PP, N] = vem.poly_projection_sample2sample(0);
    // std::cerr << "Projection operator: " << PP << std::endl;
    // std::cerr << "Null space: " << N << std::endl;
    // std::cerr << "Nul check: " << P * N << std::endl;
    // vem.vertices.setRandom();
    // vem.centers = vem.vertices.rowwise().mean();
    // std::cerr << vem.poly_projection(0) << std::endl;
    //

    if (false) {
        auto P = vem.poly_projection_sample2poly(0);
        auto CM = vem.poly_coefficient_matrix(0);
        coeffs.resize(vem.coefficient_size());
        for (int j = 0; j < 5; ++j) {
            coeffs.setRandom();

            auto inds = vem.cell_sample_indices_vec(0);
            mtao::VecXd f(inds.size());
            for (int i = 0; i < inds.size(); ++i) {
                auto p = vem.sample_position(inds.at(i));
                f(i) = vem.polynomial_eval(0, p, coeffs)(0);
            }
            std::cerr << "Input coeffs: \n";
            std::cerr << coeffs.transpose() << std::endl;
            std::cerr << "Input fvals: \n";
            std::cerr << f.transpose() << std::endl;

            std::cerr << "myevalfvals: \n";
            std::cerr << coeffs.transpose() * CM << std::endl;

            std::cerr << "f2f projected: \n";
            std::cerr << (PP * f).transpose() << std::endl;

            std::cerr << "Projected coeffs: \n";
            std::cerr << (P * f).transpose() << std::endl << std::endl;
        }
    }
    vem.order = 3;
    std::cerr << vem.per_monomial_laplacian(0) << std::endl;
#endif
}
