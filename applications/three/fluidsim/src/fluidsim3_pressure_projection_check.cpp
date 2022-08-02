#include <fmt/format.h>
#include <igl/read_triangle_mesh.h>

#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <vem/three/fluidsim/fluidvem.hpp>
#include <vem/three/fluidsim/sim.hpp>
#include <vem/three/from_grid.hpp>
#include <vem/three/from_mandoline.hpp>

#include "vem/polynomials/utils.hpp"

void prune(auto&& A) { A = (A.array().abs() > 1e-10).select(A, 0); }
auto make_mesh(int N) {
    Eigen::AlignedBox<double, 3> bb;
    bb.min().setConstant(0);
    bb.max().setConstant(1);
    auto mesh = vem::three::from_grid(bb, N, N, N);
    return mesh;

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::read_triangle_mesh("/tmp/collision.obj", V, F);
    bb.min().setConstant(-3);
    bb.max().setConstant(3);
    // auto mesh = vem::from_mandoline(bb, N, N, N, V.transpose(),
    // F.transpose());

    return mesh;
}

void test_projection_operators(int N, int D) {
    auto mesh = make_mesh(N);
    vem::three::fluidsim::FluidVEM3 fvem(mesh, D);

    {
        auto pc = fvem.get_pressure_cell(0);

        auto Pis = pc.dirichlet_projector();
        auto G = pc.monomial_dirichlet_grammian();
    }
    {
        auto L = fvem.sample_laplacian().toDense().eval();
        auto D = fvem.sample_to_poly_codivergence();
        auto G = fvem.sample_to_poly_gradient();
        auto M = fvem.poly_velocity_l2_grammian();
        // std::cout << M << std::endl;
        // std::cout << pc << std::endl;
        // return;
        auto MM = mtao::eigen::sparse_block_diagonal_repmats(M, 3);

        prune(L);
        // std::cout << "Current laplacian:\n";
        // std::cout << L << std::endl;

        auto L2 = (G.transpose() * MM * G).transpose().toDense().eval();
        prune(L2);
        // std::cout << "Alternate Laplacian:\n" << L2 << std::endl;
        std::cout << (L - L2).norm() << std::endl;

        // std::cout << (D * G - L).norm() << std::endl;
        return;
    }

    for (int j = 0; j < mesh.cell_count(); ++j) {
        auto pc = fvem.get_pressure_cell(j);
        auto vc = fvem.get_velocity_cell(j);
        auto proj = vc.l2_projector();
        if (!proj.array().isFinite().all()) {
            spdlog::info("cell {} has a nan", j);

            // auto G = vc.regularized_monomial_dirichlet_grammian();
            auto K = vc.sample_monomial_dirichlet_grammian();
            // spdlog::info("G:");
            // std::cout << G << std::endl;
            spdlog::info("K (sample monomial dirichlet grammian):");
            std::cout << K << std::endl;
            //// for (auto&& [eidx, sa] : vc.surface_areas()) {
            ////    spdlog::info("edge {} has surface area {}", eidx, sa);
            ////}
            // auto DP = vc.dirichlet_projector();
            // std::cout << "dir_proj:\n" << DP << std::endl;
            for (auto&& [eidx, sgn] : vc.faces()) {
                // int fsize = flux_size(eidx);
                // auto integrals = monomial_face_integrals(eidx,
                // flux_degree(eidx)); mtao::Vec2d N =
                // mtao::eigen::stl2eigen(EN.at(eidx)); auto G =
                // monomial_to_monomial_gradient(flux_degree(eidx));
                // Eigen::SparseMatrix<double> GN =
                //    (N.x() * G.topRows(fsize) + N.y() * G.bottomRows(fsize))
                //        .topRows(fsize);
                auto r = vc.monomial_l2_face_grammian(eidx);
                std::cout << "face id: " << eidx << " => " << r.transpose()
                          << std::endl;
                int row_degree = vc.monomial_degree();
                int col_degree = vc.flux_degree(eidx);
                // std::cout << "Face " << eidx << " is emitting a l2 face
                // grammian\n "
                //          << r << std::endl;
                // spdlog::info("Block sizing: {}x{} vs {}x{}", MyBlock.rows(),
                //             MyBlock.cols(), r.rows(), r.cols());
                size_t row_size =
                    vem::polynomials::three::num_monomials_upto(row_degree);
                size_t col_size =
                    vem::polynomials::two::num_monomials_upto(col_degree);

                // spdlog::info("L2 grammian face size: {}x{} (from degrees {}
                // {})", row_size,
                //             col_size, row_degree, col_degree);
                mtao::MatXd R(row_size, col_size);
                R.setZero();

                auto face_center = vc.mesh().FC.col(eidx);
                std::cout << "Face center:\n"
                          << face_center.transpose() << std::endl;
                double face_diameter = vc.mesh().face_diameter(eidx);
                std::cout << "Face diameter:\n" << face_diameter << std::endl;

                auto integrals =
                    vc.monomial_face_integrals(row_degree + col_degree, eidx);
                std::vector<double> p(integrals.size());
                mtao::eigen::stl2eigen(p) = integrals;
                spdlog::info("Integrals for face {}: {}", eidx,
                             fmt::join(p, ","));
                for (int j = 0; j < row_size; ++j) {
                    auto coeffs = vc.project_monomial_to_boundary(eidx, j);
                    std::cout << "monomial " << j
                              << " is projected to coefficients "
                              << coeffs.transpose() << std::endl;
                    for (int k = 0; k < col_size; ++k) {
                        auto [xk, yk] =
                            vem::polynomials::two::index_to_exponents(k);
                        // std::cout << "XI Got coeffs for " << xi <<": " <<
                        // coeffs.transpose() << std::endl;
                        // spdlog::info("Entry {} {}", j, k);

                        double& val = R(j, k) = 0;
                        for (int l = 0; l < coeffs.size(); ++l) {
                            auto [xl, yl] =
                                vem::polynomials::two::index_to_exponents(l);
                            int idx = vem::polynomials::two::exponents_to_index(
                                xl + xk, yl + yk);
                            spdlog::info("{} {} => {}", xl + xk, yl + yk,
                                         integrals(idx));
                            val += coeffs(l) * integrals(idx);
                        }
                        // spdlog::info(
                    }
                }
            }

            const double length = vc.boundary_area();
            const double area = vc.volume();

            auto integrals =
                vc.monomial_indexer().monomial_integrals(vc.cell_index());
            std::cout << (integrals.transpose() / area) << std::endl;
        }
        // auto RC = pc.local_to_world_monomial_indices();
        // auto CC = pc.local_to_world_sample_indices();
        // std::vector<int> A;
        // for (auto &&v : RC) {
        //    A.emplace_back(v);
        //}
        // std::vector<int> B;

        // for (auto &&v : CC) {
        //    B.emplace_back(v);
        //}
        // spdlog::info("{} monomial indices: {}", A.size(), fmt::join(A, ","));
        // spdlog::info("{} world indices: {}", B.size(), fmt::join(B, ","));
    }

    auto c = fvem.get_velocity_cell(0);
    auto pc = fvem.get_pressure_cell(0);

    // auto A = pc.dirichlet_projector();
    // auto B = mtao::MatXd(fvem.sample_to_poly_dirichlet());
    // A = (A.array().abs() > 1e-10).select(A, 0);
    // B = (B.array().abs() > 1e-10).select(B, 0);
    // std::cout << "Dirichlet poly cell projector\n" << A << std::endl;
    // std::cout << "Dirichlet poly all projector\n" << B << std::endl;

    fvem.sample_codivergence();
    // std::cout << "codivergence:\n";
    // std::cout << fvem.sample_codivergence() << std::endl;
    return;

    auto evaled = pc.monomial_evaluation();
    // std::cout << "Monomials evaluated\n" << evaled << std::endl;
    // std::cout << "L2 thing that should be identity" << std::endl;
    // std::cout << c.l2_projector() * evaled << std::endl;

    // std::cout << "dirichlet thing that should be identity" << std::endl;
    // std::cout << pc.dirichlet_projector() * evaled << std::endl;

    //// std::cout << "L2 sample projector" << std::endl;
    //// std::cout << c.l2_sample_projector()
    ////          << std::endl;

    // std::cout << "dirichlet sample projector" << std::endl;
    // std::cout << pc.dirichlet_sample_projector() << std::endl;
    // std::cout << "Diameter: " << c.diameter() << std::endl;
    // std::cout << "grammian\n"
    //          << pc.regularized_monomial_dirichlet_grammian() << std::endl;
    // std::cout << "off grammian\n"
    //          << pc.sample_monomial_dirichlet_grammian() << std::endl;

    // std::cout << "l2 grammian\n" << c.monomial_l2_grammian() << std::endl;
    // std::cout << "l2 off grammian\n"
    //          << c.sample_monomial_l2_grammian() << std::endl;

    // std::cout << "velocity_stride_to_pressure_monomial_map:\n";
    // std::cout << fvem.velocity_stride_to_pressure_monomial_map() <<
    // std::endl;
    std::cout << "sample_laplacian:\n";
    std::cout << fvem.sample_laplacian() << std::endl;

    // std::cout << "dirichlet Sample projection:\n"
    //          << pc.dirichlet_sample_projector() << std::endl;

    std::cout << "sample_to_poly_l2:\n";
    std::cout << fvem.sample_to_poly_l2() << std::endl;

    std::cout << "sample_to_poly_dirichlet:\n";
    std::cout << fvem.sample_to_poly_dirichlet() << std::endl;
    // std::cout << "Cell dirichlet proj error: \n"
    //          << pc.dirichlet_projector_error() << std::endl;
    // auto E = c.dirichlet_projector_error();
    // std::cout << "Cell dirichlet proj error prod: \n"
    //          << E.transpose() * E << std::endl;
    std::cout << "poly_pressure_l2_grammian:\n";
    std::cout << fvem.poly_pressure_l2_grammian() << std::endl;

    std::cout << "codivergence:\n";
    std::cout << fvem.sample_codivergence() << std::endl;

    std::cout << "evaluation pressure:\n";
    std::cout << fvem.polynomial_to_sample_evaluation_matrix(true) << std::endl;

    std::cout << "evaluation velocity:\n";
    std::cout << fvem.polynomial_to_sample_evaluation_matrix(false)
              << std::endl;

    std::cout << "evaluation projection:\n";
    std::cout << fvem.sample_to_poly_dirichlet() *
                     fvem.polynomial_to_sample_evaluation_matrix(true)
              << std::endl;
}

// void test_projected_divergence(int N) {
//    /
//    auto mesh = make_mesh(N);
//    vem::fluidsim_2d::FluidVEM2 fvem(mesh, D);
//    auto P = fvem.velocity_indexer().point_sample_indexer().get_positions();
//    for (int j = 0; j < P.cols(); ++j) {
//        auto p = P.col(j);
//        sim.sample_velocities.col(j) = P.col(j).array();
//        sim.sample_velocities.col(j) << 0, 1;
//    }
//    // for(int cidx = 0; cidx < sim.mesh().cell_count(); ++cidx) {
//    // for (int j = P.cols(); j < sim.sample_velocities.cols(); ++j) {
//    //    int off = j - P.cols();
//    //    sim.sample_velocities.block(
//    //}
//    std::cout << "Sample velocities:\n";
//    std::cout << sim.sample_velocities << std::endl;
//
//    sim.update_velocity_divergence();
//    std::cout << "Resulting divergence\n"
//              << sim.velocity_divergence << std::endl;
//
//    // sim.boundary_conditions.pointwise_dirichlet[0] = 0.0;
//    // sim.boundary_conditions=
//    // vem::poisson_2d::linear_function_neumann(sim.mesh(), 0,
//    // mtao::Vec2d(1.0,0.0));
//
//    sim.pressure_projection();
//
//    std::cout << "Resulting pressure gradient:\n"
//              << sim.pressure_gradient << std::endl;
//
//    sim.update_velocity_divergence();
//    std::cout << "Final divergence\n" << sim.velocity_divergence << std::endl;
//}

int main(int argc, char* argv[]) {  // test_projected_divergence(3);
    // test_projection_operators(2, 1);
    // test_projection_operators(2, 2);
    // test_projection_operators(2, 1);
    test_projection_operators(3, 1);
}
