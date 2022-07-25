#include <vem/fluidsim_2d/fluidvem2.hpp>
#include <vem/fluidsim_2d/sim.hpp>
#include <vem/from_grid.hpp>
#include <vem/poisson_2d/example_constraints.hpp>

#include "vem/point_moment_indexer.hpp"

auto make_mesh(int N) {
    Eigen::AlignedBox<double, 2> bb;
    bb.min().setConstant(0);
    bb.max().setConstant(1);
    auto mesh = vem::from_grid(bb, N, N);

    return mesh;
}

void get_eigen_decomposition(int N, int D) {
    spdlog::info("N = {}, D = {}", N, D);
    auto mesh = make_mesh(N);
    vem::FluxMomentIndexer flux_vem(mesh, D);
    vem::PointMomentIndexer point_vem(mesh, D);

    auto get_eigenvalues = [](const Eigen::MatrixXd& L) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(L);
        if (eigs.info() != Eigen::Success) {
            spdlog::error("Eigen decomposition failed");
            return;
        }
        std::cout << eigs.eigenvalues().transpose() << std::endl;
    };

    Eigen::MatrixXd flux_L = flux_vem.sample_laplacian();
    Eigen::MatrixXd point_L = point_vem.sample_laplacian();

    get_eigenvalues(point_L);
    get_eigenvalues(flux_L);
}

void test_projection_operators(int N, int D) {
    auto mesh = make_mesh(N);
    vem::fluidsim_2d::FluxMomentFluidVEM2 fvem(mesh, D);
    auto c = fvem.get_velocity_cell(0);
    auto pc = fvem.get_pressure_cell(0);

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
    test_projection_operators(2, 1);
    test_projection_operators(2, 2);
    // test_projection_operators(2, 1);
    // get_eigen_decomposition(2, 1);
    // get_eigen_decomposition(3, 1);
    // get_eigen_decomposition(4, 1);
    // get_eigen_decomposition(5, 1);

    // get_eigen_decomposition(3, 2);
    // get_eigen_decomposition(3, 3);
    // get_eigen_decomposition(3, 4);
    // get_eigen_decomposition(50, 1);
    // get_eigen_decomposition(25, 2);
}
