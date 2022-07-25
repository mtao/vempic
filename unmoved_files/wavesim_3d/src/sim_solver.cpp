
#include <mtao/eigen/partition_vector.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/solvers/linear/conjugate_gradient.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>
#include <vem/normals.hpp>
#include <vem/utils/parent_maps.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "vem/wavesim_3d/sim.hpp"

namespace vem::wavesim_3d {

void Sim::semiimplicit_step(double dt) {
    auto& c = sample_velocities;
    auto stacked_V =
        mtao::eigen::hstack(c.row(0), c.row(1), c.row(2)).transpose().eval();
    velocity_divergence = sample_codivergence() * stacked_V;
    mtao::VecXd b = velocity_divergence;
    mtao::VecXd x = b;
    x.setZero();

    spdlog::debug("Entering pressure solve...");
    if (static_domain) {
        auto& solver = _spqr_solver;
        _cholmod_solver.cholmod().SPQR_nthreads = 0;
        _cholmod_solver.cholmod().SPQR_grain = 32;

        _spqr_solver.cholmodCommon()->SPQR_nthreads = 0;
        _spqr_solver.cholmodCommon()->SPQR_grain = 32;

        if (!solver_warm || solver.rows() != b.rows()) {
            solver_warm = true;
            spdlog::info("Recomputing decomposition");
            Eigen::SparseMatrix<double> A = sample_laplacian();
            int nonzeros = A.nonZeros();
            A.prune(1e-5);
            spdlog::info(
                "Through trivial pruning we obtain {0} nnz of {2} entries "
                "rather than {1} nnz of {2} entries",
                A.nonZeros(), nonzeros, A.size());

            solver.compute(A);
            spdlog::info(".. done");
        }
        x = solver.solve(b);
    } else {
        Eigen::SparseMatrix<double> A = sample_laplacian();
        mtao::solvers::linear::SparseCholeskyPCGSolve(A, b, x, 1e-8);
        if (!x.allFinite()) {
            mtao::solvers::linear::CGSolve(A, b, x, 1e-8);
        }
    }
    spdlog::info("Finished pressure solve... (norm = {})", x.norm());

    auto Pi = sample_to_poly_dirichlet();
    pressure = Pi * x;

    mtao::VecXd long_pg = sample_gradient() * x;
    size_t size = velocity_stride_sample_size();
    c.row(0) -= long_pg.head(size).transpose();
    c.row(1) -= long_pg.segment(size, size).transpose();
    c.row(2) -= long_pg.tail(size).transpose();
    std::cout << "Post pressure projection velocity norms: "
              << c.rowwise().norm().transpose() << std::endl;
}

void Sim::implicit_stormer_verlet_update(double dt) {
    auto [A, b] = kkt_system(dt);
    // std::cout << A << std::endl;
    // std::cout << "===\n" << b.transpose() << std::endl;
    mtao::VecXd x = b;
    x.setZero();
    mtao::solvers::linear::SparseCholeskyPCGSolve(A, b, x, 1e-10);
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
    // Eigen::SparseQR<Eigen::SparseMatrix<double>,
    // Eigen::COLAMDOrdering<int>>
    //    solver(A);
    // x = solver.solve(b);

    double pp = pressure_previous.norm();
    pressure_previous = pressure;
    pressure = x.head(poisson_vem.system_size());

    auto L = poisson_vem.point_laplacian(active_cells);
    spdlog::info("Doing the arithmetic");
    pressure_dtdt = -c * c * L * .5 * (pressure + pressure_previous);

    spdlog::info("Pressure cur/prev/prevprev {} / {} / {}", pressure.norm(),
                 pressure_previous.norm(), pp);

    // pressure_dtdt = x.head(poisson_vem.monomial_size());
}

void Sim::explicit_stormer_verlet_integration(double dt) {
    spdlog::info("Constructing the point laplacian");
    auto L = poisson_vem.point_laplacian(active_cells);
    spdlog::info("Doing the arithmetic");
    pressure_dtdt = -c * c * L * pressure_previous;
    // pressure_dtdt =-c * pressure;
    mtao::VecXd cur_p = pressure;
    pressure = 2 * cur_p - pressure_previous + dt * dt * pressure_dtdt;
    pressure_previous = cur_p;
    spdlog::info("Pressure cur/prev/dtdt {} / {} / {}", pressure.norm(),
                 pressure_previous.norm(), pressure_dtdt.norm());
}
}  // namespace vem::wavesim_3d
