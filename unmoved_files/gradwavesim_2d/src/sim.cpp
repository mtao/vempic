#include "vem/gradwavesim_2d/sim.hpp"

#include <mtao/eigen/diagonal_to_sparse.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>
#include <mtao/eigen/partition_vector.hpp>

namespace vem::gradwavesim_2d {
Sim::Sim(const VEMMesh2 &mesh, int order) : poisson_vem(mesh, order) {
    initialize();
}

void Sim::initialize() {
    pressure.resize(poisson_vem.system_size());
    pressure_previous.resize(poisson_vem.system_size());
    auto bb = mesh().bounding_box();

    mtao::Vec2d center = bb.center();

    for (int j = 0; j < poisson_vem.point_size(); ++j) {
        mtao::Vec2d p = poisson_vem.point_sample_indexer().get_position(j);

        std::cout << p.transpose() << std::endl;
        pressure(j) =
          mtao::geometry::interpolation::spline_gaussian_rbf(center, p, .1)(0);
        pressure_previous(j) =
          mtao::geometry::interpolation::spline_gaussian_rbf(
            center + .01 * mtao::Vec2d::Unit(1), p, .1)(0);
    }
    //std::cout << "Pressure values: " << pressure.transpose() << std::endl;
}

size_t Sim::pressure_sample_count() const { return poisson_vem.system_size(); }
size_t Sim::pressure_polynomial_count() const {
    return poisson_vem.monomial_size();
}

size_t Sim::vector_field_sample_count() const { return 2 * poisson_vem.system_size(); }
size_t Sim::vector_field_polynomial_count() const {
    return 2 * poisson_vem.monomial_size();
}

size_t Sim::system_size() const { return vector_field_sample_count() + ; }

std::array<size_t, 4> kkt_block_offsets() const {
    return mtao::utils::partial_sum(
      std::array<size_t, 3>(
        vector_field_sample_count(),
        pressure_polynomial_count(),
        boundary_conditions.size()));
}

void Sim::step(double dt) {
    spdlog::info("Taking a step {}", dt);
    // TODO: cfl things
    int count = 1;
    double substep = dt / count;
    for (int j = 0; j < count; ++j) {
        implicit_stormer_verlet_update(substep);
        //explicit_stormer_verlet_integration(substep);
    }
}
void Sim::implicit_stormer_verlet_update(double dt) {
    auto [A, b] = kkt_system(dt);
    // std::cout << A << std::endl;
    // std::cout << "===\n" << b.transpose() << std::endl;
    mtao::VecXd x = b;
    x.setZero();
    mtao::solvers::linear::SparseCholeskyPCGSolve(A, b, x, 1e-10);
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(A);
    // Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>
    //    solver(A);
    // mtao::VecXd x = solver.solve(b);

    gradpressure_previous = gradpressure;
    gradpressure = x.head(poisson_vem.vector_field_sample_count());
    pressure = x.segment(poisson_vem.vector_field_sample_count());

    mtao::VecXd lambda;
    std::tie(gradpressure, pressure, lambda) = mtao::eigen::partition_vector(x);
    // pressure_dtdt = x.head(poisson_vem.monomial_size());
}


std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> Sim::kkt_system(
  double dt) const {
    auto [C, Crhs] =
      poisson_vem.point_constraint_matrix(boundary_conditions, active_cells);

    // p_tt = c^2 \Delta u
    // u_new = 2 * u_cur - u_old + dt^2 c^2 \Delta .5 * (u_new + u_old)
    // (I - .5 * dt^2c^2 \Delta) = 2 * u_cur - u_old + .5 dt^2 c^2 \Delta u_old
    auto L = poisson_vem.point_laplacian(active_cells);

    Eigen::SparseMatrix<double> A =
      mtao::eigen::diagonal_to_sparse(mtao::VecXd::Ones(L.rows())) + .5 * dt * dt * c * c * L;
    // Eigen::SparseMatrix<double> A = (1 +  dt*dt*c) *
    // mtao::eigen::diagonal_to_sparse(mtao::VecXd::Ones(L.rows())) ;
    int sys_size = L.rows();

    std::vector<Eigen::Triplet<double>> trips = mtao::eigen::mat_to_triplets(A);
    trips.reserve(trips.size() + 2 * C.nonZeros());

    for (int k = 0; k < C.outerSize(); ++k) {
        for (decltype(C)::InnerIterator it(C, k); it; ++it) {
            trips.emplace_back(sys_size + it.row(), it.col(), it.value());
            trips.emplace_back(it.col(), sys_size + it.row(), it.value());
        }
    }

    size_t size = sys_size + C.rows();
    Eigen::SparseMatrix<double, Eigen::RowMajor> R(size, size);
    R.setFromTriplets(trips.begin(), trips.end());

    mtao::VecXd rhs =
      (2 * pressure - pressure_previous) - .5 * dt * dt * c * c * L * pressure_previous;
    return { R, mtao::eigen::vstack(rhs, Crhs) };
}
}// namespace vem::gradwavesim_2d
