#include "vem/wavesim_2d/sim.hpp"

#include <mtao/eigen/diagonal_to_sparse.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/interpolation/radial_basis_function.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>
#include <vem/serialization/frame_inventory.hpp>
#include <vem/serialization/serialize_eigen.hpp>

namespace vem::wavesim_2d {
Sim::Sim(const VEMMesh2& mesh, int degree, serialization::Inventory* parent)
    : poisson_vem(mesh, degree),

      inventory(parent != nullptr ? parent->make_subinventory("vem_fluidsim_2d")
                                  : serialization::Inventory::from_scratch(
                                        "vem_wavesim_2d", true))

{
    initialize();

    inventory.add_metadata(
        "visualization_manifest",
        nlohmann::json::object(
            {{"pressure", "scalar_field"}, {"pressure_dtdt", "scalar_field"}}));
    inventory.add_metadata("degree", degree);
}

void Sim::initialize() {
    pressure.resize(poisson_vem.system_size());
    pressure_previous.resize(poisson_vem.system_size());
    auto bb = mesh().bounding_box();

    mtao::Vec2d center = bb.center();

    for (int j = 0; j < poisson_vem.point_size(); ++j) {
        mtao::Vec2d p = poisson_vem.point_sample_indexer().get_position(j);

        pressure(j) = mtao::geometry::interpolation::spline_gaussian_rbf(
            center, p, .05)(0);
        pressure_previous(j) =
            mtao::geometry::interpolation::spline_gaussian_rbf(
                center + .01 * mtao::Vec2d::Unit(1), p, .05)(0);
    }
    // std::cout << "Pressure values: " << pressure.transpose() << std::endl;
}

size_t Sim::pressure_sample_count() const { return poisson_vem.system_size(); }
size_t Sim::pressure_polynomial_count() const {
    return poisson_vem.monomial_size();
}

void Sim::step(double dt) {
    spdlog::info("Taking a step {}", dt);
    auto step_inv =
        serialization::FrameInventory::for_creation(inventory, frame_index);
    step_inv.add_metadata("timestep", dt);
    step_inv.add_metadata("complete", false);

    auto Pi = poisson_vem.sample_to_polynomial_projection_matrix(active_cells);
    {
        serialization::serialize_VecXd(step_inv, "sample_pressure", pressure);
        auto& meta = step_inv.asset_metadata("sample_pressure");
        meta["type"] = "sample_field";
    }
    {
        serialization::serialize_VecXd(step_inv, "sample_pressure_dtdt",
                                       pressure);
        auto& meta = step_inv.asset_metadata("sample_pressure_dtdt");
        meta["type"] = "sample_field";
    }
    {
        serialization::serialize_VecXd(step_inv, "pressure", Pi * pressure);
        auto& meta = step_inv.asset_metadata("pressure");
        meta["type"] = "scalar_field";
    }
    {
        if (pressure_dtdt.size() == Pi.cols()) {
            serialization::serialize_VecXd(step_inv, "pressure_dtdt",
                                           Pi * pressure_dtdt);
            auto& meta = step_inv.asset_metadata("pressure_dtdt");
            meta["type"] = "scalar_field";
        }
    }
    // TODO: cfl things
    int count = 1;
    double substep = dt / count;
    for (int j = 0; j < count; ++j) {
         implicit_stormer_verlet_update(substep);
        //explicit_stormer_verlet_integration(substep);
    }
    frame_index++;
}
void Sim::implicit_stormer_verlet_update(double dt) {
    auto [A, b] = kkt_system(dt);
    // std::cout << A << std::endl;
    // std::cout << "===\n" << b.transpose() << std::endl;
    mtao::VecXd x = b;
    x.setZero();
    mtao::solvers::linear::SparseCholeskyPCGSolve(A, b, x, 1e-10);
    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
    // Eigen::SparseQR<Eigen::SparseMatrix<double>,
    // Eigen::COLAMDOrdering<int>>
    //    solver(A);
    //x = solver.solve(b);

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

std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd> Sim::kkt_system(
    double dt) const {
    auto [C, Crhs] =
        poisson_vem.point_constraint_matrix(boundary_conditions, active_cells);

    auto M = poisson_vem.mass_matrix(active_cells);
    M.setIdentity();

    // u_tt = c^2 \Delta u
    // M * u_new = M*(2 * u_cur - u_old) + dt^2 c^2 \Delta .5 * (u_new + u_old)
    // (M - .5 * dt^2c^2 \Delta) = M*(2 * u_cur - u_old) + .5 dt^2 c^2 \Delta u_old
    auto L = poisson_vem.point_laplacian(active_cells);

    Eigen::SparseMatrix<double> A =
        // mtao::eigen::diagonal_to_sparse(mtao::VecXd::Ones(L.rows())) +
        M + .5 * dt * dt * c * c * L;
    //A = (1 +  dt*dt*c) * M;
     //mtao::eigen::diagonal_to_sparse(mtao::VecXd::Ones(L.rows())) ;
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

    mtao::VecXd rhs = M * (2 * pressure - pressure_previous) -
                      .5 * dt * dt * c * c * L * pressure_previous;
    return {R, mtao::eigen::vstack(rhs, Crhs)};
}
}  // namespace vem::wavesim_2d
