
#include <mtao/eigen/masking_utils.hpp>
#include <mtao/eigen/partition_vector.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/solvers/linear/conjugate_gradient.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>
#include <vem/normals.hpp>
#include <vem/utils/parent_maps.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "vem/fluidsim_2d/sim.hpp"

namespace vem::fluidsim_2d {
std::set<int> Sim::deactivated_pressure_samples() const {
    std::set<int> ret;
    auto inv = mtao::eigen::inverse_mask(cell_count(), active_cells());
    for (auto&& c : inv) {
        auto cell = pressure_indexer().get_cell(c);
        for (auto&& index : cell.local_to_world_sample_indices()) {
            ret.emplace(index);
        }
    }
    for (auto&& face :
         velocity.boundary_intersector().boundary_edge_indices()) {
        for (auto&& index :
             pressure_indexer().flux_indexer().coefficient_range(face)) {
            ret.emplace(face);
        }
    }
    return ret;
}

/*
std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
Sim::point_constraint_matrix(const poisson_2d::ScalarConstraints& constraints) {
auto vertex_face_map = vem::utils::vertex_faces(mesh());
auto edge_face_map = vem::utils::edge_faces(mesh());
std::vector<Eigen::Triplet<double>> trips;

std::list<double> rhs_values;
auto cur_constraint_pos = [&]() -> size_t { return rhs_values.size(); };
auto add_rhs_value = [&](double value) {
    spdlog::info("Adding a constraint with value {}", value);
    rhs_values.emplace_back(value);
};
// for (auto &&[vidx, value] : constraints.pointwise_dirichlet) {
//    trips.emplace_back(cur_constraint_pos(), vidx, 1);
//    add_rhs_value(value);
//}
// if (constraints.mean_value) {
//}

auto N = normals(mesh());
for (auto&& [edge_idx, value] : constraints.edge_integrated_flux_neumann) {
    //spdlog::info("Constructing constraints around edge {}", edge_idx);
    auto inds =
        poisson_vem.point_sample_indexer().ordered_edge_indices(edge_idx);
    auto W =
        mtao::quadrature::gauss_lobatto_sample_weights<double>(inds.size());
    auto n = N.col(edge_idx);
    auto e = mesh().E.col(edge_idx);
    auto a = mesh().V.col(e(0));
    auto b = mesh().V.col(e(1));
    // factor because gauss lobatto is defined on [-1,1]
    double weight_scale = .5 * (b - a).norm();

    int vfield_size = 2 * velocity_sample_count();
    for (auto&& [fidx, sign] : edge_face_map.at(edge_idx)) {
        if (!active_cells.empty() && !active_cells.contains(fidx)) {
            continue;
        }
        //spdlog::info("  Edge {} has child {}", edge_idx, fidx);
        auto c = poisson_vem.get_cell(fidx);
        auto reindexer = c.world_to_local_point_indices();
        auto Pi = c.Pis();

        for (auto&& [weight, point_ind] : mtao::iterator::zip(W, inds)) {
            trips.emplace_back(cur_constraint_pos(), point_ind,
                               weight * weight_scale * n(0));

            trips.emplace_back(cur_constraint_pos(),
                               point_ind + pressure_sample_count(),
                               weight * weight_scale * n(1));
        }

        //spdlog::info("Added more triplets, now i have {}", trips.size());
        // for (auto&& [ind, w] : mtao::iterator::zip(inds, v)) {
        //    trips.emplace_back(cur_constraint_pos(), ind, w);
        //}

        add_rhs_value(value);
    }
}

Eigen::SparseMatrix<double> R(
    cur_constraint_pos(),
    pressure_sample_count() + velocity_sample_count());
R.setFromTriplets(trips.begin(), trips.end());
mtao::VecXd Rv(cur_constraint_pos());
std::copy(rhs_values.begin(), rhs_values.end(), Rv.data());
//spdlog::info("MAde constraints. C{}x{} and c{}", R.rows(), R.cols(),
//             Rv.size());
// std::cout << R << std::endl;
return {R, Rv};
}
*/
void Sim::pressure_projection() {
    // auto [C, c] = point_constraint_matrix(boundary_conditions);
    // std::cout << "Constraint matrix:\n";
    // std::cout << C << std::endl;
    // std::cout << "constraint data:\n";
    // std::cout << c << std::endl;

    update_velocity_divergence();
    update_pressure();
    update_pressure_gradient();
}

void Sim::update_pressure() {
    auto& c = sample_velocities;
    auto stacked_V = mtao::eigen::hstack(c.row(0), c.row(1)).transpose().eval();
    velocity_divergence = sample_codivergence() * stacked_V;
    mtao::VecXd b = velocity_divergence;
    mtao::VecXd x = b;
    x.setZero();

    spdlog::debug("Entering pressure solve...");
    if (static_domain) {
        if (_qr_solver.rows() != b.rows()) {
            spdlog::info("Recomputing qr decomposition");
            Eigen::SparseMatrix<double> A = sample_laplacian();
            _qr_solver.compute(A);
            spdlog::info(".. done");
        }
        x = _qr_solver.solve(b);
    } else {
        Eigen::SparseMatrix<double> A = sample_laplacian();

        // auto inactive = deactivated_pressure_samples();
        // auto active = mtao::eigen::inverse_mask(A.rows(), inactive);
        // auto Pi = mtao::eigen::projected_mask_matrix<double>(A.rows(),
        // active); auto CPi =
        //    mtao::eigen::projected_mask_matrix<double>(A.rows(), inactive);

        // Eigen::SparseMatrix<double> D = Pi * A * Pi.transpose();
        // mtao::VecXd bb = Pi * b;
        // mtao::VecXd xx = bb;
        // xx.setZero();

        // mtao::solvers::linear::SparseCholeskyPCGSolve(D, bb, xx, 1e-8);
        mtao::solvers::linear::SparseCholeskyPCGSolve(A, b, x, 1e-8);
        if (!x.allFinite()) {
            // mtao::solvers::linear::CGSolve(D, bb, xx, 1e-8);
            mtao::solvers::linear::CGSolve(A, b, x, 1e-8);
        }
        // x = Pi.transpose() * xx;
    }
    spdlog::debug("Finished pressure solve...");
    pressure = std::move(x);
}
void Sim::update_pressure_gradient() {
    auto& c = velocity.coefficients();

    auto psg = sample_to_poly_gradient();
    mtao::VecXd long_ppg = psg * pressure;
    size_t size = velocity_stride_monomial_size();
    auto a = long_ppg.head(size).transpose();
    auto b = long_ppg.tail(size).transpose();
    c.row(0) -= a;
    c.row(1) -= b;
}

void Sim::update_velocity_divergence() {
    // const auto& c = sample_velocities;
    // auto stacked_V = mtao::eigen::hstack(c.row(0),
    // c.row(1)).transpose().eval(); auto Div = sample_to_poly_codivergence();
    // velocity_divergence_poly = Div * stacked_V;
}
}  // namespace vem::fluidsim_2d
