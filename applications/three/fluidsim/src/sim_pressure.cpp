
#include <Eigen/Sparse>
#include <mtao/eigen/partition_vector.hpp>
#include <mtao/eigen/sparse_block_diagonal_repmats.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/logging/stopwatch.hpp>
#include <mtao/solvers/linear/conjugate_gradient.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>
#include <mtao/types.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "vem/three/fluidsim/sim.hpp"

namespace Eigen::internal {

template <>
struct sparse_time_dense_product_impl<
    Eigen::SparseMatrix<double, Eigen::RowMajor>, mtao::VecXd, mtao::VecXd,
    double, Eigen::RowMajor, true> {
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseLhsType;
    typedef mtao::VecXd DenseRhsType;
    typedef mtao::VecXd DenseResType;
    typedef typename internal::remove_all<SparseLhsType>::type Lhs;
    typedef typename internal::remove_all<DenseRhsType>::type Rhs;
    typedef typename internal::remove_all<DenseResType>::type Res;
    typedef typename evaluator<Lhs>::InnerIterator LhsInnerIterator;
    typedef evaluator<Lhs> LhsEval;
    static void run(const SparseLhsType& lhs, const DenseRhsType& rhs,
                    DenseResType& res, const Res::Scalar& alpha) {
        LhsEval lhsEval(lhs);

        Index n = lhs.outerSize();

        static tbb::affinity_partitioner ap;

        for (Index c = 0; c < rhs.cols(); ++c) {
            // This 20000 threshold has been found experimentally on 2D and 3D
            // Poisson problems. It basically represents the minimal amount of
            // work to be done to be worth it.
            if (lhsEval.nonZerosEstimate() > 20000) {
                tbb::parallel_for(
                    int(0), int(n),
                    [&](int i) { processRow(lhsEval, rhs, res, alpha, i, c); },
                    ap);
            } else {
                for (Index i = 0; i < n; ++i)
                    processRow(lhsEval, rhs, res, alpha, i, c);
            }
        }
    }

    static void processRow(const LhsEval& lhsEval, const DenseRhsType& rhs,
                           DenseResType& res, const typename Res::Scalar& alpha,
                           Index i, Index col) {
        typename Res::Scalar tmp(0);
        for (LhsInnerIterator it(lhsEval, i); it; ++it)
            tmp += it.value() * rhs.coeff(it.index(), col);
        res.coeffRef(i, col) += alpha * tmp;
    }
};
}  // namespace Eigen::internal
struct ConjGrad {
    using MultImpl = Eigen::internal::sparse_time_dense_product_impl<
        Eigen::SparseMatrix<double, Eigen::RowMajor>, mtao::VecXd, mtao::VecXd,
        double, Eigen::RowMajor, true>;
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& A;
    const mtao::VecXd& b;

    double error() { return rsnorm; }

    void compute() {
        x = mtao::VecXd::Zero(A.rows());
        r = mtao::VecXd::Zero(A.rows());
        Ap = mtao::VecXd::Zero(A.rows());

        MultImpl::run(A, x, r, 1);
        r = b - r;

        p = r;
        MultImpl::run(A, p, Ap, 1);
        rsnorm = r.squaredNorm();
    }
    void step() {
        alpha = (rsnorm) / (p.dot(Ap));
        x += alpha * p;
        r -= alpha * Ap;
        beta = 1 / rsnorm;
        rsnorm = r.squaredNorm();
        beta *= rsnorm;
        p = r + beta * p;
        MultImpl::run(A, p, Ap, 1);
    }

    mtao::VecXd x;
    mtao::VecXd r;
    mtao::VecXd p;
    mtao::VecXd Ap;
    double rsnorm;
    double alpha, beta;
};

namespace mtao::solvers::linear {
template <typename Matrix, typename Vector>
void CGSolve_ramp(const Matrix& A, const Vector& b, Vector& x,
                  typename Matrix::Scalar threshold = 1e-5) {
    auto sw = mtao::logging::hierarchical_stopwatch("CGSolve_ramp");
    double residual =
        (b - A * x).norm();  //.template lpNorm<Eigen::Infinity>();
    double thresh = std::max(threshold, threshold * residual);
    // ConjGrad cg{A, b};
    // cg.compute();

    // spdlog::info("cgloopstart");
    // for (int iter = 0; cg.error() > thresh && iter < A.rows(); ++iter) {
    //    cg.step();
    //}
    auto solver =
        ConjugateGradientLinearSolver<Matrix, Vector>(A.rows(), thresh);
    // solver.set_logger(mtao::logging::HierarchicalStopwatch::current_logger());
    //// auto solver =
    ////
    /// IterativeLinearSolver<PreconditionedConjugateGradientCapsule<Matrix,Vector,
    //// Preconditioner> >(A.rows(), 1e-5);
    solver.solve(A, b, x);
    x = solver.x();
    // spdlog::info("cgx assign");
    // x = cg.x;
    // spdlog::info("Finished cgramp");
}

template <typename Preconditioner, typename Matrix, typename Vector>
void PCGSolve_ramp(const Matrix& A, const Vector& b, Vector& x,
                   typename Matrix::Scalar threshold = 1e-5) {
    double residual =
        (b - A * x).norm();  //.template lpNorm<Eigen::Infinity>();
    auto solver = PCGSolver<Matrix, Vector, Preconditioner>(
        .5 * A.rows(), std::max(threshold, threshold * residual));
    // auto solver =
    // IterativeLinearSolver<PreconditionedConjugateGradientCapsule<Matrix,Vector,
    // Preconditioner> >(A.rows(), 1e-5);
    solver.solve(A, b, x);
    x = solver.x();
}

template <typename Matrix, typename Vector>
void SparseCholeskyPCGSolve_ramp(const Matrix& A, const Vector& b, Vector& x,
                                 typename Matrix::Scalar threshold = 1e-5) {
    PCGSolve_ramp<cholesky::SparseLDLT_MIC0<Matrix, Vector>>(A, b, x,
                                                             threshold);
}
}  // namespace mtao::solvers::linear

namespace vem::three::fluidsim {

/*
std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
Sim::point_constraint_matrix(const poisson_2d::ScalarConstraints& constraints) {
auto vertex_face_map = vem::utils::vertex_faces(mesh());
auto edge_face_map = vem::utils::edge_faces(mesh());
std::vector<Eigen::Triplet<double>> trips;

std::list<double> rhs_values;
auto cur_constraint_pos = [&]() -> size_t { return rhs_values.size(); };
auto add_rhs_value = [&](double value) {
    spdlog::trace("Adding a constraint with value {}", value);
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
    //spdlog::trace("Constructing constraints around edge {}", edge_idx);
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
        //spdlog::trace("  Edge {} has child {}", edge_idx, fidx);
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

        //spdlog::trace("Added more triplets, now i have {}", trips.size());
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
//spdlog::trace("MAde constraints. C{}x{} and c{}", R.rows(), R.cols(),
//             Rv.size());
// std::cout << R << std::endl;
return {R, Rv};
}
*/
void Sim::update_pressure_from_divergence() {
    if (pressure.size() != divergence.size()) {
        pressure.resize(divergence.size());
        pressure.setZero();
    }
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::update_pressure_from_divergence");
    const auto& A = _operator_cache.sample_laplacian();

    spdlog::trace(
        "Entering poisson solve... {} threads and {} degrees of freedom",
        Eigen::nbThreads(), pressure.size());
    mtao::solvers::linear::CGSolve_ramp(A, divergence, pressure, 1e-5);
    // mtao::solvers::linear::SparseCholeskyPCGSolve_ramp(A, divergence,
    // pressure,
    //                                                   1e-5);
    if (!pressure.allFinite()) {
        spdlog::warn("Poisson solver failed!");
    }
    spdlog::info("Finished poisson solve... (norm = {})", pressure.norm());
}
void Sim::update_divergence_from_velocity() {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::update_divergence_from_velocity");
    auto& c = sample_velocities;
    auto stacked_V =
        mtao::eigen::hstack(c.row(0), c.row(1), c.row(2)).transpose().eval();

    const auto& scod = _operator_cache.sample_codivergence();
    divergence = scod * stacked_V;
}
void Sim::pressure_projection() {
    auto sw = mtao::logging::hierarchical_stopwatch("Sim::pressure_projection");
    // x.setZero();

    update_divergence_from_velocity();
    update_pressure_from_divergence();

    update_pressure_gradient_from_pressure();
}

void Sim::update_pressure_gradient_from_pressure() {
    auto sw = mtao::logging::hierarchical_stopwatch(
        "Sim::update_pressure_gradient_from_pressure");
    auto& c = velocity.coefficients();

    const auto& psg = _operator_cache.sample_to_poly_gradient();
    mtao::VecXd long_ppg = psg * pressure;
    size_t size = velocity_stride_monomial_size();
    c.row(0) -= long_ppg.head(size).transpose();
    c.row(1) -= long_ppg.segment(size, size).transpose();
    c.row(2) -= long_ppg.tail(size).transpose();
}

}  // namespace vem::fluidsim_3d
