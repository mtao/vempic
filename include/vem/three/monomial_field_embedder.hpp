#pragma once

#if !defined(NO_TBB_FOR)

#include <tbb/parallel_for.h>
#endif

#include <Eigen/Core>
#include <set>

#include "monomial_basis_indexer.hpp"
#include "boundary_intersector.hpp"
namespace vem::three {

template <typename Der>
class MonomialFieldEmbedderBase3 {
   public:
    const static int D = 3;
    Der &derived() { return *static_cast<Der *>(this); }
    const Der &derived() const { return *static_cast<const Der *>(this); }
    MonomialFieldEmbedderBase3(const MonomialBasisIndexer3 &mbi)
        : _monomial_indexer(mbi),
          _boundary_intersector(mbi.mesh(), active_cells) {}
    //_boundary_intersector(mbi.mesh(), active_cells) {}

    template <typename Derived>
    mtao::Vector<double, D> get_vector(const Eigen::MatrixBase<Derived> &p,
                                       int cell) const {
        return derived().get_vector(p, cell);
    }

    template <typename Derived>
    double get_value(const Eigen::MatrixBase<Derived> &p, int cell) const {
        return derived().get_value(p, cell);
    }

    // cell getters

    template <typename Derived>
    int get_cell(const Eigen::MatrixBase<Derived> &p) const {
        return _boundary_intersector.get_cell(p);
    }
    template <typename Derived>
    mtao::VecXi get_cells(const Eigen::MatrixBase<Derived> &p) const {
        return mesh().get_cells(p);
    }

    template <typename Derived>
    int get_cell_projected(const Eigen::MatrixBase<Derived> &p) const {
        return _boundary_intersector.get_projected_cell(p);
    }
    template <typename Derived>
    mtao::VecXi get_cells_projected(const Eigen::MatrixBase<Derived> &p) const {
        mtao::VecXi indices(p.cols());
        for (int j = 0; j < indices.size(); ++j) {
            indices(j) = get_cell_projected(p.col(j));
        }
        return indices;
    }

    // values

    template <typename Derived>
    double get_value(const Eigen::MatrixBase<Derived> &p) const {
        int cell = get_cell_projected(p);
        return get_value(p, cell);
    }
    template <typename Derived, typename IDerived>
    mtao::VecXd get_values(const Eigen::MatrixBase<Derived> &p,
                           const Eigen::MatrixBase<IDerived> &cells) const {
        mtao::VecXd V(p.rows(), p.cols());
        for (int j = 0; j < p.cols(); ++j) {
            auto u = p.col(j);
            int cell = cells(j);
            V(j) = get_value(u, cell);
        }
        return V;
    }
    template <typename Derived>
    mtao::VecXd get_values(const Eigen::MatrixBase<Derived> &p) const {
        mtao::VecXi cells = get_cells_projected(p);
        return get_values(p, cells);
    }

    // Vector operations

    template <typename Derived>
    mtao::Vector<double, D> get_vector(
        const Eigen::MatrixBase<Derived> &p) const {
        int cell = get_cell_projected(p);
        return get_vector(p, cell);
    }
    template <typename Derived, typename IDerived>
    mtao::ColVectors<double, D> get_vectors(
        const Eigen::MatrixBase<Derived> &p,
        const Eigen::MatrixBase<IDerived> &cells) const {
        mtao::ColVectors<double, D> V(p.rows(), p.cols());

        tbb::parallel_for(int(0), int(p.cols()), [&](int j) {
            auto u = p.col(j);
            int cell = cells(j);
            V.col(j) = get_vector(u, cell);
        });
        return V;
    }
    template <typename Derived>
    mtao::ColVectors<double, D> get_vectors(
        const Eigen::MatrixBase<Derived> &p) const {
        mtao::VecXi cells = get_cells_projected(p);
        return get_vectors(p, cells);
    }

    const MonomialBasisIndexer3 &monomial_indexer() const {
        return _monomial_indexer;
    }

    template <typename Derived, typename IDerived, typename QDerived,
              typename VDerived>
    void reflecting_advection_with_vel(const Eigen::MatrixBase<Derived> &P,
                                       VDerived &V,
                                       const Eigen::MatrixBase<IDerived> &cells,
                                       Eigen::PlainObjectBase<QDerived> &Q,
                                       double elasticity = .5) const {
        double elasticity_val = 1 + elasticity;
#if defined(NO_TBB_FOR)
        for (int j = 0; j < Q.cols(); ++j) {
#else
        tbb::parallel_for(int(0), int(Q.cols()), [&](int j) {
#endif
            int cell = cells(j);
            auto q = Q.col(j);
            // started off in an invalid position
            // project it to a valid boundary point
            if (!is_active_cell(cell)) {
                spdlog::info("started out of domain (cell {})", j);
                auto v = V.col(j);
                int face_index;
                std::tie(q, face_index) =
                    _boundary_intersector.closest_boundary_point_with_face(q);
                _boundary_intersector.reflect_vector(face_index, v, 0.0);
#if defined(NO_TBB_FOR)
                continue;
#else
                return;
#endif
            } else if (int qcell = get_cell(q); mesh().collision_free(cell) &&
                                                mesh().collision_free(qcell)) {
#if defined(NO_TBB_FOR)
                continue;
#else
                return;
#endif
            } else if (/*_boundary_intersector.has_open_boundaries() ||*/
                       !is_active_cell(qcell)) {
                auto p = P.col(j);
                auto v = V.col(j);
                if ((p - q).norm() < 1e-5) {
                    q = _boundary_intersector.closest_boundary_point(q);
                } else {
                    auto bisect = _boundary_intersector.raycast(p, q);
                    if (bisect) {
                        auto &plane = *bisect;
                        // spdlog::error("Boundary crossing seen; these
                        // should have opposite signs: {} {}",
                        // p.homogeneous().dot(plane),q.homogeneous().dot(plane));
                        plane /= plane.template head<3>().norm();
                        auto N = plane.template head<3>();
                        mtao::Vec3d protrusion =
                            q.homogeneous().dot(plane) * N;  // rem is now the
                                                             // orthogonal
                        // spdlog::error("Moved particle {}, without isect
                        // would have moved {}", 2*protrusion.norm(),
                        // (p-q).norm());
                        q = q - 2 * protrusion;

                        double proj = v.dot(N);
                        v -= N * elasticity;

                        // q.setConstant(.4);
                    } else {
                        q = _boundary_intersector.closest_boundary_point(q);
                        // q = p; q.setConstant(.4);
                    }
                }
            }
#if defined(NO_TBB_FOR)
        }
#else
        });
#endif
        // std::cout << P << std::endl;
        // spdlog::info("Reflecting advection started with {} outside, ended
        // with {} outside  with {} interventions", int((cells.array() ==
        // -1).count()), int((get_cells(Q).array() == -1).count()), count);
    }

    template <typename Derived, typename IDerived, typename QDerived>
    void reflecting_advection(const Eigen::MatrixBase<Derived> &P,
                              const Eigen::MatrixBase<IDerived> &cells,
                              Eigen::PlainObjectBase<QDerived> &Q) const {
#if defined(NO_TBB_FOR)
        for (int j = 0; j < Q.cols(); ++j) {
#else
        tbb::parallel_for(int(0), int(Q.cols()), [&](int j) {
#endif
            int cell = cells(j);
            auto q = Q.col(j);
            // started off in an invalid position
            // project it to a valid boundary point
            if (!is_active_cell(cell)) {
                spdlog::info("started out of domain (cell {})", j);
                q = _boundary_intersector.closest_boundary_point(q);
#if defined(NO_TBB_FOR)
                continue;
#else
                return;
#endif
            } else if (int qcell = get_cell(q); mesh().collision_free(cell) &&
                                                mesh().collision_free(qcell)) {
#if defined(NO_TBB_FOR)
                continue;
#else
                return;
#endif
            } else if (/*_boundary_intersector.has_open_boundaries() ||*/
                       !is_active_cell(qcell)) {
                auto p = P.col(j);
                if ((p - q).norm() < 1e-5) {
                    q = _boundary_intersector.closest_boundary_point(q);
                } else {
                    auto bisect = _boundary_intersector.raycast(p, q);
                    if (bisect) {
                        auto &plane = *bisect;
                        // spdlog::error("Boundary crossing seen; these
                        // should have opposite signs: {} {}",
                        // p.homogeneous().dot(plane),q.homogeneous().dot(plane));
                        plane /= plane.template head<3>().norm();
                        mtao::Vec3d protrusion =
                            q.homogeneous().dot(plane) *
                            plane.template head<3>();  // rem is now the
                                                       // orthogonal
                        // spdlog::error("Moved particle {}, without isect
                        // would have moved {}", 2*protrusion.norm(),
                        // (p-q).norm());
                        q = q - 2 * protrusion;

                        // q.setConstant(.4);
                    } else {
                        q = _boundary_intersector.closest_boundary_point(q);
                        // q = p; q.setConstant(.4);
                    }
                }
            }
#if defined(NO_TBB_FOR)
        }
#else
        });
#endif
        // std::cout << P << std::endl;
        // spdlog::info("Reflecting advection started with {} outside, ended
        // with {} outside  with {} interventions", int((cells.array() ==
        // -1).count()), int((get_cells(Q).array() == -1).count()), count);
    }

    const BoundaryIntersectionDetector3 &boundary_intersector() const {
        return _boundary_intersector;
    }
    const VEMMesh3 &mesh() const { return monomial_indexer().mesh(); }
    template <typename Derived, typename IDerived>
    mtao::ColVectors<double, D> advect_forward_euler(
        const Eigen::MatrixBase<Derived> &P,
        const Eigen::MatrixBase<IDerived> &cells, double dt) const {
        mtao::ColVectors<double, D> R = P + dt * get_vectors(P, cells);
        reflecting_advection(P, cells, R);
        return R;
    }
    template <typename Derived>
    mtao::ColVectors<double, D> advect_forward_euler(
        const Eigen::MatrixBase<Derived> &P, double dt) const {
        return advect_forward_euler(P, get_cells(P), dt);
    }
    template <typename Derived, typename IDerived>
    mtao::ColVectors<double, D> advect_rk2(
        const Eigen::MatrixBase<Derived> &P,
        const Eigen::MatrixBase<IDerived> &cells, double dt) const {
        auto NP = advect_forward_euler(P, cells, .5 * dt);
        // if (((NP - P).colwise().norm().array() > mesh().dx()).any()) {
        //    spdlog::error("first forward euler step too long in advect_rk2");
        //}
        auto vec = get_vectors(NP);
        // if (((dt * vec.colwise().norm()).array() > mesh().dx()).any()) {
        //    spdlog::error("Got long vectors in advect_rk2");
        //}
        mtao::ColVectors<double, D> V = P + dt * vec;
        mtao::ColVectors<double, D> V2 = V;
        reflecting_advection(P, cells, V);

        // if (((V - P).colwise().norm().array() > mesh().dx()).any()) {
        //    spdlog::error("final update was too long in advect_rk2");
        //}
        return V;
    }

    template <typename Derived>
    mtao::ColVectors<double, D> advect_rk2(const Eigen::MatrixBase<Derived> &P,
                                           double dt) const {
        spdlog::info("Rk2 getting cells");
        auto cells = get_cells(P);
        spdlog::info("Moving on");
        return advect_rk2(P, cells, dt);
    }

    template <typename Derived, typename Derived2, typename IDerived>
    mtao::ColVectors<double, D> advect_forward_euler_with_vel(
        const Eigen::MatrixBase<Derived> &P, Derived2 &V,
        const Eigen::MatrixBase<IDerived> &cells, double dt) const {
        mtao::ColVectors<double, D> R = P + dt * get_vectors(P, cells);
        reflecting_advection(P, V, cells, R);
        return R;
    }
    template <typename Derived, typename Derived2>
    mtao::ColVectors<double, D> advect_forward_euler_with_vel(
        const Eigen::MatrixBase<Derived> &P, Derived2 &V, double dt) const {
        return advect_forward_euler_with_vel(P, V, get_cells(P), dt);
    }
    template <typename Derived, typename Derived2, typename IDerived>
    mtao::ColVectors<double, D> advect_rk2_with_vel(
        const Eigen::MatrixBase<Derived> &P, Derived2 &Vel,
        const Eigen::MatrixBase<IDerived> &cells, double dt) const {
        auto NP = advect_forward_euler(P, cells, .5 * dt);
        // if (((NP - P).colwise().norm().array() > mesh().dx()).any()) {
        //    spdlog::error("first forward euler step too long in advect_rk2");
        //}
        auto vec = get_vectors(NP);
        // if (((dt * vec.colwise().norm()).array() > mesh().dx()).any()) {
        //    spdlog::error("Got long vectors in advect_rk2");
        //}
        mtao::ColVectors<double, D> V = P + dt * vec;
        mtao::ColVectors<double, D> V2 = V;
        reflecting_advection_with_vel(P, Vel, cells, V);

        // if (((V - P).colwise().norm().array() > mesh().dx()).any()) {
        //    spdlog::error("final update was too long in advect_rk2");
        //}
        return V;
    }

    template <typename Derived, typename Derived2>
    mtao::ColVectors<double, D> advect_rk2_with_vel(
        const Eigen::MatrixBase<Derived> &P, Derived2 &V, double dt) const {
        auto cells = get_cells(P);
        return advect_rk2_with_vel(P, V, cells, dt);
    }

    void set_active_cells(const std::set<int> &cells) {
        spdlog::info("Setting active cells in embedder {}", cells.size());
        active_cells = cells;
        _boundary_intersector.make_boundaries();
    }

   protected:
    bool is_valid_cell_index(int cell) const {
        return cell >= 0 || cell < _monomial_indexer.num_partitions();
    }
    bool is_active_cell(int cell) const {
        if (active_cells.empty()) {
            return is_valid_cell_index(cell);
        } else {
            return active_cells.contains(cell);
        }
    }

   public:
    // assumes you passed it a valid cell
    template <typename Derived, typename VDerived>
    double evaluate_monomial(const Eigen::MatrixBase<Derived> &p, int cell,
                             const Eigen::MatrixBase<VDerived> &coeffs) const {
        auto [start, end] = _monomial_indexer.coefficient_range(cell);
        int num_coeffs = end - start;
        auto coeff_block = coeffs.segment(start, num_coeffs);
        double ret = 0;
        for (int k = 0; k < num_coeffs; ++k) {
            ret += coeff_block(k) *
                   _monomial_indexer.evaluate_monomial(cell, k, p);
        }
        return ret;
    }
    template <typename Derived, typename VDerived>
    double evaluate_monomial(const Eigen::MatrixBase<Derived> &p,
                             const Eigen::MatrixBase<VDerived> &coeffs) const {
        int cell = get_cell_projected(p);
        return evaluate_monomial(p, cell, coeffs);
    }

   private:
    const MonomialBasisIndexer3 &_monomial_indexer;
    std::set<int> active_cells;
    BoundaryIntersectionDetector3 _boundary_intersector;
};

class MonomialScalarFieldEmbedder3
    : public MonomialFieldEmbedderBase3<MonomialScalarFieldEmbedder3> {
   public:
    using Base = MonomialFieldEmbedderBase3<MonomialScalarFieldEmbedder3>;
    using Base::get_value;
    MonomialScalarFieldEmbedder3(const MonomialBasisIndexer3 &mbi)
        : Base(mbi),
          _monomial_coefficients(mtao::VecXd::Zero(mbi.num_coefficients())) {}
    void set_coefficients(const mtao::VecXd &coeffs) {
        _monomial_coefficients = coeffs;
    }
    template <typename Derived>
    double get_value(const Eigen::MatrixBase<Derived> &p, int cell) const {
        if (!Base::is_active_cell(cell)) {
            return 0;
        }
        return Base::evaluate_monomial(p, cell, _monomial_coefficients);
    }

    const auto &coefficients() const { return _monomial_coefficients; }
    auto &coefficients() { return _monomial_coefficients; }

   private:
    mtao::VecXd _monomial_coefficients;
};

class MonomialVectorFieldEmbedder3
    : public MonomialFieldEmbedderBase3<MonomialVectorFieldEmbedder3> {
   public:
    using Base = MonomialFieldEmbedderBase3<MonomialVectorFieldEmbedder3>;
    using Base::get_vector;
    const static int D = 3;
    MonomialVectorFieldEmbedder3(const MonomialBasisIndexer3 &mbi)
        : Base(mbi),
          _monomial_coefficients(
              mtao::ColVectors<double, D>::Zero(D, mbi.num_coefficients())) {}
    void set_coefficients(const mtao::ColVectors<double, D> &coeffs) {
        _monomial_coefficients = coeffs;
    }
    template <typename Derived>
    mtao::Vector<double, D> get_vector(const Eigen::MatrixBase<Derived> &p,
                                       int cell) const {
        if (!Base::is_active_cell(cell)) {
            return mtao::Vector<double, D>::Zero();
        }
        return mtao::Vector<double, D>(
            Base::evaluate_monomial(p, cell, _monomial_coefficients.row(0)),
            Base::evaluate_monomial(p, cell, _monomial_coefficients.row(1)),
            Base::evaluate_monomial(p, cell, _monomial_coefficients.row(2)));
    }

    const auto &coefficients() const { return _monomial_coefficients; }
    auto &coefficients() { return _monomial_coefficients; }

   private:
    mtao::ColVectors<double, D> _monomial_coefficients;
};
}  // namespace vem
