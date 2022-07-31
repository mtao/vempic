#pragma once

#include <Eigen/Core>
#include <set>

#include "monomial_basis_indexer.hpp"
#include "boundary_intersector.hpp"
namespace vem::two {

template <typename Der>
class MonomialFieldEmbedderBase {
   public:
    const static int D = 2;
    Der &derived() { return *static_cast<Der *>(this); }
    const Der &derived() const { return *static_cast<const Der *>(this); }
    MonomialFieldEmbedderBase(const MonomialBasisIndexer &mbi)
        : _monomial_indexer(mbi),
          _boundary_intersector(mbi.mesh(), active_cells) {}

    template <typename Derived>
    mtao::Vector<double, D> get_vector(const Eigen::MatrixBase<Derived> &p,
                                       int cell) const {
        return derived().get_vector(p, cell);
    }
    template <typename Derived>
    mtao::Vector<double, D> get_vector(
        const Eigen::MatrixBase<Derived> &p) const {
        // int cell = get_cell(p);
        int cell = _boundary_intersector.get_projected_cell(p);
        return get_vector(p, cell);
    }
    template <typename Derived, typename IDerived>
    mtao::ColVectors<double, D> get_vectors(
        const Eigen::MatrixBase<Derived> &p,
        const Eigen::MatrixBase<IDerived> &cells) const {
        mtao::ColVectors<double, D> V(p.rows(), p.cols());
        for (int j = 0; j < p.cols(); ++j) {
            auto u = p.col(j);
            int cell = cells(j);
            V.col(j) = get_vector(u, cell);
        }
        return V;
    }
    template <typename Derived>
    mtao::ColVectors<double, D> get_vectors(
        const Eigen::MatrixBase<Derived> &p) const {
        mtao::VecXi cells(p.cols());
        for (int j = 0; j < p.cols(); ++j) {
            cells(j) = _boundary_intersector.get_projected_cell(p);
        }
        // cells = get_cells(p);
        return get_vectors(p, cells);
    }
    template <typename Derived>
    int get_cell(const Eigen::MatrixBase<Derived> &p) const {
        return _monomial_indexer.mesh().get_cell(p);
    }
    template <typename Derived>
    mtao::VecXi get_cells(const Eigen::MatrixBase<Derived> &p) const {
        mtao::VecXi indices(p.cols());
        for (int j = 0; j < indices.size(); ++j) {
            indices(j) = get_cell(p.col(j));
        }
        return indices;
    }
    const MonomialBasisIndexer &monomial_indexer() const {
        return _monomial_indexer;
    }

    template <typename Derived, typename IDerived, typename QDerived>
    void reflecting_advection(const Eigen::MatrixBase<Derived> &P,
                              const Eigen::MatrixBase<IDerived> &cells,
                              Eigen::PlainObjectBase<QDerived> &Q) const {
        int count = 0;
        double radius = 1e-3;
        for (int j = 0; j < Q.cols(); ++j) {
            int cell = cells(j);
            auto q = Q.col(j);
            if (!is_active_cell(cell)) {
                q = _boundary_intersector.closest_boundary_point(q);
                continue;
            } else {
                int qcell = get_cell(Q);
                if (true || !is_active_cell(qcell)) {
                    auto p = P.col(j);
                    if ((p - q).norm() < 1e-5) {
                        q = _boundary_intersector.closest_boundary_point(q);
                    } else {
                        // spdlog::info("Active to inactive crossing found");

                        // std::cout << p.transpose() << " to " << q.transpose()
                        //          << std::endl;
                        auto bisect = _boundary_intersector.raycast(
                            p, q, cell, qcell, radius);
                        if (bisect) {
                            auto [eidx, t] = *bisect;
                            // std::cout << "Isect on edge " << eidx << " at
                            // point "
                            //          << t << std::endl;
                            const auto &mesh = _monomial_indexer.mesh();
                            if (eidx < 0) {
                            } else {
                                auto e = mesh.E.col(eidx);
                                auto a = mesh.V.col(e(0));
                                auto b = mesh.V.col(e(1));
                                // std::cout << "Edge goes from " <<
                                // a.transpose()
                                //          << " to " << b.transpose() <<
                                //          std::endl;
                                auto T = (b - a).normalized().eval();
                                mtao::Vec2d isect_pt = (1 - t) * p + t * q;
                                mtao::Vec2d rem = q - isect_pt;
                                rem -= rem.dot(T) *
                                       T;  // rem is now the orthogonal
                                // thing that went into T

                                q = q - 2 * rem;
                            }

                            // q.setConstant(.4);
                        } else {
                            if (!is_active_cell(qcell)) {
                                q = _boundary_intersector
                                        .closest_boundary_point(q);
                            }
                            // q = p; q.setConstant(.4);
                        }
                    }
                }
            }
        }
        // std::cout << P << std::endl;
        // spdlog::info("Reflecting advection started with {} outside, ended
        // with {} outside  with {} interventions", int((cells.array() ==
        // -1).count()), int((get_cells(Q).array() == -1).count()), count);
    }

    const BoundaryIntersectionDetector &boundary_intersector() const {
        return _boundary_intersector;
    }
    const VEMMesh2 &mesh() const { return monomial_indexer().mesh(); }
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
        auto vec = get_vectors(NP);
        mtao::ColVectors<double, D> V = P + dt * vec;
        mtao::ColVectors<double, D> V2 = V;
        reflecting_advection(P, cells, V);
        return V;
    }
    template <typename Derived>
    mtao::ColVectors<double, D> advect_rk2(const Eigen::MatrixBase<Derived> &P,
                                           double dt) const {
        return advect_rk2(P, get_cells(P), dt);
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
    Eigen::Vector<double, D> evaluate_monomial_gradient(
        const Eigen::MatrixBase<Derived> &p, int cell,
        const Eigen::MatrixBase<Derived> &coeffs) const {
        auto [start, end] = _monomial_indexer.coefficient_range(cell);
        int num_coeffs = end - start;
        auto coeff_block = coeffs.segment(start, num_coeffs);

        mtao::Vector<double, D> ret = mtao::Vector<double, D>::Zero();
        for (int k = 0; k < num_coeffs; ++k) {
            ret += coeffs(k) * _monomial_indexer.evaluate_monomial(cell, k, p);
        }
        return ret;
    }

   private:
    const MonomialBasisIndexer &_monomial_indexer;
    std::set<int> active_cells;
    BoundaryIntersectionDetector _boundary_intersector;
};

class MonomialVectorFieldEmbedder
    : public MonomialFieldEmbedderBase<MonomialVectorFieldEmbedder> {
   public:
    using Base = MonomialFieldEmbedderBase<MonomialVectorFieldEmbedder>;
    using Base::get_vector;
    const static int D = 2;
    MonomialVectorFieldEmbedder(const MonomialBasisIndexer &mbi)
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
            Base::evaluate_monomial(p, cell, _monomial_coefficients.row(1)));
    }

    const auto &coefficients() const { return _monomial_coefficients; }
    auto &coefficients() { return _monomial_coefficients; }

   private:
    mtao::ColVectors<double, D> _monomial_coefficients;
};
class MonomialGradientFieldEmbedder
    : public MonomialFieldEmbedderBase<MonomialGradientFieldEmbedder> {
   public:
    using Base = MonomialFieldEmbedderBase<MonomialGradientFieldEmbedder>;
    using Base::get_vector;
    const static int D = 2;
    MonomialGradientFieldEmbedder(const MonomialBasisIndexer &mbi)
        : Base(mbi),
          _monomial_coefficients(mtao::VecXd::Zero(mbi.num_coefficients())) {}
    void set_coefficients(const mtao::VecXd &coeffs) {
        _monomial_coefficients = coeffs;
    }
    template <typename Derived>
    mtao::Vector<double, D> get_vector(const Eigen::MatrixBase<Derived> &p,
                                       int cell) const {
        if (!Base::is_active_cell(cell)) {
            return mtao::Vector<double, D>::Zero();
        }
        return Base::evaluate_monomial_gradient(p, cell,
                                                _monomial_coefficients);
    }

   private:
    mtao::VecXd _monomial_coefficients;
};

}  // namespace vem
