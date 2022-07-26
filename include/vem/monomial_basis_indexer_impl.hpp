#pragma once
#include "vem/monomial_basis_indexer.hpp"
#include "vem/polynomials/gradient.hpp"

namespace vem {

namespace {
    template<int D>
    std::vector<size_t> make_num_coeffs(const std::vector<size_t> &s) {
        for (auto &&v : s) {
            if (v == size_t(-1)) {
                v = 0;
            } else {
                if constexpr (D == 1) {
                    v = v + 1;
                } else if constexpr (D == 2) {
                    v = polynomials::two::num_monomials_upto(v);
                } else if constexpr (D == 3) {
                    v = polynomials::three::num_monomials_upto(v);
                }
            }
        }
        return s;
    }
}// namespace

namespace detail {
    template<int D, int E>
    MonomialBasisIndexer<D, E> MonomialBasisIndexer<D, E>::relative_degree_indexer(
      int degree_offset) const {
        std::vector<size_t> degrees = _degrees;
        std::transform(degrees.begin(), degrees.end(), degrees.begin(), [degree_offset](size_t v) -> size_t {
            bool val = int(v) >= -degree_offset;
            if (val) {
                double a = v + degree_offset;
                return a;
            } else {
                return -1;
            }
        });
        return MonomialBasisIndexer<D, E>(_mesh, std::move(degrees), _diameters);
    }
    template<int D, int E>
    MonomialBasisIndexer<D, E> MonomialBasisIndexer<D, E>::derivative_indexer()
      const {
        return relative_degree_indexer(-1);
    }
    template<int D, int E>
    MonomialBasisIndexer<D, E> MonomialBasisIndexer<D, E>::antiderivative_indexer()
      const {
        return relative_degree_indexer(1);
    }
    template<int D, int E>
    MonomialBasisIndexer<D, E>::MonomialBasisIndexer(const MeshType &mesh,
                                                     size_t max_degree)
      : PartitionedCoefficientIndexer(
        mesh.cell_count(),
        polynomials::num_monomials_upto<D>(max_degree)),
        _mesh(mesh) {
        if constexpr (D == E) {
            _degrees.resize(_mesh.cell_count(), max_degree);
        } else if constexpr (D == 1 && E == 2) {
            _degrees.resize(_mesh.edge_count(), max_degree);
        } else if constexpr (D == 2 && E == 3) {
            _degrees.resize(_mesh.face_count(), max_degree);
        }
        fill_diameters();
    }
    template<int D, int E>
    MonomialBasisIndexer<D, E>::MonomialBasisIndexer(
      const MeshType &mesh,
      std::vector<size_t> per_degrees,
      std::vector<double> diameters)
      : PartitionedCoefficientIndexer(make_num_coeffs<D>(per_degrees)),
        _mesh(mesh),
        _degrees(std::move(per_degrees)),
        _diameters(std::move(diameters)) {
        if (_diameters.empty()) {
            fill_diameters();
        }
        if (_diameters.size() != _degrees.size()) {
            // spdlog::warn(
            //    "Cell diameter list must match up with the number of "
            //    "polynomial "
            //    "degrees");
        }
    }

    template<int D, int E>
    double MonomialBasisIndexer<D, E>::diameter(size_t index) const {
        return _diameters.at(index);
    }
    template<int D, int E>
    size_t MonomialBasisIndexer<D, E>::coefficient_offset(size_t index) const {
        return PartitionedCoefficientIndexer::partition_offset(index);
    }
    template<int D, int E>
    size_t MonomialBasisIndexer<D, E>::num_monomials(size_t index) const {
        return PartitionedCoefficientIndexer::num_coefficients(index);
    }

    template<int D, int E>
    size_t MonomialBasisIndexer<D, E>::num_coefficients(size_t index) const {
        return num_monomials(index);
    }
    template<int D, int E>
    size_t MonomialBasisIndexer<D, E>::num_coefficients() const {
        return PartitionedCoefficientIndexer::num_coefficients();
    }

    template<int D, int E>
    size_t MonomialBasisIndexer<D, E>::degree(size_t index) const {
        return _degrees.at(index);
    }
    template<int D, int E>
    std::array<size_t, 2> MonomialBasisIndexer<D, E>::coefficient_range(
      size_t cell_index) const {
        return PartitionedCoefficientIndexer::coefficient_range(cell_index);
    }

    template<int D, int E>
    auto MonomialBasisIndexer<D, E>::monomial(size_t cell, size_t index) const
      -> std::function<double(const Vec &)> {
        if (index == 0) {
            return [](const Vec &p) -> double { return 1; };
        }
        using Vec = typename mtao::Vector<double, D>;
        Vec C = center(cell).eval();
        Vec exps = mtao::eigen::stl2eigen(polynomials::index_to_exponents<D>(index))
                     .template cast<double>();
        ;
        double diameter = _diameters.at(cell);
        return [exps, C, diameter](const Vec &p) -> double {
            return ((p - C) / diameter).array().pow(exps.array()).prod();
        };
    }
    template<int D, int E>
    auto MonomialBasisIndexer<D, E>::center(size_t index) const -> Vec {
        if constexpr (D == E) {
            return _mesh.C.col(index);
        } else if constexpr (D == 1) {
            auto e = _mesh.E.col(index);
            auto a = _mesh.V.col(e(0));
            auto b = _mesh.V.col(e(1));
            return (a + b) / 2;
        } else if constexpr (D == 2 && E == 3) {
            return _mesh.FC.col(index);
        }
    }

    template<int D, int E>
    void MonomialBasisIndexer<D, E>::fill_diameters() {
        static_assert(D == E);
        // spdlog::info("Filling diameter of a {} {}", D, E);
        _diameters.resize(_mesh.cell_count());
        // first compute the radii - i.e the furthest distance from teh center
        for (auto &&[idx, d] : mtao::iterator::enumerate(_diameters)) {
            d = _mesh.diameter(idx);
        }
    }
    template<int E, int D>
    Eigen::SparseMatrix<double> MonomialBasisIndexer<E, D>::gradient() const {
        size_t size = num_coefficients();
        Eigen::SparseMatrix<double> R(D * size, size);
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(size * E);

        size_t max_degree = *std::max_element(_degrees.begin(), _degrees.end());
        auto G = polynomials::gradients_as_tuples<D>(max_degree);

        for (size_t cell_index = 0; cell_index < _mesh.cell_count(); ++cell_index) {
            size_t num_mon = num_monomials(cell_index);
            size_t cell_offset = coefficient_offset(cell_index);
            double d = diameter(cell_index);
            for (size_t o = 0; o < num_mon; ++o) {
                for (auto &&[axis, pr] : mtao::iterator::enumerate(G.at(o))) {
                    auto &&[pval, pindex] = pr;
                    if (pindex >= 0 && pval != 0) {
                        trips.emplace_back(axis * size + cell_offset + pindex,
                                           cell_offset + o,
                                           pval / d);
                    }
                }
            }
        }
        R.setFromTriplets(trips.begin(), trips.end());
        return R;
    }

    template<int E, int D>
    Eigen::SparseMatrix<double> MonomialBasisIndexer<E, D>::divergence() const {
        size_t size = num_coefficients();
        Eigen::SparseMatrix<double> R(size, D * size);
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(size * D);

        size_t max_degree = *std::max_element(_degrees.begin(), _degrees.end());
        auto G = polynomials::gradients_as_tuples<D>(max_degree);

        for (size_t cell_index = 0; cell_index < _mesh.cell_count(); ++cell_index) {
            size_t num_mon = num_monomials(cell_index);
            size_t cell_offset = coefficient_offset(cell_index);
            double d = diameter(cell_index);
            for (size_t o = 0; o < num_mon; ++o) {
                for (auto &&[axis, pr] : mtao::iterator::enumerate(G.at(o))) {
                    auto &&[pval, pindex] = pr;
                    if (pindex >= 0 && pval != 0) {
                        trips.emplace_back(cell_offset + pindex,
                                           axis * size + cell_offset + o,
                                           pval / d);
                    }
                }
            }
        }
        R.setFromTriplets(trips.begin(), trips.end());
        return R;
    }

    template<int D, int E>
    mtao::VecXd MonomialBasisIndexer<D, E>::monomial_integrals(
      size_t cell_index) const {
        return monomial_integrals(cell_index, degree(cell_index));
    }
    template<int D, int E>
    mtao::VecXd MonomialBasisIndexer<D, E>::monomial_integrals(
      size_t cell_index,
      int max_degree) const {
        return scaled_monomial_cell_integrals(_mesh, cell_index, diameter(cell_index), max_degree);
    }

}// namespace detail
}// namespace vem
