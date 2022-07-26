#pragma once

#include "vem/monomial_basis_indexer.hpp"

namespace vem {
class FluxBasisIndexer : public detail::MonomialBasisIndexer<1, 2> {
   public:
    using Base = detail::MonomialBasisIndexer<1, 2>;
    using MonomialBasisIndexer::MonomialBasisIndexer;

    mtao::Matrix<double, 1, 3> point_to_t(int edge_index) const {
        auto c = center(edge_index);
        auto d = diameter(edge_index);
        auto e = mesh().E.col(edge_index);
        auto a = mesh().V.col(e(0));
        auto b = mesh().V.col(e(1));

        mtao::Matrix<double, 1, 3> r;
        auto T = r.head<2>() = (b - a).normalized() / d;
        r(2) = -T.dot(c);
        return r;
    }

    double monomial_evaluation(int edge_index, int poly_index,
                               const mtao::Vec2d& p) const {
        double pos = point_to_t(edge_index).dot(p.homogeneous());
        return std::pow<double>(pos, poly_index);
    }
    mtao::VecXd evaluate_monomials_by_size(int edge_index, int count,
                               const mtao::Vec2d& p) const {

        mtao::VecXd R(count);
        double pos = point_to_t(edge_index).dot(p.homogeneous());
        for(int j = 0; j < count; ++j) {
        R(j) = std::pow<double>(pos, j);

        }
        return R;
    }
};
}  // namespace vem
