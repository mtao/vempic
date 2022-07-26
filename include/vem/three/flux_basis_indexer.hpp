#pragma once

#include <mtao/eigen/shape_checks.hpp>

#include "vem/monomial_basis_indexer.hpp"
#include "vem/polynomial_utils.hpp"

namespace vem {
class FluxBasisIndexer3 : public detail::MonomialBasisIndexer<2, 3> {
   public:
    using Base = detail::MonomialBasisIndexer<2, 3>;
    using MonomialBasisIndexer::MonomialBasisIndexer;

    mtao::Matrix<double, 2, 4> point_to_st(int face_index) const {
        auto c = center(face_index);
        auto d = diameter(face_index);
        auto F = mesh().face_frames.at(face_index);

        mtao::Matrix<double, 2, 4> r;
        auto T = r.leftCols<3>() = F.transpose() / d;
        r.col(3) = -T * c;
        return r;
    }

    template <mtao::eigen::concepts::Vec2Compatible Vec>
    static double monomial_evaluation_local(int poly_index, const Vec& st) {
        // recall that st should come from a frame and therefore the diameter
        // already is part of the evaluation

        double r = 1;
        auto [a, b] = polynomials::two::index_to_exponents(poly_index);
        if (a > 0) {
            r = std::pow<double>(st.x(), a);
        }
        if (b > 0) {
            r *= std::pow<double>(st.y(), b);
        }
        return r;
    }

    template <mtao::eigen::concepts::Vec2Compatible Vec>
    static mtao::VecXd evaluate_monomials_by_size_local(int count,
                                                        const Vec& st) {
        mtao::VecXd R(count);
        for (int j = 0; j < count; ++j) {
            R(j) = monomial_evaluation_local(j, st);
        }
        return R;
    }

    template <mtao::eigen::concepts::Vec3Compatible Vec>
    double monomial_evaluation(int face_index, int poly_index,
                               const Vec& p) const {
        mtao::Vec2d st = point_to_st(face_index) * p.homogeneous();
        return monomial_evaluation_local(poly_index, st);
    }
    template <mtao::eigen::concepts::Vec3Compatible Vec>
    mtao::VecXd evaluate_monomials_by_size(int face_index, int count,
                                           const Vec& p) const {
        mtao::Vec2d st = point_to_st(face_index) * p.homogeneous();

        return evaluate_monomials_by_size_local(count, st);
    }
};
}  // namespace vem
