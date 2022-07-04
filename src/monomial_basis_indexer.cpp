#include "vem/monomial_basis_indexer.hpp"

#include <spdlog/spdlog.h>

#include <mtao/iterator/enumerate.hpp>
#include <mtao/quadrature/gauss_lobatto.hpp>
#include <set>
#include <vector>

#include "vem/monomial_cell_integrals.hpp"
#include "vem/monomial_edge_integrals.hpp"
#include "vem/polynomial_gradient.hpp"
#include "vem/polynomial_utils.hpp"

namespace vem {
namespace {
std::vector<size_t> make_num_coeffs(std::vector<size_t> s) {
    for (auto &&v : s) {
        if (v == size_t(-1)) {
            v = 0;
        } else {
            v = polynomials::two::num_monomials_upto(v);
        }
    }
    return s;
}
}  // namespace

MonomialBasisIndexer MonomialBasisIndexer::relative_degree_indexer(
    int degree_offset) const {
    std::vector<size_t> degrees = _cell_degrees;
    std::transform(degrees.begin(), degrees.end(), degrees.begin(),
                   [degree_offset](size_t v) -> size_t {
                       bool val = int(v) >= -degree_offset;
                       if (val) {
                           double a = v + degree_offset;
                           return a;
                       } else {
                           return -1;
                       }
                   });
    return MonomialBasisIndexer(_mesh, std::move(degrees), _cell_diameters);
}
MonomialBasisIndexer MonomialBasisIndexer::derivative_indexer() const {
    return relative_degree_indexer(-1);
}
MonomialBasisIndexer MonomialBasisIndexer::antiderivative_indexer() const {
    return relative_degree_indexer(1);
}
MonomialBasisIndexer::MonomialBasisIndexer(const VEMMesh2 &mesh,
                                           size_t max_degree)
    : PartitionedCoefficientIndexer(
          mesh.cell_count(), polynomials::two::num_monomials_upto(max_degree)),
      _mesh(mesh) {
    _cell_degrees.resize(_mesh.cell_count(), max_degree);
    fill_diameters();
}
MonomialBasisIndexer::MonomialBasisIndexer(const VEMMesh2 &mesh,
                                           std::vector<size_t> per_cell_degrees,
                                           std::vector<double> diameters)
    : PartitionedCoefficientIndexer(make_num_coeffs(per_cell_degrees)),
      _mesh(mesh),
      _cell_degrees(std::move(per_cell_degrees)),
      _cell_diameters(std::move(diameters)) {
    if (_cell_degrees.size() != _mesh.cell_count()) {
        spdlog::warn(
            "Internal cell sizes must match up with the number of cells");
    }
    if (_cell_diameters.empty()) {
        fill_diameters();
    }
    if (_cell_diameters.size() != _mesh.cell_count()) {
        spdlog::warn(
            "Cell diameter list must match up with the number of cells");
    }
}

double MonomialBasisIndexer::diameter(size_t index) const {
    return _cell_diameters.at(index);
}
void MonomialBasisIndexer::fill_diameters() {
    _cell_diameters.resize(_mesh.cell_count());
    // first compute the radii - i.e the furthest distance from teh center
    for (auto &&[idx, d, fbm] :
         mtao::iterator::enumerate(_cell_diameters, _mesh.face_boundary_map)) {
        std::set<size_t> verts;
        for (auto &&[eidx, sgn] : fbm) {
            auto e = _mesh.E.col(eidx);
            verts.emplace(e(0));
            verts.emplace(e(1));
        }
        d = 0;
        auto c = _mesh.C.col(idx);
        // we're doing things twice, which is stupid. but who cares
        for (auto &&v : verts) {
            for (auto &&u : verts) {
                d = std::max(d, (_mesh.V.col(v) - _mesh.V.col(u)).norm());
            }
        }
    }
}

size_t MonomialBasisIndexer::coefficient_offset(size_t index) const {
    return PartitionedCoefficientIndexer::partition_offset(index);
}
size_t MonomialBasisIndexer::num_monomials(size_t index) const {
    return PartitionedCoefficientIndexer::num_coefficients(index);
}

size_t MonomialBasisIndexer::num_coefficients(size_t index) const {
    return num_monomials(index);
}

size_t MonomialBasisIndexer::num_coefficients() const {
    return PartitionedCoefficientIndexer::num_coefficients();
}

size_t MonomialBasisIndexer::degree(size_t index) const {
    return _cell_degrees.at(index);
}
std::array<size_t, 2> MonomialBasisIndexer::coefficient_range(
    size_t cell_index) const {
    return PartitionedCoefficientIndexer::coefficient_range(cell_index);
}

std::function<double(const mtao::Vec2d &)> MonomialBasisIndexer::monomial(
    size_t cell, size_t index) const {
    if (index == 0) {
        return [](const mtao::Vec2d &p) -> double { return 1; };
    }
    auto C = _mesh.C.col(cell);
    double cx = C.x();
    double cy = C.y();
    auto [xexp, yexp] = polynomials::two::index_to_exponents(index);
    double diameter = _cell_diameters.at(cell);
    return [xexp, yexp, cx, cy, diameter](const mtao::Vec2d &p) -> double {
        double x = (p.x() - cx) / diameter;
        double y = (p.y() - cy) / diameter;

        double val = std::pow<double>(x, xexp) * std::pow<double>(y, yexp);
        // spdlog::info("{} = ({}-{})^{} ({} - {})^{}",
        // val,p.x(),cx,xexp,p.y(),cy,yexp);
        return val;
    };
}
std::function<mtao::Vec2d(const mtao::Vec2d &)>
MonomialBasisIndexer::monomial_gradient(size_t cell, size_t index) const {
    auto C = _mesh.C.col(cell);
    double cx = C.x();
    double cy = C.y();
    auto [xexp, yexp] = polynomials::two::index_to_exponents(index);
    auto [a, b] = polynomials::two::gradient_single_index(index);
    auto &&[ac, ai] = a;
    auto &&[bc, bi] = b;
    auto [xexp1, yexp1] = polynomials::two::index_to_exponents(ai);
    auto [xexp2, yexp2] = polynomials::two::index_to_exponents(bi);
    // note that xexp2 is the original exp of index and yexp2 is the original
    // exp as well
    bool bad_x = xexp <= 0;
    bool bad_y = yexp <= 0;
    const double diameter = _cell_diameters.at(cell);
    if (bad_x && bad_y) {
        spdlog::trace("zero vector: 0,0");
        return [](const mtao::Vec2d &p) -> mtao::Vec2d {
            return mtao::Vec2d::Zero();
        };
    } else if (bad_x) {
        spdlog::trace("y_onl 0, {0}x^{1}y^{2}", yexp, xexp2, yexp2);
        return [bc, xexp2, yexp2, cx, cy,
                diameter](const mtao::Vec2d &p) -> mtao::Vec2d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            return mtao::Vec2d(0, bc * std::pow<double>(x, xexp2) *
                                      std::pow<double>(y, yexp2)) /
                   diameter;
        };
    } else if (bad_y) {
        spdlog::trace("x_only {0}x^{1}y^{2},0", xexp, xexp1, yexp1);
        return [ac, xexp1, yexp1, cx, cy,
                diameter](const mtao::Vec2d &p) -> mtao::Vec2d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            return mtao::Vec2d(ac * std::pow<double>(x, xexp1) *
                                   std::pow<double>(y, yexp1),
                               0) /
                   diameter;
        };
    } else {
        spdlog::trace("{2}x^{0}y^{1} + {1}x^{2}y^{3}", xexp1, yexp1, xexp2,
                      yexp2);
        return [ac, bc, xexp1, yexp1, xexp2, yexp2, cx, cy,
                diameter](const mtao::Vec2d &p) -> mtao::Vec2d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            return mtao::Vec2d(ac * std::pow<double>(x, xexp1) *
                                   std::pow<double>(y, yexp1),
                               bc * std::pow<double>(x, xexp2) *
                                   std::pow<double>(y, yexp2)) /
                   diameter;
        };
    }
}
mtao::VecXd MonomialBasisIndexer::monomial_integrals(size_t cell_index) const {
    return monomial_integrals(cell_index, _cell_degrees.at(cell_index));
}
mtao::VecXd MonomialBasisIndexer::monomial_integrals(size_t cell_index,
                                                     int max_degree) const {
    return scaled_monomial_cell_integrals(
        _mesh, cell_index, _cell_diameters.at(cell_index), max_degree);
}
mtao::VecXd MonomialBasisIndexer::monomial_edge_integrals(
    size_t cell_index) const {
    return scaled_monomial_edge_integrals(_mesh, cell_index,
                                          _cell_diameters.at(cell_index),
                                          _cell_degrees.at(cell_index));
}
Eigen::SparseMatrix<double> MonomialBasisIndexer::gradient() const {
    size_t size = num_coefficients();
    Eigen::SparseMatrix<double> R(2 * size, size);
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(size * 2);

    size_t max_degree =
        *std::max_element(_cell_degrees.begin(), _cell_degrees.end());
    auto G = polynomials::two::gradients_as_tuples(max_degree);

    for (size_t cell_index = 0; cell_index < _mesh.cell_count(); ++cell_index) {
        size_t num_mon = num_monomials(cell_index);
        size_t cell_offset = coefficient_offset(cell_index);
        double d = diameter(cell_index);
        for (size_t o = 0; o < num_mon; ++o) {
            for (auto &&[axis, pr] : mtao::iterator::enumerate(G.at(o))) {
                auto &&[pval, pindex] = pr;
                if (pindex >= 0 && pval != 0) {
                    trips.emplace_back(axis * size + cell_offset + pindex,
                                       cell_offset + o, pval / d);
                }
            }
        }
    }
    R.setFromTriplets(trips.begin(), trips.end());
    return R;
}

Eigen::SparseMatrix<double> MonomialBasisIndexer::divergence() const {
    size_t size = num_coefficients();
    Eigen::SparseMatrix<double> R(size, 2 * size);
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(size * 2);

    size_t max_degree =
        *std::max_element(_cell_degrees.begin(), _cell_degrees.end());
    auto G = polynomials::two::gradients_as_tuples(max_degree);

    for (size_t cell_index = 0; cell_index < _mesh.cell_count(); ++cell_index) {
        size_t num_mon = num_monomials(cell_index);
        size_t cell_offset = coefficient_offset(cell_index);
        double d = diameter(cell_index);
        for (size_t o = 0; o < num_mon; ++o) {
            for (auto &&[axis, pr] : mtao::iterator::enumerate(G.at(o))) {
                auto &&[pval, pindex] = pr;
                if (pindex >= 0 && pval != 0) {
                    trips.emplace_back(cell_offset + pindex,
                                       axis * size + cell_offset + o, pval / d);
                }
            }
        }
    }
    R.setFromTriplets(trips.begin(), trips.end());
    return R;
}

}  // namespace vem
