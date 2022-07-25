#include <spdlog/spdlog.h>

#include <mtao/algebra/pascal_triangle.hpp>
#include <vem/cell.hpp>

#include "vem/clang_hacks.hpp"
#include "vem/edge_lengths.hpp"
#include "vem/monomial_cell_integrals.hpp"
#include "vem/monomial_edge_integrals.hpp"
#include "vem/polynomial_gradient.hpp"
#include "vem/utils/volumes.hpp"

namespace vem {
using namespace polynomials::two;
VEM2Cell::VEM2Cell(const VEMMesh2& mesh, const size_t cell_index)
    : VEM2Cell(mesh, cell_index, mesh.diameter(cell_index)) {}
VEM2Cell::VEM2Cell(const VEMMesh2& mesh, const size_t cell_index,
                   double diameter)
    : _mesh(mesh), _cell_index(cell_index), _diameter(diameter) {}

const VEMMesh2& VEM2Cell::mesh() const { return _mesh; }
size_t VEM2Cell::cell_index() const { return _cell_index; }
double VEM2Cell::diameter() const { return _diameter; }
size_t VEM2Cell::edge_count() const {
    return _mesh.face_boundary_map.at(_cell_index).size();
}
std::set<size_t> VEM2Cell::vertices() const {
    std::set<size_t> ret;
    for (auto&& [eidx, sgn] : edges()) {
        auto e = _mesh.E.col(eidx);
        ret.emplace(e(0));
        ret.emplace(e(1));
    }
    return ret;
}
Eigen::AlignedBox<double, 2> VEM2Cell::bounding_box() const {
    Eigen::AlignedBox<double, 2> bbox;

    for (auto&& v : vertices()) {
        bbox.extend(_mesh.V.col(v));
    }
    return bbox;
}

size_t VEM2Cell::vertex_count() const { return vertices().size(); }
double VEM2Cell::boundary_area() const {
    double sum = 0;
    for (auto&& [a, b] : edge_lengths()) {
        sum += b;
    }
    return sum;
}
double VEM2Cell::volume() const { return utils::volume(_mesh, _cell_index); }
double VEM2Cell::edge_length(size_t edge_index) const {
    return vem::edge_length(_mesh, edge_index);
}
std::map<size_t, double> VEM2Cell::edge_lengths() const {
    std::map<size_t, double> ret;
    for (auto&& [eidx, sgn] : edges()) {
        ret[eidx] = edge_length(eidx);
    }
    return ret;
}

const std::map<int, bool>& VEM2Cell::edges() const {
    return _mesh.face_boundary_map.at(_cell_index);
}

std::map<size_t, std::array<double, 2>> VEM2Cell::edge_normals() const {
    std::map<size_t, std::array<double, 2>> ret;

    // std::ranges::copy(std::ranges::views::transform(
    //                      edges(), [&](const std::pair<size_t, bool>& pr) ->
    //                      std::pair<size_t,std::array<double,2>>{ auto&&
    //                      [idx,sgn] = pr; std::array<double,2> val;
    //
    //                      return {idx,val};

    //                      }),
    //                  std::inserter(ret, ret.end()));

    auto edges = this->edges();
    std::transform(edges.begin(), edges.end(), std::inserter(ret, ret.end()),
                   [&](const std::pair<size_t, bool>& pr)
                       -> std::pair<size_t, std::array<double, 2>> {
                       auto&& [idx, sgn] = pr;
                       std::array<double, 2> val;
                       auto e = _mesh.E.col(idx);
                       auto a = _mesh.V.col(e(0));
                       auto b = _mesh.V.col(e(1));
                       auto ab = (a - b).normalized().eval();
                       // auto ab = (b - a).normalized().eval();
                       val[0] = -ab.y();
                       val[1] = ab.x();

                       return {idx, val};
                   });
    return ret;
}
mtao::MatXd VEM2Cell::monomial_l2_edge_grammian(size_t edge_index,
                                                size_t row_degree,
                                                size_t col_degree) const {
    size_t row_size = polynomials::two::num_monomials_upto(row_degree);
    size_t col_size = polynomials::one::num_monomials_upto(col_degree);

    mtao::Vec2d edge_center = mesh().boundary_center(edge_index);
    double edge_diameter = mesh().boundary_diameter(edge_index);

    mtao::MatXd R(row_size, col_size);

    mtao::algebra::PascalTriangle pt(row_degree);
    // spdlog::info(
    //    "Computing edge l2 grammian for edge {} with edge deg {} and cell "
    //    "degree {}",
    //    edge_index, col_degree, row_degree);

    const double diam = diameter();
    const auto e = mesh().E.col(edge_index);
    const auto a = mesh().V.col(e(0));
    const auto b = mesh().V.col(e(1));
    if constexpr (false) {
        spdlog::info("Cell center is: {},{}", center().x(), center().y());
        spdlog::info("Raw Edge goes from {},{} to {},{}", a.x(), a.y(), b.x(),
                     b.y());
        spdlog::info("Diameter: {}, edge_diam: {}", diam, edge_diameter);
    }
    if constexpr (false) {
        auto a = (mesh().V.col(e(0)) - center()) / diam;
        auto b = (mesh().V.col(e(1)) - center()) / diam;
        spdlog::info("Edge goes from {},{} to {},{}", a.x(), a.y(), b.x(),
                     b.y());
    }

    mtao::Vec2d ba = b - a;
    mtao::Vec2d ac = a - center();
    // std::cout << "BA: " << ba.transpose() << std::endl;
    // std::cout << "AC: " << ac.transpose() << std::endl;
    R.setZero();
    R(0, 0) = 1;
    for (int j = 0; j < row_size; ++j) {
        auto [xj, yj] = index_to_exponents(j);
        // if ((xj == yj && xj == 1)) {
        //} else {
        //    continue;
        //}
        spdlog::debug("cell poly x^{} y^{}", xj, yj);
        double jterm = 1.0 / std::pow<double>(diam, (xj + yj));

        spdlog::debug("JTERM {} {} {} {} = {}", jterm, diam, xj, yj,
                      std::pow<double>(diam, xj + yj));
        for (int k = 0; k < col_size; ++k) {
            spdlog::debug("edge poly t^{}", k);
            double& v = R(j, k) = 0;
            // \sum_a \sum_b
            for (int aj = 0; aj <= xj; ++aj) {
                double ajterm = pt(xj, aj) * std::pow<double>(ac.x(), xj - aj) *
                                std::pow(ba.x(), aj);
                for (int bj = 0; bj <= yj; ++bj) {
                    double bjterm = pt(yj, bj) *
                                    std::pow<double>(ac.y(), yj - bj) *
                                    std::pow(ba.y(), bj);
                    double jprod = jterm * ajterm * bjterm;
                    spdlog::debug("Term x^{} y^{} got {} {} {} = {}", aj, bj,
                                  jterm, ajterm, bjterm, jprod);
                    double kterm =
                        ba.norm() / std::pow<double>(edge_diameter, k);
                    for (int dk = 0; dk <= k; ++dk) {
                        double dkterm = pt(k, dk) *
                                        std::pow<double>(-.5, k - dk) /
                                        (aj + bj + dk + 1);
                        double kprod = kterm * dkterm;

                        spdlog::debug("Term t^{} got {} {} = {}", dk, kterm,
                                      dkterm, kprod);

                        spdlog::debug("V += {}", jprod * kprod);
                        v += jprod * kprod;
                        /*
                        for (int ak = 0; ak <= dk; ++ak) {
                            int bk = dk - ak;

                            int idx = exponents_to_index(ak + xj, bk + yj);
                            mtao::Vec2d P(double(dk - ak), double(dk - bk));
                            double term =
                                pt(k, ak) * D.array().pow(P.array()).prod();
                            spdlog::debug(
                                "re-aclimating x^{} y^{}, got an exp of {} {} "
                                "over an "
                                "integral {}",
                                ak, bk, pt(k, ak),
                                D.array().pow(P.array()).prod(),
                                integrals(idx));
                            v += term * ratio * integrals(idx);
                        }
                        */
                    }
                }
            }
            spdlog::debug("R(x^{}y^{},t^{}) = {}", xj, yj, k, v);

            // std::cout << std::endl;
        }
    }
    // std::cout << R << std::endl;
    // std::cout << std::endl;
    return R;
}

mtao::MatXd VEM2Cell::monomial_l2_grammian(size_t monomial_degree) const {
    size_t monomial_size =
        polynomials::two::num_monomials_upto(monomial_degree);
    mtao::MatXd R(monomial_size, monomial_size);

    auto integrals = monomial_integrals(2 * monomial_degree);
    double diam = diameter();
    double diam2 = diam * diam;
    for (int j = 0; j < monomial_size; ++j) {
        for (int k = j; k < monomial_size; ++k) {
            auto [xj, yj] = index_to_exponents(j);
            auto [xk, yk] = index_to_exponents(k);
            double val = 0;
            int idx = exponents_to_index(xj + xk, yj + yk);
            if (idx < 0 || idx > integrals.size()) {
                continue;
            }
            R(k, j) = R(j, k) = integrals(idx);
        }
    }
    return R;
}
mtao::MatXd VEM2Cell::monomial_dirichlet_grammian(size_t max_degree) const {
    int monomial_size = num_monomials_upto(max_degree);
    mtao::MatXd R(monomial_size, monomial_size);

    auto integrals = monomial_integrals(2 * max_degree);
    // std::cout << "Integrals: " << integrals.transpose() << std::endl;

    double diam = diameter();
    double diam2 = diam * diam;
    for (int j = 0; j < monomial_size; ++j) {
        for (int k = j; k < monomial_size; ++k) {
            auto [xj, yj] = index_to_exponents(j);
            auto [xk, yk] = index_to_exponents(k);
            // spdlog::info("Computing <G x^{}y^{}, Gx^{}y^{}>", xj, yj, xk,
            // yk);
            auto g0 = gradient_single_index(j);
            auto g1 = gradient_single_index(k);
            double val = 0;
            for (auto&& [pr1, pr2] : mtao::iterator::zip(g0, g1)) {
                auto&& [c1, i1] = pr1;
                auto&& [c2, i2] = pr2;

                if (i1 < 0 || i2 < 0) {
                    continue;
                }
                auto [x1, y1] = index_to_exponents(i1);
                auto [x2, y2] = index_to_exponents(i2);
                int idx = exponents_to_index(x1 + x2, y1 + y2);
                if (idx < 0 || idx >= integrals.size()) {
                    continue;
                }
                val += c1 * c2 * integrals(idx) / diam2;
            }

            R(k, j) = R(j, k) = val;
            // spdlog::info("<G x^{}y^{}, Gx^{}y^{}> = {}", xj, yj, xk, yk,
            // val);
        }
    }
    return R;
}

Eigen::SparseMatrix<double> VEM2Cell::monomial_to_monomial_gradient(
    size_t max_degree) const {
    return polynomials::two::gradient(max_degree) / diameter();
}

mtao::VecXd VEM2Cell::monomial_integrals(size_t max_degree) const {
    return scaled_monomial_cell_integrals(_mesh, _cell_index, diameter(),
                                          max_degree);
}
mtao::VecXd VEM2Cell::monomial_edge_integrals(size_t max_degree,
                                              size_t edge_index) const {
    return mtao::eigen::stl2eigen(single_edge_scaled_monomial_edge_integrals(
        _mesh, _cell_index, edge_index, diameter(), max_degree));
}

mtao::VecXd VEM2Cell::monomial_boundary_integrals(size_t max_degree) const {
    return scaled_monomial_edge_integrals(_mesh, _cell_index, diameter(),
                                          max_degree);
}

std::function<double(const mtao::Vec2d&)> VEM2Cell::monomial(
    size_t index) const {
    if (index == 0) {
        return [](const mtao::Vec2d& p) -> double { return 1; };
    }
    auto C = center();
    double cx = C.x();
    double cy = C.y();
    // auto [xexp, yexp] = polynomials::two::index_to_exponents(index);
    size_t xexp, yexp;
    assign_array_to_tuple(polynomials::two::index_to_exponents(index),
                          std::tie(xexp, yexp));
    double diameter = _diameter;
    return [xexp, yexp, cx, cy, diameter](const mtao::Vec2d& p) -> double {
        double x = (p.x() - cx) / diameter;
        double y = (p.y() - cy) / diameter;

        double val = std::pow<double>(x, xexp) * std::pow<double>(y, yexp);
        // spdlog::info("{} = ({}-{})^{} ({} - {})^{}",
        // val,p.x(),cx,xexp,p.y(),cy,yexp);
        return val;
    };
}
std::function<mtao::Vec2d(const mtao::Vec2d&)> VEM2Cell::monomial_gradient(
    size_t index) const {
    auto C = center();
    double cx = C.x();
    double cy = C.y();
    // auto [xexp, yexp] = polynomials::two::index_to_exponents(index);
    auto [a, b] = polynomials::two::gradient_single_index(index);
    // auto&& [ac, ai] = a;
    // auto&& [bc, bi] = b;
    // auto [xexp1, yexp1] = polynomials::two::index_to_exponents(ai);
    // auto [xexp2, yexp2] = polynomials::two::index_to_exponents(bi);
    size_t xexp, yexp;
    assign_array_to_tuple(polynomials::two::index_to_exponents(index),
                          std::tie(xexp, yexp));
    double ac, bc;
    int ai, bi;
    assign_array_to_tuple(a, std::tie(ac, ai));
    assign_array_to_tuple(b, std::tie(bc, bi));

    size_t xexp1, yexp1;
    size_t xexp2, yexp2;
    assign_array_to_tuple(polynomials::two::index_to_exponents(ai),
                          std::tie(xexp1, yexp1));
    assign_array_to_tuple(polynomials::two::index_to_exponents(bi),
                          std::tie(xexp2, yexp2));

    // note that xexp2 is the original exp of index and yexp2 is the original
    // exp as well
    bool bad_x = xexp <= 0;
    bool bad_y = yexp <= 0;
    const double diameter = _diameter;
    if (bad_x && bad_y) {
        spdlog::trace("zero vector: 0,0");
        return [](const mtao::Vec2d& p) -> mtao::Vec2d {
            return mtao::Vec2d::Zero();
        };
    } else if (bad_x) {
        spdlog::trace("y_onl 0, {0}x^{1}y^{2}", yexp, xexp2, yexp2);
        return [bc, xexp2, yexp2, cx, cy,
                diameter](const mtao::Vec2d& p) -> mtao::Vec2d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            return mtao::Vec2d(0, bc * std::pow<double>(x, xexp2) *
                                      std::pow<double>(y, yexp2)) /
                   diameter;
        };
    } else if (bad_y) {
        spdlog::trace("x_only {0}x^{1}y^{2},0", xexp, xexp1, yexp1);
        return [ac, xexp1, yexp1, cx, cy,
                diameter](const mtao::Vec2d& p) -> mtao::Vec2d {
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
                diameter](const mtao::Vec2d& p) -> mtao::Vec2d {
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

}  // namespace vem
