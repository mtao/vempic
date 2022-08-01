
#include <spdlog/spdlog.h>

#include <mtao/algebra/pascal_triangle.hpp>
#include <vem/three/cell.hpp>
#include <vem/three/cell_boundary_facets.hpp>

#include "vem/utils/clang_hacks.hpp"
#include "vem/three/monomial_cell_integrals.hpp"
#include "vem/three/monomial_face_integrals.hpp"
#include "vem/polynomials/gradient.hpp"
#include "vem/polynomials/utils.hpp"
#include "vem/three/volumes.hpp"

namespace vem::three {
using namespace polynomials::three;
VEM3Cell::VEM3Cell(const VEMMesh3& mesh, const size_t cell_index)
    : VEM3Cell(mesh, cell_index, mesh.diameter(cell_index)) {}
VEM3Cell::VEM3Cell(const VEMMesh3& mesh, const size_t cell_index,
                   double diameter)
    : _mesh(mesh), _cell_index(cell_index), _diameter(diameter) {}

const VEMMesh3& VEM3Cell::mesh() const { return _mesh; }
size_t VEM3Cell::cell_index() const { return _cell_index; }
double VEM3Cell::diameter() const { return _diameter; }
size_t VEM3Cell::face_count() const { return faces().size(); }

std::optional<int> VEM3Cell::cell_category() const {
    return mesh().cell_category(cell_index());
}
std::set<size_t> VEM3Cell::vertices() const {
    return cell_boundary_vertices(_mesh, cell_index());
}
Eigen::AlignedBox<double, 3> VEM3Cell::bounding_box() const {
    Eigen::AlignedBox<double, 3> bbox;

    for (auto&& v : vertices()) {
        bbox.extend(_mesh.V.col(v));
    }
    return bbox;
}

// size_t VEM3Cell::vertex_count() const { return vertices().size(); }
double VEM3Cell::boundary_area() const {
    double v = 0;
    for (auto&& [fidx, sgn] : faces()) {
        v += surface_area(fidx);
    }
    return v;
}
double VEM3Cell::volume() const { return three::volume(_mesh, _cell_index); }
double VEM3Cell::surface_area(size_t face_index) const {
    return mesh().surface_area(face_index);
}
std::map<size_t, double> VEM3Cell::surface_areas() const {
    std::map<size_t, double> ret;
    for (auto&& [fidx, sgn] : faces()) {
        ret[fidx] = surface_area(fidx);
    }
    return ret;
}

const std::map<int, bool>& VEM3Cell::faces() const {
    return _mesh.cell_boundary_map.at(_cell_index);
}

mtao::Vec3d VEM3Cell::face_normal(size_t face_index) const {
    return (faces().at(face_index) ? -1 : 1) * _mesh.normal(face_index);
}

const mtao::Matrix<double, 3, 2>& VEM3Cell::face_frame(
    size_t face_index) const {
    return mesh().face_frames.at(face_index);
}
mtao::Vec3d VEM3Cell::face_center(size_t face_index) const {
    return mesh().FC.col(face_index);
}

std::map<size_t, std::array<double, 3>> VEM3Cell::face_normals() const {
    std::map<size_t, std::array<double, 3>> ret;

    // std::ranges::copy(std::ranges::views::transform(
    //                      edges(), [&](const std::pair<size_t, bool>& pr)
    //                      -> std::pair<size_t,std::array<double,3>>{
    //                      auto&& [idx,sgn] = pr; std::array<double,3> val;
    //
    //                      return {idx,val};

    //                      }),
    //                  std::inserter(ret, ret.end()));

    const auto& Fs = faces();
    std::transform(Fs.begin(), Fs.end(), std::inserter(ret, ret.end()),
                   [&](const std::pair<size_t, bool>& pr)
                       -> std::pair<size_t, std::array<double, 3>> {
                       auto&& [idx, sgn] = pr;

                       std::array<double, 3> r;
                       mtao::eigen::stl2eigen(r) = face_normal(idx);
                       return {idx, r};
                   });
    return ret;
}

mtao::MatXd VEM3Cell::monomial_l2_grammian(size_t monomial_degree) const {
    size_t monomial_size =
        polynomials::three::num_monomials_upto(monomial_degree);
    mtao::MatXd R(monomial_size, monomial_size);

    auto integrals = monomial_integrals(2 * monomial_degree);
    double diam = diameter();
    double diam3 = diam * diam;
    for (int j = 0; j < monomial_size; ++j) {
        for (int k = j; k < monomial_size; ++k) {
            auto [xj, yj, zj] = index_to_exponents(j);
            auto [xk, yk, zk] = index_to_exponents(k);
            double val = 0;
            int idx = exponents_to_index(xj + xk, yj + yk, zj + zk);
            if (idx < 0 || idx > integrals.size()) {
                continue;
            }
            R(k, j) = R(j, k) = integrals(idx);
        }
    }
    return R;
}
mtao::VecXd VEM3Cell::face_monomial_integrals(size_t face_index,
                                              size_t max_degree) const {
    return monomial_face_integrals(max_degree, face_index);
}
mtao::MatXd VEM3Cell::monomial_l2_face_grammian(size_t face_index,
                                                size_t row_degree,
                                                size_t col_degree) const {
    size_t row_size = polynomials::three::num_monomials_upto(row_degree);
    size_t col_size = polynomials::two::num_monomials_upto(col_degree);

    // spdlog::info("L2 grammian face size: {}x{} (from degrees {} {})",
    // row_size,
    //             col_size, row_degree, col_degree);
    mtao::MatXd R(row_size, col_size);
    R.setZero();

    auto face_center = mesh().FC.col(face_index);
    // std::cout << "Face center:\n" << face_center.transpose() << std::endl;
    double face_diameter = mesh().face_diameter(face_index);
    // std::cout << "Face diameter:\n" << face_diameter << std::endl;

    auto integrals =
        monomial_face_integrals(row_degree + col_degree, face_index);
    // std::vector<double> p(integrals.size());
    // mtao::eigen::stl2eigen(p) = integrals;
    // spdlog::info("Integrals for face {}: {}", face_index, fmt::join(p, ","));
    for (int j = 0; j < row_size; ++j) {
        auto coeffs = project_monomial_to_boundary(face_index, j);
        for (int k = 0; k < col_size; ++k) {
            auto [xk, yk] = polynomials::two::index_to_exponents(k);
            // std::cout << "XI Got coeffs for " << xi <<": " <<
            // coeffs.transpose() << std::endl;
            // spdlog::info("Entry {} {}", j, k);

            double& val = R(j, k) = 0;
            for (int l = 0; l < coeffs.size(); ++l) {
                auto [xl, yl] = polynomials::two::index_to_exponents(l);
                int idx =
                    polynomials::two::exponents_to_index(xl + xk, yl + yk);
                /// spdlog::info("{} {} => {}", xl + xk, yl + yk,
                /// integrals(idx));
                val += coeffs(l) * integrals(idx);
            }
            // spdlog::info(
        }
    }
    // std::cout << R.rows() << " " << R.cols() << std::endl;
    // std::cout << R << std::endl;
    return R;
}
mtao::MatXd VEM3Cell::monomial_dirichlet_grammian(size_t max_degree) const {
    int monomial_size = num_monomials_upto(max_degree);
    mtao::MatXd R(monomial_size, monomial_size);

    auto integrals = monomial_integrals(2 * max_degree);

    double diam = diameter();
    double diam3 = diam * diam;
    for (int j = 0; j < monomial_size; ++j) {
        for (int k = j; k < monomial_size; ++k) {
            auto [xj, yj, zj] = index_to_exponents(j);
            auto [xk, yk, zk] = index_to_exponents(k);
            // spdlog::info("Computing <G x^{}y^{}, Gx^{}y^{}>", xj, yj, xk,
            // yk);
            auto g0 = gradient_single_index(j);
            auto g1 = gradient_single_index(k);
            double val = 0;
            for (auto&& [pr1, pr3] : mtao::iterator::zip(g0, g1)) {
                auto&& [c1, i1] = pr1;
                auto&& [c3, i3] = pr3;

                if (i1 < 0 || i3 < 0) {
                    continue;
                }
                auto [x1, y1, z1] = index_to_exponents(i1);
                auto [x3, y3, z3] = index_to_exponents(i3);
                int idx = exponents_to_index(x1 + x3, y1 + y3, z1 + z3);
                if (idx < 0 || idx > integrals.size()) {
                    continue;
                }
                val += c1 * c3 * integrals(idx) / diam3;
            }

            R(k, j) = R(j, k) = val;
            // spdlog::info("<G x^{}y^{}, Gx^{}y^{}> = {}", xj, yj, xk, yk,
            // val);
        }
    }
    return R;
}

Eigen::SparseMatrix<double> VEM3Cell::monomial_to_monomial_gradient(
    size_t max_degree) const {
    return polynomials::three::gradient(max_degree) / diameter();
}

mtao::VecXd VEM3Cell::monomial_integrals(size_t max_degree) const {
    return scaled_monomial_cell_integrals(_mesh, _cell_index, diameter(),
                                          max_degree);
}
mtao::VecXd VEM3Cell::monomial_face_integrals(size_t max_degree,
                                              size_t face_index) const {
    return mtao::eigen::stl2eigen(scaled_face_monomial_face_integrals(
        _mesh, face_index, mesh().face_diameter(face_index), max_degree));
}

mtao::VecXd VEM3Cell::monomial_boundary_integrals(size_t max_degree) const {
    mtao::VecXd R(polynomials::three::num_monomials_upto(max_degree));
    R.setZero();
    for (auto&& [f, sgn] : faces()) {
        R += monomial_face_integrals(max_degree, f);
    }
    return R;
}

std::function<double(const mtao::Vec3d&)> VEM3Cell::monomial(
    size_t index) const {
    if (index == 0) {
        return [](const mtao::Vec3d& p) -> double { return 1; };
    }
    auto C = center();
    double cx = C.x();
    double cy = C.y();
    double cz = C.z();
    //auto [xexp, yexp, zexp] = polynomials::three::index_to_exponents(index);
    size_t xexp, yexp, zexp;
    utils::assign_array_to_tuple(polynomials::three::index_to_exponents(index),
                          std::tie(xexp, yexp, zexp));

    double diameter = _diameter;
    return [xexp, yexp, zexp, cx, cy, cz,
            diameter](const mtao::Vec3d& p) -> double {
        double x = (p.x() - cx) / diameter;
        double y = (p.y() - cy) / diameter;
        double z = (p.z() - cz) / diameter;

        double val = std::pow<double>(x, xexp) * std::pow<double>(y, yexp) *
                     std::pow<double>(z, zexp);
        // spdlog::info("{} = ({}-{})^{} ({} - {})^{}",
        // val,p.x(),cx,xexp,p.y(),cy,yexp);
        return val;
    };
}

mtao::VecXd VEM3Cell::project_monomial_to_boundary(
    size_t face_index, size_t cell_monomial_index) const {
    //auto [xexp, yexp, zexp] = index_to_exponents(cell_monomial_index);
    size_t xexp, yexp, zexp;
    utils::assign_array_to_tuple(index_to_exponents(cell_monomial_index),
                          std::tie(xexp, yexp, zexp));

    mtao::algebra::PascalTriangle pt(std::max({xexp, yexp, zexp}));

    // i want to convert something of the form
    // (x - c_x) ^ xexp (y - c_y)  ^ yexp (z - c_z) ^ zexp / diameter^{xexp +
    // yexp + zexp} to \sum_{uexp=0}^{xexp + yexp + zexp} \sum_{vexp=0}^{vexp}
    // coeff_i ((u - C_u)^uexp (v - C_v)^vexp /D^{uexp + vexp})
    //
    // Let st = (uv - C)/D
    // so uv = Dst  + C
    //
    // (p - c) ^ exp  / diameter^{exp}
    // (P uv + C_{uv}) - c) ^ exp  / diameter^{exp}
    // Assume C = 0
    // (P (Dst + C) + C_{uv}) - c) ^ exp  / diameter^{exp}
    // (PDst + (PC + C_{uv} - c)) ^ exp  / diameter^{exp}
    // \sum_{j=0}^exp  (exp choose j) (PC + aC_{uv} - c)^{exp-j} (PDst)^j /
    // diameter^exp
    //
    // A = (PC + C_{uv} - c)
    // B = PD
    //
    // \sum_{j=0}^exp  (exp choose j) (A)^{exp-j} (Bst)^j /
    // diameter^exp

    const auto& P = mesh().face_frames.at(face_index);
    size_t max_degree = xexp + yexp + zexp;
    mtao::VecXd coeffs(polynomials::two::num_monomials_upto(max_degree));
    coeffs.setZero();
    // spdlog::info(
    //    "Projecting monomial to boundary for cell poly {} (x^{} y^{}, which "
    //    "should have {} edge coefficients",
    //    cell_monomial_index, xexp, yexp, coeffs.size());

    double face_diameter = mesh().face_diameter(face_index);
    auto C = mesh().FC.col(face_index);
    mtao::Vec3d A = C - center();
    auto B = P * face_diameter;

    // spdlog::warn("Projecting x^{} y^{} z^{} ; max_deg = {}", xexp, yexp,
    // zexp,
    //             max_degree);
    mtao::Vec3d mexp = mtao::Vec3d(double(xexp), double(yexp), double(zexp));
    // if (beware_issues)
    for (size_t j = 0; j <= xexp; ++j) {
        double xterm = pt(xexp, j);
        for (size_t k = 0; k <= yexp; ++k) {
            double yterm = pt(yexp, k);
            for (size_t l = 0; l <= zexp; ++l) {
                double zterm = pt(zexp, l);
                size_t degree = j + k + l;
                mtao::Vec3d lexp = mtao::Vec3d(double(j), double(k), double(l));
                double term = xterm * yterm * zterm *
                              A.array().pow((mexp - lexp).array()).prod();
                // spdlog::info("Term for x^{} y^{} z^{} = {} {} {} {} = {}", j,
                // k,
                //             l, xterm, yterm, zterm,
                //             A.array().pow((mexp - lexp).array()).prod(),
                //             term);
                for (size_t a = 0; a <= j; ++a) {
                    auto arow = B.row(0);
                    double aterm = pt(j, a) * std::pow<double>(arow.x(), a) *
                                   std::pow<double>(arow.y(), j - a);
                    for (size_t b = 0; b <= k; ++b) {
                        auto brow = B.row(1);
                        double bterm = pt(k, b) *
                                       std::pow<double>(brow.x(), b) *
                                       std::pow<double>(brow.y(), k - b);
                        for (size_t c = 0; c <= l; ++c) {
                            auto crow = B.row(2);
                            double cterm = pt(l, c) *
                                           std::pow<double>(crow.x(), c) *
                                           std::pow<double>(crow.y(), l - c);
                            size_t uexp = a + b + c;
                            size_t vexp = degree - uexp;
                            double& val =
                                coeffs(polynomials::two::exponents_to_index(
                                    uexp, vexp));
                            val += term * aterm * bterm * cterm;
                        }
                    }
                }
            }
        }
    }
    coeffs /= std::pow<double>(diameter(), xexp + yexp + zexp);
    return coeffs;
}

std::function<mtao::Vec3d(const mtao::Vec3d&)> VEM3Cell::monomial_gradient(
    size_t index) const {
    auto C = center();
    double cx = C.x();
    double cy = C.y();
    double cz = C.z();

    //auto [xexp, yexp, zexp] = polynomials::three::index_to_exponents(index);
    size_t xexp, yexp, zexp;
    // double a, b, c;
    utils::assign_array_to_tuple(polynomials::three::index_to_exponents(index),
                          std::tie(xexp, yexp, zexp));

    auto [a, b, c] = polynomials::three::gradient_single_index(index);
    //auto&& [ac, ai] = a;
    //auto&& [bc, bi] = b;
    //auto&& [cc, ci] = c;
    //auto [xexp1, yexp1, zexp1] = polynomials::three::index_to_exponents(ai);
    //auto [xexp2, yexp2, zexp2] = polynomials::three::index_to_exponents(bi);
    //auto [xexp3, yexp3, zexp3] = polynomials::three::index_to_exponents(ci);
    double ac, bc, cc;
    int ai, bi, ci;
    // std::tie(ac,ai) = a;
    // std::tie(bc,bi) = b;
    // std::tie(cc,ci) = c;
    utils::assign_array_to_tuple(a, std::tie(ac, ai));
    utils::assign_array_to_tuple(b, std::tie(bc, bi));
    utils::assign_array_to_tuple(c, std::tie(cc, ci));

    size_t xexp1, yexp1, zexp1;
    size_t xexp2, yexp2, zexp2;
    size_t xexp3, yexp3, zexp3;
    utils::assign_array_to_tuple(polynomials::three::index_to_exponents(ai),
                          std::tie(xexp1, yexp1, zexp1));
    utils::assign_array_to_tuple(polynomials::three::index_to_exponents(bi),
                          std::tie(xexp2, yexp2, zexp2));
    utils::assign_array_to_tuple(polynomials::three::index_to_exponents(ci),
                          std::tie(xexp3, yexp3, zexp3));

    // note that xexp2 is the original exp of index and yexp2 is the original
    // exp as well
    bool bad_x = xexp <= 0;
    bool bad_y = yexp <= 0;
    bool bad_z = zexp <= 0;
    const double diameter = _diameter;
    if (bad_x && bad_y && bad_z) {
        spdlog::trace("zero vector: 0,0");
        return [](const mtao::Vec3d& p) -> mtao::Vec3d {
            return mtao::Vec3d::Zero();
        };
    } else if (bad_y && bad_z) {
        spdlog::trace("x_onl 0, {0}x^{1}y^{2}^{3}", xexp, xexp1, yexp1, zexp1);
        return [ac, xexp1, yexp1, zexp1, cx, cy, cz,
                diameter](const mtao::Vec3d& p) -> mtao::Vec3d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            double z = (p.z() - cz) / diameter;
            return mtao::Vec3d(ac * std::pow<double>(x, xexp1) *
                                   std::pow<double>(y, yexp1) *
                                   std::pow<double>(z, zexp1),
                               0, 0) /
                   diameter;
        };
    } else if (bad_x && bad_z) {
        spdlog::trace("y_onl 0, {0}x^{1}y^{2}^{3}", yexp, xexp2, yexp2, zexp2);
        return [bc, xexp2, yexp2, zexp2, cx, cy, cz,
                diameter](const mtao::Vec3d& p) -> mtao::Vec3d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            double z = (p.z() - cz) / diameter;
            return mtao::Vec3d(0,
                               bc * std::pow<double>(x, xexp2) *
                                   std::pow<double>(y, yexp2) *
                                   std::pow<double>(z, zexp2),
                               0) /
                   diameter;
        };
    } else if (bad_x && bad_y) {
        spdlog::trace("z_onl 0, {0}x^{1}y^{2}^{3}", zexp, xexp3, yexp3, zexp3);
        return [cc, xexp3, yexp3, zexp3, cx, cy, cz,
                diameter](const mtao::Vec3d& p) -> mtao::Vec3d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            double z = (p.z() - cz) / diameter;
            return mtao::Vec3d(0, 0,
                               cc * std::pow<double>(x, xexp3) *
                                   std::pow<double>(y, yexp3) *
                                   std::pow<double>(z, zexp3)) /
                   diameter;
        };
    } else if (bad_z) {
        spdlog::trace("x_onl 0, {0}x^{1}y^{2}^{3}", xexp, xexp1, yexp1, zexp1);
        return [ac, bc, xexp1, yexp1, zexp1, xexp2, yexp2, zexp2, cx, cy, cz,
                diameter](const mtao::Vec3d& p) -> mtao::Vec3d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            double z = (p.z() - cz) / diameter;
            return mtao::Vec3d(

                       ac * std::pow<double>(x, xexp1) *
                           std::pow<double>(y, yexp1) *
                           std::pow<double>(z, zexp1),
                       bc * std::pow<double>(x, xexp2) *
                           std::pow<double>(y, yexp2) *
                           std::pow<double>(z, zexp2),
                       0) /
                   diameter;
        };
    } else if (bad_y) {
        spdlog::trace("y_onl 0, {0}x^{1}y^{2}^{3}", yexp, xexp2, yexp2, zexp2);
        return [ac, cc, xexp1, yexp1, zexp1, xexp3, yexp3, zexp3, cx, cy, cz,
                diameter](const mtao::Vec3d& p) -> mtao::Vec3d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            double z = (p.z() - cz) / diameter;
            return mtao::Vec3d(ac * std::pow<double>(x, xexp1) *
                                   std::pow<double>(y, yexp1) *
                                   std::pow<double>(z, zexp1),
                               0,
                               cc * std::pow<double>(x, xexp3) *
                                   std::pow<double>(y, yexp3) *
                                   std::pow<double>(z, zexp3)) /
                   diameter;
        };
    } else if (bad_x) {
        spdlog::trace("z_onl 0, {0}x^{1}y^{2}^{3}", zexp, xexp3, yexp3, zexp3);
        return [bc, cc, xexp2, yexp2, zexp2, xexp3, yexp3, zexp3, cx, cy, cz,
                diameter](const mtao::Vec3d& p) -> mtao::Vec3d {
            double x = (p.x() - cx) / diameter;
            double y = (p.y() - cy) / diameter;
            double z = (p.z() - cz) / diameter;
            return mtao::Vec3d(0,
                               bc * std::pow<double>(x, xexp2) *
                                   std::pow<double>(y, yexp2) *
                                   std::pow<double>(z, zexp2),
                               cc * std::pow<double>(x, xexp3) *
                                   std::pow<double>(y, yexp3) *
                                   std::pow<double>(z, zexp3)) /
                   diameter;
        };
    } else {
        spdlog::trace("{2}x^{0}y^{1} + {1}x^{2}y^{3}", xexp1, yexp1, xexp2,
                      yexp2);
        return
            [ac, bc, cc, xexp1, yexp1, zexp1, xexp2, yexp2, zexp2, xexp3, yexp3,
             zexp3, cx, cy, cz, diameter](const mtao::Vec3d& p) -> mtao::Vec3d {
                double x = (p.x() - cx) / diameter;
                double y = (p.y() - cy) / diameter;
                double z = (p.z() - cz) / diameter;
                return mtao::Vec3d(ac * std::pow<double>(x, xexp1) *
                                       std::pow<double>(y, yexp1) *
                                       std::pow<double>(z, zexp1),
                                   bc * std::pow<double>(x, xexp2) *
                                       std::pow<double>(y, yexp2) *
                                       std::pow<double>(z, zexp2),
                                   cc * std::pow<double>(x, xexp3) *
                                       std::pow<double>(y, yexp3) *
                                       std::pow<double>(z, zexp3)) /
                       diameter;
            };
    }
}
}  // namespace vem
