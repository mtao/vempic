#include "vem/poisson_2d/poisson_vem_cell.hpp"

#include <spdlog/spdlog.h>

#include <mtao/quadrature/gauss_lobatto.hpp>
#include <ranges>
#include <vem/edge_lengths.hpp>
#include <vem/monomial_cell_integrals.hpp>
#include <vem/monomial_edge_integrals.hpp>
#include <vem/polynomial_gradient.hpp>
#include <vem/polynomial_utils.hpp>
#include <vem/utils/volumes.hpp>
using namespace vem::polynomials::two;

namespace vem::poisson_2d {

PoissonVEM2Cell::PoissonVEM2Cell(const VEMMesh2 &mesh, size_t index,
                                 const RKHSBasisIndexer &a,
                                 const MonomialBasisIndexer &b,
                                 const MomentBasisIndexer &c)
    : PoissonVEM2Cell(VEM2Cell(mesh, index, b.diameter(index)), a, b, c) {}

PoissonVEM2Cell::PoissonVEM2Cell(const VEM2Cell &cell,
                                 const RKHSBasisIndexer &a,
                                 const MonomialBasisIndexer &b,
                                 const MomentBasisIndexer &c)
    : VEM2Cell(cell),
      point_indexer(a),
      monomial_indexer(b),
      moment_indexer(c) {}
int PoissonVEM2Cell::system_size() const {
    return point_indexer.num_coefficients() + moment_indexer.num_coefficients();
}
int PoissonVEM2Cell::local_system_size() const {
    size_t size = vertex_count() + moment_size();
    // size_t size = vertex_count() + edge_interior_sample_count() +
    // moment_size();
    return size;
}
// mtao::MatXd PoissonVEM2Cell::KE() const {
//    auto gt = monomial_grammian();
//    auto pis = Pis();
//    auto pi = Pi();
//    auto id = mtao::MatXd::Identity(pis.rows(),pis.cols()) - pis;
//
//    return pis.transpose() * gt * pis + id.transpose() * gt * id;
//}
mtao::MatXd PoissonVEM2Cell::PErr() const {
    auto pi = Pi();
    auto id = mtao::MatXd::Identity(pi.rows(), pi.cols()) - pi;

    return id.transpose() * id;
}
mtao::MatXd PoissonVEM2Cell::KEH() const {
    auto gt = monomial_dirichlet_grammian(monomial_degree());
    auto pis = Pis();

    return pis.transpose() * gt * pis + PErr();
}

mtao::MatXd PoissonVEM2Cell::CoGrad_mIn() const {
    // Pis^* G^* M
    size_t size = local_system_size();
    size_t mon_size = monomial_size();
    auto MG = local_monomial_gradient();
    auto M = monomial_l2_grammian();

    auto pis = Pis();

    mtao::MatXd D(size, 2 * mon_size);
    auto Dx = D.leftCols(mon_size);
    auto Dy = D.rightCols(mon_size);

    Dx = pis.transpose() * MG.topRows(mon_size).transpose() * M;
    Dy = pis.transpose() * MG.bottomRows(mon_size).transpose() * M;

    return D;
}
mtao::MatXd PoissonVEM2Cell::CoGrad() const {
    // Pis^* G^* M Diag(Pis,2)
    size_t size = local_system_size();
    size_t mon_size = monomial_size();
    mtao::MatXd D(size, 2 * size);
    auto Dm = CoGrad_mIn();
    auto Dx = D.leftCols(size);
    auto Dy = D.rightCols(size);

    auto Dmx = Dm.leftCols(mon_size);
    auto Dmy = Dm.rightCols(mon_size);

    auto pis = Pis();

    auto PS = PErr();
    Dx = Dmx * pis + PS;
    Dy = Dmy * pis + PS;
    // Dx.setConstant(1);
    // Dy.setConstant(2);
    return D;
}

mtao::MatXd PoissonVEM2Cell::Grad_mOut() const {
    auto MG = local_monomial_gradient();
    auto pis = Pis();

    mtao::MatXd G = MG * pis;
    return G;
}
mtao::MatXd PoissonVEM2Cell::Grad() const {
    size_t size = local_system_size();
    size_t mon_size = monomial_size();
    mtao::MatXd Gm = Grad_mOut();
    mtao::MatXd G(2 * size, size);

    auto pi = Pi();
    auto Kpq = D();
    auto perr = mtao::MatXd::Identity(pi.rows(), pi.cols()) - pi;

    auto Gx = G.topRows(size);
    auto Gy = G.bottomRows(size);
    auto Gmx = Gm.topRows(mon_size);
    auto Gmy = Gm.bottomRows(mon_size);
    Gx = Kpq * Gmx + .5 * perr;
    Gy = Kpq * Gmy + .5 * perr;
    // Gx.setConstant(1);
    // Gy.setConstant(2);
    return G;
}
Eigen::SparseMatrix<double> PoissonVEM2Cell::local_monomial_gradient() const {
    return monomial_to_monomial_gradient(monomial_degree());
}

// mtao::MatXd PoissonVEM2Cell::KEH_sqrt() const {
//    auto gt = monomial_grammian();
//    auto pis = Pis();
//    auto pi = Pi();
//
//    auto id = mtao::MatXd::Identity(pi.rows(), pi.cols()) - pi;
//
//    return gt * pis + id;
//}
mtao::MatXd PoissonVEM2Cell::Pis() const {
    mtao::MatXd b = B();
    mtao::MatXd g = G();
    mtao::MatXd PS(g.cols(), b.cols());
    auto glu = g.lu();
    for (int j = 0; j < b.cols(); ++j) {
        PS.col(j) = glu.solve(b.col(j));
    }
    return PS;
}
mtao::MatXd PoissonVEM2Cell::Pi() const {
    mtao::MatXd d = D();

    return d * Pis();
}

mtao::MatXd PoissonVEM2Cell::Pis0() const {
    mtao::MatXd b = C();
    mtao::MatXd g = H();
    mtao::MatXd PS(g.cols(), b.cols());
    auto glu = g.lu();
    for (int j = 0; j < b.cols(); ++j) {
        PS.col(j) = glu.solve(b.col(j));
    }
    return PS;
}

mtao::MatXd PoissonVEM2Cell::Pi0() const {
    mtao::MatXd d = D();

    return d * Pis0();
}
mtao::MatXd PoissonVEM2Cell::monomial_l2_grammian() const {
    return VEM2Cell::monomial_l2_grammian(monomial_degree());
}
mtao::MatXd PoissonVEM2Cell::C() const {
    mtao::MatXd R(monomial_size(), local_system_size());
    R.setZero();

    double vol = volume();
    // auto integrals =
    //    monomial_indexer.monomial_integrals(index, 2 * moment_degree());
    size_t mom_off = local_moment_index_offset();
    // for row 1... n_{k-2} we use the moment inner product
    // n_{k-2} = moment size

    for (size_t j = 0; j < moment_size(); ++j) {
        int col = mom_off + j;
        R(j, col) = vol;
        // auto [mxexp, myexp] = index_to_exponents(j);
        // for (int row = 0; row < moment_size(); ++row) {
        //    // auto [Mxexp, Myexp] = index_to_exponents(row);
        //    R(row, col) = vol;
        //}
    }
    size_t off = moment_size();  // k-2 polynomials
    mtao::MatXd Cp = H() * Pis();

    int block_size = monomial_size() - off;
    R.bottomRows(block_size) = Cp.bottomRows(block_size);
    return R;
}
mtao::MatXd PoissonVEM2Cell::H() const { return monomial_l2_grammian(); }

mtao::MatXd PoissonVEM2Cell::monomial_grammian() const {
    return monomial_dirichlet_grammian(monomial_degree());
}
size_t PoissonVEM2Cell::local_moment_index_offset() const {
    size_t size = edges().size();
    for (auto &&[e, s] : edges()) {
        size += point_indexer.num_internal_edge_indices(e);
    }
    return size;
}

mtao::iterator::detail::range_container<size_t>
PoissonVEM2Cell::local_to_world_monomial_indices() const {
    return monomial_indexer.coefficient_indices(cell_index());
}

size_t PoissonVEM2Cell::global_moment_index_offset() const {
    return point_indexer.num_coefficients();
}
size_t PoissonVEM2Cell::global_monomial_index_offset() const {
    return monomial_indexer.coefficient_offset(cell_index());
}
mtao::MatXd PoissonVEM2Cell::G() const {
    // [ ... P_0 m_i ... ]
    // [ ... \nabla m_i, \nabla m_j ... ]
    // [              ... ...           ]
    mtao::MatXd R = monomial_grammian();
    auto P0 = R.row(0);
    R.col(0).setZero();
    if (monomial_degree() == 1) {
        for (int j = 0; j < monomial_size(); ++j) {
            P0(j) =
                point_indexer.evaluate_coefficients(cell_index(), monomial(j))
                    .mean();
        }

    } else {
        const double length = boundary_area();
        const double area = volume();

        auto integrals = monomial_indexer.monomial_integrals(cell_index());
        // std::cout << "CELL INTEGRALS: " << integrals.transpose() <<
        // std::endl;
        P0 = integrals.transpose() / area;
    }

    return R;
}

std::map<size_t, size_t> PoissonVEM2Cell::world_to_local_point_indices() const {
    std::map<size_t, size_t> ret;
    auto pi = point_indices();
    for (auto &&[idx, i] : mtao::iterator::enumerate(pi)) {
        ret[i] = idx;
    }
    return ret;
}
std::map<size_t, size_t> PoissonVEM2Cell::world_to_local_sample_indices()
    const {
    auto ret = world_to_local_point_indices();
    size_t local_offset = local_moment_index_offset();
    size_t global_offset = global_moment_index_offset();
    auto [start, end] = moment_indexer.coefficient_range(cell_index());
    // spdlog::info("Moment indexer gave range {} {} for cell {}", start, end,
    //             index);
    for (size_t j = 0; j < end - start; ++j) {
        ret[j + start + global_offset] = j + local_offset;
    }
    return ret;
}
std::vector<size_t> PoissonVEM2Cell::local_to_world_point_indices() const {
    return point_indices();
}
std::vector<size_t> PoissonVEM2Cell::local_to_world_sample_indices() const {
    return sample_indices();
}
Eigen::SparseMatrix<double> PoissonVEM2Cell::local_to_world_monomial_map()
    const {
    Eigen::SparseMatrix<double> R(monomial_indexer.num_coefficients(),
                                  monomial_size());
    auto [start, end] = monomial_indexer.coefficient_range(cell_index());
    for (size_t j = 0; j < end - start; ++j) {
        R.insert(j + start, j) = 1;
    }

    return R;
}
Eigen::SparseMatrix<double> PoissonVEM2Cell::local_to_world_sample_map() const {
    Eigen::SparseMatrix<double> R(system_size(), local_system_size());
    for (auto &&[r, c] : world_to_local_sample_indices()) {
        R.insert(r, c) = 1;
    }
    // std::cout << R << std::endl;

    return R;
}

mtao::MatXd PoissonVEM2Cell::B() const {
    mtao::MatXd R(monomial_size(), local_system_size());
    R.setZero();

    std::map<size_t, size_t> w2l_pi = world_to_local_point_indices();

    auto P0 = R.row(0);
    auto indices = point_indices();
    std::map<size_t, double> edge_lengths = this->edge_lengths();
    double total_edge_length = boundary_area();

    if (monomial_degree() == 1) {
        P0.setConstant(1.0 / indices.size());
    } else {
        P0.setUnit(local_moment_index_offset());
    }

    auto EN = edge_normals();
    double diameter = this->diameter();
    double diam2 = diameter * diameter;
    for (auto &&[eidx, sgn] : edges()) {
        auto edge_indices = point_indexer.ordered_edge_indices(eidx);
        mtao::Vec2d N = (sgn ? -1 : 1) * mtao::eigen::stl2eigen(EN[eidx]);
        double edge_length = edge_lengths[eidx];
        for (size_t poly_idx = 1; poly_idx < monomial_size(); ++poly_idx) {
            auto g = gradient_single_index(poly_idx);
            auto [xc, xi] = g[0];
            auto [yc, yi] = g[1];
            if (xi < 0) {
                xc = 0;
            } else {
                xc *= N.x();
            }
            if (yi < 0) {
                yc = 0;
            } else {
                yc *= N.y();
            }
            mtao::Vec2d xexp =
                mtao::eigen::stl2eigen(index_to_exponents(xi)).cast<double>();

            mtao::Vec2d yexp =
                mtao::eigen::stl2eigen(index_to_exponents(yi)).cast<double>();
            if (xi < 0) {
                xexp.setZero();
            }

            if (yi < 0) {
                yexp.setZero();
            }

            auto &&[P, W] = mtao::quadrature::gauss_lobatto_data<double>(
                edge_indices.size());

            for (auto &&[vidx, weight] : mtao::iterator::zip(edge_indices, W)) {
                size_t local_index = w2l_pi.at(vidx);
                auto p =
                    (point_indexer.get_position(vidx) - center()) / diameter;
                double dmdn = (xc * p.array().pow(xexp.array()).prod() +
                               yc * p.array().pow(yexp.array()).prod()) /
                              diameter;
                // factor of 2 due to gauss lobatto being defined on [-1,1]
                R(poly_idx, local_index) += dmdn * weight * edge_length / 2;
            }
        }
    }
    double vol = volume();
    // spdlog::info("Cell volume: {}", vol);
    size_t mom_off = local_moment_index_offset();
    auto L = polynomials::two::laplacian(monomial_degree());
    double d2 = std::pow(diameter, 2.0);
    // std::cout <<"Laplacian\n" << L << std::endl;
    for (int o = 0; o < L.outerSize(); ++o) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, o); it; ++it) {
            int col = it.row() + mom_off;
            int row = it.col();
            R(row, col) = -it.value() * vol / d2;
        }
    }
    return R;
}
mtao::MatXd PoissonVEM2Cell::D() const {
    mtao::MatXd R(local_system_size(), monomial_size());
    R.setZero();
    double vol = volume();
    auto p = point_indices();
    for (auto &&[local_vsample_idx, vsample_idx] :
         mtao::iterator::enumerate(p)) {
        // for (int vsample_idx = 0; vsample_idx < boundary_sample_count();
        //     ++vsample_idx) {
        for (size_t poly_idx = 0; poly_idx < monomial_size(); ++poly_idx) {
            R(local_vsample_idx, poly_idx) = evaluate_monomial(
                poly_idx, point_indexer.get_position(vsample_idx));
        }
    }
    size_t mom_off = local_moment_index_offset();
    auto integrals =
        VEM2Cell::monomial_integrals(monomial_degree() + moment_degree());
    for (size_t j = 0; j < moment_size(); ++j) {
        int row = j + mom_off;
        auto [mxexp, myexp] = index_to_exponents(j);
        for (int col = 0; col < monomial_size(); ++col) {
            auto [Mxexp, Myexp] = index_to_exponents(col);

            R(row, col) =
                integrals(exponents_to_index(mxexp + Mxexp, myexp + Myexp)) /
                vol;
        }
    }
    return R;
}

// size_t PoissonVEM2Cell::edge_count() const {
//    return mesh.face_boundary_map.at(cell_index()).size();
//}
size_t PoissonVEM2Cell::edge_interior_sample_count() const {
    size_t size = 0;

    for (auto &&[idx, sgn] : mesh().face_boundary_map.at(cell_index())) {
        size += point_indexer.num_internal_edge_indices(idx);
    }
    return size;
}

mtao::MatXd PoissonVEM2Cell::M() const {
    auto c = C();
    auto pis0 = Pis();
    auto d = D();
    size_t size = system_size();
    mtao::MatXd id = mtao::MatXd::Identity(d.rows(), pis0.cols()) - d * pis0;

    return c.transpose() * pis0 + volume() * id.transpose() * id;
    // return c.transpose() * pis0;
}
size_t PoissonVEM2Cell::vertex_count() const { return point_indices().size(); }
size_t PoissonVEM2Cell::boundary_sample_count() const {
    return edge_count() + edge_interior_sample_count();
}
size_t PoissonVEM2Cell::moment_size() const {
    return moment_indexer.num_monomials(cell_index());
}
size_t PoissonVEM2Cell::monomial_degree() const {
    return monomial_indexer.degree(cell_index());
}
size_t PoissonVEM2Cell::moment_degree() const {
    return moment_indexer.degree(cell_index());
}
size_t PoissonVEM2Cell::monomial_size() const {
    return monomial_indexer.num_monomials(cell_index());
}
std::vector<size_t> PoissonVEM2Cell::point_indices() const {
    auto s = point_indexer.cell_indices(cell_index());
    return {s.begin(), s.end()};
}
std::vector<size_t> PoissonVEM2Cell::sample_indices() const {
    auto s = point_indices();
    s.reserve(s.size() + moment_size());
    auto [start, end] = moment_indexer.coefficient_range(cell_index());
    start += global_moment_index_offset();
    end += global_moment_index_offset();
    for (size_t j = start; j < end; ++j) {
        s.emplace_back(j);
    }

    return {s.begin(), s.end()};
}
// double PoissonVEM2Cell::boundary_area() const {
//    double sum = 0;
//    for (auto &&[a, b] : edge_lengths()) {
//        sum += b;
//    }
//    return sum;
//}
// double PoissonVEM2Cell::volume() const { return utils::volume(mesh, index); }
// double PoissonVEM2Cell::edge_length(size_t edge_index) const {
//    return vem::edge_length(mesh, edge_index);
//}
// std::map<size_t, double> PoissonVEM2Cell::edge_lengths() const {
//    std::map<size_t, double> ret;
//    for (auto &&[eidx, sgn] : edges()) {
//        ret[eidx] = edge_length(eidx);
//    }
//    return ret;
//}
//
// const std::map<int, bool> &PoissonVEM2Cell::edges() const {
//    return mesh.face_boundary_map.at(cell_index());
//}
//
// std::map<size_t, std::array<double, 2>> PoissonVEM2Cell::edge_normals() const
// {
//    std::map<size_t, std::array<double, 2>> ret;
//
//    // std::ranges::copy(std::ranges::views::transform(
//    //                      edges(), [&](const std::pair<size_t, bool>& pr) ->
//    //                      std::pair<size_t,std::array<double,2>>{ auto&&
//    //                      [idx,sgn] = pr; std::array<double,2> val;
//    //
//    //                      return {idx,val};
//
//    //                      }),
//    //                  std::inserter(ret, ret.end()));
//
//    auto edges = this->edges();
//    std::transform(edges.begin(), edges.end(), std::inserter(ret, ret.end()),
//    [&](const std::pair<size_t, bool> &pr) -> std::pair<size_t,
//    std::array<double, 2>> {
//        auto &&[idx, sgn] = pr;
//        std::array<double, 2> val;
//        auto e = mesh.E.col(idx);
//        auto a = mesh.V.col(e(0));
//        auto b = mesh.V.col(e(1));
//        auto ab = (a - b).normalized().eval();
//        // auto ab = (b - a).normalized().eval();
//        val[0] = -ab.y();
//        val[1] = ab.x();
//
//        return { idx, val };
//    });
//    return ret;
//}
// std::set<size_t> PoissonVEM2Cell::vertices() const {
//    std::set<size_t> ret;
//    for (auto &&[eidx, sgn] : edges()) {
//        auto e = mesh.E.col(eidx);
//        ret.emplace(e(0));
//        ret.emplace(e(1));
//    }
//    return ret;
//}
// double PoissonVEM2Cell::diameter() const {
//    return monomial_indexer.diameter(cell_index());
//}
//
// std::function<double(const mtao::Vec2d &)> PoissonVEM2Cell::monomial(
//  size_t index) const {
//    return monomial_indexer.monomial(this->index, index);
//}
// std::function<mtao::Vec2d(const mtao::Vec2d &)>
//  PoissonVEM2Cell::monomial_gradient(size_t index) const {
//    return monomial_indexer.monomial_gradient(this->index, index);
//}
}  // namespace vem::poisson_2d
