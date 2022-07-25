#include "diwlevin_vem_mesh2.hpp"

#include <fmt/format.h>
#include <igl/copyleft/quadprog.h>
#include <spdlog/spdlog.h>

#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/triangle_monomial_integrals.hpp>
#include <mtao/iterator/enumerate.hpp>
#include <numeric>
//#include <mtao/eigen/interweave.h>
mtao::MatXd DIWLevinVEMMesh2::per_cell_per_monomial_linear_integral(
    size_t cell_index) const {
    auto indices = cell_sample_indices_vec(cell_index);
    mtao::MatXd R(coefficient_size(), indices.size());
    R.setZero();
    for (auto&& [e, s] : cells.at(cell_index)) {
        auto A = per_edge_per_monomial_linear_integral(cell_index, e);
        auto T = index_map(indices, boundary_sample_indices_vec(e));
        // fmt::print("A{}x{} T{}x{}\n", A.rows(),A.cols(),T.rows(),T.cols());
        R += A * T;
    }
    return R;
}
mtao::MatXd DIWLevinVEMMesh2::per_boundary_per_monomial_product(
    size_t index, size_t edge_index) const {
    auto e = E(edge_index);
    mtao::Vec2d a = V(e(0)) - C(index);
    mtao::Vec2d dir = V(e(1)) - V(e(0));

    size_t size = coefficient_size();
    mtao::MatXd R(size, size);
    R.setZero();
    mtao::algebra::PascalTriangle pt(2 * order);

    // \int_{0,1} (o.x + td.x)^i (o.y + td.y)^j (o.x + td.x)^k (o.y + td.y)^l
    // \int_{0,1} (o.x + td.x)^{i+k} (o.y + td.y)^{j+l}
    // \int_{0,1} \sum_{m=0}^{i+k} \sum_{n=0}^j+l}{i+k \choose m}{j+l \choose n}
    //                  o.x^m d.x^{i+k-m} o.y^{n} d.y^{j+l-n} t^{i+k+j+l-m-n}
    // \sum_{m=0}^{i+k} \sum_{n=0}^j+l}{i+k \choose m}{j+l \choose n}
    //                  o.x^m d.x^{i+k-m} o.y^{n} d.y^{j+l-n}
    //                  \frac{1}{i+k+j+l-m-n+1}
    for (size_t d1 = 0; d1 <= order; ++d1) {
        int off1 = (d1 * (d1 + 1)) / 2;
        for (size_t d2 = 0; d2 <= order; ++d2) {
            int off2 = (d2 * (d2 + 1)) / 2;
            for (size_t i = 0; i <= d1; ++i) {
                size_t j = d1 - i;
                for (size_t k = 0; k <= d2; ++k) {
                    size_t l = d2 - k;
                    double value = 0.0;
                    size_t row = off1 + i;
                    size_t col = off2 + k;
                    // fmt::print(
                    //    "{0}x{1}: D1={2} D2={3}  i={4} j={5} k={6} l={7} "
                    //    "x^({4}+{6})y^({5}+{7})\n",
                    //    row, col, d1, d2, i  // 4
                    //    ,
                    //    j  // 5
                    //    ,
                    //    k  // 6
                    //    ,
                    //    l);  // 7

                    for (size_t m = 0; m <= i + k; ++m) {
                        size_t o = i + k - m;
                        double m_coeff = std::pow<double>(a.y(), m) *
                                         pt(i + k, m) *
                                         std::pow<double>(dir.y(), o);
                        ;
                        for (size_t n = 0; n <= j + l; ++n) {
                            size_t p = j + l - n;
                            double n_coeff = std::pow<double>(a.x(), n) *
                                             pt(j + l, n) *
                                             std::pow<double>(dir.x(), p);
                            size_t torder = o + p;

                            double v = 1.0 / (torder + 1) * m_coeff * n_coeff;
                            value += v;
                            /*
                            fmt::print("m={} n={}\n", m  // 4
                                       ,
                                       n  // 5
                            );
                            fmt::print(
                                "{11} = ({0}+{2} choose {4})({1}+{3} choose "
                                "{5}) {7}^{4} {9}^({0}+{2}-{4}) {8}^{5} "
                                "{10}^({1} + {3} - {5}) ( 1 / ({6})\n",
                                i  // 0
                                ,
                                j  // 1
                                ,
                                k  // 2
                                ,
                                l  // 3
                                ,
                                m  // 4
                                ,
                                n  // 5
                                ,
                                torder + 1  // 6
                                ,
                                a.x()  // 7
                                ,
                                a.y()  // 8
                                ,
                                dir.x()  // 9
                                ,
                                dir.y()  // 10
                                ,
                                v);  // 111
                            fmt::print("{} {} {}\n", m_coeff, n_coeff,
                                       torder + 1);
                            continue;

                            fmt::print("{} {} {}  {} {} {}\n",
                                       std::pow<double>(a.x(), m), pt(i + k, m),
                                       std::pow<double>(dir.x(), o),
                                       std::pow<double>(a.y(), n), pt(j + l, n),
                                       std::pow<double>(dir.y(), p));

                            fmt::print("mcoeff={} ncoeff={}\n", m_coeff,
                                       n_coeff);
                                       */
                        }
                    }
                    // fmt::print("Val{}x{} = {}\n\n", row, col, value);
                    R(row, col) = value;
                    // R(row,col) = R(col,row) = value;
                }
            }
        }
    }
    R *= dir.norm();
    return R;
}
mtao::MatXd DIWLevinVEMMesh2::per_cell_gramian(size_t cell_index) const {
    mtao::MatXd R(coefficient_size(), coefficient_size());
    R.setZero();
    for (auto&& [e, s] : cells.at(cell_index)) {
        auto A = per_boundary_per_monomial_product(cell_index, e);
        R += A;
        // auto T = index_map(indices,boundary_sample_indices_vec(e));
        // fmt::print("A{}x{} T{}x{}\n", A.rows(),A.cols(),T.rows(),T.cols());
        // R += A * T;
    }
    return R;
}
mtao::MatXd DIWLevinVEMMesh2::poly_projection(size_t cell_index) const {
    auto indices = cell_sample_indices_vec(cell_index);

    auto A = per_cell_gramian(cell_index);
    auto B = per_cell_per_monomial_linear_integral(cell_index);

    return A.inverse() * B;
}
mtao::MatXd DIWLevinVEMMesh2::per_cell_poly_gradients_sample2sample(
    size_t index) const {
    return per_cell_gradients_poly2sample(index) * poly_projection(index);
    // return per_cell_gradient_poly2sample(index) *
    // poly_projection_sample2poly(index);
}
mtao::MatXd DIWLevinVEMMesh2::per_cell_hybrid_gradient(
    size_t cell_index) const {
    // auto PolyGrad =
    // per_cell_integrated_projected_boundary_gradient(cell_index);
    // spdlog::info("Creating hybrid gradient");
    auto PolyGrad = per_cell_poly_gradients_sample2sample(cell_index);
    // spdlog::info("Polygrad size: {} {}", PolyGrad.rows(), PolyGrad.cols());
    auto cell_indices = cell_sample_indices_vec(cell_index);
    std::map<size_t, size_t> cell_index_inverter;
    size_t num_verts = 0;
    for (auto&& [idx, ind] : mtao::iterator::enumerate(cell_indices)) {
        cell_index_inverter[ind] = idx;
    }

    const auto& boundary = cells.at(cell_index);

    // spdlog::info("Creating pertinent edge segments");
    // generate all the pertinent edge segments
    std::vector<std::array<size_t, 2>> cut_edges;
    std::map<size_t, std::set<size_t>> edge_cut_edge_ownership;
    for (auto&& [idx, pr] : mtao::iterator::enumerate(boundary)) {
        auto [eidx, sgn] = pr;
        auto e = E(eidx);
        std::vector<size_t> vertices;
        auto [start, end] = boundary_internal_index_range(eidx);
        if (start >= end) {
            edge_cut_edge_ownership[eidx].emplace(cut_edges.size());
            cut_edges.emplace_back(
                std::array<size_t, 2>{{size_t(e(0)), size_t(e(1))}});

        } else {
            start += num_vertices();
            end += num_vertices();
            edge_cut_edge_ownership[eidx].emplace(cut_edges.size());
            cut_edges.emplace_back(
                std::array<size_t, 2>{{size_t(e(0)), start}});
            edge_cut_edge_ownership[eidx].emplace(cut_edges.size());
            cut_edges.emplace_back(
                std::array<size_t, 2>{{end - 1, size_t(e(1))}});
            for (size_t i = start; i < end - 1; ++i) {
                edge_cut_edge_ownership[eidx].emplace(cut_edges.size());
                cut_edges.emplace_back(std::array<size_t, 2>{{i, i + 1}});
            }
        }
    }

    mtao::MatXd D(2 * cut_edges.size(), PolyGrad.cols());
    spdlog::info("The grad operator is of size {} {}", D.rows(), D.cols());

    D.setZero();
    for (auto&& [idx, pr] : mtao::iterator::enumerate(boundary)) {
        auto [eidx, sgn] = pr;
        auto&& my_cut_edges = edge_cut_edge_ownership.at(eidx);
        auto e = E(eidx);

        // construct basis
        mtao::Mat2d basis;
        auto T = basis.col(0);
        auto N = basis.col(1);
        T = V(e(1)) - V(e(0));
        double edge_length = T.norm();
        T.normalize();
        N << -T.y(), T.x();

        mtao::MatXd N_integrated_edge_gradients;
        if (integrated_edges) {
            // TODO: this doesn't work for cut-edges
            // for each edge segment we want the integrated gradient.
            // in other words
            N_integrated_edge_gradients.resize(cut_edges.size(),
                                               coefficient_size());
            N_integrated_edge_gradients.setZero();
            // columns are the integrals for each cut-edge
            auto edge_integrals =
                per_edge_per_monomial_integrals(cell_index, eidx);
            auto G = poly_gradient();
            Eigen::SparseMatrix<double> Ndot =
                (N.x() * G.topRows(coefficient_size()) +
                 N.y() * G.bottomRows(coefficient_size()));
            N_integrated_edge_gradients = edge_integrals * Ndot;
        }

        mtao::MatXd PP = poly_projection(cell_index);
        double dx = edge_length / (my_cut_edges.size());
        for (auto&& [local_ceidx, ceidx] :
             mtao::iterator::enumerate(my_cut_edges)) {
            auto De = D.block(2 * (ceidx), 0, 2, D.cols());
            auto Tstuff = De.row(0);
            auto Nstuff = De.row(1);
            for (auto&& [ewhich, sample_index] :
                 mtao::iterator::enumerate(cut_edges.at(ceidx))) {
                // tangent component
                if (cell_index_inverter.find(sample_index) ==
                    cell_index_inverter.end()) {
                    spdlog::warn("Inverter fail {}", sample_index);
                }
                double sgn = (ewhich == 0) ? -1 : 1;

                if (integrated_edges) {
                    Tstuff(cell_index_inverter.at(sample_index)) = sgn;
                } else {
                    // std::cout << sgn << " : " << sample_index << std::endl;
                    // take the centered difference gradient at this point
                    Tstuff(cell_index_inverter.at(sample_index)) = sgn / dx;
                }
                // Nstuff *= sgn?-1:1;
            }
            if (integrated_edges) {
                Nstuff = N_integrated_edge_gradients.row(local_ceidx) * PP;
            } else {
                auto ce = cut_edges.at(ceidx);
                auto a = sample_position(ce[0]);
                auto b = sample_position(ce[1]);

                mtao::Vec2d p = .5 * (a + b);
                Nstuff =
                    N.transpose() * polynomial_grad_entries(cell_index, p) * PP;

                // take average polygon gradient from either side
                // auto PDe = PolyGrad.block(2 *
                // cell_index_inverter.at(sample_index), 0, 2, D.cols()); Nstuff
                // += .5 * N.transpose() * PDe;
            }
        }
    }
    return D;
}
mtao::MatXd DIWLevinVEMMesh2::per_cell_integrated_projected_boundary_gradient(
    size_t cell_index) const {
    const auto& boundary = cells.at(cell_index);
    auto cell_indices = cell_sample_indices_vec(cell_index);

    // invert the per-cell coordinates
    std::map<size_t, size_t> cell_index_inverter;
    for (auto&& [idx, ind] : mtao::iterator::enumerate(cell_indices)) {
        cell_index_inverter[ind] = idx;
        // fmt::print("Cell indexer: {}=>{}\n", ind, idx);
    }

    // inver the per-cell edge coordinates
    std::map<size_t, size_t> edge_indexer;
    for (auto&& [eidx, sgn] : boundary) {
        size_t size = edge_indexer.size();
        edge_indexer[eidx] = size;
        // fmt::print("Edge indexer: {}=>{}\n", eidx, size);
    }

    Eigen::SparseMatrix<double> A(2 * boundary.size(),
                                  2 * cell_index_inverter.size());
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(2 * cell_indices.size());
    for (auto&& [eidx, sgn] : boundary) {
        std::vector<size_t> bound_sample_inds =
            boundary_sample_indices_vec(eidx);

        auto e = E(eidx);
        auto a = V(e(0));
        auto b = V(e(1));
        double length = (b - a).norm();
        double dx = length / (bound_sample_inds.size() - 1);

        const size_t row = edge_indexer.at(eidx);

        // write rows using a piecewise linear approximation
        for (auto&& v : bound_sample_inds) {
            const size_t col = cell_index_inverter.at(v);
            double val = dx;
            if (v < num_vertices()) {
                val /= 2;  // endpoints have half value
            }
            trips.emplace_back(2 * row, 2 * col, val);
            trips.emplace_back(2 * row + 1, 2 * col + 1, val);
            // fmt::print("Inserting {}x{} and {}x{} out of {}x{}\n",
            // 2*row,2*col,2*row+1,2*col+1,A.rows(),A.cols());
        }
    }
    A.setFromTriplets(trips.begin(), trips.end());
    return A * per_cell_poly_gradients_sample2sample(cell_index);
}
mtao::MatXd DIWLevinVEMMesh2::per_cell_gradients_poly2sample(
    size_t cell_index) const {
    auto cell_indices = cell_sample_indices_vec(cell_index);

    mtao::vector<mtao::ColVecs2d> grads(cell_indices.size());

    std::transform(cell_indices.begin(), cell_indices.end(), grads.begin(),
                   [&](size_t idx) {
                       return polynomial_grad_entries(cell_index,
                                                      sample_position(idx));
                   });
    return mtao::eigen::vstack_iter(grads.begin(), grads.end());
}
mtao::MatXd DIWLevinVEMMesh2::per_edge_per_monomial_integrals(
    size_t cell_index, size_t edge_index) const {
    // for a given cell
    // mtao::VecXd
    // DIWLevinVEMMesh2::per_edge_per_cutedge_per_monomial_integrals(size_t
    //        index, size_t edge_index, size_t cutedge_index) const {
    auto e = E(edge_index);
    mtao::Vec2d a = V(e(0)) - C(cell_index);
    mtao::Vec2d dir = V(e(1)) - V(e(0));
    double length = dir.norm();

    size_t size = coefficient_size();
    std::vector<size_t> indices = edge_samples(edge_index);
    double dx = length / (indices.size() - 1);
    double dt = 1.0 / (indices.size() - 1);
    mtao::MatXd R(indices.size() - 1, size);
    assert(indices.size() == 2);
    R.setZero();
    mtao::algebra::PascalTriangle pt(order + 1);

    // \int_a^b m(x) f(x) for data f(x)
    // the basis integrals are
    // \int_a^b ( o_x + t d_x)^i ( o_y + t d_y)^j
    //
    // well we can evaluate  the integral to just do the latter one
    // \int_a^b ( o_x + t d_x)^i ( o_y + t d_y)^j
    // \int_a^b {i \choose m}{j \choose n} o_x^{i-m} d_x^{m} o_y^{j-n} d_y^{n}
    // t^{m+n} \frac{1}{m+n+1}{i \choose m}{j \choose n}( o_x^{i-m} d_x^{m}
    // o_y^{j-n} d_y^{n} t^{m+n+1} |_a^b
    //
    for (size_t d = 0; d <= order; ++d) {
        int off = (d * (d + 1)) / 2;
        for (size_t i = 0; i <= d; ++i) {
            size_t j = d - i;

            for (size_t m = 0; m <= i; ++m) {
                size_t o = i - m;
                double m_coeff = std::pow<double>(a.y(), m) * pt(i, m) *
                                 std::pow<double>(dir.y(), o);
                ;
                for (size_t n = 0; n <= j; ++n) {
                    size_t p = j - n;
                    double n_coeff = std::pow<double>(a.x(), n) * pt(j, n) *
                                     std::pow<double>(dir.x(), p);
                    size_t torder = o + p;
                    // fmt::print("{}-{} {}-{} => {} {}\n",
                    // i,o,j,n,m_coeff,n_coeff);

                    for (size_t segment = 0; segment < indices.size() - 1;
                         ++segment) {
                        double start = dt * segment;
                        double end = dt * (segment + 1);
                        double tpow = std::pow<double>(end, torder + 1) -
                                      std::pow<double>(start, torder + 1);
                        double value =
                            1.0 / (torder + 1) * m_coeff * n_coeff * tpow;

                        R(segment, off + i) += value;
                    }
                }
            }
        }
    }
    return dx * R;
}
mtao::MatXd DIWLevinVEMMesh2::per_edge_per_monomial_linear_integral(
    size_t cell_index, size_t edge_index) const {
    auto e = E(edge_index);
    mtao::Vec2d a = V(e(0)) - C(cell_index);
    mtao::Vec2d dir = V(e(1)) - V(e(0));
    double length = dir.norm();

    size_t size = coefficient_size();
    std::vector<size_t> indices = edge_samples(edge_index);
    double dx = length / (indices.size() - 1);
    double dt = 1.0 / (indices.size() - 1);
    mtao::MatXd R(size, indices.size());
    R.setZero();
    mtao::algebra::PascalTriangle pt(order + 1);
    // \int_a^b ( o_x + t d_x)^i ( o_y + t d_y)^j (1-t)
    // \int_a^b ( o_x + t d_x)^i ( o_y + t d_y)^j t
    //
    // well we can evaluate  the integral to just do the latter one
    // \int_a^b ( o_x + t d_x)^i ( o_y + t d_y)^j
    // \int_a^b {i \choose m}{j \choose n}( o_x^{i-m} d_x^{m} o_y^{j-n} d_y^{n}
    // t^{m+n} \frac{1}{m+n+1}{i \choose m}{j \choose n}( o_x^{i-m} d_x^{m}
    // o_y^{j-n} d_y^{n} t^{m+n+1} |_a^b
    //
    // \int_a^b ( o_x + t d_x)^i ( o_y + t d_y)^j t
    // \int_a^b {i \choose m}{j \choose n}( o_x^{i-m} d_x^{m} o_y^{j-n} d_y^{n}
    // t^{m+n+1} \frac{1}{m+n+2}{i \choose m}{j \choose n}( o_x^{i-m} d_x^{m}
    // o_y^{j-n} d_y^{n} t^{m+n+2} |_a^b the analytical iresult from teh
    // monomial
    for (size_t d = 0; d <= order; ++d) {
        int off = (d * (d + 1)) / 2;
        for (size_t i = 0; i <= d; ++i) {
            size_t j = d - i;

            for (size_t m = 0; m <= i; ++m) {
                size_t o = i - m;
                double m_coeff = std::pow<double>(a.y(), m) * pt(i, m) *
                                 std::pow<double>(dir.y(), o);
                ;
                for (size_t n = 0; n <= j; ++n) {
                    size_t p = j - n;
                    double n_coeff = std::pow<double>(a.x(), n) * pt(j, n) *
                                     std::pow<double>(dir.x(), p);
                    size_t torder = o + p;
                    // fmt::print("{}-{} {}-{} => {} {}\n",
                    // i,o,j,n,m_coeff,n_coeff);

                    for (size_t segment = 0; segment < indices.size() - 1;
                         ++segment) {
                        double start = dt * segment;
                        double end = dt * (segment + 1);
                        double tpow = std::pow<double>(end, torder + 1) -
                                      std::pow<double>(start, torder + 1);
                        double tpowp = std::pow<double>(end, torder + 2) -
                                       std::pow<double>(start, torder + 2);
                        double value =
                            1.0 / (torder + 1) * m_coeff * n_coeff * tpow;
                        double valuet =
                            1.0 / (torder + 2) * m_coeff * n_coeff * tpowp;

                        R(off + i, segment) += value - valuet;
                        R(off + i, segment + 1) += valuet;
                    }
                }
            }
        }
    }
    return dx * R * index_map(boundary_sample_indices_vec(edge_index), indices);
}

mtao::MatXd DIWLevinVEMMesh2::per_cell_laplacian(size_t index) const {
    // std::cout << "BOUNMDARY OFFSET SIZE: " << boundary_sample_offsets.size()
    //          << std::endl;
    auto B = per_cell_hybrid_gradient(index);
    mtao::VecXd w = per_cell_edge_quadrature_weights(index);

    mtao::VecXd ww(2 * w.size());  //= mtao::eigen::interweaveRows(w,w);
    for (int j = 0; j < w.size(); ++j) {
        ww(2 * j) = ww(2 * j + 1) = std::max(1e-2, w(j));
    }

    // std::cout << "Weights: " << w.transpose() << std::endl;
    // std::cout << "wweights: " << ww.transpose() << std::endl;
    // std::cout << B << std::endl;
    // std::cout << B.rows() << "x" << B.cols() << ": " << ww.size() <<
    // std::endl;

    // return B.transpose() * B;

    return B.transpose() * ww.asDiagonal() * B;
}
mtao::VecXd DIWLevinVEMMesh2::per_cell_quadrature_weights(
    size_t cell_index) const {
    auto A = poly_coefficient_matrix(cell_index);
    auto b = per_cell_per_monomial_integral(cell_index);

    Eigen::MatrixXd G = A.transpose() * A;
    Eigen::VectorXd g0 = A.transpose() * b;
    Eigen::MatrixXd CI = Eigen::MatrixXd::Identity(G.rows(), G.cols());
    Eigen::VectorXd ci0 = Eigen::VectorXd::Zero(CI.rows());

    Eigen::VectorXd x;
    igl::copyleft::quadprog(G, g0, {}, {}, CI, ci0, x);
    return x;

    return (A.transpose() * A).ldlt().solve(A.transpose() * b);
}
mtao::VecXd DIWLevinVEMMesh2::per_cell_edge_quadrature_weights(
    size_t cell_index) const {
    std::vector<size_t> edge_indices = boundary_indices_vec(cell_index);
    // columns are the integrals of the monomial for each cut-edge
    mtao::MatXd A(coefficient_size(), edge_indices.size());
    mtao::VecXd b = per_cell_per_monomial_integral(cell_index);
    for (auto&& [ind, eidx] : mtao::iterator::enumerate(edge_indices)) {
        if (integrated_edges) {
            A.col(ind) =
                per_edge_per_monomial_integrals(cell_index, eidx).transpose();
        } else {
            auto e = E(eidx);
            auto a = V(e(0));
            auto b = V(e(1));
            mtao::Vec2d p = .5 * (a + b);
            mtao::VecXd pe = polynomial_entries(cell_index, p).transpose();
            A.col(ind) = pe;
        }
    }

    Eigen::MatrixXd G = A.transpose() * A;
    Eigen::VectorXd g0 = A.transpose() * b;
    Eigen::MatrixXd CI = Eigen::MatrixXd::Identity(G.rows(), G.cols());
    Eigen::VectorXd ci0 = Eigen::VectorXd::Zero(CI.rows());

    Eigen::VectorXd x;
    igl::copyleft::quadprog(G, g0, {}, {}, CI, ci0, x);
    return x;

    mtao::VecXd R = (A.transpose() * A).ldlt().solve(A.transpose() * b);

    return R;
}

std::vector<std::array<size_t, 3>> DIWLevinVEMMesh2::cell_cut_edges(
    size_t cell_index) const {
    std::vector<std::array<size_t, 3>> cut_edges;
    const auto& boundary = cells.at(cell_index);
    for (auto&& [idx, pr] : mtao::iterator::enumerate(boundary)) {
        auto [eidx, sgn] = pr;
        auto e = E(eidx);
        std::vector<size_t> vertices;
        auto [start, end] = boundary_internal_index_range(eidx);
        if (start >= end) {
            cut_edges.emplace_back(std::array<size_t, 3>{
                {size_t(idx), size_t(e(0)), size_t(e(1))}});
        } else {
            start += num_vertices();
            end += num_vertices();
            cut_edges.emplace_back(
                std::array<size_t, 3>{{size_t(idx), size_t(e(0)), start}});
            cut_edges.emplace_back(
                std::array<size_t, 3>{{size_t(idx), end - 1, size_t(e(1))}});
            for (size_t i = start; i < end - 1; ++i) {
                cut_edges.emplace_back(
                    std::array<size_t, 3>{{size_t(idx), i, i + 1}});
            }
        }
    }
    return cut_edges;
}
double DIWLevinVEMMesh2::per_edge_cut_edge_length(size_t edge_index) const {
    auto e = E(edge_index);
    auto a = V(e(0));
    auto b = V(e(1));
    return double((b - a).norm()) / per_edge_num_cut_edges(edge_index);
}
size_t DIWLevinVEMMesh2::per_edge_num_cut_edges(size_t edge_index) const {
    // there's one more cut-edge than the number of cut-edges
    return num_interior_samples_on_boundary(edge_index) + 1;
}

std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
DIWLevinVEMMesh2::orthogonal_neumann_entries(
    const std::map<size_t, double>& target_fluxes) const {
    if (target_fluxes.size() == 0) {
        return {};
    }
    // flatten the edge indices into a coherent order
    std::vector<size_t> edge_indices(target_fluxes.size());
    std::transform(target_fluxes.begin(), target_fluxes.end(),
                   edge_indices.begin(),
                   [](auto&& pr) { return std::get<0>(pr); });

    // get the number of indices for each segment
    std::vector<size_t> boundary_sizes(target_fluxes.size());
    std::transform(edge_indices.begin(), edge_indices.end(),
                   boundary_sizes.begin(),
                   std::bind(&VEMMeshBase::num_interior_samples_on_boundary,
                             dynamic_cast<const VEMMeshBase*>(this),
                             std::placeholders::_1));

    // get offsets for easy acccess
    std::vector<size_t> constraint_matrix_offsets(target_fluxes.size() + 1);
    std::partial_sum(edge_indices.begin(), edge_indices.end(),
                     constraint_matrix_offsets.begin() + 1);

    auto COB = coboundary();
    // for (auto it = COB.begin(); it != COB.end();) {
    //    if (target_fluxes.find(it->first) != target_fluxes.end()) {
    //        ++it;
    //    } else {
    //        COB.erase(it);
    //    }
    //}
    for (auto&& [c, b] : COB) {
        fmt::print("COB {}: {} {}\n", c, b[0], b[1]);
    }
    std::map<size_t, mtao::MatXd> cell_projectors;
    for (auto&& [eidx, pr] : COB) {
        for (auto&& a : pr) {
            if (a >= 0) {
                if (cell_projectors.find(a) == cell_projectors.end()) {
                    cell_projectors[a] = poly_projection(a);
                }
            }
        }
    }
    for (auto&& [a, b] : cell_projectors) {
        std::cout << "Have projector for " << a << std::endl;
    }

    // first we construct a map from polynomials -> satisfaction of the neumann
    // constraint
    size_t size = constraint_matrix_offsets.back();
    Eigen::SparseMatrix<double> A(target_fluxes.size(), num_samples());
    std::vector<Eigen::Triplet<double>> trips;

    for (auto&& [local_edge_index, edge_index] :
         mtao::iterator::enumerate(edge_indices)) {
        size_t offset = constraint_matrix_offsets.at(local_edge_index);
        auto e = E(edge_index);

        // a bit of error checking for unused edges:
        auto it = COB.find(edge_index);
        if (it == COB.end()) {
            continue;
        }
        auto [neg_cell, pos_cell] = it->second;

        double target_value = target_fluxes.at(edge_index);
        auto samples = boundary_sample_indices_vec(edge_index);
        auto B = boundary_basis(edge_index);
        auto N = B.col(1);

        for (auto&& [cidx, sign] : {std::make_tuple(neg_cell, false),
                                    std::make_tuple(pos_cell, true)}) {
            if (cidx < 0) {
                continue;
            }
            std::cout << "Getting projector for " << cidx << std::endl;
            const auto& projector = cell_projectors.at(cidx);
            double orient = sign ? -1 : 1;
            for (auto&& [local_ind, sample_index] :
                 mtao::iterator::enumerate(samples)) {
                size_t coeff_col = coefficient_size() * (local_ind + offset);

                auto g = polynomial_grad_entries(cidx,
                                                 sample_position(sample_index));
                mtao::RowVecXd Nd = N.transpose() * g * projector;

                double dx =
                    ((local_ind == 0 || local_ind == (samples.size() - 1))
                         ? .5
                         : 1.) /
                    (samples.size() - 1);
                for (auto&& [local_ind, sample_index] :
                     mtao::iterator::enumerate(samples)) {
                    trips.emplace_back(local_edge_index, sample_index,
                                       dx * orient * Nd(local_ind));
                }
            }
        }
    }
    A.setFromTriplets(trips.begin(), trips.end());
    mtao::VecXd r(target_fluxes.size());
    for (auto&& [a, b] : mtao::iterator::enumerate(edge_indices)) {
        r(a) = target_fluxes.at(b);
    }
    return {A, r};
}

mtao::MatXd DIWLevinVEMMesh2::laplacian_sample2sample(
    const std::set<size_t>& disengaged_cells) const {
    return {};
}

Eigen::SparseMatrix<double> DIWLevinVEMMesh2::gradient_sample2poly() const {
    return {};
}
