#include "vem/vem_mesh2.hpp"

#include <spdlog/spdlog.h>

#include <mtao/eigen/mat_to_triplets.hpp>
#include <mtao/eigen/stack.hpp>
#include <mtao/eigen/stl2eigen.hpp>
#include <mtao/geometry/triangle_monomial_integrals.hpp>
#include <mtao/geometry/winding_number.hpp>
#include <mtao/quadrature/simpsons.hpp>
#include <numeric>
size_t VEMMesh2::num_vertices() const { return vertices.cols(); }

size_t VEMMesh2::num_edges() const { return edges.cols(); }
size_t VEMMesh2::num_boundaries() const { return num_edges(); }

void VEMMesh2::initialize_interior_offsets() {
    initialize_interior_offsets(desired_edge_counts);
}

void VEMMesh2::initialize_interior_offsets(size_t edge_count) {
    boundary_sample_offsets.resize(num_edges() + 1);
    for (size_t idx = 0; idx < num_edges() + 1; ++idx) {
        boundary_sample_offsets[idx] = edge_count * idx;
    }
}
mtao::Vec2d VEMMesh2::sample_position(size_t index) const {
    auto edge_index = get_sample_parent_boundary(index);
    if (edge_index) {
        index -= num_vertices();

        size_t edge_sub_index =
            index - boundary_internal_index_offset(*edge_index);
        ;
        return edge_sample(*edge_index, edge_sub_index);
    } else {
        return V(index);
    }
}

double VEMMesh2::polynomial_entry(size_t order, size_t index,
                                  const mtao::Vec2d& v) {
    return std::pow<double>(v(1), index) *
           std::pow<double>(v(0), order - index);
}
mtao::Vec2d VEMMesh2::polynomial_grad_entry(size_t order, size_t index,
                                            const mtao::Vec2d& v) {
    mtao::Vec2d g = mtao::Vec2d::Zero();
    size_t oindex = order - index;
    if (index > 0) {
        g(1) = index * std::pow<double>(v(1), index - 1) *
               std::pow<double>(v(0), oindex);
    }
    if (oindex > 0) {
        g(0) = (oindex)*std::pow<double>(v(1), index) *
               std::pow<double>(v(0), oindex - 1);
    }
    return g;
}

double VEMMesh2::polynomial_laplacian_entry(size_t order, size_t index,
                                            const mtao::Vec2d& v) {
    double val = 0.0;
    size_t oindex = order - index;
    if (index > 1) {
        val += index * (index - 1) * std::pow<double>(v(1), index - 2) *
               std::pow<double>(v(0), oindex);
    }
    if (oindex > 1) {
        val += (oindex) * (oindex - 1) * std::pow<double>(v(1), index) *
               std::pow<double>(v(0), oindex - 2);
    }
    return val;
}
mtao::ColVecs2d VEMMesh2::cell_sample_positions(size_t cell_index) const {
    auto indices = cell_sample_indices_vec(cell_index);
    mtao::ColVecs2d V(2, indices.size());
    for (int i = 0; i < indices.size(); ++i) {
        V.col(i) = sample_position(indices.at(i));
    }
    return V;
}
mtao::Vec2d VEMMesh2::edge_sample(size_t edge_index,
                                  size_t sample_index) const {
    auto e = E(edge_index);
    auto a = V(e(0));
    auto b = V(e(1));

    size_t num_samples = num_interior_samples_on_boundary(edge_index);

    double t = (sample_index + 1) / (num_samples + 1.);
    return (1 - t) * a + t * b;
}
std::vector<size_t> VEMMesh2::edge_samples(size_t boundary_index) const {
    std::vector<size_t> edge;
    auto e = E(boundary_index);
    auto [start, end] = boundary_internal_index_range(boundary_index);
    edge.resize(end - start + 2);

    edge.front() = e(0);
    edge.back() = e(1);
    if (end > start) {
        std::iota(edge.begin() + 1, edge.begin() + end - start - 1,
                  start + num_vertices());
    }
    return edge;
}
/*
mtao::MatXd VEMMesh2::per_cell_boundary_gradient(
    size_t cell_index) const {
    auto cell_indices = cell_sample_indices_vec(cell_index);
    std::map<size_t,size_t> cell_index_inverter;
    std::map<size_t,size_t> sample_index_inverter;
    size_t num_verts = 0;
    for (auto&& [idx, ind] : mtao::iterator::enumerate(cell_indices)) {
        cell_index_inverter[ind] = idx;
        if(ind < num_vertices()) {
            num_verts++;
        } else {
            size_t s = sample_index_inverter.size();
            sample_index_inverter[ind] = s;
        }
    }
    size_t num_interior_samples = sample_index_inverter.size();


    std::map<size_t,size_t> cell_edge_index_inverter;




    const auto& boundary = cells.at(cell_index);
    mtao::MatXd D(2 * boundary.size(), cell_indices.size());
    //mtao::MatXd D(2 * num_interior_samples, indices.size());
    mtao::MatXd A(coefficient_size(),coefficient_size());
    mtao::MatXd B(coefficient_size(), cell_indices.size());
    for (auto&& [eidx, sgn] : boundary) {
        fmt::print("Creating AB matrix for edge{} sgn={}\n", eidx,sgn);
        size_t local_edge_index = cell_edge_index_inverter.size();
        {
            cell_edge_index_inverter[eidx] = local_edge_index;
        }
        continue;

        std::vector<size_t> edge_indices = boundary_sample_indices_vec(eidx);
        Eigen::SparseMatrix<double> E2C = index_map(edge_indices,cell_indices);


        // coeffs -> coeffs
        mtao::MatXd Ae = per_boundary_per_monomial_product(cell_index, eidx);
        // samples -> coeffs
        mtao::MatXd Be = per_edge_per_monomial_linear_update(cell_index, eidx);


        A += E2C * Ae * E2C.transpose();
        B += Be * E2C.transpose();


        //Eigen::SelfAdjointEigenSolver<mtao::MatXd> solver(A);
        //std::cout << "Eigenvalues: " << solver.eigenvalues().transpose()
<<std::endl;
        //size_t usable_eigenvals =
        //    A.rows()-1;
        //mtao::VecXd invdiag =
        //    1.0 / solver.eigenvalues().tail(usable_eigenvals).array();

        //mtao::MatXd Ae =
        //solver.eigenvectors().rightCols(usable_eigenvals);
        //mtao::MatXd Ainv = Ae * invdiag.asDiagonal() * Ae.transpose();



        //mtao::MatXd AinvB = Ae * B;
    }
    //std::cout << "A: \n" << A << std::endl;
    mtao::MatXd AinvB = A.inverse() * B;
    AinvB = poly_projection_sample2poly(cell_index);

    //for(size_t poly_idx = 0; poly_idx < coefficient_size(); ++poly_idx) {
    //    fmt::print("Poly index: {}\n", poly_idx);
    //    mtao::VecXd C = mtao::VecXd::Unit(coefficient_size(),poly_idx);

    //    mtao::VecXd dat(cell_indices.size());
    //    for(auto&& [idx,ind]: mtao::iterator::enumerate(cell_indices)) {
    //        mtao::Vec2d p = sample_position(ind);
    //        dat(idx) = polynomial_eval(0,p,C)(0);
    //    }
    //    std::cout << "Dat: " << dat.transpose() << std::endl;

    //    std::cout << dat.transpose() * A * dat << std::endl;
    //    std::cout << "abinv dat: " << (AinvB * dat).transpose() << std::endl;

    //}

    D.setZero();
    for (auto&& [eidx, sgn] : boundary) {
            size_t local_edge_index = cell_edge_index_inverter.at(eidx);
        fmt::print("Making boundary {} sgn={} local_edge_idx={}\n", eidx,sgn,
local_edge_index);

        // the pertinent block

        auto De = D.block(2*(local_edge_index), 0, 2, D.cols());
        //fmt::print("DIWLEVIN A{}x{} B{}x{}\n ",
A.rows(),A.cols(),B.rows(),B.cols()); std::vector<size_t> edge_indices; auto
[start, end] = boundary_internal_index_range(eidx);
        edge_indices.reserve(end-start + 2);
        // edge
        auto e = E(eidx);
        // place vertices in
        edge_indices.emplace_back(e(0));
        size_t off = num_vertices();
        for (size_t i = start; i < end; ++i) {
            edge_indices.emplace_back(i + off);
        }
        edge_indices.emplace_back(e(1));


        //construct basis
        mtao::Mat2d basis;
        auto T = basis.col(0);
        auto N = basis.col(1);
        T = V(e(1)) - V(e(0));
        double edge_length = T.norm();
        T.normalize();
        N << -T.y(), T.x();




        // tangent component
        auto Tstuff = De.row(0);
        Tstuff(cell_index_inverter.at(e(0))) = -1./edge_length;
        Tstuff(cell_index_inverter.at(e(1))) = 1./edge_length;
        Tstuff *= sgn?-1:1;

        auto Nstuff = De.row(1);
        mtao::ColVecs2d grads(2,coefficient_size());
        //fmt::print("grads shape: {}x{}\n", grads.rows(),grads.cols());
        grads.setZero();
        double dt = edge_length / (edge_indices.size()-1);

        for (size_t i = 1; i < edge_indices.size()-1; ++i) {
            bool ends = i == 1 || i == edge_indices.size()-2;
            mtao::Vec2d p = sample_position(edge_indices.at(i));
            mtao::ColVecs2d T =  polynomial_grad_entries(cell_index,p);
            //std::cout << "p: " << p.transpose() << " => " << (p -
C(cell_index)).transpose() << std::endl; grads += (ends?1.5:1.) * T;
         }
        // these live along the edge samples, gotta reindex to the cell
        mtao::RowVecXd BB = dt * N.transpose() * grads * AinvB;


        Eigen::SparseMatrix<double> E2C = index_map(edge_indices,cell_indices);
        //std::cout << dt * grads << std::endl;
        //std::cout << N << std::endl;
        //std::cout << BB.rows() << "x" << BB.cols() << " <= " << E2C.cols() <<
"x" << E2C.rows() << std::endl;
        //std::cout << Nstuff.rows() << "x" << Nstuff.cols() << std::endl;
        Nstuff = BB;// * E2C.transpose();

        //std::cout << "Grads: \n" << dt * grads << std::endl;
        //std::cout << "Ngrads: \n" << N.transpose() * grads << std::endl;
        //std::cout << "ainvv: \n" << AinvB << std::endl;
        //fmt::print("operators: n{}x{} grad{}x{} ainvb{}x{}\n",
        //        N.rows(),N.cols(),
        //        grads.rows(),grads.cols(),
        //        AinvB.rows(),AinvB.cols());
        //std::cout << grads * AinvB << std::endl;
        //    std::cout << "Nthing: " << Nstuff << std::endl;

    }
    return D;
}
*/

mtao::RowVecXd VEMMesh2::polynomial_entries(size_t cell_index,
                                            const mtao::Vec2d& p) const {
    mtao::Vec2d v = p - C(cell_index);
    mtao::RowVecXd r(coefficient_size());
    r(0) = 1.0;  // set the constant values

    for (int d = 1; d <= order; ++d) {
        int off = (d * (d + 1)) / 2;
        for (int j = 0; j < d + 1; ++j) {
            double& coeff = r(off + j);
            coeff = polynomial_entry(d, j, v);
        }
    }
    return r;
}
mtao::ColVecs2d VEMMesh2::polynomial_grad_entries(size_t cell_index,
                                                  const mtao::Vec2d& p) const {
    mtao::Vec2d v = p - C(cell_index);
    mtao::ColVecs2d r(2, coefficient_size());
    r.setConstant(0);  // set the constant values

    for (int d = 1; d <= order; ++d) {
        int off = (d * (d + 1)) / 2;
        for (int j = 0; j < d + 1; ++j) {
            auto g = r.col(off + j);
            g = polynomial_grad_entry(d, j, v);
            // double& coeff = r(off + j) = 1.0;
        }
    }
    return r;
}
mtao::RowVecXd VEMMesh2::polynomial_laplacian_entries(
    size_t cell_index, const mtao::Vec2d& p) const {
    mtao::Vec2d v = p - C(cell_index);
    mtao::RowVecXd r(coefficient_size());
    r.setConstant(0);  // set the constant values

    for (int d = 1; d <= order; ++d) {
        int off = (d * (d + 1)) / 2;
        for (int j = 0; j < d + 1; ++j) {
            double& l = r(off + j);
            l = polynomial_laplacian_entry(d, j, v);
            // double& coeff = r(off + j) = 1.0;
        }
    }
    return r;
}
mtao::ColVecs2d VEMMesh2::polynomial_grad(
    size_t index, const mtao::Vec2d& p,
    const mtao::RowVecXd& coefficients) const {
    return polynomial_grad_entries(index, p) * coefficients.transpose();
}

mtao::ColVecs2d VEMMesh2::polynomial_grad(
    size_t index, int sample_index, const mtao::RowVecXd& coefficients) const {
    return polynomial_grad(index, sample_position(sample_index), coefficients);
}
mtao::RowVecXd VEMMesh2::polynomial_laplacian(
    size_t index, int sample_index, const mtao::RowVecXd& coefficients) const {
    return polynomial_laplacian(index, sample_position(sample_index),
                                coefficients);
}
mtao::RowVecXd VEMMesh2::polynomial_laplacian(
    size_t index, const mtao::Vec2d& p,
    const mtao::RowVecXd& coefficients) const {
    return polynomial_laplacian_entries(index, p) * coefficients.transpose();
}
double VEMMesh2::winding_number(size_t cell_size, const mtao::Vec2d& p) const {
    std::map<int, bool> a;
    for (auto&& [i, s] : cells.at(cell_size)) {
        a[i] = s;
    }
    return mtao::geometry::winding_number(vertices, edges, a, p);
}
bool VEMMesh2::is_inside(size_t size, const mtao::Vec2d& p) const {
    return std::abs(winding_number(size, p)) > 1e-3;
}
std::set<size_t> VEMMesh2::boundary_vertex_indices(size_t edge_index) const {
    auto e = E(edge_index);
    return {size_t(e(0)), size_t(e(1))};
}

std::set<size_t> VEMMesh2::boundary_indices(size_t cell_index) const {
    std::set<size_t> edge_indices;
    for (auto&& [k, v] : cells.at(cell_index)) {
        edge_indices.emplace(k);
    }
    return edge_indices;
}

mtao::RowVecXd VEMMesh2::polynomial_entries(size_t cell_index,
                                            size_t sample_index) const {
    return polynomial_entries(cell_index, sample_position(sample_index));
}
size_t VEMMesh2::coefficient_size(size_t order) const {
    // for each order d we can pick 0...d i.e d+1 options for the first var,
    // second is forced by this \sum_{d=0}^order d+1 = \sum_{d=1}^{order+1} =
    // (order+1)(order+1+1)/2
    return (order + 1) * (order + 2) / 2;
}

mtao::VecXd VEMMesh2::per_cell_per_monomial_integral(size_t index) const {
    return per_cell_per_monomial_integral(index, order);
}
mtao::VecXd VEMMesh2::per_cell_per_monomial_integral(size_t index,
                                                     size_t max_order) const {
    Eigen::VectorXd monomial_integrals((max_order + 1) * (max_order + 2) / 2);
    monomial_integrals.setZero();
    auto a = C(index);
    for (auto&& [eidx, sgn] : cells.at(index)) {
        auto e = E(eidx);
        mtao::Vec2d b = V(e(sgn ? 1 : 0)) - a;
        mtao::Vec2d c = V(e(sgn ? 0 : 1)) - a;

        auto v = mtao::geometry::triangle_monomial_integrals<double>(
            max_order, mtao::Vec2d::Zero(), b, c);
        monomial_integrals += mtao::eigen::stl2eigen(v);
    }
    return monomial_integrals;
}

mtao::VecXd VEMMesh2::monomial_integrals(
    const std::set<size_t>& disengaged_cells) const {
    mtao::VecXd R = mtao::VecXd::Zero(num_cells() * coefficient_size());
    int off = coefficient_size();
    for (int j = 0; j < num_cells(); ++j) {
        if (disengaged_cells.find(j) != disengaged_cells.end()) {
            continue;
        }
        auto integrals = per_cell_per_monomial_integral(j);
        R.segment(off * j, integrals.size()) = integrals;
    }
    return R;
}

Eigen::SparseMatrix<double> VEMMesh2::poly_gradient() const {
    size_t size = coefficient_size();
    int D = 2;
    Eigen::SparseMatrix<double> R(D * size, size);
    std::vector<Eigen::Triplet<double>> trips =
        mtao::eigen::mat_to_triplets(poly_d(0));
    trips.reserve(D * trips.size());

    for (int d = 1; d < D; ++d) {
        auto dt = mtao::eigen::mat_to_triplets(poly_d(d));
        int off = d * size;
        std::transform(dt.begin(), dt.end(), dt.begin(),
                       [off](const Eigen::Triplet<double>& t) {
                           return Eigen::Triplet<double>{t.row() + off, t.col(),
                                                         t.value()};
                       });
        trips.insert(trips.end(), dt.begin(), dt.end());
    }
    R.setFromTriplets(trips.begin(), trips.end());
    return R;
}

Eigen::SparseMatrix<double> VEMMesh2::poly_d(int dim) const {
    size_t size = coefficient_size();
    Eigen::SparseMatrix<double> R(size, size);
    for (int d = 1; d <= order; ++d) {
        int off = (d * (d + 1)) / 2;
        int prevoff = ((d - 1) * (d)) / 2;
        for (int j = 0; j < d + 1; ++j) {
            if (dim == 0 && j < d) {
                int x = prevoff + j;
                auto [a, b] = monomial_index_to_powers(off + j);
                auto [c, f] = monomial_index_to_powers(x);
                R.insert(x, off + j) = d - j;
            }
            if (dim == 1 && j > 0) {
                int y = prevoff + j - 1;
                auto [a, b] = monomial_index_to_powers(off + j);
                auto [c, f] = monomial_index_to_powers(y);
                R.insert(y, off + j) = j;
            }
        }
    }
    return R;
}

Eigen::SparseMatrix<double> VEMMesh2::poly_dx() const { return poly_d(0); }
Eigen::SparseMatrix<double> VEMMesh2::poly_dy() const { return poly_d(1); }
mtao::VecXd VEMMesh2::per_cell_divergence(const mtao::VecXd& a,
                                          const mtao::VecXd& b) const {
    auto G = poly_gradient();
    mtao::VecXd div(coefficient_size());
    div = (G * a).head(coefficient_size());
    div += (G * b).tail(coefficient_size());
    return div;
}
mtao::VecXd VEMMesh2::divergence(const mtao::RowVecs2d& u) const {
    size_t off = coefficient_size();
    mtao::VecXd ret(off * num_cells());
    assert(u.rows() == ret.size());
    auto G = poly_gradient();
    auto a = u.col(0);
    auto b = u.col(1);
    for (size_t cidx = 0; cidx < num_cells(); ++cidx) {
        auto div = ret.segment(off * cidx, off);
        auto x = a.segment(off * cidx, off);
        auto y = b.segment(off * cidx, off);
        div = (G * x).head(coefficient_size());
        div += (G * y).tail(coefficient_size());
    }
    return ret;
}
Eigen::SparseMatrix<double> VEMMesh2::gradient() const {
    size_t off = coefficient_size();
    std::vector<Eigen::Triplet<double>> trips;
    auto G = poly_gradient();
    auto grad_trips = mtao::eigen::mat_to_triplets(G);

    Eigen::SparseMatrix<double> ret(2 * off * num_cells(), off * num_cells());
    trips.reserve(2 * num_cells() * grad_trips.size());

    for (int j = 0; j < num_cells(); ++j) {
        int col_off = off * j;
        int x_row_off = col_off;
        int y_row_off = col_off + ret.cols() - off;
        for (auto&& trip : grad_trips) {
            bool x_case = trip.row() < off;
            int row_off = x_case ? x_row_off : y_row_off;

            trips.emplace_back(row_off + trip.row(), col_off + trip.col(),
                               trip.value());
        }
    }
    ret.setFromTriplets(trips.begin(), trips.end());

    return ret;
}
Eigen::SparseMatrix<double> VEMMesh2::laplacian(
    const std::set<size_t>& disengaged_cells) const {
    size_t off = coefficient_size();
    std::vector<Eigen::Triplet<double>> trips;
    auto G = poly_gradient();
    auto grad_trips = mtao::eigen::mat_to_triplets(G);

    Eigen::SparseMatrix<double> ret(off * num_cells(), off * num_cells());
    trips.reserve(2 * num_cells() * grad_trips.size());

    for (int j = 0; j < num_cells(); ++j) {
        if (disengaged_cells.find(j) != disengaged_cells.end()) {
            continue;
        }
        auto L = per_monomial_laplacian(j);
        // std::cout << "percelllap\n" << L << std::endl;
        int row_off = off * j;
        int col_off = off * j;
        for (int k = 0; k < off; ++k) {
            for (int l = 0; l < off; ++l) {
                double val = L(k, l);
                if (val != 0) {
                    trips.emplace_back(row_off + k, col_off + l, val);
                }
            }
        }
    }
    ret.setFromTriplets(trips.begin(), trips.end());

    return ret;
}
// Eigen::SparseMatrix<double> VEMMesh2::laplacian_sample2sample() const {
mtao::MatXd VEMMesh2::laplacian_sample2sample(
    const std::set<size_t>& disengaged_cells) const {
    auto G = gradient_sample2poly();
    auto L = monomial_integrals(disengaged_cells);
    mtao::VecXd L2 = mtao::eigen::vstack(L, L);
    return G.transpose() * L2.asDiagonal() * G;

    // auto S2C = sample2cell_coefficients();
    // return S2C.transpose() * laplacian(disengaged_cells) * S2C;
}

Eigen::SparseMatrix<double> VEMMesh2::divergence() const {
    size_t off = coefficient_size();
    std::vector<Eigen::Triplet<double>> trips;
    auto G = poly_gradient();
    auto grad_trips = mtao::eigen::mat_to_triplets(G);

    Eigen::SparseMatrix<double> ret(off * num_cells(), 2 * off * num_cells());
    trips.reserve(2 * num_cells() * grad_trips.size());

    for (int j = 0; j < num_cells(); ++j) {
        int x_row_off = off * j;
        int y_row_off = off * j - off;
        int x_col_off = x_row_off;
        int y_col_off = ret.rows() + x_col_off;
        for (auto&& trip : grad_trips) {
            bool x_case = trip.row() < off;
            int row_off = x_case ? x_row_off : y_row_off;
            int col_off = x_case ? x_col_off : y_col_off;

            trips.emplace_back(row_off + trip.row(), col_off + trip.col(),
                               trip.value());
        }
    }
    ret.setFromTriplets(trips.begin(), trips.end());

    return ret;
}

mtao::RowVecs2d VEMMesh2::gradient(const mtao::VecXd& a) const {
    size_t off = coefficient_size();
    mtao::RowVecs2d ret(off * num_cells(), 2);
    auto x = ret.col(0);
    auto y = ret.col(1);
    assert(x.size() == a.size());
    auto G = poly_gradient();
    for (size_t cidx = 0; cidx < num_cells(); ++cidx) {
        auto A = a.segment(off * cidx, off);
        auto X = x.segment(off * cidx, off);
        auto Y = y.segment(off * cidx, off);
        auto D = G * A;
        X = (D).head(coefficient_size());
        Y = (D).tail(coefficient_size());
    }
    return ret;
}

std::array<size_t, 2> VEMMesh2::monomial_index_to_powers(size_t index) const {
    std::array<size_t, 2> ret;
    size_t d = std::ceil(std::sqrt(index));
    size_t off;
    if (off = (d + 1) * (d + 2) / 2; index >= off) {
        d++;
    } else if (off = d * (d + 1) / 2; index < off) {
        d--;
        off = d * (d + 1) / 2;
    } else {
        off = d * (d + 1) / 2;
    }
    size_t j = index - off;
    return std::array<size_t, 2>{{d - j, j}};
}

size_t VEMMesh2::powers_to_monomial_index(size_t a, size_t b) const {
    size_t d = a + b;
    size_t offset = d * (d + 1) / 2;
    return offset + b;
}

mtao::Mat2d VEMMesh2::boundary_basis(size_t boundary_index) const {
    auto e = E(boundary_index);
    // construct basis
    mtao::Mat2d basis;
    auto T = basis.col(0);
    auto N = basis.col(1);
    T = V(e(1)) - V(e(0));
    double edge_length = T.norm();
    T.normalize();
    N << -T.y(), T.x();
    return basis;
}

Eigen::SparseMatrix<double> VEMMesh2::poly2sample(
    const std::set<int>& disengaged_cells) const {
    mtao::VecXi cell_counts = mtao::VecXi::Zero(num_samples());

    int off = coefficient_size();
    Eigen::SparseMatrix<double> R(num_samples(), num_cells() * off);

    std::vector<Eigen::Triplet<double>> trips;
    for (size_t cidx = 0; cidx < num_cells(); ++cidx) {
        if (disengaged_cells.find(cidx) != disengaged_cells.end()) {
            continue;
        }
        int cell_off = off * cidx;
        for (auto&& pidx : cell_sample_indices(cidx)) {
            cell_counts(pidx)++;
            auto entries = polynomial_entries(cidx, pidx);
            for (int j = 0; j < entries.size(); ++j) {
                trips.emplace_back(pidx, cell_off + j, entries(j));
            }
        }
    }
    R.setFromTriplets(trips.begin(), trips.end());
    mtao::VecXd C = (cell_counts.array() != 0)
                        .select(1. / cell_counts.array().cast<double>(), 0.);
    return C.asDiagonal() * R;
}

mtao::MatXd VEMMesh2::per_monomial_laplacian(size_t index) const {
    mtao::MatXd M(coefficient_size(), coefficient_size());

    auto monomial_integrals =
        per_cell_per_monomial_integral(index, 2 * order - 1);
    for (int d1 = 0; d1 <= order; ++d1) {
        int off1 = (d1 * (d1 + 1)) / 2;
        for (int j1 = 0; j1 < d1 + 1; ++j1) {
            for (int d2 = d1; d2 <= order; ++d2) {
                int off2 = (d2 * (d2 + 1)) / 2;
                for (int j2 = (d1 == d2) ? (j1) : 0; j2 < d2 + 1; ++j2) {
                    // \int \nabla (x^j y^k)  \cdot \nabla (x^l y^m
                    // = j * l \int x^{l+j-2} y^{k+m} + km x^{j+l} y^{k+m-2}
                    size_t j = j1;
                    size_t k = (d1 - j1);
                    size_t l = j2;
                    size_t m = (d2 - j2);
                    size_t d = d1 + d2 - 2;
                    // fmt::print("j{} + k{} = d{}; l{} + m{} = d{}; total order
                    // = {}\n", j,k,d1,l,m,d2, d);

                    size_t off = (d * (d + 1)) / 2;
                    double val = 0;
                    if (j >= 1 && l >= 1) {
                        val += j * l * monomial_integrals[off + j + l - 2];
                    }
                    if (k >= 1 && m >= 1) {
                        val += k * m * monomial_integrals[off + k + m - 2];
                    }
                    M(off2 + j2, off1 + j1) = M(off1 + j1, off2 + j2) = val;
                }
            }
        }
    }
    return M;
}
