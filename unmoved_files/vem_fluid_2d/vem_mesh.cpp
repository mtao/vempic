#include "vem_mesh.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <array>
#include <iostream>
#include <mtao/eigen/stack.hpp>
#include <mtao/iterator/enumerate.hpp>
#include <mtao/solvers/linear/preconditioned_conjugate_gradient.hpp>

VEMMeshBase::VEMMeshBase() {}
VEMMeshBase::~VEMMeshBase() {}

size_t VEMMeshBase::num_samples() const {
    return num_vertices() + num_boundary_samples();
}
size_t VEMMeshBase::num_boundary_samples() const {
    return boundary_sample_offsets.back();
}
std::array<size_t, 2> VEMMeshBase::boundary_internal_index_range(
    size_t boundary_index) const {
    return {{boundary_sample_offsets.at(boundary_index),
             boundary_sample_offsets.at(boundary_index + 1)}};
}

size_t VEMMeshBase::boundary_internal_index_offset(
    size_t boundary_index) const {
    return boundary_sample_offsets.at(boundary_index);
}

size_t VEMMeshBase::num_interior_samples_on_boundary(
    size_t boundary_index) const {
    assert(boundary_index < boundary_sample_offsets.size());
    auto [start, end] = boundary_internal_index_range(boundary_index);
    assert(end >= start);
    return end - start;
}
std::set<size_t> VEMMeshBase::cell_sample_indices(size_t cell_index) const {
    std::set<size_t> ret;
    for (auto&& [eidx, sgn] : cells.at(cell_index)) {
        ret.merge(boundary_sample_indices(eidx));
    }
    return ret;
}
std::vector<size_t> VEMMeshBase::cell_sample_indices_vec(
    size_t cell_index) const {
    auto indset = cell_sample_indices(cell_index);
    std::vector<size_t> indices(indset.begin(), indset.end());
    return indices;
}

size_t VEMMeshBase::boundary_internal_sample_index(size_t bound_index,
                                                   size_t sample_index) const {
    auto [start, end] = boundary_internal_index_range(bound_index);
    assert(sample_index + start < end);
    return start + sample_index;
}
std::vector<size_t> VEMMeshBase::boundary_indices_vec(
    size_t bound_index) const {
    auto a = boundary_indices(bound_index);
    return {a.begin(), a.end()};
}
std::vector<size_t> VEMMeshBase::boundary_sample_indices_vec(
    size_t bound_index) const {
    auto a = boundary_sample_indices(bound_index);
    return {a.begin(), a.end()};
}
std::set<size_t> VEMMeshBase::boundary_sample_indices(
    size_t bound_index) const {
    auto ret = boundary_vertex_indices(bound_index);
    auto [start, end] = boundary_internal_index_range(bound_index);
    size_t off = num_vertices();
    for (size_t i = start; i < end; ++i) {
        ret.emplace(i + off);
    }
    return ret;
}

Eigen::SparseMatrix<double> VEMMeshBase::boundary_facets_to_world(
    size_t cell_index) const {
    std::vector<size_t> bi = boundary_indices_vec(cell_index);
    Eigen::SparseMatrix<double> A(num_boundaries(), bi.size());
    A.reserve(bi.size());
    for (auto&& [col, row] : mtao::iterator::enumerate(bi)) {
        A.insert(row, col) = 1.;
    }
    return A;
}

size_t VEMMeshBase::num_cells() const { return cells.size(); }

mtao::MatXd VEMMeshBase::poly_coefficient_matrix(size_t index) const {
    auto indices = cell_sample_indices_vec(index);
    return poly_coefficient_matrix(index, indices);
}

mtao::MatXd VEMMeshBase::poly_coefficient_matrix(
    size_t index, const std::vector<size_t>& indices) const {
    mtao::MatXd coeff_mat(coefficient_size(), indices.size());

    for (auto&& [idx, ind] : mtao::iterator::enumerate(indices)) {
        mtao::VecXd pe = polynomial_entries(index, ind).transpose();
        coeff_mat.col(idx) = pe;
    }
    return coeff_mat;
}

Eigen::SparseMatrix<double> VEMMeshBase::cell_sample_to_world_sample(
    size_t cell_index) const {
    auto indices = cell_sample_indices_vec(cell_index);
    Eigen::SparseMatrix<double> L(num_samples(), indices.size());
    std::vector<Eigen::Triplet<int>> trips;
    trips.reserve(indices.size());
    for (auto&& [local, world] : mtao::iterator::enumerate(indices)) {
        trips.emplace_back(world, local, 1);
    }
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}
mtao::MatXd VEMMeshBase::laplacian_sample2sample(
    const std::set<size_t>& disengaged_cells) const {
    size_t system_size = num_samples();
    mtao::MatXd LD(system_size, system_size);  // = L;
    LD.setZero();
    for (size_t cidx = 0; cidx < num_cells(); ++cidx) {
        if (disengaged_cells.find(cidx) != disengaged_cells.end()) {
            continue;
        }
        if (cells.at(cidx).size() > 0) {
            auto C = cell_sample_to_world_sample(cidx);
            auto PL = per_cell_laplacian(cidx);
            // std::cout << "Permutation:\n";
            // std::cout << C << std::endl;

            LD.topLeftCorner(num_samples(), num_samples()) +=
                C * PL * C.transpose();
        }
    }
    return LD;
}
mtao::VecXd VEMMeshBase::laplace_problem(
    const std::map<size_t, double>& constrained_vertices,
    const std::map<size_t, double>& orthogonal_neumann_samples,
    const std::set<size_t>& disengaged_cells) const {
    return poisson_problem(mtao::VecXd::Zero(num_samples()),
                           constrained_vertices, orthogonal_neumann_samples,
                           disengaged_cells);
}
mtao::VecXd VEMMeshBase::poisson_problem(
    const mtao::VecXd& coefficient_rhs,
    const std::map<size_t, double>& constrained_vertices,
    const std::map<size_t, double>& orthogonal_neumann_samples,
    const std::set<size_t>& disengaged_cells) const {
    // auto [P, V] = dirichlet_constraints(constrained_vertices);

    size_t constraint_size =
        constrained_vertices.size() + orthogonal_neumann_samples.size();
    size_t system_size = num_samples() + constraint_size;
    // size_t system_size = num_samples() + V.rows();
    // Eigen::SparseMatrix<double> L(system_size, system_size);
    auto LD = laplacian_sample2sample(disengaged_cells);
    fmt::print("Laplacian size: {}x{}\n", LD.rows(), LD.cols());
    LD.conservativeResize(system_size, system_size);
    fmt::print("System size: {}x{}\n", LD.rows(), LD.cols());
    // mtao::VecXd rhs = mtao::eigen::vstack(mtao::VecXd::Zero(num_samples()),
    // V);
    mtao::VecXd rhs = mtao::VecXd::Zero(system_size);
    rhs.head(num_samples()) = coefficient_rhs;
    if (constrained_vertices.size() > 0) {
        auto [C, CR] = dirichlet_entries(constrained_vertices);
        fmt::print("Dirichlet size: {}x{}\n", C.rows(), C.cols());
        fmt::print("Dirichlet block: {}x{} => {}x{}\n", 0, num_samples(),
                   C.cols(), num_samples() + C.rows());
        LD.block(0, num_samples(), C.cols(), C.rows()) = C.transpose();
        LD.block(num_samples(), 0, C.rows(), C.cols()) = C;
        rhs.segment(num_samples(), CR.size()) = CR;
    }
    if (orthogonal_neumann_samples.size() > 0) {
        auto [N, NR] = orthogonal_neumann_entries(orthogonal_neumann_samples);
        fmt::print("Neumann size: {}x{}\n", N.rows(), N.cols());
        std::cout << N << std::endl;
        size_t off = num_samples() + constrained_vertices.size();
        LD.block(0, off, N.cols(), N.rows()) = N.transpose();
        LD.block(off, 0, N.rows(), N.cols()) = N;
        rhs.segment(off, NR.size()) = NR;
    }
    LD.bottomRightCorner(constraint_size, constraint_size).setZero();
    mtao::VecXd sol(rhs.rows());
    sol.setZero();
    // std::cout << "L\n" << LD << std::endl;
    // std::cout << "rhs\n" << rhs.transpose() << std::endl;
    // auto solver = LD.householderQr();
    // auto solver = LD.ldlt();
    // if (solver.info() != Eigen::Success) {
    //    spdlog::error("LDLT failed to compute!");
    //}
    // sol = solver.solve(rhs);
    // if (solver.info() != Eigen::Success) {
    //    spdlog::error("LDLT failed to solve!");
    //}

    mtao::solvers::linear::CholeskyPCGSolve(LD, rhs, sol);
    // std::cout << "LD:\n" << LD << std::endl;
    // std::cout << "RHS: " << rhs.transpose() << std::endl;
    // std::cout << "sol: " << sol.transpose() << std::endl;
    std::cout << "sol error: " << (LD * sol - rhs).transpose() << std::endl;
    // LD.block(0, num_samples(), P.rows(), P.cols()) = P;
    // LD.block(num_samples(), 0, P.cols(), P.rows()) = P.transpose();
    // Eigen::SparseMatrix<double> ConstOff(system_size, system_size);
    // Eigen::SparseMatrix<double> PS = P.sparseView();
    // for (int j = 0; j < P.cols(); ++j) {
    //    ConstOff.coeffRef(j, coefficient_size() + j) = 1;
    //}
    // L += (ConstOff * PS).transpose();
    // L += PS * ConstOff;

    return sol.topRows(num_samples());
}

/*
std::tuple<mtao::MatXd, mtao::VecXd>
VEMMeshBase::schur_complement_dirichlet_constraints( const std::map<size_t,
double>& constrained_vertices) const { std::vector<mtao::MatXd> mats;
    std::vector<mtao::VecXd> vecs;
    mats.resize(num_cells());
    vecs.resize(num_cells());

    for (auto&& [cell_index, mat, vec] :
         mtao::iterator::enumerate(mats, vecs)) {
        auto PL = per_monomial_laplacian(cell_index);

        std::vector<size_t> indices = cell_sample_indices_vec(cell_index);
        std::map<size_t, double> local_dirichlet_vertices;
        for (size_t idx = 0; idx < PL.cols(); ++idx) {
            size_t sample_index = indices.at(idx);
            if (constrained_vertices.find(sample_index) !=
                constrained_vertices.end()) {
                local_dirichlet_vertices[idx] =
                    constrained_vertices.at(sample_index);
            }
        }
        size_t dsize = local_dirichlet_vertices.size();
        if (dsize == 0) {
            continue;
        }
        mat.resize(coefficient_size(), dsize);
        vec.resize(dsize);

        std::map<size_t, size_t> index_inverter;
        for (auto&& [idx, pr] :
             mtao::iterator::enumerate(local_dirichlet_vertices)) {
            index_inverter[std::get<0>(pr)] = idx;
        }
        auto coeff_mat = poly_coefficient_matrix(cell_index, indices);
        for (auto&& [idx, val] : local_dirichlet_vertices) {
            size_t index = index_inverter.at(idx);
            vec(index) = val;
            auto p = coeff_mat.col(idx);
            mat.col(index) = p;
        }
    }
    return {mtao::eigen::hstack_iter(mats.begin(), mats.end()),
            mtao::eigen::vstack_iter(vecs.begin(), vecs.end())};
}
*/
std::tuple<Eigen::SparseMatrix<double>, mtao::VecXd>
VEMMeshBase::dirichlet_entries(
    const std::map<size_t, double>& dirichlet_vertices) const {
    size_t system_size = num_samples();
    size_t constraint_size = dirichlet_vertices.size();
    Eigen::SparseMatrix<double> A(constraint_size, system_size);
    A.reserve(dirichlet_vertices.size());
    mtao::VecXd rhs = mtao::VecXd::Zero(constraint_size);
    for (auto&& [idx, pr] : mtao::iterator::enumerate(dirichlet_vertices)) {
        int r = idx;
        auto [c, v] = pr;
        spdlog::info("r{0},c{1} := v{2} c{1}<{3}", r, c, v, system_size);
        A.insert(r, c) = 1;
        rhs(r) = v;
    }
    return {A, rhs};
}

size_t VEMMeshBase::coefficient_size() const { return coefficient_size(order); }
std::optional<size_t> VEMMeshBase::get_sample_parent_boundary(
    size_t sample_index) const {
    if (sample_index < num_vertices()) {
        return {};
    } else {
        sample_index -= num_vertices();

        auto lb = std::lower_bound(boundary_sample_offsets.begin(),
                                   boundary_sample_offsets.end(), sample_index);
        if (*lb > sample_index) {
            lb--;
        }
        return std::distance(boundary_sample_offsets.begin(), lb);
    }
}

Eigen::SparseMatrix<double> VEMMeshBase::index_map(
    const std::vector<size_t>& from, const std::vector<size_t>& to,
    size_t size) {
    if (to.empty()) {
        size_t max_coeff = *std::max_element(from.begin(), from.end());
        Eigen::SparseMatrix<double> R(size, from.size());
        for (auto&& [col, row] : mtao::iterator::enumerate(from)) {
            R.coeffRef(row, col) =
                1;  // this operation is fast for this type of matrix!
        }
        return R;
    } else {
        Eigen::SparseMatrix<double> R(to.size(), from.size());
        std::map<size_t, size_t> tomap;
        for (auto&& [a, b] : mtao::iterator::enumerate(to)) {
            tomap[b] = a;
        }
        for (auto&& [col, internal] : mtao::iterator::enumerate(from)) {
            if (auto it = tomap.find(internal); it != tomap.end()) {
                R.coeffRef(it->second, col) = 1;
            }
        }
        return R;
    }
}

Eigen::SparseMatrix<double> VEMMeshBase::local_to_world(
    const std::vector<size_t>& from) const {
    return index_map(from, {}, num_samples());
}
Eigen::SparseMatrix<double> VEMMeshBase::cell_to_world(size_t index) const {
    return local_to_world(cell_sample_indices_vec(index));
}
Eigen::SparseMatrix<double> VEMMeshBase::boundary_to_world(size_t index) const {
    return local_to_world(boundary_sample_indices_vec(index));
}

std::map<size_t, std::array<int, 2>> VEMMeshBase::coboundary() const {
    constexpr static int bad = (-1);
    const std::array<int, 2> def{{bad, bad}};
    std::map<size_t, std::array<int, 2>> ret;
    for (auto&& [cell_index, c] : mtao::iterator::enumerate(cells)) {
        const size_t ci = cell_index;
        for (auto&& [fidx, s] : c) {
            auto [it, new_item] = ret.emplace(fidx, def);
            auto& arr = it->second;  // hte array that stores the entries
            int& entry = arr[s ? 1 : 0];
            // if (!new_item && entry == bad) {
            //    spdlog::warn(
            //        "Nonomanifold edge: cell {} has two edges with the same "
            //        "sign ({} {} with sign {})",
            //        cell_index, fidx, entry, s);
            //}
            entry = cell_index;
        }
    }
    return ret;
}

size_t VEMMeshBase::polynomial_size() const {
    return coefficient_size() * num_cells();
}
