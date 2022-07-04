#include "vem/flux_moment_cell3.hpp"

#include <mtao/algebra/pascal_triangle.hpp>
#include <vem/polynomial_gradient.hpp>
#include <vem/polynomial_utils.hpp>

#include "vem/flux_moment_indexer3.hpp"
using namespace vem::polynomials::three;
namespace vem {

FluxMomentVEM3Cell::FluxMomentVEM3Cell(const FluxMomentIndexer3& mom,
                                       size_t index)
    : VEM3Cell(mom.mesh(), index), _indexer(&mom) {
    size_t offset = 0;
    for (auto&& [eidx, sgn] : faces()) {
        _flux_index_offsets[eidx] = offset;
        offset += flux_size(eidx);
    }
}

// the number of DOFs in the whole problem
size_t FluxMomentVEM3Cell::sample_size() const {
    return indexer().sample_size();
}
// the number of DOFs in the local sample
size_t FluxMomentVEM3Cell::local_sample_size() const {
    return flux_size() + moment_size();
}

const FluxMomentIndexer3::MonomialIndexer&
FluxMomentVEM3Cell::monomial_indexer() const {
    return indexer().monomial_indexer();
}
const FluxMomentIndexer3& FluxMomentVEM3Cell::indexer() const {
    return *_indexer;
}

const FluxMomentIndexer3::MomentIndexer& FluxMomentVEM3Cell::moment_indexer()
    const {
    return indexer().moment_indexer();
}
const FluxMomentIndexer3::FluxIndexer& FluxMomentVEM3Cell::flux_indexer()
    const {
    return indexer().flux_indexer();
}

size_t FluxMomentVEM3Cell::flux_size() const {
    size_t size = 0;

    for (auto&& [idx, sgn] : mesh().cell_boundary_map.at(cell_index())) {
        size += flux_indexer().num_coefficients(idx);
    }
    return size;
}
size_t FluxMomentVEM3Cell::flux_degree(size_t face_index) const {
    return flux_indexer().degree(face_index);
}

size_t FluxMomentVEM3Cell::local_flux_index_offset(size_t face_index) const {
    return _flux_index_offsets.at(face_index);
}
size_t FluxMomentVEM3Cell::flux_size(size_t face_index) const {
    return flux_indexer().num_coefficients(face_index);
}
size_t FluxMomentVEM3Cell::moment_degree() const {
    return moment_indexer().degree(cell_index());
}
size_t FluxMomentVEM3Cell::moment_size() const {
    return moment_indexer().num_coefficients(cell_index());
}
size_t FluxMomentVEM3Cell::monomial_degree() const {
    return monomial_indexer().degree(cell_index());
}
size_t FluxMomentVEM3Cell::monomial_size() const {
    return monomial_indexer().num_coefficients(cell_index());
}

size_t FluxMomentVEM3Cell::local_moment_index_offset() const {
    return flux_size();
}
size_t FluxMomentVEM3Cell::moment_only_global_moment_index_offset() const {
    return moment_indexer().coefficient_offset(cell_index());
}
size_t FluxMomentVEM3Cell::global_moment_index_offset() const {
    return flux_indexer().num_coefficients() +
           moment_only_global_moment_index_offset();
}
size_t FluxMomentVEM3Cell::global_monomial_index_offset() const {
    return monomial_indexer().coefficient_offset(cell_index());
}

mtao::MatXd FluxMomentVEM3Cell::monomial_l2_grammian() const {
    return monomial_l2_grammian(monomial_degree());
}
mtao::MatXd FluxMomentVEM3Cell::monomial_dirichlet_grammian() const {
    return monomial_dirichlet_grammian(monomial_degree());
}

mtao::MatXd FluxMomentVEM3Cell::monomial_l2_face_grammian(
    size_t face_index) const {
    return VEM3Cell::monomial_l2_face_grammian(face_index, monomial_degree(),
                                               flux_degree(face_index));
}

mtao::MatXd FluxMomentVEM3Cell::l2_projector() const {
    auto G = monomial_l2_grammian();
    auto K = sample_monomial_l2_grammian();

    mtao::MatXd PS(G.cols(), K.rows());
    auto Glu = G.lu();
    for (int j = 0; j < K.rows(); ++j) {
        PS.col(j) = Glu.solve(K.row(j).transpose());
    }
    return PS;
}
mtao::MatXd FluxMomentVEM3Cell::dirichlet_projector() const {
    auto G = regularized_monomial_dirichlet_grammian();
    auto K = sample_monomial_dirichlet_grammian();

    mtao::MatXd PS(G.cols(), K.rows());
    auto Glu = G.lu();
    for (int j = 0; j < K.rows(); ++j) {
        PS.col(j) = Glu.solve(K.row(j).transpose());
    }
    return PS;
}

mtao::MatXd FluxMomentVEM3Cell::l2_sample_projector() const {
    auto Pis = l2_projector();
    auto B = monomial_evaluation();
    return B * Pis;
}
mtao::MatXd FluxMomentVEM3Cell::dirichlet_sample_projector() const {
    auto Pis = dirichlet_projector();
    auto B = monomial_evaluation();
    return B * Pis;
}
// max degree seen among all edges in cell
size_t FluxMomentVEM3Cell::flux_max_degree() const {
    size_t max_deg = -1;
    for (auto&& [e, sgn] : faces()) {
        size_t deg = flux_degree(e);
        if (max_deg == -1 || deg > max_deg) {
            max_deg = deg;
        }
    }
    return max_deg;
}

mtao::MatXd FluxMomentVEM3Cell::sample_monomial_l2_grammian() const {
    mtao::MatXd R(local_sample_size(), monomial_size());
    R.setZero();

    // R.topRows(flux_size()) =
    // monomial_evaluation().topRows(flux_size());

    auto P0 = R.col(0);
    if (moment_size() == 0) {
        // if moment size is 0, then flux sizes are 1, so the offsets are hte
        // indices
        for (auto&& [eidx, loc] : _flux_index_offsets) {
            P0(loc) = surface_area(eidx);
        }
        P0 /= P0.sum();
    } else {
        P0.setUnit(local_moment_index_offset());
    }
    // std::cout << "P0 filled \n" << R << std::endl;
    double vol = volume();
    // auto integrals =
    //    monomial_indexer().monomial_integrals(index, 2 * moment_degree());
    size_t mom_size = moment_size();
    size_t mom_off = local_moment_index_offset();
    // for row 1... n_{k-2} we use the moment inner product
    // n_{k-2} = moment size

    for (size_t j = 0; j < moment_size(); ++j) {
        int row = mom_off + j;
        R(row, j) = vol;
    }
    mtao::MatXd Cp = monomial_l2_grammian() * dirichlet_projector();

    size_t off = moment_size();  // k-2 polynomials
    // int block_size = monomial_size() - off - 1;
    int block_size = monomial_size() - off;
    R.rightCols(block_size) = Cp.transpose().rightCols(block_size);
    return R;
}

mtao::ColVecs3d FluxMomentVEM3Cell::monomial_gradient_in_face_coefficients(
    int face_index, int monomial_index) const {
    int face_degree = flux_indexer().degree(face_index);
    // mtao::ColVecs3d R(3,
    // polynomials::two::num_monomials_upto(polynomials::two::monomial_index_degree(monomial_index)
    // ));
    mtao::ColVecs3d R(3, flux_size(face_index));
    R.setZero();
    if (monomial_index <= 0) {
        return R;
    }

    auto g = gradient_single_index(monomial_index);

    double d = this->diameter();
    for (auto&& [dim, pr] : mtao::iterator::enumerate(g)) {
        const auto& [c, i] = pr;
        if (i >= 0) {
            auto pmi = project_monomial_to_boundary(face_index, i);
            mtao::RowVecXd A = c * pmi / d;
            R.row(dim).head(A.cols()) = A;
        }
    }
    return R;
}
mtao::MatXd FluxMomentVEM3Cell::sample_monomial_dirichlet_grammian() const {
    mtao::MatXd R(local_sample_size(), monomial_size());
    R.setZero();

    auto indices = flux_indices();
    std::map<size_t, double> surface_areas = this->surface_areas();

    auto P0 = R.col(0);
    if (moment_size() == 0) {
        for (auto&& [fidx, sgn] : faces()) {
            int fsize = flux_size(fidx);
            P0.segment(local_flux_index_offset(fidx), fsize)
                .setConstant(surface_areas.at(fidx));
        }
        P0 /= P0.sum();
    } else {
        P0.setUnit(local_moment_index_offset());
    }

    size_t max_degree = flux_max_degree();
    assert(max_degree != size_t(-1));
    auto EN = face_normals();
    double diameter = this->diameter();

    for (auto&& [fidx, sgn] : faces()) {
        mtao::Vec3d N = mtao::eigen::stl2eigen(EN[fidx]);
        double surface_area = surface_areas[fidx];
        int fsize = flux_size(fidx);
        int flux_monomial_offset = local_flux_index_offset(fidx);
        // spdlog::info("face {} got offset {} and normal {} {} {}", fidx,
        //             flux_monomial_offset, N.x(), N.y(), N.z());
        for (size_t poly_idx = 1; poly_idx < monomial_size(); ++poly_idx) {
            // col = poly_index

            // face_coefss has 3 rows and some number of poly wide
            // local_coeffs = coeffs * poly_coeffs
            auto face_coeffs =
                monomial_gradient_in_face_coefficients(fidx, poly_idx);
            auto row = (surface_area * N.transpose() * face_coeffs).eval();
            // spdlog::info("Wriging a row of size {} into {}x{}:{}x{} in a
            // {}x{}", row.size(),
            // fsize,1,flux_monomial_offset,poly_idx,R.rows(),R.cols());
            auto B = R.col(poly_idx);
            B.segment(flux_monomial_offset, row.size()) += row.transpose();
        }
    }
    double vol = volume();
    // spdlog::info("Cell volume: {}", vol);
    size_t mom_off = local_moment_index_offset();
    auto L = polynomials::three::laplacian(monomial_degree());
    // std::cout <<"Laplacian\n" << L << std::endl;
    double d2 = diameter * diameter;
    for (int o = 0; o < L.outerSize(); ++o) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, o); it; ++it) {
            int row = it.row() + mom_off;
            int col = it.col();
            R(row, col) = -it.value() * vol / d2;
        }
    }
    return R;
}

mtao::MatXd FluxMomentVEM3Cell::regularized_monomial_dirichlet_grammian()
    const {
    mtao::MatXd R = monomial_dirichlet_grammian();
    auto P0 = R.row(0);
    R.col(0).setZero();
    if (monomial_degree() == 1) {
        for (auto&& [fidx, sgn] : faces()) {
            // int fsize = flux_size(fidx);
            // auto integrals = monomial_face_integrals(fidx,
            // flux_degree(fidx)); mtao::Vec2d N =
            // mtao::eigen::stl2eigen(EN.at(fidx)); auto G =
            // monomial_to_monomial_gradient(flux_degree(fidx));
            // Eigen::SparseMatrix<double> GN =
            //    (N.x() * G.topRows(fsize) + N.y() * G.bottomRows(fsize))
            //        .topRows(fsize);
            auto R = monomial_l2_face_grammian(fidx);
            // std::cout << "Face " << fidx << " is emitting a l2 face
            // grammian\n "
            //          << r << std::endl;
            // spdlog::info("Block sizing: {}x{} vs {}x{}", MyBlock.rows(),
            //             MyBlock.cols(), r.rows(), r.cols());
            P0 += R.transpose();
        }
        P0 /= boundary_area();

    } else {
        const double length = boundary_area();
        const double area = volume();

        auto integrals = monomial_indexer().monomial_integrals(cell_index());
        P0 = integrals.transpose() / area;
    }

    return R;
}

mtao::MatXd FluxMomentVEM3Cell::l2_projector_error() const {
    auto Pi = l2_sample_projector();
    Pi.noalias() = mtao::MatXd::Identity(Pi.rows(), Pi.cols()) - Pi;
    return Pi;
}
mtao::MatXd FluxMomentVEM3Cell::dirichlet_projector_error() const {
    auto Pi = dirichlet_sample_projector();
    Pi.noalias() = mtao::MatXd::Identity(Pi.rows(), Pi.cols()) - Pi;
    return Pi;
}

std::vector<size_t> FluxMomentVEM3Cell::flux_indices() const {
    std::vector<size_t> indices(flux_size());
    for (auto&& [fidx, sgn] : faces()) {
        auto [start, end] = flux_indexer().coefficient_range(fidx);
        size_t offset = _flux_index_offsets.at(fidx);
        for (size_t j = 0; j < end - start; ++j) {
            indices[offset + j] = start + j;
        }
    }
    // spdlog::info("Given flux indices we've constructed: {}",
    //             fmt::join(indices, ","));
    return indices;
}

std::vector<size_t> FluxMomentVEM3Cell::sample_indices() const {
    auto s = flux_indices();
    s.reserve(s.size() + moment_size());
    auto [start, end] = moment_indexer().coefficient_range(cell_index());
    size_t off = flux_indexer().num_coefficients();
    start += off;
    end += off;
    for (size_t j = start; j < end; ++j) {
        s.emplace_back(j);
    }

    return {s.begin(), s.end()};
}

mtao::MatXd FluxMomentVEM3Cell::monomial_evaluation() const {
    mtao::MatXd R(local_sample_size(), monomial_size());
    R.setZero();
    double vol = volume();
    auto EN = face_normals();
    int index_offset = 0;
    for (auto&& [fidx, sgn] : faces()) {
        int fsize = flux_size(fidx);
        // auto integrals = monomial_face_integrals(eidx, flux_degree(eidx));
        // mtao::Vec2d N = mtao::eigen::stl2eigen(EN.at(eidx));
        // auto G = monomial_to_monomial_gradient(flux_degree(eidx));
        // Eigen::SparseMatrix<double> GN =
        //    (N.x() * G.topRows(fsize) + N.y() * G.bottomRows(fsize))
        //        .topRows(fsize);
        auto MyBlock =
            R.block(local_flux_index_offset(fidx), 0, fsize, R.cols());
        // spdlog::info("Block size: {}x{}", fsize, R.cols());
        MyBlock =
            monomial_l2_face_grammian(fidx).transpose() / surface_area(fidx);

        // std::cout << "MYBLOCK" << MyBlock << std::endl;
    }
    size_t mom_size = moment_size();
    if (mom_size == 0) {
        assert(moment_degree() == -1);
        return R;
    }
    assert(moment_degree() != -1);
    size_t mom_off = local_moment_index_offset();
    auto integrals =
        VEM3Cell::monomial_integrals(monomial_degree() + moment_degree());
    for (size_t j = 0; j < moment_size(); ++j) {
        int row = j + mom_off;
        auto [mxexp, myexp, mzexp] = index_to_exponents(j);
        for (int col = 0; col < monomial_size(); ++col) {
            auto [Mxexp, Myexp, Mzexp] = index_to_exponents(col);

            const double& ival = integrals(exponents_to_index(
                mxexp + Mxexp, myexp + Myexp, mzexp + Mzexp));
            R(row, col) = ival / vol;
            // spdlog::info("{} {} {} + {} {} {} => {} / {} = {}", mxexp, myexp,
            //             mzexp, Mxexp, Myexp, Mzexp, ival, vol, R(row, col));
        }
    }
    return R;
}

std::map<size_t, size_t> FluxMomentVEM3Cell::world_to_local_flux_indices()
    const {
    std::map<size_t, size_t> ret;
    auto pi = flux_indices();
    for (auto&& [idx, i] : mtao::iterator::enumerate(pi)) {
        ret[i] = idx;
    }
    return ret;
}
std::map<size_t, size_t> FluxMomentVEM3Cell::world_to_local_sample_indices()
    const {
    auto ret = world_to_local_flux_indices();
    size_t local_offset = local_moment_index_offset();
    size_t global_offset = global_moment_index_offset();
    auto [start, end] = moment_indexer().coefficient_range(cell_index());
    for (size_t j = 0; j < end - start; ++j) {
        ret[j + start + global_offset] = j + local_offset;
    }
    return ret;
}

mtao::iterator::detail::range_container<size_t>
FluxMomentVEM3Cell::local_to_world_monomial_indices() const {
    return monomial_indices();
}

mtao::iterator::detail::range_container<size_t>
FluxMomentVEM3Cell::flux_indices(size_t face_index) const {
    return flux_indexer().coefficient_indices(face_index);
}
mtao::iterator::detail::range_container<size_t>
FluxMomentVEM3Cell::moment_indices() const {
    return moment_indexer().coefficient_indices(cell_index());
}
mtao::iterator::detail::range_container<size_t>
FluxMomentVEM3Cell::monomial_indices() const {
    return monomial_indexer().coefficient_indices(cell_index());
}
std::vector<size_t> FluxMomentVEM3Cell::local_to_world_flux_indices() const {
    return flux_indices();
}
std::vector<size_t> FluxMomentVEM3Cell::local_to_world_sample_indices() const {
    return sample_indices();
}
Eigen::SparseMatrix<double> FluxMomentVEM3Cell::local_to_world_monomial_map()
    const {
    Eigen::SparseMatrix<double> R(monomial_indexer().num_coefficients(),
                                  monomial_size());
    auto [start, end] = monomial_indexer().coefficient_range(cell_index());
    for (size_t j = 0; j < end - start; ++j) {
        R.insert(j + start, j) = 1;
    }

    return R;
}
Eigen::SparseMatrix<double> FluxMomentVEM3Cell::local_to_world_sample_map()
    const {
    Eigen::SparseMatrix<double> R(sample_size(), local_sample_size());
    for (auto&& [r, c] : world_to_local_sample_indices()) {
        R.insert(r, c) = 1;
    }
    // std::cout << R << std::endl;

    return R;
}

mtao::VecXd FluxMomentVEM3Cell::project_monomial_to_boundary(
    size_t face_index, size_t cell_monomial_index) const {
    return VEM3Cell::project_monomial_to_boundary(face_index,
                                                  cell_monomial_index);
}
}  // namespace vem
