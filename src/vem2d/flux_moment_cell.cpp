
#include "vem/flux_moment_cell.hpp"

#include <mtao/algebra/pascal_triangle.hpp>
#include <vem/polynomial_gradient.hpp>
#include <vem/polynomial_utils.hpp>

#include "vem/flux_moment_indexer.hpp"
using namespace vem::polynomials::two;
namespace vem {

FluxMomentVEM2Cell::FluxMomentVEM2Cell(const FluxMomentIndexer& mom,
                                       size_t index)
    : VEM2Cell(mom.mesh(), index), _indexer(&mom) {
    size_t offset = 0;
    for (auto&& [eidx, sgn] : edges()) {
        _flux_index_offsets[eidx] = offset;
        offset += flux_size(eidx);
    }
}

// the number of DOFs in the whole problem
size_t FluxMomentVEM2Cell::sample_size() const {
    return _indexer->sample_size();
}
// the number of DOFs in the local sample
size_t FluxMomentVEM2Cell::local_sample_size() const {
    return flux_size() + moment_size();
}

const MonomialBasisIndexer& FluxMomentVEM2Cell::monomial_indexer() const {
    return _indexer->monomial_indexer();
}

const MomentBasisIndexer& FluxMomentVEM2Cell::moment_indexer() const {
    return _indexer->moment_indexer();
}
const FluxBasisIndexer& FluxMomentVEM2Cell::flux_indexer() const {
    return _indexer->flux_indexer();
}

size_t FluxMomentVEM2Cell::flux_size() const {
    size_t size = 0;

    for (auto&& [idx, sgn] : mesh().face_boundary_map.at(cell_index())) {
        size += flux_indexer().num_coefficients(idx);
    }
    return size;
}
size_t FluxMomentVEM2Cell::flux_degree(size_t edge_index) const {
    return flux_indexer().degree(edge_index);
}

size_t FluxMomentVEM2Cell::local_flux_index_offset(size_t edge_index) const {
    return _flux_index_offsets.at(edge_index);
}
size_t FluxMomentVEM2Cell::flux_size(size_t edge_index) const {
    return flux_indexer().num_coefficients(edge_index);
}
size_t FluxMomentVEM2Cell::moment_degree() const {
    return moment_indexer().degree(cell_index());
}
size_t FluxMomentVEM2Cell::moment_size() const {
    return moment_indexer().num_coefficients(cell_index());
}
size_t FluxMomentVEM2Cell::monomial_degree() const {
    return monomial_indexer().degree(cell_index());
}
size_t FluxMomentVEM2Cell::monomial_size() const {
    return monomial_indexer().num_coefficients(cell_index());
}

size_t FluxMomentVEM2Cell::local_moment_index_offset() const {
    return flux_size();
}
size_t FluxMomentVEM2Cell::moment_only_global_moment_index_offset() const {
    return moment_indexer().coefficient_offset(cell_index());
}
size_t FluxMomentVEM2Cell::global_moment_index_offset() const {
    return flux_indexer().num_coefficients() +
           moment_only_global_moment_index_offset();
}
size_t FluxMomentVEM2Cell::global_monomial_index_offset() const {
    return monomial_indexer().coefficient_offset(cell_index());
}

mtao::MatXd FluxMomentVEM2Cell::monomial_l2_grammian() const {
    return monomial_l2_grammian(monomial_degree());
}
mtao::MatXd FluxMomentVEM2Cell::monomial_dirichlet_grammian() const {
    return monomial_dirichlet_grammian(monomial_degree());
}

mtao::MatXd FluxMomentVEM2Cell::l2_projector() const {
    auto G = monomial_l2_grammian();
    auto K = sample_monomial_l2_grammian();

    mtao::MatXd PS(G.cols(), K.rows());
    auto Glu = G.lu();
    for (int j = 0; j < K.rows(); ++j) {
        PS.col(j) = Glu.solve(K.row(j).transpose());
    }
    return PS;
}
mtao::MatXd FluxMomentVEM2Cell::dirichlet_projector() const {
    auto G = regularized_monomial_dirichlet_grammian();
    auto K = sample_monomial_dirichlet_grammian();

    mtao::MatXd PS(G.cols(), K.rows());
    auto Glu = G.lu();
    for (int j = 0; j < K.rows(); ++j) {
        PS.col(j) = Glu.solve(K.row(j).transpose());
    }
    return PS;
}

mtao::MatXd FluxMomentVEM2Cell::l2_sample_projector() const {
    auto Pis = l2_projector();
    auto B = monomial_evaluation();
    return B * Pis;
}
mtao::MatXd FluxMomentVEM2Cell::dirichlet_sample_projector() const {
    auto Pis = dirichlet_projector();
    auto B = monomial_evaluation();
    return B * Pis;
}
// max degree seen among all edges in cell
size_t FluxMomentVEM2Cell::flux_max_degree() const {
    size_t max_deg = -1;
    for (auto&& [e, sgn] : edges()) {
        size_t deg = flux_degree(e);
        if (max_deg == -1 || deg > max_deg) {
            max_deg = deg;
        }
    }
    return max_deg;
}

mtao::MatXd FluxMomentVEM2Cell::sample_monomial_l2_grammian() const {
    mtao::MatXd R(local_sample_size(), monomial_size());
    R.setZero();

    // R.topRows(flux_size()) =
    // monomial_evaluation().topRows(flux_size());

    auto P0 = R.col(0);
    if (moment_size() == 0) {
        // if moment size is 0, then flux sizes are 1, so the offsets are hte
        // indices
        for (auto&& [eidx, loc] : _flux_index_offsets) {
            P0(loc) = edge_length(eidx);
        }
        P0 /= P0.sum();
    } else {
        P0.setUnit(local_moment_index_offset());
    }
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
    int block_size = monomial_size() - off;
    R.rightCols(block_size) = Cp.transpose().rightCols(block_size);
    return R;
}
mtao::MatXd FluxMomentVEM2Cell::sample_monomial_dirichlet_grammian() const {
    mtao::MatXd R(local_sample_size(), monomial_size());
    R.setZero();

    auto indices = flux_indices();
    std::map<size_t, double> edge_lengths = this->edge_lengths();

    auto P0 = R.col(0);
    if (moment_size() == 0) {
        for (auto&& [eidx, sgn] : edges()) {
            int fsize = flux_size(eidx);
            P0.segment(local_flux_index_offset(eidx), fsize)
                .setConstant(edge_lengths.at(eidx));
        }
        P0 /= P0.sum();
    } else {
        P0.setUnit(local_moment_index_offset());
    }

    size_t max_degree = flux_max_degree();
    assert(max_degree != size_t(-1));
    auto EN = edge_normals();
    double diameter = this->diameter();

    if constexpr (false) {
        const auto& fi = flux_indexer();
        const auto& mi = moment_indexer();
        const auto& Mi = monomial_indexer();

        spdlog::info(
            "Flux moments: {} elements, {} total DOFs of values [{}] and "
            "degrees [{}]",
            fi.num_partitions(), fi.num_coefficients(),
            fmt::join(fi.partition_offsets(), ","),
            fmt::join(fi.degrees(), ","));
        spdlog::info(
            "cell moments: {} elements, {} total DOFs of values [{}] and "
            "degrees [{}]",
            mi.num_partitions(), mi.num_coefficients(),
            fmt::join(mi.partition_offsets(), ","),
            fmt::join(mi.degrees(), ","));
        spdlog::info(
            "cell monomials: {} elements, {} total DOFs of values [{}] and "
            "degrees [{}]",
            Mi.num_partitions(), Mi.num_coefficients(),
            fmt::join(Mi.partition_offsets(), ","),
            fmt::join(Mi.degrees(), ","));
    }
    for (auto&& [eidx, sgn] : edges()) {
        // mtao::Vec2d N = mtao::eigen::stl2eigen(EN[eidx]);
        mtao::Vec2d N = (sgn ? -1 : 1) * mtao::eigen::stl2eigen(EN[eidx]);
        // spdlog::warn("Outward facing normal: {} {}", N.x(), N.y());
        // spdlog::warn("Getting an edge monomial, mono deg = {}, flux deg =
        // {}",
        //             monomial_degree(), flux_degree(eidx));

        double edge_length = edge_lengths[eidx];

        const auto e = mesh().E.col(eidx);
        const auto a = mesh().V.col(e(0));
        const auto b = mesh().V.col(e(1));
        // std::cout << "edge goes from " << a.transpose() << " to "
        //          << b.transpose() << std::endl;
        int flux_monomial_offset = local_flux_index_offset(eidx);
        // \int m_\alpha N \cdot dm_\beta
        for (size_t beta = 0; beta < monomial_size(); ++beta) {
            auto [bx, by] = index_to_exponents(beta);
            auto g = gradient_single_index(beta);
            auto [xc, xi] = g[0];
            auto [yc, yi] = g[1];
            if (xi >= 0) {
                auto coeffs = project_monomial_to_boundary(eidx, xi);
                // std::cout << "XI Got coeffs for " << xi <<": " <<
                // coeffs.transpose() << std::endl;

                for (int j = 0; j < coeffs.size(); ++j) {
                    int local_index = flux_monomial_offset + j;
                    double v = edge_length * N.x() * xc * coeffs(j) / diameter;
                    // spdlog::info(
                    //        "Wrote to R({},{}) += {} {} {} / {} = {}",
                    //        local_index, beta, N.x(), xc, coeffs(j),
                    //        diameter, v);
                    R(local_index, beta) += v;
                }
            }

            if (yi >= 0) {
                auto coeffs = project_monomial_to_boundary(eidx, yi);
                // std::cout << "YI Got coeffs for " << yi <<": " <<
                // coeffs.transpose() << std::endl;

                for (int j = 0; j < coeffs.size(); ++j) {
                    int local_index = flux_monomial_offset + j;
                    double v = edge_length * N.y() * yc * coeffs(j) / diameter;
                    // spdlog::info(
                    //        "Wrote to R({},{}) += {} {} {} / {} = {}",
                    //        local_index, beta, N.y(), yc, coeffs(j),
                    //        diameter, v);
                    R(local_index, beta) += v;
                }
            }
        }
    }
    double vol = volume();
    // spdlog::info("Cell volume: {}", vol);
    size_t mom_off = local_moment_index_offset();
    auto L = polynomials::two::laplacian(monomial_degree());
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

mtao::MatXd FluxMomentVEM2Cell::regularized_monomial_dirichlet_grammian()
    const {
    mtao::MatXd R = monomial_dirichlet_grammian();
    auto P0 = R.row(0);
    P0.setZero();
    if (monomial_degree() == 1) {
        for (auto&& [eidx, sgn] : edges()) {
            int fsize = flux_size(eidx);
            // auto integrals = monomial_edge_integrals(eidx,
            // flux_degree(eidx)); mtao::Vec2d N =
            // mtao::eigen::stl2eigen(EN.at(eidx)); auto G =
            // monomial_to_monomial_gradient(flux_degree(eidx));
            // Eigen::SparseMatrix<double> GN =
            //    (N.x() * G.topRows(fsize) + N.y() * G.bottomRows(fsize))
            //        .topRows(fsize);
            // spdlog::info("Flux degree on edge {} is {} vs mnomial degree of
            // {}",
            //             eidx, flux_degree(eidx), monomial_degree());
            mtao::VecXd R = monomial_l2_edge_grammian(eidx);
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

mtao::MatXd FluxMomentVEM2Cell::l2_projector_error() const {
    auto Pi = l2_sample_projector();
    Pi.noalias() = mtao::MatXd::Identity(Pi.rows(), Pi.cols()) - Pi;
    return Pi;
}
mtao::MatXd FluxMomentVEM2Cell::dirichlet_projector_error() const {
    auto Pi = dirichlet_sample_projector();
    Pi.noalias() = mtao::MatXd::Identity(Pi.rows(), Pi.cols()) - Pi;
    return Pi;
}

std::vector<size_t> FluxMomentVEM2Cell::flux_indices() const {
    std::set<int> indices;
    for (auto&& [fidx, sgn] : edges()) {
        auto [start, end] = flux_indexer().coefficient_range(fidx);
        for (size_t j = start; j < end; ++j) {
            indices.emplace(j);
        }
    }
    return {indices.begin(), indices.end()};
}

std::vector<size_t> FluxMomentVEM2Cell::sample_indices() const {
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

mtao::MatXd FluxMomentVEM2Cell::monomial_evaluation() const {
    mtao::MatXd R(local_sample_size(), monomial_size());
    R.setZero();
    double vol = volume();
    auto EN = edge_normals();
    int index_offset = 0;
    for (auto&& [eidx, sgn] : edges()) {
        int fsize = flux_size(eidx);
        // auto integrals = monomial_edge_integrals(eidx, flux_degree(eidx));
        // mtao::Vec2d N = mtao::eigen::stl2eigen(EN.at(eidx));
        // auto G = monomial_to_monomial_gradient(flux_degree(eidx));
        // Eigen::SparseMatrix<double> GN =
        //    (N.x() * G.topRows(fsize) + N.y() * G.bottomRows(fsize))
        //        .topRows(fsize);
        auto MyBlock =
            R.block(local_flux_index_offset(eidx), 0, fsize, R.cols());
        auto B = monomial_l2_edge_grammian(eidx);
        // std::cout << "Monomial evaluation: Edge " << eidx << std::endl;
        // std::cout << B.transpose() << std::endl;
        MyBlock = B.transpose() / B(0, 0);
    }
    size_t mom_size = moment_size();
    if (mom_size == 0) {
        assert(moment_degree() == -1);
        return R;
    }
    assert(moment_degree() != -1);
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

std::map<size_t, size_t> FluxMomentVEM2Cell::world_to_local_flux_indices()
    const {
    std::map<size_t, size_t> ret;
    auto pi = flux_indices();
    for (auto&& [idx, i] : mtao::iterator::enumerate(pi)) {
        ret[i] = idx;
    }
    return ret;
}
std::map<size_t, size_t> FluxMomentVEM2Cell::world_to_local_sample_indices()
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
FluxMomentVEM2Cell::local_to_world_monomial_indices() const {
    return monomial_indices();
}

mtao::iterator::detail::range_container<size_t>
FluxMomentVEM2Cell::flux_indices(size_t edge_index) const {
    return flux_indexer().coefficient_indices(edge_index);
}
mtao::iterator::detail::range_container<size_t>
FluxMomentVEM2Cell::moment_indices() const {
    return moment_indexer().coefficient_indices(cell_index());
}
mtao::iterator::detail::range_container<size_t>
FluxMomentVEM2Cell::monomial_indices() const {
    return monomial_indexer().coefficient_indices(cell_index());
}
std::vector<size_t> FluxMomentVEM2Cell::local_to_world_flux_indices() const {
    return flux_indices();
}
std::vector<size_t> FluxMomentVEM2Cell::local_to_world_sample_indices() const {
    return sample_indices();
}
Eigen::SparseMatrix<double> FluxMomentVEM2Cell::local_to_world_monomial_map()
    const {
    Eigen::SparseMatrix<double> R(monomial_indexer().num_coefficients(),
                                  monomial_size());
    auto [start, end] = monomial_indexer().coefficient_range(cell_index());
    for (size_t j = 0; j < end - start; ++j) {
        R.insert(j + start, j) = 1;
    }

    return R;
}
Eigen::SparseMatrix<double> FluxMomentVEM2Cell::local_to_world_sample_map()
    const {
    Eigen::SparseMatrix<double> R(sample_size(), local_sample_size());
    for (auto&& [r, c] : world_to_local_sample_indices()) {
        R.insert(r, c) = 1;
    }
    // std::cout << R << std::endl;

    return R;
}

mtao::MatXd FluxMomentVEM2Cell::monomial_l2_edge_grammian(
    size_t edge_index) const {
    size_t fdeg = flux_degree(edge_index);
    auto old = VEM2Cell::monomial_l2_edge_grammian(edge_index,
                                                   monomial_degree(), fdeg);
    return old;
}

mtao::VecXd FluxMomentVEM2Cell::project_monomial_to_boundary(
    size_t edge_index, size_t cell_monomial_index) const {
    auto [xexp, yexp] = index_to_exponents(cell_monomial_index);
    mtao::algebra::PascalTriangle pt(std::max(xexp, yexp));
    bool beware_issues = cell_monomial_index == 7;

    // i want to convert something of the form
    // (x - c_x) ^ xexp (y - c_y)  ^ yexp  / diameter^{xexp + yexp}
    // to
    // \sum_{d=0}^{xexp + yexp} coeff_i ((t - C)/D)^d
    //
    // Let s = (t - C)/D
    // so t = sD + C
    //
    // (p - c) ^ exp  / diameter^{exp}
    // ((a + (b-a)t) - c) ^ exp  / diameter^{exp}
    // (((b-a)t) + a - c) ^ exp  / diameter^{exp}
    // (((b-a)(sD + C)) + a - c) ^ exp  / diameter^{exp}
    // (((b-a)Ds + (C(b-a) + a - c)) ^ exp  / diameter^{exp}
    // \sum_{j=0}^exp  (exp choose j) (C(b-a) + a - c)^{exp-j} (D(b-a))^j s^j /
    // diameter^exp A = (C(b-a) + a - c) B = (D(b-a))

    const auto e = mesh().E.col(edge_index);
    const auto a = mesh().V.col(e(0));
    const auto b = mesh().V.col(e(1));
    size_t max_degree = xexp + yexp;
    mtao::VecXd coeffs(max_degree + 1);
    // spdlog::info(
    //    "Projecting monomial to boundary for cell poly {} (x^{} y^{}, which "
    //    "should have {} edge coefficients",
    //    cell_monomial_index, xexp, yexp, coeffs.size());

    double edge_diameter = flux_indexer().diameter(edge_index);
    mtao::Vec2d edge_center = flux_indexer().center(edge_index);
    mtao::Vec2d A = edge_center - center();
    mtao::Vec2d B = edge_diameter * (b - a);
    coeffs.setZero();

    mtao::Vec2d mexp = mtao::Vec2d(double(xexp), double(yexp));
    // if (beware_issues)
    //    spdlog::warn("x^{} y^{} / {}", xexp, yexp,
    //                 std::pow<double>(diameter, max_degree));
    for (size_t j = 0; j <= xexp; ++j) {
        double xterm = pt(xexp, j);
        for (size_t k = 0; k <= yexp; ++k) {
            double yterm = pt(yexp, k);
            size_t degree = j + k;
            double& val = coeffs(degree);
            mtao::Vec2d lexp = mtao::Vec2d(double(j), double(k));

            // if (beware_issues)
            //    spdlog::warn("({} C {})({} C {}) / {}", xexp, yexp,
            //                 std::pow<double>(diameter, max_degree));
            val += xterm * yterm * A.array().pow((mexp - lexp).array()).prod() *
                   B.array().pow(lexp.array()).prod();
        }
    }
    coeffs /= std::pow<double>(diameter(), xexp + yexp);
    return coeffs;
}
}  // namespace vem
