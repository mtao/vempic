#include "vem/fluidsim_2d/fluidvem2_cell.hpp"

#include <vem/polynomial_gradient.hpp>
#include <vem/polynomial_utils.hpp>
using namespace vem::polynomials::two;
namespace vem::fluidsim_2d {

FluidVEM2Cell::FluidVEM2Cell(const PointMomentIndexer& mom, size_t index)
    : VEM2Cell(mom.mesh(), index), _indexer(mom) {}

// the number of DOFs in the whole problem
size_t FluidVEM2Cell::sample_size() const { return indexer().sample_size(); }
// the number of DOFs in the local sample
size_t FluidVEM2Cell::local_sample_size() const {
    return point_sample_size() + moment_size();
}

const MonomialBasisIndexer& FluidVEM2Cell::monomial_indexer() const {
    return indexer().monomial_indexer();
}

const MomentBasisIndexer& FluidVEM2Cell::moment_indexer() const {
    return indexer().moment_indexer();
}
const PointSampleIndexer& FluidVEM2Cell::point_sample_indexer() const {
    return indexer().point_sample_indexer();
}

// size_t FluidVEM2Cell:: edge_count() const{}
size_t FluidVEM2Cell::edge_interior_sample_count() const {
    size_t size = 0;

    for (auto&& [idx, sgn] : mesh().face_boundary_map.at(cell_index())) {
        size += point_sample_indexer().num_internal_edge_indices(idx);
    }
    return size;
}
size_t FluidVEM2Cell::point_sample_size() const {
    return vertex_count() + edge_interior_sample_count();
}
size_t FluidVEM2Cell::moment_degree() const {
    return moment_indexer().degree(cell_index());
}
size_t FluidVEM2Cell::moment_size() const {
    return moment_indexer().num_coefficients(cell_index());
}
size_t FluidVEM2Cell::monomial_degree() const {
    return monomial_indexer().degree(cell_index());
}
size_t FluidVEM2Cell::monomial_size() const {
    return monomial_indexer().num_coefficients(cell_index());
}

size_t FluidVEM2Cell::local_moment_index_offset() const {
    return point_sample_size();
}
size_t FluidVEM2Cell::global_moment_index_offset() const {
    return point_sample_indexer().num_coefficients() +
           moment_indexer().coefficient_offset(cell_index());
}
size_t FluidVEM2Cell::global_monomial_index_offset() const {
    return _indexer.monomial_indexer().coefficient_offset(cell_index());
}

mtao::MatXd FluidVEM2Cell::monomial_l2_grammian() const {
    return monomial_l2_grammian(monomial_degree());
}
mtao::MatXd FluidVEM2Cell::monomial_dirichlet_grammian() const {
    return monomial_dirichlet_grammian(monomial_degree());
}

mtao::MatXd FluidVEM2Cell::l2_projector() const {
    auto G = monomial_l2_grammian();
    auto K = sample_monomial_l2_grammian();

    mtao::MatXd PS(G.cols(), K.rows());
    auto Glu = G.lu();
    for (int j = 0; j < K.rows(); ++j) {
        PS.col(j) = Glu.solve(K.row(j).transpose());
    }
    return PS;
}
mtao::MatXd FluidVEM2Cell::dirichlet_projector() const {
    auto G = regularized_monomial_dirichlet_grammian();
    auto K = sample_monomial_dirichlet_grammian();

    mtao::MatXd PS(G.cols(), K.rows());
    auto Glu = G.lu();
    for (int j = 0; j < K.rows(); ++j) {
        PS.col(j) = Glu.solve(K.row(j).transpose());
    }
    return PS;
}

mtao::MatXd FluidVEM2Cell::l2_sample_projector() const {
    auto Pis = l2_projector();
    auto B = monomial_evaluation();
    return B * Pis;
}
mtao::MatXd FluidVEM2Cell::dirichlet_sample_projector() const {
    auto Pis = dirichlet_projector();
    auto B = monomial_evaluation();
    return B * Pis;
}

mtao::MatXd FluidVEM2Cell::sample_monomial_l2_grammian() const {
    mtao::MatXd R(local_sample_size(), monomial_size());
    R.setZero();

    // R.topRows(point_sample_size()) =
    // monomial_evaluation().topRows(point_sample_size());

    auto P0 = R.col(0);
    if (moment_size() == 0) {
        P0.setConstant(1.0 / point_sample_size());
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
        // auto [mxexp, myexp] = index_to_exponents(j);
        // for (int row = 0; row < moment_size(); ++row) {
        //    // auto [Mxexp, Myexp] = index_to_exponents(row);
        //    R(row, col) = vol;
        //}
    }
    size_t off = moment_size();  // k-2 polynomials
    mtao::MatXd Cp = monomial_l2_grammian() * dirichlet_projector();

    int block_size = monomial_size() - off;
    R.rightCols(block_size) = Cp.transpose().rightCols(block_size);
    return R;
}
mtao::MatXd FluidVEM2Cell::sample_monomial_dirichlet_grammian() const {
    mtao::MatXd R(local_sample_size(), monomial_size());
    R.setZero();

    std::map<size_t, size_t> w2l_pi = world_to_local_point_sample_indices();

    auto indices = point_sample_indices();
    std::map<size_t, double> edge_lengths = this->edge_lengths();
    double total_edge_length = boundary_area();

    auto P0 = R.col(0);
    if (moment_size() == 0) {
        P0.setConstant(1.0 / indices.size());
    } else {
        P0.setUnit(local_moment_index_offset());
    }

    auto EN = edge_normals();
    double diameter = this->diameter();
    double diam2 = diameter * diameter;
    for (auto&& [eidx, sgn] : edges()) {
        auto edge_indices = point_sample_indexer().ordered_edge_indices(eidx);
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

            auto&& [P, W] = mtao::quadrature::gauss_lobatto_data<double>(
                edge_indices.size());

            for (auto&& [vidx, weight] : mtao::iterator::zip(edge_indices, W)) {
                size_t local_index = w2l_pi.at(vidx);
                auto p =
                    (point_sample_indexer().get_position(vidx) - center()) /
                    diameter;
                double dmdn = (xc * p.array().pow(xexp.array()).prod() +
                               yc * p.array().pow(yexp.array()).prod()) /
                              diameter;
                // factor of 2 due to gauss lobatto being defined on [-1,1]
                R(local_index, poly_idx) += dmdn * weight * edge_length / 2;
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
            int row = it.row() + mom_off;
            int col = it.col();
            R(row, col) = -it.value() * vol / d2;
        }
    }
    return R;
}

mtao::MatXd FluidVEM2Cell::regularized_monomial_dirichlet_grammian() const {
    mtao::MatXd R = monomial_dirichlet_grammian();
    auto P0 = R.row(0);
    R.col(0).setZero();
    if (monomial_degree() == 1) {
        for (int j = 0; j < monomial_size(); ++j) {
            P0(j) = point_sample_indexer()
                        .evaluate_coefficients(cell_index(), monomial(j))
                        .mean();
        }

    } else {
        const double length = boundary_area();
        const double area = volume();

        auto integrals = monomial_indexer().monomial_integrals(cell_index());
        P0 = integrals.transpose() / area;
    }

    return R;
}

mtao::MatXd FluidVEM2Cell::l2_projector_error() const {
    auto Pi = l2_sample_projector();
    Pi.noalias() = mtao::MatXd::Identity(Pi.rows(), Pi.cols()) - Pi;
    return Pi;
}
mtao::MatXd FluidVEM2Cell::dirichlet_projector_error() const {
    auto Pi = dirichlet_sample_projector();
    Pi.noalias() = mtao::MatXd::Identity(Pi.rows(), Pi.cols()) - Pi;
    return Pi;
}

std::vector<size_t> FluidVEM2Cell::point_sample_indices() const {
    auto s = point_sample_indexer().cell_indices(cell_index());
    return {s.begin(), s.end()};
}

std::vector<size_t> FluidVEM2Cell::sample_indices() const {
    auto s = point_sample_indices();
    s.reserve(s.size() + moment_size());
    auto [start, end] = moment_indexer().coefficient_range(cell_index());
    size_t off = point_sample_indexer().num_coefficients();
    start += off;
    end += off;
    for (size_t j = start; j < end; ++j) {
        s.emplace_back(j);
    }

    return {s.begin(), s.end()};
}

mtao::MatXd FluidVEM2Cell::monomial_evaluation() const {
    mtao::MatXd R(local_sample_size(), monomial_size());
    R.setZero();
    double vol = volume();
    auto p = point_sample_indices();
    for (auto&& [local_vsample_idx, vsample_idx] :
         mtao::iterator::enumerate(p)) {
        // for (int vsample_idx = 0; vsample_idx < boundary_sample_count();
        //     ++vsample_idx) {
        for (size_t poly_idx = 0; poly_idx < monomial_size(); ++poly_idx) {
            R(local_vsample_idx, poly_idx) = evaluate_monomial(
                poly_idx, point_sample_indexer().get_position(vsample_idx));
        }
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

std::map<size_t, size_t> FluidVEM2Cell::world_to_local_point_sample_indices()
    const {
    std::map<size_t, size_t> ret;
    auto pi = point_sample_indices();
    for (auto&& [idx, i] : mtao::iterator::enumerate(pi)) {
        ret[i] = idx;
    }
    return ret;
}
std::map<size_t, size_t> FluidVEM2Cell::world_to_local_sample_indices() const {
    auto ret = world_to_local_point_sample_indices();
    size_t local_offset = local_moment_index_offset();
    size_t global_offset = global_moment_index_offset();
    auto [start, end] = moment_indexer().coefficient_range(cell_index());
    // spdlog::info("Moment indexer gave range {} {} for cell {}", start, end,
    //             index);
    for (size_t j = 0; j < end - start; ++j) {
        ret[j + start + global_offset] = j + local_offset;
    }
    return ret;
}

mtao::iterator::detail::range_container<size_t>
FluidVEM2Cell::local_to_world_monomial_indices() const {
    return monomial_indexer().coefficient_indices(cell_index());
}
std::vector<size_t> FluidVEM2Cell::local_to_world_point_sample_indices() const {
    return point_sample_indices();
}
std::vector<size_t> FluidVEM2Cell::local_to_world_sample_indices() const {
    return sample_indices();
}
Eigen::SparseMatrix<double> FluidVEM2Cell::local_to_world_monomial_map() const {
    Eigen::SparseMatrix<double> R(monomial_indexer().num_coefficients(),
                                  monomial_size());
    auto [start, end] = monomial_indexer().coefficient_range(cell_index());
    for (size_t j = 0; j < end - start; ++j) {
        R.insert(j + start, j) = 1;
    }

    return R;
}
Eigen::SparseMatrix<double> FluidVEM2Cell::local_to_world_sample_map() const {
    Eigen::SparseMatrix<double> R(sample_size(), local_sample_size());
    for (auto&& [r, c] : world_to_local_sample_indices()) {
        R.insert(r, c) = 1;
    }
    // std::cout << R << std::endl;

    return R;
}

}  // namespace vem::fluidsim_2d
