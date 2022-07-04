#include "vem/utils/point_moment_to_flux_moment.hpp"
namespace vem::utils {
Eigen::SparseMatrix<double> point_moment_to_flux_moment(
    const PointMomentIndexer& pmi, const FluxMomentIndexer& fmi) {
    Eigen::SparseMatrix<double> R(fmi.sample_size(), pmi.sample_size());

    size_t pmi_moment_off = pmi.point_sample_size();
    size_t fmi_moment_off = pmi.flux_size();

    for (auto&& [edge_index, pr] :
         mtao::iterator::enumerate(mtao::iterator::interval<2>(
             pmi.point_sample_indexer().edge_offsets()))) {
        auto&& [start, end] = pr;
        auto e = pmi.mesh().E.col(edge_index);
        auto a = pmi.mesh().V.col(e(0));
        auto b = pmi.mesh().V.col(e(1));
        mtao::VecXd D(end - start + 2);
        D(0) = coefficients(e(0));
        D(D.size() - 1) = coefficients(e(1));
        D.segment(1, end - start) = coefficients.segment(start, end - start);
        R(edge_index) = mtao::quadrature::gauss_lobatto(D, (b - a).norm());
    }
    for (size_t edge_index = 0; edge_index < pmi.mesh().edge_count();
         ++edge_index) {
        int sample_count =
            pmi.point_sample_indexer().num_edge_indices(edge_index);
        auto&& [P, W] =
            mtao::quadrature::gauss_lobatto_data<double>(sample_count);

        for (auto&& [vidx, weight] : mtao::iterator::zip(edge_indices, W)) {
            weight* edge_length / 2;
        }
    }
    for (size_t cell_index = 0; cell_index < pmi.mesh().cell_count();
         ++cell_index) {
        auto pcell = pmi.get_cell(cell_index);
        auto fcell = fmi.get_cell(cell_index);
    }
}
}  // namespace vem::utils
