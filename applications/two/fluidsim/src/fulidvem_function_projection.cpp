

#include <vem/utils/boundary_facets.hpp>
#include <vem/utils/cell_identifier.hpp>
#include <vem/utils/loop_over_active.hpp>
#include <vem/utils/parent_maps.hpp>

#include "mtao/eigen/mat_to_triplets.hpp"
#include "vem/fluidsim_2d/fluidvem2.hpp"

namespace vem::fluidsim_2d {

mtao::VecXd FluidVEM2::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec2d &)> &f) const {
    double val = (double)(pressure_monomial_size()) / cell_count() + 2;
    return coefficients_from_point_sample_function(f, val * val);
}
mtao::VecXd FluidVEM2::coefficients_from_point_sample_function(
    const std::function<double(const mtao::Vec2d &)> &f,
    int samples_per_cell) const {
    auto [P, O] = sample_active_cells(samples_per_cell);
    return coefficients_from_point_sample_function(f, P, O);
}

mtao::ColVecs2d FluidVEM2::coefficients_from_point_sample_vector_function(
    const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f) const {
    double val = (double)(pressure_monomial_size()) / cell_count() + 2;
    return coefficients_from_point_sample_vector_function(f, val * val);
}
mtao::ColVecs2d FluidVEM2::coefficients_from_point_sample_vector_function(
    const std::function<mtao::Vec2d(const mtao::Vec2d &)> &f,
    int samples_per_cell) const {
    auto [P, O] = sample_active_cells(samples_per_cell);
    return coefficients_from_point_sample_vector_function(f, P, O);
}

std::tuple<mtao::ColVecs2d, std::vector<std::set<int>>>
FluidVEM2::sample_active_cells(size_t samples_per_cell) const {
    std::vector<std::set<int>> ownerships(cell_count());
    mtao::ColVecs2d points(2, samples_per_cell * cell_count());
    tbb::parallel_for(size_t(0), ownerships.size(), [&](size_t idx) {
        // for (auto &&[idx, own] : mtao::iterator::enumerate(ownerships)) {
        auto &own = ownerships[idx];
        if (is_active_cell(idx)) {
            auto c = get_pressure_cell(idx);
            auto bb = c.bounding_box();
            int offset = idx * samples_per_cell;
            for (int j = 0; j < samples_per_cell; ++j) {
                own.emplace(j + offset);
                auto p = points.col(j + offset) = bb.sample();
                while (!c.is_inside(p)) {
                    p = bb.sample();
                }
            }
        }
    });
    return {points, ownerships};
}

}  // namespace vem::fluidsim_2d
