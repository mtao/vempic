#include "vem/two/poisson/constraints.hpp"

#include <set>

namespace vem::two::poisson {
void ScalarConstraints::clear() { *this = ScalarConstraints(); }
bool ScalarConstraints::empty() const {
    return !(pointwise_dirichlet.empty() && pointwise_neumann.empty() && edge_integrated_flux_neumann.empty() && edge_dirichlet.empty() && edge_flux_neumann.empty() && !bool(mean_value));
}
void ScalarConstraints::merge(const ScalarConstraints &o) {
    pointwise_dirichlet.insert(o.pointwise_dirichlet.begin(),
                               o.pointwise_dirichlet.end());
    pointwise_neumann.insert(o.pointwise_neumann.begin(),
                             o.pointwise_neumann.end());
    edge_integrated_flux_neumann.insert(o.edge_integrated_flux_neumann.begin(),
                                        o.edge_integrated_flux_neumann.end());
    edge_dirichlet.insert(o.edge_dirichlet.begin(), o.edge_dirichlet.end());
    edge_flux_neumann.insert(o.edge_flux_neumann.begin(),
                             o.edge_flux_neumann.end());
    if (mean_value) {
        if (o.mean_value) {
            mean_value = *mean_value + *o.mean_value;
        }
    } else {
        mean_value = o.mean_value;
    }
}
size_t ScalarConstraints::size() const {
    return pointwise_dirichlet.size() + pointwise_neumann.size() + edge_integrated_flux_neumann.size() + edge_dirichlet.size() + edge_flux_neumann.size() + (mean_value ? 1 : 0);
}
}// namespace vem::poisson_2d
