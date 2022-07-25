#pragma once
#include <map>
#include <mtao/types.hpp>
#include <optional>

namespace vem::poisson_2d {

struct ScalarConstraints {
    std::map<size_t, double> pointwise_dirichlet;
    // for each edge we want \int \nabla f \cdot N = svalue
    std::map<size_t, mtao::Vec2d> pointwise_neumann;

    // for each edge we want \int \nabla f \cdot N = svalue
    std::map<size_t, double> edge_integrated_flux_neumann;

    // for each edge we want pointwise f = some polynomial
    std::map<size_t, mtao::VecXd> edge_dirichlet;
    // for each edge we want pointwise \nabla f \cdot N = some polynomial
    std::map<size_t, mtao::VecXd> edge_flux_neumann;
    std::optional<double> mean_value;
    void clear();
    bool empty() const;

    void merge(const ScalarConstraints &other);
    size_t size() const;
};
}// namespace vem::poisson_2d
