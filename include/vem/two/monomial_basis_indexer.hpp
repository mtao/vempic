#pragma once

#include "cell_boundary_facets.hpp"
#include "monomial_cell_integrals.hpp"
#include "vem/monomial_basis_indexer.hpp"

namespace vem {

namespace detail {
template <>
std::function<mtao::Vec2d(const mtao::Vec2d &)>
MonomialBasisIndexer<2, 2>::monomial_gradient(size_t cell, size_t index) const;
}
}  // namespace vem
