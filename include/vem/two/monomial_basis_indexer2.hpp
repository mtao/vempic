#pragma once

#include "vem/monomial_basis_indexer_new.hpp"

namespace vem {

namespace detail {
template <>
std::function<mtao::Vec2d(const mtao::Vec2d &)>
MonomialBasisIndexer<2, 2>::monomial_gradient(size_t cell, size_t index) const;
}
}  // namespace vem
