
#pragma once

#include "vem/monomial_basis_indexer_new.hpp"

namespace vem {

namespace detail {
template <>
std::function<mtao::Vec3d(const mtao::Vec3d &)>
MonomialBasisIndexer<3, 3>::monomial_gradient(size_t cell, size_t index) const;
}
}  // namespace vem
