
#pragma once

#include "cell_boundary_facets.hpp"
#include "face_boundary_facets.hpp"
#include "monomial_cell_integrals.hpp"
#include "vem/monomial_basis_indexer.hpp"

namespace vem {

namespace detail {
    template<>
    std::function<mtao::Vec3d(const mtao::Vec3d &)>
      MonomialBasisIndexer<3, 3>::monomial_gradient(size_t cell, size_t index) const;
template<>
void MonomialBasisIndexer<2,3>::fill_diameters();
template<>
void MonomialBasisIndexer<1, 3>::fill_diameters();
}

namespace three {
using MonomialBasisIndexer = detail::MonomialBasisIndexer<3,3>;
using MonomialBasisFaceIndexer = detail::MonomialBasisIndexer<2,2>;
using MonomialBasisEdgeIndexer = detail::MonomialBasisIndexer<1,3>;
}
}// namespace vem
