#pragma once

#include "monomial_basis_indexer.hpp"

namespace vem::two {
class MomentBasisIndexer : public detail::MonomialBasisIndexer<2, 2> {
    using Base = detail::MonomialBasisIndexer<2, 2>;
    using Base::Base;
};
}// namespace vem
