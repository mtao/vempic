#pragma once

#include "vem/monomial_basis_indexer.hpp"

namespace vem {
class MomentBasisIndexer : public MonomialBasisIndexer {
    using MonomialBasisIndexer = detail::MonomialBasisIndexer<2,2>;
    using MonomialBasisIndexer::MonomialBasisIndexer;
};
}// namespace vem
