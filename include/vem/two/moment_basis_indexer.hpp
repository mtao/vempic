#pragma once

#include "vem/monomial_basis_indexer.hpp"

namespace vem {
class MomentBasisIndexer : public detail::MonomialBasisIndexer<2, 2> {
    using Base = detail::MonomialBasisIndexer<2, 2>;
    using Base::Base;
};
}// namespace vem
