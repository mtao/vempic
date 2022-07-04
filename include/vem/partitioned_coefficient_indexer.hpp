#pragma once
#include <compare>
#include <limits>
#include <mtao/iterator/range.hpp>
#include <range/v3/view/span.hpp>

#include "vem/mesh.hpp"

namespace vem {
// an indexer into coefficients that are _JUST_ coefficients
// coefficients are combnied together according to some unit, either edges or
// partitions or whatnot
class PartitionedCoefficientIndexer {
   public:
    // pass in the number of samples are held on the interior of each edge
    // perhaps these constructors are a bit bizarre as in-place initialization
    // creates a very different result from construction
    PartitionedCoefficientIndexer(size_t number_partitions,
                                  size_t partition_size, size_t offset = 0);
    PartitionedCoefficientIndexer(const std::vector<size_t> &partition_sizes,
                                  size_t offset = 0);

    PartitionedCoefficientIndexer(
        std::nullopt_t tag_for_direct_offset_construction,
        std::vector<size_t> partition_offsets);

    PartitionedCoefficientIndexer(const PartitionedCoefficientIndexer &) =
        default;
    PartitionedCoefficientIndexer(PartitionedCoefficientIndexer &&) = default;

    PartitionedCoefficientIndexer &operator=(
        const PartitionedCoefficientIndexer &) = default;
    PartitionedCoefficientIndexer &operator=(PartitionedCoefficientIndexer &&) =
        default;

    size_t partition_offset(size_t partition_index) const;
    // the number of coefficients in a single unit
    size_t num_coefficients(size_t partition_index) const;
    // the number of coefficients total
    size_t num_coefficients() const { return _partition_offsets.back(); }
    // the total number of partitions available
    size_t num_partitions() const { return _partition_offsets.size() - 1; }
    std::array<size_t, 2> coefficient_range(size_t partition_index) const;

    mtao::iterator::detail::range_container<size_t> coefficient_indices(
        size_t partition_index) const;

    const std::vector<size_t> &partition_offsets() const {
        return _partition_offsets;
    }

    // if no active partitions are mentioned we will assuem this means they should all be active. this is sillly, but conforms to laziness in other places
    mtao::VecXd partition_mask(const std::set<int>& active_partitions) const;

    // ordering represents how the index corresponds with the range available
    // for this PCI
    std::tuple<std::strong_ordering, size_t> get_partition(size_t index) const;
    std::strong_ordering get_index_validity(size_t index) const;

    // constructs edge offsets assuming each edge has the prescribed number
    // of internal nodes
    void construct_partition_offsets(const std::vector<size_t> &partition_sizes,
                                     size_t offset = 0);
    // constructs edge offsets assuming each edge has the same number of
    // internal nodes
    void construct_partition_offsets(size_t number_partitions,
                                     size_t partition_size, size_t offset = 0);

   private:
    // offsets according to the numbers of coefficients stored in each
    // partition
    std::vector<size_t> _partition_offsets;
};
}  // namespace vem
