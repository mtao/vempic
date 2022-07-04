#include "vem/partitioned_coefficient_indexer.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <compare>
#include <mtao/iterator/enumerate.hpp>
#include <mtao/iterator/interval.hpp>

namespace vem {
PartitionedCoefficientIndexer::PartitionedCoefficientIndexer(
    size_t num_samples, size_t partition_size, size_t offset) {
    construct_partition_offsets(num_samples, partition_size, offset);
}
PartitionedCoefficientIndexer::PartitionedCoefficientIndexer(
    const std::vector<size_t> &partition_sizes, size_t offset) {
    construct_partition_offsets(partition_sizes, offset);
}
PartitionedCoefficientIndexer::PartitionedCoefficientIndexer(
    std::nullopt_t tag_for_direct_offset_construction,
    std::vector<size_t> partition_offsets)
    : _partition_offsets(std::move(partition_offsets)) {}

size_t PartitionedCoefficientIndexer::partition_offset(size_t index) const {
    return _partition_offsets.at(index);
}
size_t PartitionedCoefficientIndexer::num_coefficients(size_t index) const {
    return partition_offset(index + 1) - partition_offset(index);
}

std::array<size_t, 2> PartitionedCoefficientIndexer::coefficient_range(
    size_t partition_index) const {
    return {{_partition_offsets.at(partition_index),
             _partition_offsets.at(partition_index + 1)}};
}
mtao::iterator::detail::range_container<size_t>
PartitionedCoefficientIndexer::coefficient_indices(
    size_t partition_index) const {
    auto [a, b] = coefficient_range(partition_index);
    return mtao::iterator::detail::range_container<size_t>(a, b, 1);
}

// Construction stuff
void PartitionedCoefficientIndexer::construct_partition_offsets(
    const std::vector<size_t> &partition_sizes, size_t offset) {
    _partition_offsets.resize(partition_sizes.size() + 1);
    for (auto &&[eoff, size] :
         mtao::iterator::zip(_partition_offsets, partition_sizes)) {
        eoff = offset;
        offset += size;
    }
    _partition_offsets.back() = offset;
}
void PartitionedCoefficientIndexer::construct_partition_offsets(
    size_t number_partitions, size_t partition_size, size_t offset) {
    _partition_offsets.resize(number_partitions + 1);
    for (auto &&[idx, eoff] : mtao::iterator::enumerate(_partition_offsets)) {
        eoff = offset + partition_size * idx;
    }
    _partition_offsets.back() = offset + partition_size * number_partitions;
}

std::tuple<std::strong_ordering, size_t>
PartitionedCoefficientIndexer::get_partition(size_t index) const {
    std::strong_ordering order = get_index_validity(index);
    if (order != std::strong_ordering::equivalent) {
        return {order, 0};
    } else {
        auto it = std::upper_bound(_partition_offsets.begin(),
                                   _partition_offsets.end(), index);
        return {order, std::distance(_partition_offsets.begin(), it) - 1};
    }
}
std::strong_ordering PartitionedCoefficientIndexer::get_index_validity(
    size_t index) const {
    if (index < _partition_offsets.front()) {
        return std::strong_ordering::less;
    } else if (index >= _partition_offsets.back()) {
        return std::strong_ordering::greater;
    } else {
        return std::strong_ordering::equivalent;
    }
}

mtao::VecXd PartitionedCoefficientIndexer::partition_mask(
    const std::set<int> &active_partitions) const {
    mtao::VecXd R(num_partitions());
    if (active_partitions.size() == 0) {
        R.setConstant(1);
        return R;
    } else {
        R.setZero();
    }
    for (auto &&p : active_partitions) {
        auto [start, end] = coefficient_range(p);
        R.segment(start, end - start).setOnes();
    }
    return R;
}
}  // namespace vem
