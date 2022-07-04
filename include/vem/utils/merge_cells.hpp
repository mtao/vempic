#pragma once
#include <map>
#include <mtao/types.hpp>
#include <tuple>
#include <vector>

namespace vem::utils {
std::map<int, bool> merge_two_cells(const std::map<int, bool>& a,
                                    const std::map<int, bool>& b);

std::tuple<std::vector<std::map<int, bool>>, std::vector<size_t>> merge_cells(
    const std::vector<std::map<int, bool>>& cells, const mtao::VecXd& volumes,
    const mtao::VecXi& regions, double threshold);

}  // namespace vem::utils
