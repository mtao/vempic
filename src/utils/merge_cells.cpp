#include "vem/utils/merge_cells.hpp"

#include <algorithm>
#include <map>
#include <mtao/data_structures/disjoint_set.hpp>

#include "vem/utils/face_neighboring_cells.hpp"

namespace vem::utils {
std::map<int, bool> merge_two_cells(const std::map<int, bool>& a,
                                    const std::map<int, bool>& b) {
    // std::cout << "A)))";
    // for (auto&& [eidx, sgn] : a) {
    //    std::cout << (sgn ? '-' : '+') << eidx << " ";
    //}
    // std::cout << std::endl;
    // std::cout << "B)))";
    // for (auto&& [eidx, sgn] : b) {
    //    std::cout << (sgn ? '-' : '+') << eidx << " ";
    //}
    // std::cout << std::endl;
    std::map<int, bool> ret;
    for (auto&& [ai, as] : a) {
        if (b.find(ai) == b.end()) {
            ret[ai] = as;
        }
    }

    for (auto&& [bi, bs] : b) {
        if (a.find(bi) == a.end()) {
            ret[bi] = bs;
        }
    }

    // std::cout << "Ret)))";
    // for (auto&& [eidx, sgn] : ret) {
    //    std::cout << (sgn ? '-' : '+') << eidx << " ";
    //}
    // std::cout << std::endl;
    // std::cout << std::endl;
    return ret;
}
std::tuple<std::vector<std::map<int, bool>>, std::vector<size_t>> merge_cells(
    const std::vector<std::map<int, bool>>& cells, const mtao::VecXd& volumes,
    const mtao::VecXi& regions, double threshold) {
    auto face_neighbors = vem::utils::face_neighboring_cells(cells);

    // flag small cells

    if (cells.size() != volumes.size()) {
        spdlog::info(
            "cannot merge with a different number of cells {} and regions {}",
            cells.size(), volumes.size());
        return {};
    }
    if (cells.size() != regions.size()) {
        spdlog::info(
            "cannot merge with a different number of cells {} and regions {}",
            cells.size(), regions.size());
        return {};
    }
    std::vector<std::map<int, bool>> new_cells;
    new_cells.reserve(cells.size());
    std::vector<size_t> old_to_new_map(cells.size(), 0);

    mtao::data_structures::DisjointSet<size_t> ds;
    ds.nodes.reserve(cells.size());
    for (size_t j = 0; j < cells.size(); ++j) {
        ds.add_node(j);
    }

    for (size_t j = 0; j < volumes.size(); ++j) {
        if (volumes(j) < threshold) {
            int best_idx = -1;
            double best_vol = 0;
            for (auto&& n : face_neighbors.at(j)) {
                if (regions(n) == regions(j)) {
                    if (volumes(n) > best_vol) {
                        best_idx = n;
                        best_vol = volumes(n);
                    }
                }
            }
            if (best_idx >= 0) {
                // note this assumes taht join sets j's parent to best_idex
                ds.join(j, best_idx);
            }
        }
    }
    std::vector<std::set<int>> children(cells.size());
    for (size_t j = 0; j < volumes.size(); ++j) {
        children[ds.get_root(j).data].emplace(j);
    }

    children.erase(std::remove_if(children.begin(), children.end(),
                                  [](auto&& c) -> bool { return c.empty(); }),
                   children.end());

    new_cells.resize(children.size());
    for (auto&& [index, children, new_cell] :
         mtao::iterator::enumerate(children, new_cells)) {
        // spdlog::info("Cell {} comprised of {}", index,
        //             fmt::join(children, ","));
        if (children.empty()) {
            continue;
        }
        for (auto&& c : children) {
            old_to_new_map[c] = index;
            new_cell = merge_two_cells(new_cell, cells.at(c));
        }
    }

    return {new_cells, old_to_new_map};
}
}  // namespace vem::utils
