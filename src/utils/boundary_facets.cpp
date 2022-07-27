
std::map<size_t, size_t> boundary_facet_indices(
    const std::vector<std::map<int, bool>> &cells, size_t facet_count,
    const std::set<int> &active_cells = {}) {
    std::vector<int> sizes(facet_count, 0);

    std::map<size_t, size_t> ret;
    utils::loop_over_active_indices(cells.size(), active_cells,
                             [&](size_t cell_index) {
                                 auto &c = cells.at(cell_index);
                                 for (auto &&[eidx, sgn] : c) {
                                     sizes.at(eidx) += sgn ? -1 : 1;
                                     // sizes.at(eidx)++;
                                     ret[eidx] = cell_index;
                                 }
                             });

    for (auto it = ret.begin(); it != ret.end();) {
        // if (sizes.at(it->first) != 0) {
        if (sizes.at(it->first) % 2 != 0) {
            ++it;
        } else {
            it = ret.erase(it);
        }
    }
    return ret;
}

