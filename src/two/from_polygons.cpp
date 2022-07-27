#include "vem/two/from_polygons.hpp"

#include <mtao/eigen/stl2eigen.hpp>
#include <mtao/geometry/winding_number.hpp>
#include <mtao/iterator/enumerate.hpp>
#include <mtao/iterator/interval.hpp>
#include <mtao/reindex/compressing_reindexer.hpp>
#include <utility>


namespace vem::two {


PolygonVEMMesh2 from_polygons(const mtao::ColVecs2d &V,
                              const std::vector<std::vector<int>> &polygons) {
    return {V, polygons};
}
PolygonVEMMesh2 from_polygons(const mtao::vector<mtao::ColVecs2d> &polygons) {
    mtao::reindex::CompressingReindexer<std::array<double, 2>> vert_indexer;
    std::vector<std::vector<int>> poly_indices(polygons.size());
    for (auto &&[V, inds] : mtao::iterator::zip(polygons, poly_indices)) {
        std::array<double, 2> P;
        inds.resize(V.cols());

        for (auto &&[col, ind] : mtao::iterator::enumerate(inds)) {
            auto v = V.col(col);
            mtao::eigen::stl2eigen(P) = v;
            ind = vert_indexer.add(P);
        }
    }

    return from_polygons(mtao::eigen::stl2eigen(vert_indexer.unindex_vec()),
                         poly_indices);
    // VEMMesh2 vem = from_polygons(, poly_indices);
}

}  // namespace vem
