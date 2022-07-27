
#include "vem/two/polygon_mesh.hpp"

#include <mtao/eigen/stl2eigen.hpp>
#include <mtao/geometry/winding_number.hpp>
#include <mtao/iterator/enumerate.hpp>
#include <mtao/iterator/interval.hpp>
#include <mtao/reindex/compressing_reindexer.hpp>
#include <utility>

#include "vem/two/set_centroids_as_centers.hpp"

namespace vem::two {

PolygonVEMMesh2::PolygonVEMMesh2(const mtao::ColVecs2d &V,
                                 const std::vector<std::vector<int>> &polygons)
    : _polygons(polygons) {
    this->V = V;
    this->face_boundary_map.resize(polygons.size());

    mtao::reindex::CompressingReindexer<std::array<int, 2>> indexer;
    for (auto &&[poly_index, polygon, boundary_map] :
         mtao::iterator::enumerate(polygons, this->face_boundary_map)) {
        for (auto pr : mtao::iterator::cyclic_interval<2>(polygon)) {
            int a, b;
            std::tie(a, b) = pr;
            bool sign = a > b;
            if (sign) {
                std::swap(a, b);
            }

            int index = indexer.add(std::array<int, 2>{{a, b}});
            boundary_map[index] = sign;
        }
    }
    this->E = mtao::eigen::stl2eigen(indexer.unindex_vec());
    set_centroids_as_centers(*this);
}

bool PolygonVEMMesh2::in_cell(const mtao::Vec2d &p, int cell_index) const {
    return mtao::geometry::interior_winding_number(V, _polygons.at(cell_index),
                                                   p);
}

int PolygonVEMMesh2::get_cell(const mtao::Vec2d &p, int last_known) const {
    for (auto &&[idx, polygon] : mtao::iterator::enumerate(_polygons)) {
        if (mtao::geometry::interior_winding_number(V, polygon, p)) {
            return idx;
        }
    }
    return -1;
}

}  // namespace vem
