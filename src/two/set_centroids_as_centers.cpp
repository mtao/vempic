#include "vem/two/set_centroids_as_centers.hpp"
#include <mtao/geometry/centroid.hpp>
#include <mtao/iterator/enumerate.hpp>

namespace vem::two {

void set_centroids_as_centers(VEMMesh2 &vem) {
    vem.C.resize(2, vem.cell_count());
    for (auto &&[cell_index, cell] :
         mtao::iterator::enumerate(vem.face_boundary_map)) {
        vem.C.col(cell_index) = mtao::geometry::centroid(vem.V, vem.E, cell);
    }
}
}// namespace vem
