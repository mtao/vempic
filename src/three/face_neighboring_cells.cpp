
#include "vem/utils/face_neighboring_cells.hpp"

#include "vem/three/face_neighboring_cells.hpp"
namespace vem::three {

std::vector<std::set<int>> face_neighboring_cells(const VEMMesh3& mesh) {
    return utils::face_neighboring_cells(mesh.cell_boundary_map,
                                         mesh.face_count());
}
}  // namespace vem::three
