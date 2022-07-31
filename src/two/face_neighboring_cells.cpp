
#include "vem/utils/face_neighboring_cells.hpp"

#include "vem/two/face_neighboring_cells.hpp"
namespace vem::two {
std::vector<std::set<int>> face_neighboring_cells(const VEMMesh2& mesh) {
    return utils::face_neighboring_cells(mesh.face_boundary_map,
                                         mesh.edge_count());
}
}  // namespace vem::two
