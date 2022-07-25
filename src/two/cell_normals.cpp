#include "vem/cell_normals.hpp"
#include "vem/normals.hpp"
namespace vem {
mtao::Vec2d normal(const VEMMesh2 &mesh, size_t cell_index, size_t edge_index) {
    bool sign = mesh.face_boundary_map.at(cell_index).at(edge_index);
    auto R = normal(mesh, edge_index);
    if (sign) {
        R *= -1;
    }
    return R;
}
}// namespace vem
