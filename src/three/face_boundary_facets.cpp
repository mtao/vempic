
#include "vem/there/face_boundary_facets.hpp"

namespace vem::three {

std::set<size_t> face_boundary_vertices(const VEMMesh3& mesh, int face_index) {
    std::set<size_t> ret;
        auto l = mesh.face_loops(face_index);
        for (auto&& v : l) {
            ret.emplace(v);
        }
        for (auto&& l : l.holes) {
            for (auto&& v : l) {
                ret.emplace(v);
            }
        }
    return ret;
}

}  // namespace vem::utils
