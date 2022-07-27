#include "vem/two/parent_maps.hpp"
namespace vem::two{

std::map<size_t, std::set<size_t>> vertex_faces(const VEMMesh2 &mesh) {
    std::map<size_t, std::set<size_t>> ret;
    for (auto &&[eidx, faces] : edge_faces(mesh)) {
        auto e = mesh.E.col(eidx);
        for (int j = 0; j < 2; ++j) {
            auto &myfaces = ret[e(j)];
            std::transform(faces.begin(), faces.end(), std::inserter(myfaces, myfaces.end()), [](auto &&v) { return v.first; });
            // myfaces.insert(faces.begin(), faces.end());
        }
    }
    return ret;
}

std::map<size_t, std::map<size_t, bool>> edge_faces(const VEMMesh2 &mesh) {
    std::map<size_t, std::map<size_t, bool>> ret;
    for (auto &&[fidx, mp] :
         mtao::iterator::enumerate(mesh.face_boundary_map)) {
        for (auto &&[eidx, sgn] : mp) {
            ret[eidx].emplace(fidx, sgn);
        }
    }
    return ret;
}
}// namespace vem::utils
