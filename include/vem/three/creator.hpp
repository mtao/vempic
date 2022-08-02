
#pragma once
#include <nlohmann/json.hpp>

#include "mesh.hpp"

namespace vem::three {
class VEMMesh3Creator {
   public:
    enum class MeshType { Grid, TriangleMesh, Cutmesh, None };
    static std::string MeshType2string(MeshType m);
    bool make_grid_mesh();
    bool load_boundary_mesh();
    bool make_mandoline_mesh(bool load_boundary = true);

    std::shared_ptr<const VEMMesh3> stored_mesh() const { return _stored_mesh; }

    nlohmann::json serialize_to_json() const;
    void configure_from_json(const nlohmann::json &,
                             bool make_if_available = false);
    void configure_from_json_file(const std::string &filename,
                                  bool make_if_available = false);

    Eigen::AlignedBox3f grid_mesh_bbox =
        Eigen::AlignedBox3f(mtao::Vec3f::Constant(0), mtao::Vec3f::Constant(1));
    std::array<int, 3> grid_mesh_dimensions = std::array<int, 3>{{3,3, 3}};
    std::optional<std::tuple<mtao::ColVecs3d,mtao::ColVecs3i>> _held_boundary_mesh = {};
    int adaptive_grid_level = 0;
    std::shared_ptr<VEMMesh3> _stored_mesh = nullptr;
    MeshType last_made_mesh;
    std::string mesh_filename;
};
}  // namespace vem
