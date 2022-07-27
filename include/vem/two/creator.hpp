#pragma once
#include <nlohmann/json.hpp>

#include "mesh.hpp"

namespace vem::two {
class VEMMesh2Creator {
   public:
    enum class MeshType { Grid, TriangleMesh, Cutmesh, None };
    static std::string MeshType2string(MeshType m);
    bool make_grid_mesh();
    bool make_obj_mesh();
    bool load_boundary_mesh();
    bool make_mandoline_mesh(bool load_boundary = true);

    std::shared_ptr<const VEMMesh2> stored_mesh() const { return _stored_mesh; }

    // follows the specification seen in proto/vem_creator.proto
    nlohmann::json serialize_to_json() const;
    void configure_from_json(const nlohmann::json &,
                             bool make_if_available = false);
    void configure_from_json_file(const std::string &filename,
                                  bool make_if_available = false);

    Eigen::AlignedBox2f grid_mesh_bbox =
        Eigen::AlignedBox2f(mtao::Vec2f::Constant(0), mtao::Vec2f::Constant(1));
    std::array<int, 2> grid_mesh_dimensions = std::array<int, 2>{{2, 2}};
    std::optional<std::tuple<mtao::ColVecs2d, mtao::ColVecs2i>>
        _held_boundary_mesh = {};
    std::shared_ptr<VEMMesh2> _stored_mesh = nullptr;
    std::optional<int> active_region_index;
    bool do_delaminate = false;
    MeshType last_made_mesh;
    std::string mesh_filename;
};
}  // namespace vem
