#pragma once
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>
#include <mtao/opengl/drawables.h>
#include <mtao/opengl/objects/types.h>

#include <mtao/opengl/shaders/vector_field.hpp>
#include <set>
#include <vem/visualize/vem_scalar_field_viewer.hpp>

#include "vem/serialization/frame_inventory.hpp"
#include "vem/serialization/inventory.hpp"
#include "vem/visualize/inventory_viewer.hpp"

namespace vem::visualize {

class InventoryViewer : public mtao::opengl::Object2D,
                        Magnum::SceneGraph::Drawable2D {
   public:
    InventoryViewer(const std::shared_ptr<serialization::Inventory> &inventory,
                    mtao::opengl::Object2D *parent = nullptr,
                    Magnum::SceneGraph::DrawableGroup2D *group = nullptr);
    ~InventoryViewer();

    void draw(const Magnum::Matrix3 &transformationMatrix,
              Magnum::SceneGraph::Camera2D &camera) override;

    void gui();
    // returns true if new data was found
    bool reload();

    // every item in is an associated pair {"name": "expected_type"} specifies a
    // potential visualization asset
    const nlohmann::json &visualization_manifest() const;

    void load_visualizations() const;

    bool load_current_frame();
    bool load_frame(int index);
    bool set_current_frame(int index);
    void update_vis();
    void clear_vis();

    void create_mesh();

   private:
    // shaders
    mtao::opengl::VectorFieldShader<2> vf_shader;
    Magnum::Shaders::Flat2D flat_shader;
    Magnum::Shaders::VertexColor2D vertex_color_shader;

    Magnum::SceneGraph::DrawableGroup2D drawables;
    Magnum::SceneGraph::DrawableGroup2D foreground_drawables;
    Magnum::SceneGraph::DrawableGroup2D background_drawables;
    bool draw_background = true;

    std::shared_ptr<serialization::Inventory> _inventory;
    std::vector<std::unique_ptr<const serialization::FrameInventory>>
        _frame_inventories;

    std::shared_ptr<const VEMMesh2> _mesh;
    int frame_index = 0;
    std::shared_ptr<vem::MonomialBasisIndexer> monomial_indexer;
    std::shared_ptr<vem::MonomialBasisIndexer> monomial_indexer_up;

    std::map<std::string, std::vector<std::string>> assets_by_type;

    // mesh boundary
    mtao::opengl::objects::Mesh<2> boundary_mesh;
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>
        *boundary_drawable = nullptr;

    // actual vis
    mtao::opengl::objects::Mesh<2> point_sample_mesh;
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>
        *point_sample_node_drawable = nullptr;
    void update_point();
    std::map<std::string, mtao::ColVecs2f> point_data;
    std::optional<int> current_point_index = {};
    const std::string &current_point() const;

    void update_point_density();
    mtao::opengl::MeshDrawable<Magnum::Shaders::VertexColor2D>
        *point_density_node_drawable = nullptr;
    mtao::opengl::objects::Mesh<2> point_density_mesh;
    std::map<std::string, mtao::ColVecs3f> point_density_data;
    std::optional<int> current_point_density_index = {};
    const std::string &current_point_density() const;

    mtao::opengl::objects::Mesh<2> point_vector_mesh;
    mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>>
        *point_vector_drawable = nullptr;
    // vis setup
    void update_point_vector();
    // vis data
    std::map<std::string, mtao::ColVecs4f> point_vector_data;
    // vis selection
    std::optional<int> current_point_vector_index = {};
    const std::string &current_point_vector() const;

    // actual vis
    vem::visualize::VEM2ScalarFieldViewer scalar_field_viewer;
    // vis setup
    void update_scalar_field();
    // vis data
    std::map<std::string, mtao::VecXf> scalar_field_data;
    // vis selection
    std::optional<int> current_scalar_field_index = {};
    const std::string &current_scalar_field() const;
    std::set<std::string> increased_degree_fields;

    template <typename DataMap>
    auto update_data(const std::string &current, const DataMap &data) const
        -> std::optional<typename DataMap::mapped_type>;
    const std::string &get_asset_name_from_index(
        const std::string &name, const std::optional<size_t> &index) const;
};
}  // namespace vem::visualize
