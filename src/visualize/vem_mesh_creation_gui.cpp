#include "vem/visualize/vem_mesh_creation_gui.hpp"

#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>

#include <filesystem>
#include <fstream>

namespace vem::visualize {
VEMMesh2CreationGui::VEMMesh2CreationGui(
    mtao::opengl::Object2D *parent, Magnum::SceneGraph::DrawableGroup2D *group)
    : mtao::opengl::Object2D{parent},
      Magnum::SceneGraph::Drawable2D{*this, group} {
    grid.setParent(this);
    _grid_preview_drawable =
        new mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>{
            grid, _flat_shader, post_mesh_drawables};

    _grid_preview_drawable->deactivate();
    update_grid_previewer();
}
void VEMMesh2CreationGui::draw(const Magnum::Matrix3 &transformationMatrix,
                               Magnum::SceneGraph::Camera2D &camera) {
    camera.draw(mesh_drawable);
    camera.draw(post_mesh_drawables);
}
bool VEMMesh2CreationGui::gui(const std::string &name_) {
    bool ret = false;
    std::string name = name_;
    if (name == "") {
        name = "VEMMesh2 Creation";
    }
    ImGui::Begin(name.c_str());
    ImGui::BulletText("Grid Mesh Creation");
    ImGui::Indent();
    {
        if (ImGui::Checkbox("Show Guide grid", &show_guide_grid)) {
            if (show_guide_grid) {
                _grid_preview_drawable->activate_edges();
            } else {
                _grid_preview_drawable->deactivate();
            }
        }
        if (ImGui::InputInt2("Grid Dimensions", grid_mesh_dimensions.begin())) {
            for (auto &&v : grid_mesh_dimensions) {
                v = std::max<int>(2, v);
            }
            update_grid_previewer();
        }
        if (ImGui::InputFloat2("Grid Min", grid_mesh_bbox.min().data())) {
            grid_mesh_bbox.min() =
                grid_mesh_bbox.min().cwiseMin(grid_mesh_bbox.max());
            update_grid_previewer();
        }
        if (ImGui::InputFloat2("Grid Max", grid_mesh_bbox.max().data())) {
            grid_mesh_bbox.max() =
                grid_mesh_bbox.max().cwiseMax(grid_mesh_bbox.min());
            update_grid_previewer();
        }
        if (ImGui::Button("Create from BBOx")) {
            ret |= make_grid_mesh();
        }
    }
    ImGui::Unindent();
    ImGui::BulletText("Grid Mesh Creation");
    ImGui::Indent();
    {
        ImGui::InputText("Mesh File Path", &mesh_filename);
        ImGui::SameLine();
        std::filesystem::path path(mesh_filename);
        if (std::filesystem::is_regular_file(path)) {
            ImGui::Text("File Exists");
        } else if (std::filesystem::exists(path.parent_path())) {
            ImGui::Text("Parent path exists");
        } else {
            ImGui::Text("File not Found");
        }
        if (ImGui::Button("Create from OBJ")) {
            ret |= make_obj_mesh();
        }
        if (ImGui::Button("Create CutcellMesh")) {
            ret |= make_mandoline_mesh();
        }
    }
    ImGui::Unindent();
    if (ImGui::Checkbox("Delaminate", &do_delaminate)) {
    }
    ImGui::End();
    return ret;
}

void VEMMesh2CreationGui::update_grid_previewer() {
    auto g = mtao::geometry::grid::Grid2f::from_bbox(grid_mesh_bbox,
                                                     grid_mesh_dimensions);
    grid.set(g);
}
}  // namespace vem::visualize
