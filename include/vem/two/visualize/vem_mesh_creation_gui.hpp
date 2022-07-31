#pragma once
#include <mtao/opengl/objects/grid.h>
#include <mtao/opengl/objects/types.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>
#include <mtao/opengl/drawables.h>

#include "vem/two/creator.hpp"


namespace vem::two::visualize {
class VEMMesh2CreationGui : public mtao::opengl::Object2D,
                            Magnum::SceneGraph::Drawable2D, public VEMMesh2Creator {
   public:
    VEMMesh2CreationGui(mtao::opengl::Object2D *parent = nullptr,
                        Magnum::SceneGraph::DrawableGroup2D *group = nullptr);
    // return strue when the underlying mesh object was modified
    bool gui(const std::string &name = "");

    void draw(const Magnum::Matrix3 &transformationMatrix,
              Magnum::SceneGraph::Camera2D &camera) override;
    void update_grid_previewer();


   private:
    Magnum::SceneGraph::DrawableGroup2D mesh_drawable;
    Magnum::SceneGraph::DrawableGroup2D post_mesh_drawables;

    bool show_guide_grid = false;
    mtao::opengl::objects::Grid<2> grid;
    Magnum::Shaders::Flat2D _flat_shader;
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>
        *_grid_preview_drawable = nullptr;

};
}  // namespace vem::visualize
