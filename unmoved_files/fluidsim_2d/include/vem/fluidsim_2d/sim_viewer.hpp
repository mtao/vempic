#pragma once
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>
#include <mtao/opengl/drawables.h>
#include <mtao/opengl/objects/types.h>

#include <mtao/opengl/shaders/vector_field.hpp>
#if defined(VEM_USE_PYTHON)
#include <mtao/python/load_python_function.hpp>
#endif
#include <vem/poisson_2d/constraint_viewer.hpp>
#include <vem/visualize/vem_mesh_creation_gui.hpp>
#include <vem/visualize/vem_scalar_field_viewer.hpp>

#include "vem/fluidsim_2d/sim_scene.hpp"

namespace vem::fluidsim_2d {

class SimViewer : public mtao::opengl::Object2D,
                  Magnum::SceneGraph::Drawable2D,
                  public SimScene {
   public:
    SimViewer(mtao::opengl::Object2D *parent = nullptr,
              Magnum::SceneGraph::DrawableGroup2D *group = nullptr);

    // when the sim object is changed we have to reload everything
    void refresh_from_sim();

    void draw(const Magnum::Matrix3 &transformationMatrix,
              Magnum::SceneGraph::Camera2D &camera) override;

    void gui();
    void update_particle_vis();
    void update_sample_vis();
    void remake_sim();

#if defined(VEM_USE_PYTHON)
    void set_initial_velocity(const std::string& str);
#endif
    void update_active_region();

    

   //private:
    // shaders
    mtao::opengl::VectorFieldShader<2> vf_shader;
    Magnum::Shaders::Flat2D flat_shader;

    Magnum::SceneGraph::DrawableGroup2D drawables;
    Magnum::SceneGraph::DrawableGroup2D foreground_drawables;

    vem::visualize::VEMMesh2CreationGui mesh_gui;
    // render the resulting pressure field
    vem::visualize::VEM2ScalarFieldViewer scalar_field_viewer;
    vem::poisson_2d::ScalarConstraintsGui constraint_viewer;

    // Data for rendering the domain boundary
    mtao::opengl::objects::Mesh<2> boundary_mesh;
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>
        *boundary_mesh_drawable = nullptr;

    //  data for rendering the vertex-valued vector field
    mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>>
        *vfield_drawable = nullptr;
    mtao::opengl::objects::Mesh<2> vfield_mesh;

    //  data for rendering the vertex-valued vector field
    mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>>
        *particle_vector_drawable = nullptr;
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>
        *particle_position_drawable = nullptr;
    mtao::opengl::objects::Mesh<2> particle_mesh;

    //  data for rendering the vertex-valued vector field
    mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>>
        *sample_vector_drawable = nullptr;
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>
        *sample_position_drawable = nullptr;
    mtao::opengl::objects::Mesh<2> sample_mesh;

    bool sim_modified_since_creation = true;

    bool show_particles = true;
    bool show_samples = false;

    bool show_mesh_selection_window = true;

};
}  // namespace vem::fluidsim_2d
