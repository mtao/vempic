#pragma once
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/Object.h>
#include <mtao/opengl/drawables.h>
#include <mtao/opengl/objects/types.h>

#include <mtao/opengl/shaders/vector_field.hpp>
#include <vem/two/mesh.hpp>
#include <vem/two/poisson/constraint_viewer.hpp>
#include <vem/two/visualize/vem_scalar_field_viewer.hpp>
#include <vem/two/wavesim/sim.hpp>

#include <mtao/python/load_python_function.hpp>

namespace vem::two::wavesim {

class SimViewer : public mtao::opengl::Object2D,
                  Magnum::SceneGraph::Drawable2D {
   public:
    SimViewer(Sim &sim, mtao::opengl::Object2D *parent = nullptr,
              Magnum::SceneGraph::DrawableGroup2D *group = nullptr);

    // when the sim object is changed we have to reload everything
    void refresh_from_sim();

    void draw(const Magnum::Matrix3 &transformationMatrix,
              Magnum::SceneGraph::Camera2D &camera) override;
    const std::set<int> &active_cell_regions() const;

    void gui();
    void remake_sim();
    void update_sample_vis();

    std::shared_ptr<mtao::python::PythonFunction> func;
   private:
    Sim &_sim;

    // shaders
    mtao::opengl::VectorFieldShader<2> vf_shader;
    Magnum::Shaders::Flat2D flat_shader;

    Magnum::SceneGraph::DrawableGroup2D drawables;
    Magnum::SceneGraph::DrawableGroup2D foreground_drawables;

    // render the resulting pressure field
    visualize::VEM2ScalarFieldViewer scalar_field_viewer;
    // vem::poisson_2d::ScalarConstraintsGui constraint_viewer;

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
        *sample_vector_drawable = nullptr;
    mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>
        *sample_position_drawable = nullptr;
    mtao::opengl::objects::Mesh<2> sample_mesh;

    // set of cells used for each region
    std::vector<std::set<int>> cell_regions;
    // our currently chosen region
    std::optional<int> active_cell_region_index;

    bool sim_modified_since_creation = true;
    float timestep = .1f;

    bool show_samples = true;
    void update_initial_conditions();
};
}  // namespace vem::wavesim_2d
