#pragma once
#include <Magnum/GL/Renderer.h>
#include <mtao/opengl/drawables.h>
#include <mtao/opengl/objects/mesh.h>

#include "sim.h"
class SimVis : public mtao::opengl::Object2D {
   public:
    std::weak_ptr<Sim> sim;
    mtao::opengl::objects::Mesh<2> particles;
    mtao::opengl::objects::Mesh<2> vector_field;
    mtao::opengl::Drawable<Magnum::Shaders::VertexColor2D>* particle_drawable =
        nullptr;
    mtao::opengl::Drawable<Magnum::Shaders::VertexColor2D>*
        vector_field_drawable = nullptr;
    Magnum::Shaders::VertexColor2D vcolor_shader;
    Magnum::SceneGraph::DrawableGroup2D drawgroup;

    float spectral_field_a = 5;
    float spectral_field_b = 5;

    // streamer setup
    float streamer_timestep = .1;
    int streamer_count = 5;
    int particle_count = 1000;
    bool draw_arrows = false;
    int subdivs = 10;
    double radius = .1;

    // simulation timestep
    float timestep = 0.02;
    bool autostep = false;

    SimVis();
    void set_sim(std::shared_ptr<Sim> sim = nullptr);
    void setSceneRooot(mtao::opengl::Object2D& obj);
    void gui();
    void update();
    void clear();

    void update_colors(const VEMMesh2ScalarField& field);
    void update_colors_divergence();
    void update_colors_pressure();

    void set_velocity_field(
        const std::function<mtao::Vec2d(const mtao::Vec2d&)>& func);
    void set_pressure_gradient();
    void set_velocity();
    void set_spectral_velocity_field();
    void set_rotation_velocity_field();
    void set_yyx_velocity_field();
};
