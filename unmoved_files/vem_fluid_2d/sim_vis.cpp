#include "sim_vis.hpp"

#include <igl/parula.h>

#include "imgui.h"
#include "particle_streamers.hpp"

SimVis::SimVis() {
    particle_drawable =
        new mtao::opengl::Drawable<Magnum::Shaders::VertexColor2D>{
            particles, vcolor_shader, drawgroup};
    vector_field_drawable =
        new mtao::opengl::Drawable<Magnum::Shaders::VertexColor2D>{
            vector_field, vcolor_shader, drawgroup};
    particles.setParent(this);
    vector_field.setParent(this);
    clear();
}
void SimVis::set_sim(std::shared_ptr<Sim> sim) { this->sim = sim; }
void SimVis::gui() {
    if (auto sim = this->sim.lock()) {
        if (ImGui::Button("Step")) {
            sim->step(timestep);
            update();
        }
        if (ImGui::Checkbox("Autostep", &autostep)) {
            update();
        }
        if (ImGui::InputFloat("Timestep", &timestep)) {
        }
        if (ImGui::InputInt("Particle count", &particle_count)) {
            sim->reinitialize_particles(particle_count);
            update();
        }
        if (ImGui::Button("Show velocity")) {
            set_velocity();
        }
        if (ImGui::Button("Show divergence")) {
            sim->update_velocity_divergence();
            update_colors_divergence();
        }
        if (ImGui::Button("Show pressure")) {
            sim->update_pressure();
            update_colors_pressure();
        }
        if (ImGui::Button("Show pressure_gradient")) {
            sim->update_pressure_gradient();
            set_pressure_gradient();
        }
        if (ImGui::Button("Reset Particles")) {
            sim->reinitialize_particles(particle_count);
            update();
        }
        if (ImGui::Button("Reset Particles w/Vel [cos(ax),sin(by)]")) {
            set_spectral_velocity_field();
            update();
        }
        if (ImGui::Button("Reset Particles w/Vel [-y,x]")) {
            sim->initialize_particles(particle_count,
                                      [](const mtao::Vec2d& p) -> mtao::Vec2d {
                                          return mtao::Vec2d(-p.y(), p.x());
                                      });
            set_rotation_velocity_field();
            update();
        }
        if (ImGui::Button("Reset Particles w/Vel [-y*y,x]")) {
            set_yyx_velocity_field();
            update();
        }
        if (ImGui::TreeNode("Streamers")) {
            if (ImGui::InputFloat("Streamer timestep", &streamer_timestep)) {
                update();
            }
            if (ImGui::InputInt("Streamer steps", &streamer_count)) {
                update();
            }

            if (ImGui::InputDouble("streamer radius", &radius)) {
                update();
            }
            // if (ImGui::Checkbox("Draw Arrows", &draw_arrows)) {
            //    update_particles_with_top();
            //}
            // if (ImGui::Button("Save streamers")) {
            //    ParticleStreamerBase<2,VEMMesh2VectorField>
            //    ps(sim->velocity_field());
            //    ps.create(sim->particle_positions(), streamer_timestep,
            //    streamer_count); auto [V, C, F] = ps.volumetric_output(radius,
            //    subdivs); mtao::geometry::mesh::write_ply(V, C, F,
            //    "streamer.ply");
            //}
            ImGui::TreePop();
        }

    } else {
        ImGui::Text("No Sim object attached to SimVis");
    }
}
/*
void SimVis::update_particles_with_top() {
    if (auto sim = this->sim.lock()) {
        ParticleStreamerBase<VEMMesh2VectorField> ps(sim->velocity_field);
        ps.create(sim->P(), streamer_timestep, streamer_count);
        if (draw_arrows) {
            auto [V, C, F] = ps.volumetric_output(radius, subdivs);
            auto C2 = mtao::eigen::vstack(C.cast<float>(),
                                          mtao::RowVecXf::Ones(C.cols()))
                          .eval();
            particle_streamers.setTriangleBuffer(V.cast<float>(),
                                                 F.cast<unsigned int>());
            particle_streamers.setColorBuffer(C2);
            if (streamer_drawable) {
                streamer_drawable->deactivate();
                streamer_drawable->activate_triangles();
            }
        } else {
            auto V = ps.vertices().cast<float>().eval();
            auto E = ps.edges().cast<unsigned int>().eval();
            auto C = ps.colors4().cast<float>().eval();
            particle_streamers.setEdgeBuffer(V, E);
            particle_streamers.setColorBuffer(C);
            if (streamer_drawable) {
                streamer_drawable->deactivate();
                streamer_drawable->activate_edges();
            }
        }
    }
}
*/

void SimVis::update_colors(const VEMMesh2ScalarField& field) {
    if (auto sim = this->sim.lock()) {
        mtao::ColVecs4d C(4, sim->num_particles());
        mtao::VecXd div(sim->num_particles());
        mtao::ColVecs2d p = sim->particle_positions();
        for (int i = 0; i < sim->num_particles(); ++i) {
            div(i) = field(p.col(i));
        }
        Eigen::MatrixXd cols;
        igl::parula(div, true, cols);
        // std::cout << cols << std::endl;

        C.topRows<3>() = cols.transpose();
        // C.setConstant(0);
        // C.row(2).setConstant(1);
        C.row(3).setConstant(1);

        // for (int i = 0; i < C.cols(); ++i) {
        //    C.col(i) =
        //        mtao::Vec3d::Constant(.5 + sim->particle_velocity(i).norm());
        //}

        particles.setColorBuffer(C.cast<float>().eval());
        particles.setVertexBuffer(p.cast<float>().eval());
        particle_drawable->activate_points();
    }
}
void SimVis::update_colors_divergence() {
    if (auto sim = this->sim.lock()) {
        auto dfield = sim->velocity_divergence_field();
        update_colors(dfield);
    }
}
void SimVis::update_colors_pressure() {
    if (auto sim = this->sim.lock()) {
        auto dfield = sim->pressure_field();
        update_colors(dfield);
    }
}
void SimVis::update() {
    if (auto sim = this->sim.lock()) {
        mtao::ColVecs3d C(3, sim->num_particles());
        for (int i = 0; i < C.cols(); ++i) {
            C.col(i) =
                mtao::Vec3d::Constant(.5 + sim->particle_velocity(i).norm());
        }

        particles.setColorBuffer(C.cast<float>().eval());
        particles.setVertexBuffer(
            sim->particle_positions().cast<float>().eval());
        particle_drawable->activate_points();
    }
    set_velocity();
}
void SimVis::set_velocity() {
    if (auto sim = this->sim.lock()) {
        ParticleStreamerBase<2, VEMMesh2VectorField> ps(sim->velocity_field());
        ps.create(sim->particle_positions(), streamer_timestep, streamer_count);
        auto V = ps.vertices().cast<float>().eval();
        auto E = ps.edges().cast<unsigned int>().eval();
        auto C = ps.colors4().cast<float>().eval();
        vector_field.setEdgeBuffer(V, E);
        vector_field.setColorBuffer(C);
        vector_field_drawable->activate_edges();
    }
}
void SimVis::set_pressure_gradient() {
    if (auto sim = this->sim.lock()) {
        ParticleStreamerBase<2, VEMMesh2VectorField> ps(
            sim->pressure_gradient_field());
        ps.create(sim->particle_positions(), streamer_timestep, streamer_count);

        {
            auto V = ps.vertices().cast<float>().eval();
            auto E = ps.edges().cast<unsigned int>().eval();
            auto C = ps.colors4().cast<float>().eval();
            vector_field.setEdgeBuffer(V, E);
            vector_field.setColorBuffer(C);
            vector_field_drawable->activate_edges();
        }
    }
}

void SimVis::clear() {
    particle_drawable->deactivate();
    vector_field_drawable->deactivate();
}
void SimVis::set_velocity_field(
    const std::function<mtao::Vec2d(const mtao::Vec2d&)>& func) {
    if (auto sim = this->sim.lock()) {
        sim->initialize_particles(particle_count, func);
        sim->update_particle_cell_cache();
        sim->particle_velocities_to_field();
    }
}
void SimVis::set_spectral_velocity_field() {
    set_velocity_field([&](const mtao::Vec2d& p) -> mtao::Vec2d {
        return mtao::Vec2d(std::cos(spectral_field_a * p.x()),
                           std::sin(spectral_field_b * p.y()));
    });
}
void SimVis::set_rotation_velocity_field() {
    set_velocity_field([](const mtao::Vec2d& p) -> mtao::Vec2d {
        return mtao::Vec2d(-p.y(), p.x());
    });
}
void SimVis::set_yyx_velocity_field() {
    set_velocity_field([](const mtao::Vec2d& p) -> mtao::Vec2d {
        return mtao::Vec2d(-p.y() * p.y(), p.x());
    });
}
