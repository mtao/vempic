#include "vem/two/fluidsim/sim_viewer.hpp"

#include <misc/cpp/imgui_stdlib.h>

#include <fstream>
#include <vem/two/poisson/example_constraints.hpp>
namespace vem::two::fluidsim {

SimViewer::SimViewer(mtao::opengl::Object2D* parent,
                     Magnum::SceneGraph::DrawableGroup2D* group)
    : mtao::opengl::Object2D{parent},
      Magnum::SceneGraph::Drawable2D{*this, group},
      mesh_gui(this, &foreground_drawables),

      scalar_field_viewer(&drawables),
      constraint_viewer(scalar_field_viewer, &foreground_drawables)

{
    scalar_field_viewer.setParent(this);
    particle_mesh.setParent(this);
    sample_mesh.setParent(this);
    boundary_mesh.setParent(this);
    constraint_viewer.setParent(this);

    boundary_mesh_drawable = new mtao::opengl::MeshDrawable{
        boundary_mesh, flat_shader, foreground_drawables};
    boundary_mesh_drawable->data().color = Magnum::Color4(0, 0, 0, 1);
    boundary_mesh_drawable->deactivate();
    boundary_mesh_drawable->line_width = 5;
    spdlog::info("Creating mesh first time ereally");

    _mesh_creator.make_grid_mesh();
    create_mesh();
    scalar_field_viewer.set_mesh(_mesh);
    remake_sim();
    update_velocity_from_func();
}

void SimViewer::update_active_region() {
    _sim->set_active_cells(active_cell_regions());
    scalar_field_viewer.active_cells = active_cell_regions();
}

#if defined(VEM_USE_PYTHON)
void SimViewer::set_initial_velocity(const std::string& str) {
    velocity_function = str;
    initial_velocity =
        std::make_shared<mtao::python::PythonFunction>(velocity_function);
}
#endif

void SimViewer::gui() {
    ImGui::Checkbox("Show Mesh Selection Window", &show_mesh_selection_window);
    if (ImGui::InputInt("Simulation degree", &max_degree)) {
        remake_sim();
    }
    if (show_mesh_selection_window) {
        if (mesh_gui.gui()) {
            set_mesh_settings(mesh_gui.serialize_to_json());
            create_mesh();
            scalar_field_viewer.set_mesh(_mesh);
            remake_sim();
        }
    }
    static bool advect_particle_mode = false;
    ImGui::Checkbox("Just advect particles", &advect_particle_mode);
    ImGui::InputFloat("Timestep", &timestep);
    static bool autostep = false;
    ImGui::Checkbox("Autostep", &autostep);
    if (ImGui::Button("Step") || autostep) {
        if (advect_particle_mode) {
            _sim->advect_particles_with_field(timestep);
            _sim->set_particle_velocities_from_grid();
        } else {
            _sim->step(timestep);
        }
        refresh_from_sim();
    }
    {
        if (active_cell_region_index) {
            if (ImGui::InputInt("Active region id",
                                &*active_cell_region_index)) {
                active_cell_region_index = std::clamp<int>(
                    *active_cell_region_index, 0, cell_regions.size() - 1);
                update_active_region();
                sim_modified_since_creation = true;
            }
            if (ImGui::Button("Use all regions")) {
                active_cell_region_index = {};
                update_active_region();
                sim_modified_since_creation = true;
            }
        } else {
            if (ImGui::Button("Select a cell region")) {
                active_cell_region_index = 0;
                update_active_region();
                sim_modified_since_creation = true;
            }
        }
    }
    if (ImGui::InputInt("Number of Particles Per Cell",
                        &desired_particle_count_per_cell)) {
        desired_particle_count_per_cell =
            std::max(desired_particle_count_per_cell, 0);
        sim_modified_since_creation = true;
    }
    if (ImGui::Button("Reinitialize Particles")) {
        _sim->reinitialize_particles(desired_particle_count_per_cell *
                                     _sim->active_cell_count());
        update_particle_vis();
    }
    {
        std::string base("Recreate Sim");
        if (sim_modified_since_creation) {
            base += "[Changes Waiting]";
        }
        if (ImGui::Button(base.c_str())) {
            remake_sim();
        }
    }
    if (ImGui::Checkbox("Assume static system", &_sim->static_domain)) {
    }
    if (constraint_viewer.gui(_sim->mesh())) {
        //_sim->boundary_conditions = constraint_viewer;
        // spdlog::info("Constraint size: {} {}",
        // _sim->boundary_conditions.size(),
        //             constraint_viewer.size());
    }
    scalar_field_viewer.gui();

    {
        // ImGui::TreeNode("Particle/sample vis");
        if (ImGui::Checkbox("Show particles", &show_particles)) {
            update_particle_vis();
        }
        if (ImGui::Checkbox("Show samples", &show_samples)) {
            update_sample_vis();
        }
        if (particle_vector_drawable) {
            particle_vector_drawable->gui("Particle Vector");
        }
        if (particle_position_drawable) {
            particle_position_drawable->gui("Particle Position");
        }
        // ImGui::TreePop();
    }

#if defined(VEM_USE_PYTHON)
    ImGui::Begin("PythonGui");
    // ImGui::Text

    ImGui::InputTextMultiline("Initial Velocity", &velocity_function);
    if (ImGui::Button("Activate Velocity")) {
        auto p = inventory->get_new_asset_path("initial_conditions", "py");
        std::ofstream(p) << velocity_function << std::endl;

        try {
            std::cout << "Trying function: \n"
                      << velocity_function << std::endl;
            initial_velocity->update_function(velocity_function);
            update_velocity_from_func();
        } catch (const std::exception& e) {
            spdlog::error(e.what());
        }
    }
    ImGui::End();
#endif
}

void SimViewer::refresh_from_sim() {
    // TODO: reload boundary data

    scalar_field_viewer.set_monomial_indexer(
        std::make_shared<MonomialBasisIndexer>(
            _sim->pressure_indexer().monomial_indexer()));
    scalar_field_viewer.update_mesh_visualization();
    if (_sim->pressure.size() > 0) {
        // auto E =
        // _sim->poisson_vem.polynomial_to_sample_evaluation_matrix(
        //    poisson_2d::PoissonVEM2::CellWeightWeightMode::AreaWeighted,
        //    _sim->active_cells);
        mtao::VecXd
            P = /*_sim->poisson_vem.sample_to_polynomial_projection_matrix(
                    active_cell_regions()) **/
            _sim->pressure;
        // std::cout << P.transpose() << std::endl;
        // scalar_field_viewer.setPointValues(P.head(_sim->mesh().vertex_count()));
        scalar_field_viewer.set_coefficients(P);
    }
    scalar_field_viewer.active_cells = _sim->active_cells();
    if (boundary_mesh_drawable) {
        boundary_mesh_drawable->deactivate();
    }

    {
        boundary_mesh.setVertexBuffer(_sim->mesh().V.cast<float>().eval());
        auto edge_indices =
            _sim->velocity.boundary_intersector().boundary_edge_indices();
        if (edge_indices.size() > 0) {
            mtao::ColVectors<unsigned int, 2> E(2, edge_indices.size());
            E.setZero();
            const auto& CE = _sim->mesh().E;
            if (E.maxCoeff() >= _sim->mesh().V.cols()) {
                spdlog::error("Coeff range {},{}, but have {} vertices",
                              E.minCoeff(), E.maxCoeff(),
                              _sim->mesh().V.cols());
                exit(1);
            }
            for (auto&& [a, b] : mtao::iterator::enumerate(edge_indices)) {
                E.col(a) = CE.col(b).cast<unsigned int>();
            }

            boundary_mesh.setEdgeBuffer(E);
            if (boundary_mesh_drawable) {
                boundary_mesh_drawable->activate_edges();
            }
        }
    }
    if (show_particles) {
        update_particle_vis();
    } else {
        if (particle_vector_drawable) {
            particle_vector_drawable->deactivate();
        }
        if (particle_position_drawable) {
            particle_position_drawable->deactivate();
        }
    }

    if (show_samples) {
        update_sample_vis();
    } else {
        if (sample_vector_drawable) {
            sample_vector_drawable->deactivate();
        }
        if (sample_position_drawable) {
            sample_position_drawable->deactivate();
        }
    }
}
void SimViewer::update_particle_vis() {
    if (particle_vector_drawable == nullptr) {
        particle_vector_drawable =
            new mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>>{
                particle_mesh, vf_shader, foreground_drawables};
        particle_vector_drawable->deactivate();
    }
    // particle_vector_drawable->deactivate();

    if (particle_position_drawable == nullptr) {
        particle_position_drawable =
            new mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>{
                particle_mesh, flat_shader, foreground_drawables};
        particle_position_drawable->deactivate();
        particle_position_drawable->point_size = 5;
    }
    // particle_position_drawable->deactivate();
    if (_sim->num_particles() > 0 && show_particles) {
        auto P = _sim->particle_positions().cast<float>().eval();
        auto V = _sim->particle_velocities().cast<float>().eval();
        particle_mesh.setVertexBuffer(P);
        particle_mesh.setVFieldBuffer(V);
        if (particle_vector_drawable) {
            particle_vector_drawable->activate_points();
        }
        if (particle_position_drawable) {
            particle_position_drawable->activate_points();
        }
    } else {
        if (particle_vector_drawable) {
            particle_vector_drawable->deactivate();
        }
        if (particle_position_drawable) {
            particle_position_drawable->deactivate();
        }
    }
}
void SimViewer::update_sample_vis() {
    if (sample_vector_drawable == nullptr) {
        sample_vector_drawable =
            new mtao::opengl::MeshDrawable<mtao::opengl::VectorFieldShader<2>>{
                sample_mesh, vf_shader, foreground_drawables};
        sample_vector_drawable->deactivate();
    }
    // sample_vector_drawable->deactivate();

    if (sample_position_drawable == nullptr) {
        sample_position_drawable =
            new mtao::opengl::MeshDrawable<Magnum::Shaders::Flat2D>{
                sample_mesh, flat_shader, foreground_drawables};
        sample_position_drawable->deactivate();
        sample_position_drawable->point_size = 5;
    }

#if !defined(VEM_FLUX_MOMENT_FLUID)
    if (show_samples) {
        auto P = _sim->velocity_indexer()
                     .point_sample_indexer()
                     .get_positions()
                     .cast<float>()
                     .eval();
        auto V = _sim->sample_velocities.cast<float>().eval();
        // spdlog::info("Velocities being uploaded with norm {}", V.norm());
        sample_mesh.setVertexBuffer(P);
        sample_mesh.setVFieldBuffer(V);
        if (sample_vector_drawable) {
            sample_vector_drawable->activate_points();
        }
        if (sample_position_drawable) {
            sample_position_drawable->activate_points();
        }
    } else {
        if (sample_vector_drawable) {
            sample_vector_drawable->deactivate();
        }
        if (sample_position_drawable) {
            sample_position_drawable->deactivate();
        }
    }
#endif
    // sample_position_drawable->deactivate();
}

void SimViewer::remake_sim() {
    if (!_mesh) {
        spdlog::error("No mesh set!");
    }
    // desired_particle_count = 10 * _mesh->cell_count();
    create_sim();

    sim_modified_since_creation = false;
    refresh_from_sim();
}

void SimViewer::draw(const Magnum::Matrix3& transformationMatrix,
                     Magnum::SceneGraph::Camera2D& camera) {
    vf_shader.setTransformationProjectionMatrix(camera.projectionMatrix() *
                                                transformationMatrix);
    flat_shader.setTransformationProjectionMatrix(camera.projectionMatrix() *
                                                  transformationMatrix);
    camera.draw(drawables);
    camera.draw(foreground_drawables);
}

}  // namespace vem::fluidsim_2d
