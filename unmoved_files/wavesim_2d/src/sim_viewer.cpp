#include "vem/wavesim_2d/sim_viewer.hpp"

#include <misc/cpp/imgui_stdlib.h>
#include <omp.h>
#include <pybind11/eigen.h>

#include <fstream>
#include <vem/utils/loop_over_active.hpp>
namespace vem::wavesim_2d {

SimViewer::SimViewer(Sim &sim, mtao::opengl::Object2D *parent,
                     Magnum::SceneGraph::DrawableGroup2D *group)
    : mtao::opengl::Object2D{parent},
      Magnum::SceneGraph::Drawable2D{*this, group},
      _sim(sim),
      scalar_field_viewer(sim.mesh(),
                          std::make_shared<MonomialBasisIndexer>(
                              sim.poisson_vem.monomial_indexer()),
                          &drawables)
// constraint_viewer(scalar_field_viewer, &foreground_drawables)

{
    scalar_field_viewer.setParent(this);
    sample_mesh.setParent(this);
    boundary_mesh.setParent(this);
    // constraint_viewer.setParent(this);

    boundary_mesh_drawable = new mtao::opengl::MeshDrawable{
        boundary_mesh, flat_shader, foreground_drawables};
    boundary_mesh_drawable->data().color = Magnum::Color4(0, 0, 0, 1);
    boundary_mesh_drawable->deactivate();
    boundary_mesh_drawable->line_width = 5;
    cell_regions = sim.mesh().cell_regions();
    remake_sim();
    refresh_from_sim();
}

void SimViewer::gui() {
    ImGui::InputFloat("Timestep", &timestep);
    {
        if (ImGui::InputDouble("C", &_sim.c)) {
        }
    }
    static bool autostep = false;
    ImGui::Checkbox("Autostep", &autostep);
    if (ImGui::Button("Step") || autostep) {
        _sim.step(timestep);
        refresh_from_sim();
    }
    {
        if (active_cell_region_index) {
            if (ImGui::InputInt(fmt::format("Active region id ({} available)", cell_regions.size()).c_str(),
                                &*active_cell_region_index)) {
                active_cell_region_index = std::clamp<int>(
                    *active_cell_region_index, 0, std::max<int>(0,cell_regions.size() - 1));
                spdlog::info("Selecting region {} with {} cells", *active_cell_region_index, active_cell_regions().size());
                sim_modified_since_creation = true;
            }
            if (ImGui::Button("Use all regions")) {
                active_cell_region_index = {};
                sim_modified_since_creation = true;
            }
        } else {
            if (ImGui::Button("Select a cell region")) {
                active_cell_region_index = 0;
                sim_modified_since_creation = true;
            }
        }
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
    // if (constraint_viewer.gui(_sim.mesh())) {
    //    _sim.boundary_conditions = constraint_viewer;
    //}
    scalar_field_viewer.gui();

    ImGui::Begin("PythonGui");
    // ImGui::Text

    static std::string text =
        "import numpy as np\n"
        "import numpy.linalg\n"
        "from math import *\n"
        "def FUNC_NAME(x,t):\n"
        "  d = np.linalg.norm(x - np.array([.2 + .01*t,.5]))\n"
        "  v = (1-5*d)**3\n"
        "  return max(0,v)";

    ImGui::InputTextMultiline("Code", &text);
    if (ImGui::Button("Activate")) {
        auto p = _sim.inventory.get_new_asset_path("initial_conditions", "py");
        std::ofstream(p) << text << std::endl;

        try {
            if (func) {
                func->update_function(text);
            } else {
                func = std::make_shared<mtao::python::PythonFunction>(text);
            }
            update_initial_conditions();
        } catch (const std::exception &e) {
            spdlog::error(e.what());
        }
    }
    ImGui::End();
}

void SimViewer::update_initial_conditions() {
    spdlog::info("Updating initial conditions");
    if (!func) {
        return;
    }
    const auto &f = *func;

    try {
        int threads = omp_get_num_threads();
        omp_set_num_threads(1);
        _sim.pressure =
            _sim.poisson_vem.coefficients_from_point_sample_function(
                [&](const mtao::Vec2d &v) -> double {
                    return f(v, 0).cast<double>();
                });
        _sim.pressure_previous =
            _sim.poisson_vem.coefficients_from_point_sample_function(
                [&](const mtao::Vec2d &v) -> double {
                    return f(v, -timestep).cast<double>();
                });
        omp_set_num_threads(threads);

    } catch (const std::exception &e) {
        spdlog::error(e.what());
        _sim.pressure.setZero();
        _sim.pressure_previous.setZero();
    }

    refresh_from_sim();
}

void SimViewer::refresh_from_sim() {
    // TODO: reload boundary data

    scalar_field_viewer.set_degrees(
        mtao::eigen::stl2eigen(
            _sim.poisson_vem.monomial_indexer().cell_degrees())
            .cast<int>());
    scalar_field_viewer.set_scales(mtao::eigen::stl2eigen(
        _sim.poisson_vem.monomial_indexer().cell_diameters()));
    scalar_field_viewer.update_mesh_visualization();
    if (_sim.pressure.size() > 0) {
        spdlog::info("SimViewer: Updating from viewer");
        mtao::VecXd P = _sim.poisson_vem.sample_to_polynomial_projection_matrix(
                            active_cell_regions()) *
                        _sim.pressure;
        scalar_field_viewer.setPointValues(P.head(_sim.mesh().vertex_count()));
        scalar_field_viewer.set_coefficients(P);
    }
    scalar_field_viewer.active_cells = _sim.active_cells;
    if (boundary_mesh_drawable) {
        boundary_mesh_drawable->deactivate();
    }

    {
        boundary_mesh.setVertexBuffer(_sim.mesh().V.cast<float>().eval());
        auto edge_indices = _sim.mesh().boundary_edge_indices();
        if (edge_indices.size() > 0) {
            mtao::ColVectors<unsigned int, 2> E(2, edge_indices.size());
            E.setZero();
            const auto &CE = _sim.mesh().E;
            for (auto &&[a, b] : mtao::iterator::enumerate(edge_indices)) {
                E.col(a) = CE.col(b).cast<unsigned int>();
            }

            boundary_mesh.setEdgeBuffer(E);
            if (boundary_mesh_drawable) {
                boundary_mesh_drawable->activate_edges();
            }
        }
    }

    if (show_samples) {
        update_sample_vis();
    } else {
        if (sample_vector_drawable) {
            sample_vector_drawable->activate_points();
        }
        if (sample_position_drawable) {
            sample_position_drawable->activate_points();
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
    // sample_position_drawable->deactivate();
    {
        auto P = _sim.poisson_vem.point_sample_indexer()
                     .get_positions()
                     .cast<float>()
                     .eval();
        sample_mesh.setVertexBuffer(P);
        if (sample_position_drawable) {
            sample_position_drawable->activate_points();
        }
    }
}

void SimViewer::remake_sim() {
    _sim.active_cells = active_cell_regions();
    _sim.initialize();

    scalar_field_viewer.active_cells = _sim.active_cells;
    sim_modified_since_creation = false;
    refresh_from_sim();
}

const std::set<int> &SimViewer::active_cell_regions() const {
    const static std::set<int> empty = {};

    if (active_cell_region_index) {
        int index = *active_cell_region_index;
        if (index >= 0 && index < cell_regions.size()) {
            return cell_regions.at(index);
        }
    }

    return empty;
}

void SimViewer::draw(const Magnum::Matrix3 &transformationMatrix,
                     Magnum::SceneGraph::Camera2D &camera) {
    vf_shader.setTransformationProjectionMatrix(camera.projectionMatrix() *
                                                transformationMatrix);
    flat_shader.setTransformationProjectionMatrix(camera.projectionMatrix() *
                                                  transformationMatrix);
    camera.draw(drawables);
    camera.draw(foreground_drawables);
}
}  // namespace vem::wavesim_2d
