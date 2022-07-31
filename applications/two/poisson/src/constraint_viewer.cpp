#include "vem/two/poisson/constraint_viewer.hpp"
#include <vem/serialization/inventory.hpp>
#include <fstream>

#include <Magnum/GL/Renderer.h>
#include <fmt/format.h>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>
#include <pybind11/eigen.h>
#include <spdlog/spdlog.h>

#include "vem/two/poisson/example_constraints.hpp"

namespace vem::two::poisson {
ScalarConstraintsGui::ScalarConstraintsGui(
    const visualize::VEM2ScalarFieldViewer &viewer,
    Magnum::SceneGraph::DrawableGroup2D *group)
    : Magnum::SceneGraph::Drawable2D(*this, group), _viewer(viewer) {
    _vertex_mesh.setParent(this);
    _vertex_mesh.setPrimitive(Magnum::GL::MeshPrimitive::Points);
    _vertex_mesh.setCount(1);
    _vertex_mesh.addVertexBuffer(_vertex_mesh.vertex_buffer, 0,
                                 Magnum::Shaders::Flat2D::Position{});

    _edge_mesh.setParent(this);
    _edge_mesh.setPrimitive(Magnum::GL::MeshPrimitive::Lines);
    _edge_mesh.setCount(2);
    _edge_mesh.addVertexBuffer(_edge_mesh.vertex_buffer, 0,
                               Magnum::Shaders::Flat2D::Position{});

    boundary_condition_function =
        "import numpy as np\n"
        "import numpy.linalg\n"
        "from math import *\n"
        "def FUNC_NAME(x,t):\n"
        "  # set boundary to 0\n"
        "  # set boundary to 0\n"
        "  return np.array([0,0]),True\n";

    boundary_conditions = std::make_shared<mtao::python::PythonFunction>(
        boundary_condition_function);
}
void ScalarConstraintsGui::draw(const TransMat &transformationMatrix,
                                Camera &camera) {
    Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::DepthTest);
    _flat_shader.setTransformationProjectionMatrix(camera.projectionMatrix() *
                                                   transformationMatrix);
    using namespace Magnum::Math::Literals;
    if (_vertex_index >= 0) {
        Magnum::GL::Renderer::setPointSize(10);
        _flat_shader.setColor(0x000000_rgbaf);
        _flat_shader.draw(_vertex_mesh);
        Magnum::GL::Renderer::setPointSize(8);
        _flat_shader.setColor(_vertex_color);
        _flat_shader.draw(_vertex_mesh);
    }
    if (_edge_index >= 0) {
        Magnum::GL::Renderer::setLineWidth(10);
        _flat_shader.setColor(0x000000_rgbaf);
        _flat_shader.draw(_edge_mesh);

        Magnum::GL::Renderer::setLineWidth(8);
        _flat_shader.setColor(_edge_color);
        _flat_shader.draw(_edge_mesh);
    }
}
ScalarConstraintsGui &ScalarConstraintsGui::operator=(
    const ScalarConstraints &constraints) {
    ScalarConstraints::operator=(constraints);
    return *this;
}
void ScalarConstraintsGui::update_mesh(const VEMMesh2 &mesh) {
    //_vertex_index = std::clamp<int>(_vertex_index, 0, mesh.vertex_count() - 1;
    //_edge_index = std::clamp<int>(_edge_index, 0, mesh.edge_count() - 1);
    _vertex_index = -1;
    _edge_index = -1;
    clear();
}
void ScalarConstraintsGui::update_vertex(const VEMMesh2 &mesh) {
    spdlog::info("Setting vertex buffer");
    _vertex_mesh.setVertexBuffer(
        mesh.V.col(_vertex_index).cast<float>().eval());
    auto c = _viewer.get_color(_vertex_value);
    _vertex_color = Magnum::Color4(c.x(), c.y(), c.z(), c.w());
}
void ScalarConstraintsGui::update_edge(const VEMMesh2 &mesh) {
    mtao::Mat2f V;
    auto e = mesh.E.col(_edge_index);
    V.col(0) = mesh.V.col(e(0)).cast<float>();
    V.col(1) = mesh.V.col(e(1)).cast<float>();
    _edge_mesh.setVertexBuffer(V);
    auto c = _viewer.get_color(_edge_value);
    _edge_color = Magnum::Color4(c.x(), c.y(), c.z(), c.w());
}
void ScalarConstraintsGui::update_boundary_condition_from_func(
    const VEMMesh2 &m) {
    if (!boundary_conditions) {
        return;
    }
    try {
        *this = neumann_from_boundary_function(
            m,
            [&](const mtao::Vec2d &a,
                double t) -> std::tuple<mtao::Vec2d, double> {
                return (*boundary_conditions)(a, t)
                    .cast<std::tuple<mtao::Vec2d, bool>>();
            },
            0);
    } catch (const std::exception &e) {
        spdlog::error(e.what());
    }
}

bool ScalarConstraintsGui::gui(const VEMMesh2 &mesh) {
    auto constraint_gadget = [](std::map<size_t, double> &values,
                                size_t max_index, const std::string &format,
                                int &index, float &value) -> bool {
        bool ret = false;
        if (ImGui::InputInt("Index", &index)) {
            index = std::clamp<int>(index, 0, max_index);
            ret = true;
        }
        if (ImGui::InputFloat("Value", &value)) {
            ret = true;
        }
        if (ImGui::Button("Add Constraint")) {
            index = std::clamp<int>(index, 0, max_index);
            values[index] = value;
            ret = true;
        }
        if (ImGui::TreeNode("Constraint List")) {
            std::set<int> toremove;
            for (auto &&[index, value] : values) {
                if (ImGui::Button(fmt::vformat(format, fmt::make_format_args(index, value)).c_str())) {
                    toremove.emplace(index);
                }
            }
            for (auto &&v : toremove) {
                values.erase(v);
            }
            if (!toremove.empty()) {
                ret = true;
            }
            ImGui::TreePop();
        }
        return ret;
    };
    bool ret = false;

    if (ImGui::TreeNode("Example Constraints")) {
        if (ImGui::Button("Clear Constraints")) {
            clear();
        }

        if (ImGui::InputFloat("Constraint Polynomial Constant",
                              &poly_constant)) {
        }
        if (ImGui::InputFloat2("Constraint Polynomial Linear",
                               poly_linear.data())) {
        }

        if (ImGui::Button("Linear Dirichlet Boundaries")) {
            *this = linear_function_dirichlet(
                mesh, poly_constant, poly_linear.cast<double>());
            ret = true;
        }
        if (ImGui::Button("Linear Neumann")) {
            *this = linear_function_neumann(
                mesh, poly_constant, poly_linear.cast<double>());
            ret = true;
        }
        if (ImGui::Button("Zero Dirichlet Boundary")) {
            *this = linear_function_dirichlet(
                mesh, 0, mtao::Vec2d::Zero());
            ret = true;
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Dirichlet Pointwise Constraints")) {
        if (constraint_gadget(pointwise_dirichlet, mesh.vertex_count() - 1,
                              "Remove f({})={}", _vertex_index,
                              _vertex_value)) {
            ret = true;
            update_vertex(mesh);
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Neumann Flux Constraints")) {
        if (constraint_gadget(edge_integrated_flux_neumann,
                              mesh.edge_count() - 1, "Remove Int_e f({}) dx={}",
                              _edge_index, _edge_value)) {
            ret = true;
            update_edge(mesh);
        }
        ImGui::TreePop();
    }

    ImGui::Begin("Boundary Condition Function");
    // ImGui::Text

    ImGui::InputTextMultiline("Boundary Condition",
                              &boundary_condition_function);
    if (ImGui::Button("Activate boundary condition")) {
        if (inventory != nullptr) {
            auto p = inventory->get_new_asset_path("boundary_conditions", "py");
            std::ofstream(p) << boundary_condition_function << std::endl;
        }

        try {
            boundary_conditions->update_function(boundary_condition_function);
            update_boundary_condition_from_func(mesh);
            ret = true;
        } catch (const std::exception &e) {
            spdlog::error(e.what());
        }
    }
    ImGui::End();
    return ret;
}  // namespace vem::poisson_2d
}  // namespace vem::poisson_2d
