#include <Magnum/EigenIntegration/Integration.h>
#include <mtao/opengl/Window.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <mtao/geometry/mesh/stack_meshes.hpp>
#include <mtao/opengl/shaders/polynomial_scalar_field.hpp>
#include <optional>
#include <vem/two/poisson/constraint_viewer.hpp>
#include <vem/two/poisson/poisson.hpp>
#include <vem/utils/boundary_facets.hpp>
#include <vem/two/visualize/vem_mesh_creation_gui.hpp>
#include <vem/two/visualize/vem_scalar_field_viewer.hpp>
#include <vem/two/wavesim/sim_viewer.hpp>
#include <mtao/python/load_python_function.hpp>

template <class T>
auto stuff_into_unique_ptr(T &&obj) {
    return std::make_unique<T>(obj);
}

class VemViewer2d : public mtao::opengl::Window2 {
   public:
    std::shared_ptr<const vem::two::VEMMesh2> mesh;
    std::optional<vem::two::wavesim::Sim> sim;
    std::optional<vem::two::wavesim::SimViewer> sim_viewer;

    // set of cells used for each region
    std::vector<std::set<int>> cell_regions;
    // our currently chosen region
    std::optional<int> active_cell_region_index;
    int system_degree = 1;
    float timestep = 0.02;

    VemViewer2d(const Arguments &args);
    void gui() override;
    void draw() override;
    void mouseMoveEvent(MouseMoveEvent &event) override;
    void set_pointwise_function(const std::function<double(double)> &f);

   private:
    Magnum::SceneGraph::DrawableGroup2D post_mesh_drawables;
    vem::two::visualize::VEMMesh2CreationGui mesh_gui;

    bool show_mesh_selection_window = true;
    bool show_sim_window = true;
    mtao::Vec2f cursor;
    const std::set<int> &active_cell_regions() const;

    void update_sim_from_mesh();
    void cell_regions_updated();
    void update_constraint_view();

    std::shared_ptr<mtao::python::PythonFunction> func;
   public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
VemViewer2d::VemViewer2d(const Arguments &args)
    : Window2(args), mesh_gui(&scene(), &post_mesh_drawables) {
    Eigen::initParallel();
    spdlog::info("Eigen is going to use {} threads", Eigen::nbThreads());
    // std::string a("../alligator_fixed.obj ");
    // std::copy(a.begin(), a.end(), mesh_filename);
    //// mesh_filename[a.size()+1] = '\0';
    // make_mandoline_mesh();

    mesh_gui.make_grid_mesh();
    mesh = mesh_gui.stored_mesh();
    update_sim_from_mesh();
}

void VemViewer2d::update_sim_from_mesh() {
    if (!mesh) {
        spdlog::info("Attempted to update sim with no mesh!");
        return;
    }
    sim.emplace(*mesh, system_degree);
    {
        auto p = sim->inventory.get_new_asset_path("mesh_info");
        std::ofstream(p) << mesh_gui.serialize_to_json();
    }
    sim->inventory.add_metadata("degree", system_degree);
    sim_viewer.emplace(*sim, &root(), &drawables());

                func = std::make_shared<mtao::python::PythonFunction>("");
    sim_viewer->func = func;
}
void VemViewer2d::draw() {
    // Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::DepthTest);
    // Magnum::GL::Renderer::disable(Magnum::GL::Renderer::Feature::FaceCulling);
    // Magnum::GL::Renderer::setPointSize(10);

    // camera().draw(background_drawgroup);
    // camera().draw(sim_vis.drawgroup);
    mtao::opengl::Window2::draw();
    camera().draw(post_mesh_drawables);
}
void VemViewer2d::mouseMoveEvent(MouseMoveEvent &event) {
    mtao::opengl::Window2::mouseMoveEvent(event);
    auto c = localPosition(event.position());
    cursor = Magnum::EigenIntegration::cast<mtao::Vec2f>(c);

    if (event.isAccepted()) {
        return;
    }

    // event.setAccepted();
}

void VemViewer2d::gui() {
    ImGui::Text("Cursor Position: (%f,%f)", cursor.x(), cursor.y());
    if (ImGui::InputInt("VEM poly degree", &system_degree)) {
        update_sim_from_mesh();
    }
    ImGui::InputFloat("Timestep", &timestep);

    ImGui::Checkbox("Show Mesh Selection Window", &show_mesh_selection_window);
    if (sim_viewer) {
        ImGui::Checkbox("Show Sim Window", &show_sim_window);
        if (show_sim_window) {
            ImGui::Begin("Sim Gui");
            sim_viewer->gui();
            ImGui::End();
        }
    }

    if (show_mesh_selection_window) {
        if (mesh_gui.gui()) {
            mesh = mesh_gui.stored_mesh();
            update_sim_from_mesh();
        }
    }
    recording_gui();
}

MAGNUM_APPLICATION_MAIN(VemViewer2d)
