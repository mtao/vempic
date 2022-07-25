#include <Magnum/EigenIntegration/Integration.h>
#include <mtao/opengl/Window.h>
#include <nlohmann/json.hpp>
#include <Corrade/Utility/Arguments.h>
#include <fstream>
#include <spdlog/spdlog.h>

#include <mtao/geometry/mesh/stack_meshes.hpp>
#include <mtao/opengl/shaders/polynomial_scalar_field.hpp>
#include <optional>
#include <vem/fluidsim_2d/sim_viewer.hpp>
#include <vem/utils/boundary_facets.hpp>
#include <vem/visualize/vem_scalar_field_viewer.hpp>

template <class T>
auto stuff_into_unique_ptr(T &&obj) {
    return std::make_unique<T>(obj);
}

class VemViewer2d : public mtao::opengl::Window2 {
   public:
    vem::fluidsim_2d::SimViewer sim_viewer;

    VemViewer2d(const Arguments &args);
    void gui() override;
    void mouseMoveEvent(MouseMoveEvent &event) override;
    void set_pointwise_function(const std::function<double(double)> &f);

    mtao::Vec2f cursor;

   private:
};
VemViewer2d::VemViewer2d(const Arguments &args)
    : Window2(args), sim_viewer(&root(), &drawables()) {
    auto ss_path = sim_viewer.inventory->real_path() / "screenshots";
    if (!std::filesystem::exists(ss_path)) {
        std::filesystem::create_directory(ss_path);
    }

    Corrade::Utility::Arguments myargs;
    myargs.addOption("base-inventory", "")
        .addOption("mesh-info", "")
        .parse(args.argc, args.argv);

    std::string bi = myargs.value("base-inventory");
    std::string mi = myargs.value("mesh-info");

    if (!bi.empty()) {
        try {
            const vem::serialization::Inventory invent(bi,nullptr,true,false);
            {
                spdlog::info("Loading mesh info");
                auto mipath = invent.get_asset_path("mesh_info");
                nlohmann::json js;
                std::ifstream(mipath) >> js;
                sim_viewer.set_mesh_settings(js);
                sim_viewer.mesh_gui.configure_from_json(js,true);
                sim_viewer.max_degree = invent.metadata()["degree"];

                spdlog::info("Loading initial conditions");
                std::ifstream in(invent.get_asset_path("initial_conditions"));
                std::stringstream buffer;
                buffer << in.rdbuf();
                sim_viewer.set_initial_velocity( buffer.str());
                sim_viewer.remake_sim();
            }
        } catch (const std::exception &e) {
            spdlog::error("Loading base inventory error: {}", e.what());
        }
    }
    if (!mi.empty()) {
        try {
            nlohmann::json js;
            std::ifstream(mi) >> js;
            sim_viewer.set_mesh_settings(js);
        } catch (const std::exception &e) {
            spdlog::error("Loading mesh info error: {}", e.what());
        }
    }

    set_recording_path(ss_path);
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
    if (ImGui::Button("Center camera")) {
        // if (mesh) {
        //    mtao::Vec2f c = mesh->bounding_box().center().cast<float>();
        //    Window2::setTranslation({c.x(), c.y()});
        //}
    }

    ImGui::Begin("Sim Gui");
    sim_viewer.gui();
    ImGui::End();

    WindowBase::recording_gui();
}

MAGNUM_APPLICATION_MAIN(VemViewer2d)
