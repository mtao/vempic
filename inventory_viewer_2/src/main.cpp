#include <Corrade/Utility/Arguments.h>

#include <vem/two/visualize/inventory_viewer.hpp>

#include "imgui.h"
#include "mtao/opengl/Window.h"

class Viewer : public mtao::opengl::Window2 {
   public:
    Viewer(const Arguments &args);
    void gui() override;
    void draw() override { Window2::draw(); }

   private:
    std::shared_ptr<vem::serialization::Inventory> _inventory;
    std::shared_ptr<vem::two::visualize::InventoryViewer> _viewer;
};
Viewer::Viewer(const Arguments &args) : Window2(args) {
    Corrade::Utility::Arguments myargs;
    myargs.addArgument("inventory-path").parse(args.argc, args.argv);
    std::string path = myargs.value("inventory-path");

    _inventory = std::make_shared<vem::serialization::Inventory>(path);
    _inventory->unset_immediate_mode();
    _viewer = std::make_shared<vem::two::visualize::InventoryViewer>(
        _inventory, &root(), &drawables());

    _viewer->setParent(&root());

    // automatically_increment_recording_frames(false);
    set_recording_frame_callback([&](int index) -> bool {
        spdlog::info("Incrementing frame!");
        return _viewer->set_current_frame(index);
    });
}

void Viewer::gui() {
    if (_viewer) {
        _viewer->gui();
    }

    recording_gui();
}
MAGNUM_APPLICATION_MAIN(Viewer)
