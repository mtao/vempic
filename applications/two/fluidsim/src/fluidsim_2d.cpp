#include <fstream>
#include <iostream>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/bounding_box.hpp>
#include <vem/two/fluidsim/sim_scene.hpp>

//#include "vem/from_mandoline3.hpp"

int main(int argc, char* argv[]) {
    std::string bi = argv[1];
    const vem::serialization::Inventory invent(bi, nullptr, true, false);
    vem::two::fluidsim::SimScene scene;
    {
        spdlog::info("Loading mesh info");
        auto mipath = invent.get_asset_path("mesh_info");
        nlohmann::json js;
        std::ifstream(mipath) >> js;
        scene.set_mesh_settings(js);
        scene.max_degree = invent.metadata()["degree"];
    }
    auto& creator = scene._mesh_creator;
    // creator.mesh_filename = argv[1];

    creator.load_boundary_mesh();
    scene.create_mesh();
    scene.create_sim();

    auto ss_path = scene.inventory->real_path() / "screenshots";
    if (!std::filesystem::exists(ss_path)) {
        std::filesystem::create_directory(ss_path);
    }

    if (!scene._sim) {
        spdlog::error("Sim object was not created");
        return 1;
    }
    vem::two::fluidsim::Sim& sim = *scene._sim;
    // std::cout << sim.sample_laplacian() << std::endl;
    // return 0;

    sim.static_domain = false;

    int num_particles = sim.num_particles();
    int num_frames = 500;
    bool just_advect = false;
    if (just_advect) {
        sim.update_polynomial_velocity();
    }

    auto root = sim.inventory->real_path();

    for (int j = 0; j < num_frames; ++j) {
        spdlog::info("Working on frame {} of {}", j, num_frames);

        if (just_advect) {
            sim.advect_particles_with_field(.02);
        } else {
            sim.step(.02);
        }
    }
    return 0;
}
