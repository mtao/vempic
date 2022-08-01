#include "vem/two/fluidsim/sim_scene.hpp"

#if defined(VEM_USE_PYTHON)
#include <pybind11/eigen.h>
#endif

#include <fstream>

#include "vem/serialization/serialize_mesh.hpp"

namespace {
#if defined(VEM_USE_PYTHON)
std::string default_velocity_function =
    "import numpy as np\n"
    "import numpy.linalg\n"
    "from math import *\n"
    "center = np.array([.5,.8])\n"
    "radius = .3\n"
    "dir = np.array([0,-1])\n"
    "def FUNC_NAME(x):\n"
    "  # cubic spline of a vector pointing to the right\n"
    "  dist_to_center = np.linalg.norm(x - center)\n"
    "  v = max((1 - dist_to_center / radius)**3, 0)\n"
    "  return v * dir\n";
#endif
}


namespace vem::two::fluidsim {
void SimScene::set_mesh_settings(const nlohmann::json& js) {
    _mesh_creator.configure_from_json(js, true);

    spdlog::info("From set_mesh_settings i see {}",
                 _mesh_creator.mesh_filename);
    if (js.contains("active_region") &&
        js["active_region"].is_number_integer()) {
        active_cell_region_index = js["active_region"];

    } else {
        active_cell_region_index = {};
    }
}

void SimScene::create_mesh() {
    _mesh = _mesh_creator.stored_mesh();
    if (!_mesh) {
        spdlog::warn("Config did not specify the type of mesh to make");
    }
    spdlog::info("Saving mesh creation data");
    auto path = inventory->get_new_asset_path("mesh_info", "json");
    std::ofstream(path) << _mesh_creator.serialize_to_json();
    cell_regions = _mesh->cell_regions();

    spdlog::info("Mesh has {} regions:", cell_regions.size());
    for (auto&& [idx, cells] : mtao::iterator::enumerate(cell_regions)) {
        spdlog::info("Region {} has {} cells", idx, cells.size());
    }
    if (cell_regions.size() > 0) {
        active_cell_region_index = 0;
    } else {
        active_cell_region_index = {};
    }
    vem::serialization::serialize_mesh(*inventory, "mesh", *_mesh);
}

void SimScene::create_sim() {
    if (!_mesh) {
        spdlog::error("Cannot make sim withotu a mesh");
        return;
    }
    spdlog::info("Creating scene with max deg {}", max_degree);
    _sim = std::make_shared<Sim>(*_mesh, max_degree, inventory);
    { inventory->add_metadata("degree", max_degree); }

    _sim->set_active_cells(active_cell_regions());
    if (active_cell_region_index) {
        spdlog::info("Sim using active region index {} has {} cells",
                     *active_cell_region_index, _sim->active_cells().size());
        inventory->add_metadata("active_region", *active_cell_region_index);
    } else {
        spdlog::info("Sim using all cells");
        inventory->add_metadata("active_region", nullptr);
    }

    _sim->initialize_particles(desired_particle_count_per_cell *
                               _sim->active_cell_count());
    _sim->velocity.coefficients().setZero();
    _sim->initialize();
    spdlog::info("Done creating scene");
}

const std::set<int>& SimScene::active_cell_regions() const {
    const static std::set<int> empty = {};

    if (active_cell_region_index) {
        int index = *active_cell_region_index;
        if (index >= 0 && index < cell_regions.size()) {
            return cell_regions.at(index);
        }
    }

    return empty;
}

void SimScene::update_velocity_from_func() {
    if (!_sim) {
        return;
    }
#if defined(VEM_USE_PYTHON)
    if (!initial_velocity) {
        return;
    }
#endif

    try {
#if defined(VEM_USE_PYTHON)
        auto f = [&](const mtao::Vec2d& a) -> mtao::Vec2d {
            return (*initial_velocity)(a).cast<mtao::Vec2d>();
        };
#else
        auto f = [&](const mtao::Vec2d& a) -> mtao::Vec2d {
            return mtao::Vec2d::Zero();
        };
#endif
        int threads = omp_get_num_threads();
        omp_set_num_threads(1);
        _sim->initialize(f);
        _sim->initialize_particles(
            desired_particle_count_per_cell * _sim->active_cell_count(), f);

        omp_set_num_threads(threads);

        //_sim->particle_velocities_to_field(_sim->mesh().dx());
        //_sim->update_velocity_using_samples();
        // update_sample_vis();
        // update_particle_vis();

    } catch (const std::exception& e) {
        spdlog::error(e.what());
    }
}

}  // namespace vem::fluidsim_2d
