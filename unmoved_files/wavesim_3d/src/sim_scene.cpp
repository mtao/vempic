#include "vem/wavesim_3d//sim_scene.hpp"

#include <pybind11/eigen.h>

#include <fstream>
#include <vem/creator3.hpp>

#include "vem/serialization/serialize_mesh.hpp"

namespace {
std::string default_pressure_function =
    "import numpy as np\n"
    "import numpy.linalg\n"
    "from math import *\n"
    "center = np.array([.5,.8])\n"
    "radius = .3\n"
    "dir = np.array([0,0,-1])\n"
    "def FUNC_NAME(x, t):\n"
    "  # cubic spline of a vector pointing to the right\n"
    "  dist_to_center = np.linalg.norm(x - center + t * dir)\n"
    "  v = max((1 - dist_to_center / radius)**3, 0)\n"
    "  return v\n"
}

namespace vem::wavesim_3d {
void SimScene::set_default_function() {
    pressure_function = default_pressure_function;
    update_pressure_from_func();
}
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
        return;
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

void SimScene::create_sim(bool default_initialize) {
    if (!_mesh) {
        spdlog::error("Cannot make sim withotu a mesh");
        return;
    }
    spdlog::info("Creating scene");
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

    if (default_initialize) {
        _sim->initialize();
    }
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

void SimScene::update_pressure_from_func() {
    if (!_sim) {
        return;
    }
    if (!pressure) {
        return;
    }

    try {
        omp_set_num_threads(1);
        auto f = [&](const mtao::Vec3d& a, double t) -> mtao::Vec3d {
            return (*pressure)(a, t).cast<double>();
        };
        int threads = omp_get_num_threads();
        _sim->initialize(f);

        omp_set_num_threads(threads);

    } catch (const std::exception& e) {
        spdlog::error(e.what());
    }
}

}  // namespace vem::wavesim_3d
