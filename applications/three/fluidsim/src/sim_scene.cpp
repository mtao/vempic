#include "vem/three/fluidsim/sim_scene.hpp"

#include <mtao/logging/json_sink.hpp>
#include <mtao/logging/stopwatch.hpp>

#if defined(VEM_USE_PYTHON)
#include <pybind11/eigen.h>
#endif

#include <fstream>
#include <mtao/eigen/stack.hpp>
#include <mtao/geometry/mesh/read_obj.hpp>
#include <mtao/geometry/point_cloud/partio_loader_impl.hpp>
#include <mtao/json/bounding_box.hpp>

#include "vem/serialization/inventory.hpp"
#include "vem/serialization/serialize_mesh.hpp"

namespace {

#if defined(VEM_USE_PYTHON)
std::string default_velocity_function =
    "import numpy as np\n"
    "import numpy.linalg\n"
    "from math import *\n"
    "center = np.array([.5,.8])\n"
    "radius = .3\n"
    "dir = np.array([0,0,-1])\n"
    "def FUNC_NAME(x):\n"
    "  # cubic spline of a vector pointing to the right\n"
    "  dist_to_center = np.linalg.norm(x - center)\n"
    "  v = max((1 - dist_to_center / radius)**3, 0)\n"
    "  return v * dir\n";
#endif
}  // namespace

namespace vem::three::fluidsim {

cxxopts::OptionAdder &SimScene::add_options(cxxopts::Options &opts) {
    serialization::Inventory::add_options(opts);

    // clang-format off
        return opts.add_options("Sim Scene")
            ("a,advect_only", "only perform advection")
            ("i,initial_field", "location of initial field used (mostly useful for advection)", cxxopts::value<std::string>())
            ("c,config", "configuration json file", cxxopts::value<std::string>())
            ("resolution", "Resolution of the cut-cell mesh, overwrites the config", cxxopts::value<int>()->default_value("0"))
            ("particles", "Set initial particle state", cxxopts::value<std::string>())
            ("frame_index", "Set initial frame index", cxxopts::value<int>()->default_value("0"))
            ("d,degree", "polynomial degree of VEM used", cxxopts::value<int>())
            ("s,samples", "number of samples in each cell", cxxopts::value<int>());

    // clang-format on
}
void SimScene::load_config(const nlohmann::json &js, int start_frame_index,
                           int resolution) {
    spdlog::info("Load config time");
    frame_start = js["frame_start"];
    frame_end = js["frame_end"];

    std::cout << js << std::endl;
    if (resolution == 0) {
        resolution = js["domain_resolution"];
    }
    spdlog::info("RESOLUTINO: {}", resolution);
    auto bbox = js["bounding_box"].get<Eigen::AlignedBox<double, 3>>();
    auto &creator = _mesh_creator;
    // creator.mesh_filename = argv[1];

    frame_start = std::max(frame_start, start_frame_index);

    if (js.contains("active_region") &&
        js["active_region"].is_number_integer()) {
        active_cell_region_index = js["active_region"];

    } else {
        active_cell_region_index = {};
    }
    if (js.contains("name")) {
        std::string name =
            fmt::format("{}_{}", std::string(js["name"]), resolution);
        this->name = name;

        inventory = std::make_shared<serialization::Inventory>(
            serialization::Inventory::from_scratch(name));
    }
    if (js.contains("desired_particle_count_per_cell")) {
        desired_particle_count_per_cell = js["desired_particle_count_per_cell"];
    }

    if (js.contains("timestep")) {
        timestep = js["timestep"];
    }
    bool force_cutmesh = js.contains("cutmesh");
    if (force_cutmesh) {
        spdlog::error("Forcing cutmesh");
        creator.mesh_filename = js["cutmesh"];
        spdlog::info("Make mandoline mesh");
        //creator.make_mandoline_mesh();
    } else {
        creator.load_boundary_mesh();

        creator.grid_mesh_bbox = bbox.cast<float>();
        creator.grid_mesh_dimensions[0] = resolution;
        creator.grid_mesh_dimensions[1] = resolution;
        creator.grid_mesh_dimensions[2] = resolution;

        if (js.contains("adaptive_grid_level")) {
            creator.adaptive_grid_level = js["adaptive_grid_level"];
        }
        bool use_mandoline = js.contains("collision_mesh");
        if (use_mandoline) {
            creator.mesh_filename = js["collision_mesh"];
            spdlog::info("Make gmandoline mesh");
            //creator.make_mandoline_mesh();
        } else {
            spdlog::info("Make grid mesh");
            creator.make_grid_mesh();
        }
    }

    if (js.contains("flow_mesh")) {
        spdlog::info("Loading initial density field");
        Eigen::MatrixXd VV;
        Eigen::MatrixXi FF;
        auto [V, F] = mtao::geometry::mesh::read_objD(js["flow_mesh"]);
        VV = V.transpose();
        FF = F.transpose();

        _in_geo_pred = std::make_unique<InTriangleMesh>(std::move(VV),
                                                               std::move(FF));
        update_density_from_predicate();
        spdlog::info("Done Loading initial density field");
    }
}
void SimScene::load_options(const cxxopts::ParseResult &result) {
    std::string js_path = result["config"].as<std::string>();
    nlohmann::json js;
    std::ifstream(js_path) >> js;
    int resolution = 0;

    int res_dims = result["resolution"].as<int>();
    if (res_dims > 0) {
        resolution = res_dims;
    }

    int start_frame_index = result["frame_index"].as<int>();
    if (!js.contains("name")) {
        name = js["name"];
        inventory = std::make_shared<serialization::Inventory>(
            serialization::Inventory::from_options(result));
    }
    load_config(js, start_frame_index, resolution);
}

void SimScene::update_density_from_predicate() {
    if (!_sim) {
        spdlog::warn(
            "Cannot update velocity from function without a function!");
        return;
    }

    if (!_in_geo_pred) {
        spdlog::warn(
            "Cannot update velocity from function without a function!");
        return;
    }
    _sim->set_particle_densities_from_func(
        [&](const mtao::Vec3d &p) -> bool { return (*_in_geo_pred)(p); });
}

void SimScene::set_default_function() {
#if defined(VEM_USE_PYTHON)
    velocity_function = default_velocity_function;
#endif
    update_velocity_from_func();
}

SimScene::SimScene(const nlohmann::json &result) : inventory(nullptr) {
    //: inventory(std::make_shared<serialization::Inventory>(
    //      serialization::Inventory::from_options(result))) {
    load_config(result);
}
SimScene::SimScene(const cxxopts::ParseResult &result) : inventory(nullptr) {
    //: inventory(std::make_shared<serialization::Inventory>(
    //      serialization::Inventory::from_options(result))) {
    load_options(result);
}
void SimScene::set_mesh_settings(const nlohmann::json &js) {
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
    if (bool(inventory)) {
        auto path = inventory->get_new_asset_path("mesh_info", "json");
        std::ofstream(path) << _mesh_creator.serialize_to_json();
    } else {
        spdlog::info("No inventory for storing mesh info!");
    }
    cell_regions = _mesh->cell_regions();

    spdlog::info("Mesh has {} regions:", cell_regions.size());
    for (auto &&[idx, cells] : mtao::iterator::enumerate(cell_regions)) {
        spdlog::info("Region {} has {} cells", idx, cells.size());
    }
    // if (cell_regions.size() > 0) {
    //    active_cell_region_index = 0;
    //} else {
    //    active_cell_region_index = {};
    //}
    vem::serialization::serialize_mesh(*inventory, "mesh", *_mesh);
}

void SimScene::create_sim(bool default_initialize) {
    if (!_mesh) {
        spdlog::error("Cannot make sim withotu a mesh");
        return;
    }
    spdlog::info("Creating scene with {} regions available",
                 cell_regions.size());
    _sim = std::make_shared<Sim>(*_mesh, max_degree, active_cell_regions(),
                                 inventory);
    if (bool(inventory)) {
        { inventory->add_metadata("degree", max_degree); }
    }
    spdlog::info("Done making base sim object");

    if (active_cell_region_index) {
        spdlog::info("Sim using active region index {} has {} cells",
                     *active_cell_region_index, _sim->active_cells().size());
        if (bool(inventory)) {
            inventory->add_metadata("active_region", *active_cell_region_index);
        }
    } else {
        spdlog::info("Sim using all cells");
        inventory->add_metadata("active_region", nullptr);
    }

    _sim->velocity.coefficients().setZero();

    if (initial_particle_state_file) {
        const std::string &filename = *initial_particle_state_file;

        mtao::geometry::point_cloud::PartioFileReader r(filename);

        _sim->particles = mtao::eigen::vstack(r.positions(), r.velocities());
        _sim->particle_density = r.attribute<float>("densities").cast<double>();
    } else {
        int target_particles =
            desired_particle_count_per_cell * _sim->active_cell_count();
        spdlog::info("Initializing {} particles", target_particles);
        _sim->initialize_particles(target_particles);
    }
    // if (default_initialize) {
    //    _sim->initialize();
    //}
    //
    if (_in_geo_pred) {
        _sim->set_particle_densities_from_func(
            [&](const mtao::Vec3d &p) -> bool { return (*_in_geo_pred)(p); });
    }
    spdlog::info("Done creating scene");
}

const std::set<int> &SimScene::active_cell_regions() const {
    const static std::set<int> empty = {};

    if (active_cell_region_index) {
        int index = *active_cell_region_index;
        if (index >= 0 && index < cell_regions.size()) {
            spdlog::info("fetching cells for active region {}", index);
            return cell_regions.at(index);
        }
    } else {
        spdlog::warn("No active cell region index set");
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
        auto f = [&](const mtao::Vec3d &a) -> mtao::Vec3d {
            return (*initial_velocity)(a).cast<mtao::Vec3d>();
        };
#else
        auto f = [&](const mtao::Vec3d &v) -> mtao::Vec3d {
            return mtao::Vec3d::Zero();
        };
#endif
        int threads = omp_get_num_threads();
        omp_set_num_threads(1);
        _sim->initialize_mesh(f);
        _sim->initialize_particles(
            desired_particle_count_per_cell * _sim->active_cell_count(), f);

        omp_set_num_threads(threads);

        //_sim->particle_velocities_to_field(_sim->mesh().dx());
        //_sim->update_velocity_using_samples();
        // update_sample_vis();
        // update_particle_vis();

    } catch (const std::exception &e) {
        spdlog::error(e.what());
    }
}

void SimScene::step(int index) {
    if (!_sim) {
        return;
    }
    if (advect_only) {
        _sim->advect_particles_with_field(.02);
    } else {
        if (index == frame_start) {
            update_velocity_from_func();
        }
        _sim->step(timestep);
    }
}
void SimScene::run() {
    for (int frame_index = frame_start; frame_index < frame_end;
         ++frame_index) {
        spdlog::info("Working on frame {}, stopping at {}", frame_index,
                     frame_end);
        step(frame_index);
    }
}

}  // namespace vem::fluidsim_3d
