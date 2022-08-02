#pragma once
#include <cxxopts.hpp>
#include <memory>
#if defined(VEM_USE_PYTHON)
#include <mtao/python/load_python_function.hpp>
#endif
#include <vem/three/creator.hpp>
#include <vem/three/fluidsim/sim.hpp>
#include <vem/three/mesh.hpp>
#include <vem/utils/inside_geometry_predicate.hpp>
#include "vem/three/in_triangle_mesh.hpp"


namespace vem::three::fluidsim {
// you can set the mesh, set sim parameters, set boundary conditions, choose
// which cells to use, and finally run the simulation
struct SimScene {
    void set_mesh_settings(const nlohmann::json& mesh);
    VEMMesh3Creator _mesh_creator;

    static cxxopts::OptionAdder& add_options(cxxopts::Options& opts);
    void load_options(const cxxopts::ParseResult& result);
    void load_config(const nlohmann::json& js, int frame_index = 0,
                     int resolution = 0);

    // every time a mesh is created we will dump it to disk
    void create_mesh();
    const std::set<int>& active_cell_regions() const;
    void create_sim(bool default_initialize = true);
    void update_velocity_from_func();

    std::shared_ptr<const VEMMesh3> _mesh;
    std::shared_ptr<Sim> _sim;

    int desired_particle_count_per_cell = 20;
    int max_degree = 1;
    float timestep = .05f;
    bool advect_only = false;
    int frame_start = 0;
    int frame_end = 250;

    std::optional<std::string> initial_particle_state_file;
    std::unique_ptr<utils::InsideGeometryPredicate> _in_geo_pred;
    void update_density_from_predicate();

    void step(int index);
    void run();

    void set_default_function();
    // set of cells used for each region
    std::vector<std::set<int>> cell_regions;
    // our currently chosen region
    std::optional<int> active_cell_region_index;
    std::shared_ptr<serialization::Inventory> inventory =
        std::make_shared<serialization::Inventory>(
            serialization::Inventory::from_scratch("3d_fluidsim"));

    std::string name;
    // SimScene(const std::string& inv_name = "3d_fluidsim");
    SimScene(const cxxopts::ParseResult& result);
    SimScene(const nlohmann::json& js);
    // maps (point) -> (vector)
#if defined(VEM_USE_PYTHON)
    std::shared_ptr<mtao::python::PythonFunction> initial_velocity;
    std::string velocity_function;
#endif
};
}  // namespace vem::fluidsim_3d
