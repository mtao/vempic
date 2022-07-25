#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "vem/fluidsim_2d/sim.hpp"

namespace vem::fluidsim_2d {
/*
void Sim::save_state(
const std::string& frame_dir_format = "frame-{:4d}") const {
std::filesystem::path base_path(sim_storage_base_path);
std::filesystem::path frame_path =
    base_path / fmt::format(frame_dir_format, frame_index);

if (std::filesystem::is_directory(frame_path)) {
    spdlog::warn("Frame path was already a directory: [{}]",
                 std::string(frame_path));
} else {
    std::filesystem::create_directory(frame_path);
}

std::filesystem::path inventory_path = frame_path / "inventory.json";
bool inventory_existed = std::filesystem::is_regular_file(inventory_path);
std::ofstream ofs(inventory_path);
nlohmann::json inventory;
if (inventory_existed) {
    inventory["frame_number"] = frame_index;

    inventory["active_cells"] = active_cells;
}

std::ofstream ofs(frame_path / "inventory.json");
ofs << inventory;
}
*/
}  // namespace vem::fluidsim_2d
