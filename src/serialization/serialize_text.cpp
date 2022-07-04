
#include "vem/serialization/serialize_text.hpp"
#include <fstream>
namespace vem::serialization {

void serialize_string(Inventory& inventory, const std::string& name,
        const std::string& file_data) {
            auto p = inventory.get_new_asset_path("boundary_conditions", "py");
            std::ofstream(p) << file_data;
}
void serialize_json(Inventory& inventory, const std::string& name,
        const nlohmann::json& js) {
            auto p = inventory.get_new_asset_path("boundary_conditions", "py");
            std::ofstream(p) << js;
}
}
